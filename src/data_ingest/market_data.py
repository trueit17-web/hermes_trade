"""Сбор рыночных данных с бирж через ccxt (REST + WebSocket планирование)."""
import asyncio
import logging
import random

import ccxt.async_support as ccxt
import pandas as pd

from src.config import settings
from src.utils.ccxt_helpers import ccxt_symbol

logger = logging.getLogger(__name__)

# Паузы между повторами при rate-limit биржи на загрузке свечей (секунды,
# плюс случайные до +50%, чтобы повторы разных символов не совпадали).
_RATE_LIMIT_RETRY_DELAYS = (1.0, 2.5, 5.0)


class MarketDataIngest:
    """
    Инжектор рыночных данных:
    - Загрузка исторических OHLCV (для инициализации)
    - Периодическое обновление свечей
    - Буферизация последних N свечей для стратегий

    self.exchange — СПОТОВЫЙ клиент, используется для динамической торговой
    вселенной (get_tradable_symbols — явно "активные SPOT-пары") и для
    свечей большинства символов (у которых спотовый и фьючерсный рынок на
    Bybit делят один и тот же тикер, так что разница на практике не видна).
    self.futures_exchange — ОТДЕЛЬНЫЙ, лениво подключаемый linear-swap
    клиент — заводится только при первом запросе свечей для символа с
    market_type="futures" (см. fetch_ohlcv). Разделены по тому же принципу,
    что и ExecutionEngine._exchanges в executor.py: реальный инцидент —
    TAO/USDT на Bybit вообще не имеет спотового листинга, и раньше
    единственный (всегда спотовый) клиент валился с "does not have market
    symbol TAO/USDT" на каждой попытке обновить свечи для реально открытой
    фьючерсной позиции; менять self.exchange целиком на swap сломало бы
    get_tradable_symbols, который именно спотовый листинг и ожидает.
    """

    def __init__(self, exchange_id: str = "binance"):
        self.exchange_id = exchange_id.lower()
        self.exchange: ccxt.Exchange | None = None
        self.futures_exchange: ccxt.Exchange | None = None
        self.candles_buffer: dict[str, pd.DataFrame] = {}
        self._running = False
        # Ленивое подключение фьючерсного клиента идёт через await
        # load_markets() (секунды) — без блокировки параллельные вызовы
        # (основной цикл + /chart/candles с дашборда сразу после рестарта)
        # создавали каждый свой клиент, последний перезаписывал предыдущий,
        # и тот уходил в GC незакрытым: "Unclosed client session" в логах.
        self._futures_lock = asyncio.Lock()

    def _build_exchange(self, futures: bool) -> ccxt.Exchange:
        options = {"defaultType": "swap", "defaultSubType": "linear"} if futures else {"defaultType": "spot"}
        if self.exchange_id in ("binance", "bybit"):
            return getattr(ccxt, self.exchange_id)({"enableRateLimit": True, "options": options})
        return getattr(ccxt, self.exchange_id)({"enableRateLimit": True})

    @staticmethod
    async def _close_quietly(exchange: ccxt.Exchange | None) -> None:
        if exchange is None:
            return
        try:
            await exchange.close()
        except Exception as e:
            logger.debug(f"Не удалось закрыть клиент биржи: {e}")

    async def initialize(self):
        """Инициализация подключения к бирже (спотовый клиент)."""
        exchange = None
        try:
            exchange = self._build_exchange(futures=False)
            await exchange.load_markets()
            self.exchange = exchange
            logger.info(f"[{self.exchange_id}] Подключено, рынки загружены")
        except Exception as e:
            logger.error(f"Ошибка инициализации {self.exchange_id}: {e}")
            await self._close_quietly(exchange)
            self.exchange = None

    async def _ensure_futures_exchange(self) -> ccxt.Exchange | None:
        """Лениво подключить отдельный linear-swap клиент — только для
        символов, у которых реально открыта фьючерсная позиция (см.
        комментарий класса)."""
        if self.futures_exchange is not None:
            return self.futures_exchange
        async with self._futures_lock:
            if self.futures_exchange is not None:
                return self.futures_exchange
            exchange = None
            try:
                exchange = self._build_exchange(futures=True)
                await exchange.load_markets()
                self.futures_exchange = exchange
                logger.info(f"[{self.exchange_id}] Фьючерсный market-data клиент подключён")
            except Exception as e:
                logger.error(f"Ошибка инициализации фьючерсного market-data клиента {self.exchange_id}: {e}")
                await self._close_quietly(exchange)
                self.futures_exchange = None
        return self.futures_exchange

    async def close(self):
        """Закрыть соединение(я)."""
        if self.exchange:
            await self.exchange.close()
            logger.info(f"[{self.exchange_id}] Соединение закрыто")
        if self.futures_exchange:
            await self.futures_exchange.close()
            logger.info(f"[{self.exchange_id}] Фьючерсное соединение закрыто")

    async def _fetch_ohlcv_with_retry(self, exchange, request_symbol, timeframe, limit, since):
        """
        Bybit отвечает "Too many visits" (10006 -> ccxt.RateLimitExceeded)
        в моменты закрытия свечей, когда за ними одновременно идут все
        клиенты биржи: из 131 такой ошибки на проде 84 пришлись на минуту
        :00, остальные — на :15/:30/:45, при нашем темпе ~1 запрос в
        несколько секунд. Раньше символ просто пропускал итерацию (цена и
        проверка SL/TP по нему — устаревшие). Короткий повтор с нарастающей
        паузой почти всегда проходит.
        """
        for attempt in range(len(_RATE_LIMIT_RETRY_DELAYS) + 1):
            try:
                return await exchange.fetch_ohlcv(request_symbol, timeframe=timeframe, limit=limit, since=since)
            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection):
                if attempt == len(_RATE_LIMIT_RETRY_DELAYS):
                    raise
                delay = _RATE_LIMIT_RETRY_DELAYS[attempt] * (1 + random.random() * 0.5)
                logger.debug(
                    f"[{self.exchange_id}] rate-limit на свечах {request_symbol} {timeframe}, "
                    f"повтор через {delay:.1f}с"
                )
                await asyncio.sleep(delay)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: int | None = None,
        market_type: str = "spot",
    ) -> pd.DataFrame | None:
        """
        Загрузить OHLCV свечи из биржи.
        Возвращает DataFrame с индексом timestamp и колонками: open, high, low, close, volume

        market_type="futures" — использовать отдельный linear-swap клиент
        (см. _ensure_futures_exchange) и перевести символ в unified-формат
        "BASE/QUOTE:QUOTE", который ccxt ожидает для этого рынка (см.
        ccxt_symbol) — без этого символы без спотового листинга (TAO/USDT)
        не резолвятся вообще, а символы с совпадающим спотовым тикером
        молча резолвились бы в спотовый рынок.
        """
        if market_type == "futures":
            exchange = await self._ensure_futures_exchange()
        else:
            exchange = self.exchange

        if not exchange:
            logger.warning(f"[{self.exchange_id}] Соединение не инициализировано")
            return None

        request_symbol = ccxt_symbol(exchange, symbol) if market_type == "futures" else symbol

        try:
            ohlcv = await self._fetch_ohlcv_with_retry(exchange, request_symbol, timeframe, limit, since)

            if not ohlcv:
                logger.warning(f"[{self.exchange_id}] Пустой ответ для {symbol} {timeframe}")
                return None

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            # DEBUG, а не INFO — этот метод дёргается на каждый символ на
            # каждой итерации основного цикла (см. TradingBot._refresh_symbol_
            # candles в main.py), одна строка на пару за раз при десятках
            # активных пар превращала лог INFO+ на дашборде в почти
            # исключительно "Загружено N свечей для X" без реального сигнала.
            # Ошибки загрузки (ниже) остаются на своих уровнях — они и есть
            # то, что операционно важно видеть.
            logger.debug(f"[{self.exchange_id}] Загружено {len(df)} свечей для {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"[{self.exchange_id}] Ошибка загрузки {symbol} {timeframe}: {e}")
            return None

    # Плечевые токены (3x/5x, автоматически "сгорающие" от ежедневного
    # ребалансинга) — исключаются из динамической торговой вселенной всегда,
    # независимо от symbol_blacklist: они формально являются spot-парами на
    # Binance, но по своей природе не подходят для стратегий этого бота.
    _LEVERAGED_TOKEN_MARKERS = ("UP/", "DOWN/", "BULL/", "BEAR/")

    async def get_tradable_symbols(
        self,
        quote: str = "USDT",
        blacklist: list[str] | None = None,
        max_symbols: int = 30,
    ) -> list[str]:
        """
        Активные spot-пары биржи с заданной quote-валютой, минус блэклист и
        плечевые токены, отсортированные по 24ч объёму (топ max_symbols).
        """
        if not self.exchange or not self.exchange.markets:
            logger.warning(f"[{self.exchange_id}] Рынки не загружены — вернуть список пар невозможно")
            return []

        blacklist_set = set(blacklist or [])
        candidates = [
            symbol for symbol, market in self.exchange.markets.items()
            if market.get("spot")
            and market.get("active")
            and market.get("quote") == quote
            and symbol not in blacklist_set
            and not any(marker in symbol for marker in self._LEVERAGED_TOKEN_MARKERS)
        ]

        if not candidates:
            return []

        try:
            # Без списка символов — один лёгкий запрос "все тикеры" (у Binance
            # это /ticker/24hr без параметра symbols). Передача сотен
            # символов в fetch_tickers(candidates) регулярно роняла запрос
            # (URL на грани лимита длины у биржи, кривые ответы, которые
            # ccxt не всегда корректно распознаёт как ошибку — отсюда
            # обманчивое "'str' object has no attribute 'keys'": в tickers
            # прилетала строка вместо dict). Локальная фильтрация по
            # candidates ниже работает так же, но без этого риска.
            tickers = await self.exchange.fetch_tickers()
            candidates.sort(key=lambda s: (tickers.get(s) or {}).get("quoteVolume") or 0, reverse=True)
        except Exception as e:
            logger.warning(f"[{self.exchange_id}] Не удалось получить объёмы для сортировки пар: {e}")

        return candidates[:max_symbols]

    async def fetch_ohlcv_batch(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        limit: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """Загрузить OHLCV для нескольких пар."""
        results = {}
        for symbol in symbols:
            df = await self.fetch_ohlcv(symbol, timeframe, limit)
            if df is not None:
                results[symbol] = df
        return results

    def update_buffer(self, symbol: str, df: pd.DataFrame):
        """Обновить буфер свечей для символа."""
        self.candles_buffer[symbol] = self.merge_candles(self.candles_buffer.get(symbol), df)

    @staticmethod
    def merge_candles(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
        """
        Слить новые свечи в существующий буфер: дедуп по индексу (новые
        значения перекрывают старые), сортировка по времени, обрезка до
        candlesticks_cache_size. Вынесено в отдельный метод (не зависящий от
        self.candles_buffer), чтобы вызывающий код мог применить ту же логику
        слияния к СВОЕМУ собственному буферу — например, TradingBot.candles_buffer
        в main.py, который является отдельным dict от MarketDataIngest.candles_buffer
        и должен обновляться напрямую, а не через побочный эффект в буфере ingest.
        """
        if existing is None:
            combined = new
        else:
            combined = pd.concat([existing, new])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        max_rows = settings.candlesticks_cache_size
        if len(combined) > max_rows:
            combined = combined.iloc[-max_rows:]
        return combined

    def get_latest_candle(self, symbol: str) -> dict | None:
        """Получить последнюю свечу из буфера."""
        if symbol not in self.candles_buffer:
            return None
        df = self.candles_buffer[symbol]
        if df.empty:
            return None
        latest = df.iloc[-1]
        return {
            "open_time": int(latest.name.timestamp() * 1000),
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "volume": float(latest["volume"]),
        }

    def get_candles_for_symbol(self, symbol: str, limit: int | None = None) -> pd.DataFrame | None:
        """Получить свечи из буфера."""
        if symbol not in self.candles_buffer:
            return None
        df = self.candles_buffer[symbol]
        if df.empty:
            return None
        if limit:
            return df.iloc[-limit:]
        return df

    async def start_periodic_update(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        interval_seconds: int = 60,
    ):
        """Запустить периодическое обновление свечей."""
        self._running = True
        logger.info(f"[{self.exchange_id}] Запущен периодический сбор для {symbols} {timeframe}")

        while self._running:
            try:
                for symbol in symbols:
                    df = await self.fetch_ohlcv(symbol, timeframe, limit=2)
                    if df is not None and not df.empty:
                        self.update_buffer(symbol, df)
                        latest = self.get_latest_candle(symbol)
                        if latest:
                            logger.debug(
                                f"[{self.exchange_id}] Обновлена {symbol}: "
                                f"O={latest['open']:.2f} H={latest['high']:.2f} "
                                f"L={latest['low']:.2f} C={latest['close']:.2f} V={latest['volume']:.2f}"
                            )
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"[{self.exchange_id}] Ошибка периодического обновления: {e}")
                await asyncio.sleep(interval_seconds)


# === Фабрика для создания инжектора ===

def create_market_data_ingest(exchange_id: str = "binance") -> MarketDataIngest:
    """Создать инжектор донных с указанной биржей."""
    ingest = MarketDataIngest(exchange_id)
    return ingest
