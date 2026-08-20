"""Сбор рыночных данных с бирж через ccxt (REST + WebSocket планирование)."""
import asyncio
import logging

import ccxt.async_support as ccxt
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)


class MarketDataIngest:
    """
    Инжектор рыночных данных:
    - Загрузка исторических OHLCV (для инициализации)
    - Периодическое обновление свечей
    - Буферизация последних N свечей для стратегий
    """

    def __init__(self, exchange_id: str = "binance"):
        self.exchange_id = exchange_id.lower()
        self.exchange: ccxt.Exchange | None = None
        self.candles_buffer: dict[str, pd.DataFrame] = {}
        self._running = False

    async def initialize(self):
        """Инициализация подключения к бирже."""
        try:
            if self.exchange_id == "binance":
                self.exchange = ccxt.binance({
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
            elif self.exchange_id == "bybit":
                self.exchange = ccxt.bybit({
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
            else:
                self.exchange = getattr(ccxt, self.exchange_id)({
                    "enableRateLimit": True,
                })

            # Загрузить рынки
            await self.exchange.load_markets()
            logger.info(f"[{self.exchange_id}] Подключено, рынки загружены")
        except Exception as e:
            logger.error(f"Ошибка инициализации {self.exchange_id}: {e}")
            self.exchange = None

    async def close(self):
        """Закрыть соединение."""
        if self.exchange:
            await self.exchange.close()
            logger.info(f"[{self.exchange_id}] Соединение закрыто")

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: int | None = None,
    ) -> pd.DataFrame | None:
        """
        Загрузить OHLCV свечи из биржи.
        Возвращает DataFrame с индексом timestamp и колонками: open, high, low, close, volume
        """
        if not self.exchange:
            logger.warning(f"[{self.exchange_id}] Соединение не инициализировано")
            return None

        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since,
            )

            if not ohlcv:
                logger.warning(f"[{self.exchange_id}] Пустой ответ для {symbol} {timeframe}")
                return None

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            logger.info(f"[{self.exchange_id}] Загружено {len(df)} свечей для {symbol} {timeframe}")
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
            tickers = await self.exchange.fetch_tickers(candidates)
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
        if symbol not in self.candles_buffer:
            self.candles_buffer[symbol] = df
        else:
            # Объединяем, удаляя дубликаты по индексу
            existing = self.candles_buffer[symbol]
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            # Ограничиваем размер буфера
            max_rows = settings.candlesticks_cache_size
            if len(combined) > max_rows:
                combined = combined.iloc[-max_rows:]
            self.candles_buffer[symbol] = combined

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
