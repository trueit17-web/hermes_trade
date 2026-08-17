"""WebSocket feed — реальные потоковые данные от бирж."""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import ccxt.async_support as ccxt

from src.config import settings
from src.event_bus import event_bus, MarketDataEvent
from src.utils.logging import logger

logger = logging.getLogger(__name__)


class WebSocketFeed:
    """
    WebSocket поток от биржи (Binance, Bybit).
    Получает реальные свечи, тики, order book в реальном времени.

    Поддерживает:
    - Каналы: kline (свечи), ticker (тикер), depth (order book)
    - Автоматическое переподключение
    - Распространение событий через event_bus
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        symbols: Optional[list[str]] = None,
        channels: Optional[list[str]] = None,
    ):
        self.exchange_id = exchange_id
        self.symbols = symbols or settings.default_symbols
        self.channels = channels or ["kline"]  # kline, ticker, depth
        self.exchange: Optional[ccxt.Exchange] = None
        self._ws_client: Any = None
        self._running = False
        self._reconnect_delay = 5  # секунды
        self._max_reconnect_attempts = 10
        self._reconnect_attempts = 0
        self._callbacks: Dict[str, List[Callable]] = {}

    async def initialize(self):
        """Инициализация WebSocket подключения."""
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
                exchange_class = getattr(ccxt, self.exchange_id, None)
                if exchange_class:
                    self.exchange = exchange_class({"enableRateLimit": True})
                else:
                    raise ValueError(f"Exchange {self.exchange_id} not supported by ccxt")

            logger.info(f"[{self.exchange_id}] WebSocket feed initialized, symbols: {self.symbols}")
            return True
        except Exception as e:
            logger.error(f"[{self.exchange_id}] WebSocket init failed: {e}")
            return False

    async def close(self):
        """Закрыть WebSocket подключение."""
        self._running = False
        if self._ws_client:
            try:
                await self._ws_client.close()
            except Exception:
                pass
        if self.exchange:
            await self.exchange.close()
        logger.info(f"[{self.exchange_id}] WebSocket closed")

    async def start(self):
        """
        Запустить WebSocket поток.
        Блокирующая операция — запускать в отдельной задаче.
        """
        if not await self.initialize():
            return

        self._running = True
        self._reconnect_attempts = 0

        # Binance WebSocket
        if self.exchange_id == "binance":
            await self._start_binance()
        elif self.exchange_id == "bybit":
            await self._start_bybit()
        else:
            logger.warning(f"[{self.exchange_id}] WebSocket start not implemented")
            return

        logger.info(f"[{self.exchange_id}] WebSocket stream started")

    async def _start_binance(self):
        """Binance WebSocket — kline stream."""
        base_url = "wss://stream.binance.com:9443/ws"

        for symbol in self.symbols:
            # Преобразовать BTC/USDT в btcusdt (lowercase, no slash)
            stream_name = f"{symbol.lower().replace('/', '')}@kline_1h"

            try:
                async with self.exchange.websocket(base_url) as ws:
                    await ws.subscribe([stream_name])

                    async for msg in ws:
                        if not self._running:
                            break

                        if isinstance(msg, dict) and msg.get("e") == "kline":
                            kline = msg.get("k", {})
                            if kline:
                                candle = {
                                    "open_time": kline.get("t", 0),
                                    "open": float(kline.get("o", 0)),
                                    "high": float(kline.get("h", 0)),
                                    "low": float(kline.get("l", 0)),
                                    "close": float(kline.get("c", 0)),
                                    "volume": float(kline.get("v", 0)),
                                    "close_time": kline.get("T", 0),
                                    "is_candle_closed": kline.get("x", False),
                                }

                                if candle["is_candle_closed"]:
                                    event = MarketDataEvent(
                                        symbol=symbol,
                                        timeframe="1h",
                                        candle=candle,
                                        timestamp=datetime.utcnow().timestamp(),
                                    )
                                    await event_bus.publish(event)
                                    logger.debug(f"[WS] Канал закрыт: {symbol} {candle['close']:.2f}")

            except Exception as e:
                logger.error(f"[Binance WS] Ошибка: {e}")
                await self._reconnect()

    async def _start_bybit(self):
        """Bybit WebSocket — доступ к рыночным данным."""
        # По умолчанию ccxt.pro имеет WebSocket поддержку
        try:
            if not self.exchange:
                await self.initialize()

            # Загрузить рынки
            await self.exchange.load_markets()

            for symbol in self.symbols:
                # Подписка на обновления
                self.exchange.on(f"trade:{symbol}", self._on_trade)
                self.exchange.on(f"ohlcv:{symbol}", self._on_ohlcV)

            # Запустить WebSocket
            if hasattr(self.exchange, 'ws'):
                await self.exchange.ws.connect()
                for symbol in self.symbols:
                    await self.exchange.ws.subscribe(f"trade:{symbol}")
                    await self.exchange.ws.subscribe(f"ohlcv:{symbol}")

            logger.info(f"[Bybit WS] подписан на: {self.symbols}")

        except Exception as e:
            logger.error(f"[Bybit WS] Ошибка: {e}")
            await self._reconnect()

    async def _on_trade(self, trade: dict):
        """Обработчик новых trades."""
        event = MarketDataEvent(
            symbol=trade.get("symbol", ""),
            candle={
                "open_time": trade.get("timestamp", 0),
                "open": trade.get("price", 0),
                "high": trade.get("price", 0),
                "low": trade.get("price", 0),
                "close": trade.get("price", 0),
                "volume": trade.get("amount", 0),
            },
            timestamp=datetime.utcnow().timestamp(),
        )
        await event_bus.publish(event)

    async def _on_ohlcV(self, ohlcv: list):
        """Обработчик OHLCV обновлений."""
        if len(ohlcv) < 6:
            return

        symbol = ohlcv[0]
        candle = {
            "open_time": ohlcv[1],
            "open": float(ohlcv[2]),
            "high": float(ohlcv[3]),
            "low": float(ohlcv[4]),
            "close": float(ohlcv[5]),
            "volume": float(ohlcv[6]) if len(ohlcv) > 6 else 0,
            "is_candle_closed": ohlcv[7] if len(ohlcv) > 7 else True,
        }

        event = MarketDataEvent(
            symbol=symbol,
            timeframe="1h",
            candle=candle,
            timestamp=datetime.utcnow().timestamp(),
        )
        await event_bus.publish(event)

        logger.debug(f"[WS] New candle: {symbol} close={candle['close']:.2f}")

    async def _reconnect(self):
        """Попытка переподключения."""
        self._reconnect_attempts += 1
        if self._reconnect_attempts > self._max_reconnect_attempts:
            logger.error(f"[{self.exchange_id}] WebSocket: превышено количество попыток переподключения ({self._max_reconnect_attempts})")
            self._running = False
            return

        delay = min(self._reconnect_delay * self._reconnect_attempts, 60)
        logger.warning(f"[{self.exchange_id}] WebSocket: переподключение через {delay}s (попытка {self._reconnect_attempts})")
        await asyncio.sleep(delay)

        try:
            await self.initialize()
            if self.exchange_id == "binance":
                await self._start_binance()
            elif self.exchange_id == "bybit":
                await self._start_bybit()
        except Exception as e:
            logger.error(f"[{self.exchange_id}] WebSocket reconnect failed: {e}")
            await self._reconnect()


# Глобальный экземпляр
ws_feed: Optional[WebSocketFeed] = None


def get_ws_feed(exchange_id: str = "binance") -> WebSocketFeed:
    """Получить или создать WebSocket feed."""
    global ws_feed
    if ws_feed is None:
        ws_feed = WebSocketFeed(exchange_id)
    return ws_feed
