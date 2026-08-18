"""Event Bus — асинхронная шина событий с pub/sub моделью."""
import asyncio
import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar, cast

from rich.console import Console
from rich.table import Table
from rich.live import Live

from src.config import settings

console = Console()
logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Event:
    """Базовое событие."""
    type: str
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    source: str = "unknown"
    payload: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"[{self.type}] from {self.source} at {self.timestamp:.3f}"


@dataclass
class MarketDataEvent(Event):
    """Событие рыночных данных: свечи, тики, обновления."""
    type: str = "market_data"
    symbol: str = ""
    timeframe: str = "1h"
    candle: dict[str, Any] | None = None  # {open, high, low, close, volume, timestamp}
    features: dict[str, float] | None = None  # вычисленные фичи
    coinglass_data: dict[str, Any] | None = None


@dataclass
class SignalGeneratedEvent(Event):
    """Сигнал от стратегии."""
    type: str = "signal_generated"
    strategy_id: str = ""
    symbol: str = ""
    side: str = ""  # long, short, flat
    confidence: float = 0.0
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size_pct: float = 0.0
    rationale: str = ""
    timestamp: float = 0.0


@dataclass
class TelegramSignalEvent(Event):
    """Сигнал из Telegram канала."""
    type: str = "telegram_signal"
    channel_id: str = ""
    raw_message: str = ""
    parsed_pair: str = ""
    parsed_side: str = ""
    parsed_entry: float = 0.0
    parsed_sl: float | None = None
    parsed_tp: float | None = None
    quality_score: float = 0.0
    decision: str = ""  # executed, rejected, pending


@dataclass
class OrderEvent(Event):
    """Событие ордера."""
    type: str = "order_event"
    exchange: str = ""
    symbol: str = ""
    order_id: str = ""
    side: str = ""
    order_type: str = ""
    amount: float = 0.0
    price: float | None = None
    status: str = ""  # open, filled, partial, cancelled, rejected
    filled_amount: float = 0.0
    filled_price: float | None = None
    fee: float = 0.0
    timestamp: float = 0.0


@dataclass
class TradeEvent(Event):
    """Событие сделки (открытие/закрытие позиции)."""
    type: str = "trade_event"
    trade_id: int = 0
    symbol: str = ""
    direction: str = ""  # long, short
    entry_price: float = 0.0
    exit_price: float = 0.0
    amount: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_seconds: int = 0
    features_at_open: dict[str, float] | None = None
    outcome: str = ""  # win, loss, break-even
    is_opening: bool = True
    timestamp: float = 0.0


class EventBus:
    """
    Асинхронная шина событий.
    Реализует pub/sub: модули подписываются на типы событий,
    при публикации события вызываются все подписчики.

    Поддерживает:
    - подписку на конкретные типы событий
    - подписку на все события (wildcard)
    - отписку
    - историю событий (для отладки и реплея)
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable[[Event], Any]]] = {}
        self._historic: list[Event] = []
        self._max_history = 10000

    def subscribe(self, event_type: str, callback: Callable[[Event], Any]) -> None:
        """Подписаться на события определённого типа."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.info(f"[{self.__class__.__name__}] Подписка на {event_type}: {callback.__name__}")

    def subscribe_all(self, callback: Callable[[Event], Any]) -> None:
        """Подписаться на все события (wildcard)."""
        self._subscribers["*"] = self._subscribers.get("*", []) + [callback]

    def unsubscribe(self, event_type: str, callback: Callable[[Event], Any]) -> None:
        """Отписаться от событий."""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)

    async def publish(self, event: Event) -> list[Any]:
        """
        Опубликовать событие. Вызывает всех подписчиков.
        Возвращает список результатов вызовов подписчиков.
        """
        # Сохраняем в историю
        self._historic.append(event)
        if len(self._historic) > self._max_history:
            self._historic = self._historic[-self._max_history:]

        results: list[Any] = []

        # Вызываем подписчиков на конкретный тип
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    result = callback(event)
                    if asyncio.iscoroutine(result):
                        result = await result
                    results.append(result)
                except Exception as e:
                    logger.error(f"Ошибка подписчика на {event.type}: {e}")

        # Вызываем подписчиков wildcard
        if "*" in self._subscribers:
            for callback in self._subscribers["*"]:
                try:
                    result = callback(event)
                    if asyncio.iscoroutine(result):
                        result = await result
                    results.append(result)
                except Exception as e:
                    logger.error(f"Ошибка wildcard подписчика: {e}")

        return results

    def get_history(self, event_type: str | None = None) -> list[Event]:
        """Получить историю событий. Опционально по типу."""
        if event_type:
            return [e for e in self._historic if e.type == event_type]
        return list(self._historic)

    def clear_history(self) -> None:
        """Очистить историю событий."""
        self._historic.clear()

    def subscribe_all_events(self, callback):
        """Подписка на все типы событий."""
        for event_type in self._subscribers:
            self.subscribe(event_type, callback)


# Глобальная шина событий
event_bus = EventBus()
