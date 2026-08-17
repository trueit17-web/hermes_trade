"""Pydantic схемы для веб-интерфейса."""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# === Схемы конфигурации ===

class ConfigValue(BaseModel):
    """Значение конфигурации."""
    value: Any
    source: str = "default"
    updated_at: Optional[datetime] = None


class RiskConfig(BaseModel):
    """Риск-конфигурация."""
    daily_loss_limit_usd: float = 500.0
    max_open_positions: int = 8
    max_position_size_pct: float = 10.0
    max_drawdown_pct: float = 15.0
    cooldown_seconds: int = 300


class StrategyConfig(BaseModel):
    """Конфигурация стратегии."""
    id: str
    name: str
    type: str  # rule, ml, ensemble
    active: bool = True
    weight: float = 1.0
    params: dict = Field(default_factory=dict)


class BotConfigResponse(BaseModel):
    """Ответ с конфигурацией бота."""
    trading_mode: str
    startup_capital_usdt: float
    symbols: list[str]
    timeframes: list[str]
    risk: RiskConfig
    strategies: list[StrategyConfig]


# === Схемы сделок ===

class TradeResponse(BaseModel):
    """Сделка."""
    id: int
    symbol: Optional[str] = None
    direction: str
    entry_price: float
    exit_price: Optional[float] = None
    amount: float
    pnl: float
    pnl_pct: float
    holding_seconds: int
    outcome: Optional[str] = None
    is_open: bool = True
    strategy_id: Optional[int] = None
    created_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None


class TradeListResponse(BaseModel):
    """Список сделок."""
    trades: list[TradeResponse]
    total: int


# === Схемы ордеров ===

class OrderResponse(BaseModel):
    """Ордер."""
    id: int
    symbol: Optional[str] = None
    side: str
    order_type: str
    amount: float
    price: Optional[float] = None
    status: str
    filled_amount: float
    filled_price: Optional[float] = None
    fee: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    client_order_id: Optional[str] = None
    created_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None


class OrderListResponse(BaseModel):
    """Список ордеров."""
    orders: list[OrderResponse]
    total: int


# === Схемы ML моделей ===

class MLModelResponse(BaseModel):
    """ML модель."""
    id: int
    model_type: str
    version: int
    model_path: Optional[str] = None
    params: dict
    metrics: dict
    is_active: bool = False
    is_shadow: bool = False
    created_at: Optional[datetime] = None


class MLModelListResponse(BaseModel):
    """Список ML моделей."""
    models: list[MLModelResponse]


# === Схемы Telegram ===

class TelegramChannelResponse(BaseModel):
    """Telegram канал."""
    id: int
    channel_id: str
    channel_title: Optional[str] = None
    parser_type: str
    auto_execute: bool = False
    active: bool = True
    quality_threshold: float = 0.5


class TelegramSignalResponse(BaseModel):
    """Telegram сигнал."""
    id: int
    channel_id: int
    raw_message: str
    parsed_pair: Optional[str] = None
    parsed_side: Optional[str] = None
    parsed_entry: Optional[float] = None
    parsed_sl: Optional[float] = None
    parsed_tp: Optional[float] = None
    quality_score: Optional[float] = None
    decision: str
    created_at: Optional[datetime] = None


class TelegramSignalListResponse(BaseModel):
    """Список Telegram сигналов."""
    signals: list[TelegramSignalResponse]
    total: int


# === Схемы производительности ===

class PerformanceSnapshotResponse(BaseModel):
    """Снимок производительности."""
    time: Optional[datetime] = None
    total_balance: float
    open_pnl: float
    realized_pnl: float
    daily_pnl: float
    weekly_pnl: float
    num_open_positions: int
    num_trades_today: int
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None


class PerformanceResponse(BaseModel):
    """Производительность."""
    snapshots: list[PerformanceSnapshotResponse]


# === Схемы для event bus ===

class EventResponse(BaseModel):
    """Событие."""
    id: int
    type: str
    source: str
    message: str
    level: str
    created_at: Optional[datetime] = None


class EventListResponse(BaseModel):
    """Список событий."""
    events: list[EventResponse]
    total: int
