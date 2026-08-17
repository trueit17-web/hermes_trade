"""Полная схема базы данных для крипто-трейдер бота."""
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, DECIMAL, Float, Integer, JSON, String, Text,
    ForeignKey, Index, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base
from src.utils.timeutils import utcnow


class Exchange(Base):
    """Биржа (Binance, Bybit и т.д.)."""
    __tablename__ = "exchanges"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    api_key_enc: Mapped[Optional[str]] = mapped_column(String(512))
    api_secret_enc: Mapped[Optional[str]] = mapped_column(String(512))
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    symbols: Mapped[list["Symbol"]] = relationship(back_populates="exchange")
    orders: Mapped[list["Order"]] = relationship(back_populates="exchange")


class Symbol(Base):
    """Торговая пара."""
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(primary_key=True)
    exchange_id: Mapped[int] = mapped_column(ForeignKey("exchanges.id"))
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    base_asset: Mapped[str] = mapped_column(String(20))
    quote_asset: Mapped[str] = mapped_column(String(20))
    tick_size: Mapped[Optional[float]] = mapped_column(Float)
    step_size: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    exchange: Mapped["Exchange"] = relationship(back_populates="symbols")
    orders: Mapped[list["Order"]] = relationship(back_populates="symbol")
    trades: Mapped[list["Trade"]] = relationship(back_populates="symbol")
    candles: Mapped[list["Candle"]] = relationship(back_populates="symbol")
    signals: Mapped[list["Signal"]] = relationship(back_populates="symbol")

    __table_args__ = (
        UniqueConstraint("exchange_id", "symbol", name="uq_exchange_symbol"),
        Index("ix_symbols_symbol", "symbol"),
    )


class Strategy(Base):
    """Стратегия торговли."""
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    strategy_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # rule, ml, ensemble
    params: Mapped[dict] = mapped_column(JSON, default={})
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    orders: Mapped[list["Order"]] = relationship(back_populates="strategy")
    trades: Mapped[list["Trade"]] = relationship(back_populates="strategy")
    signals: Mapped[list["Signal"]] = relationship(back_populates="strategy")


class Signal(Base):
    """Сигнал от стратегии (независимо от исполнения)."""
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"))
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"))
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # long, short
    confidence: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[Optional[float]] = mapped_column(DECIMAL)
    stop_loss: Mapped[Optional[float]] = mapped_column(DECIMAL)
    take_profit: Mapped[Optional[float]] = mapped_column(DECIMAL)
    position_size_pct: Mapped[float] = mapped_column(Float)
    timeframe: Mapped[str] = mapped_column(String(10))
    rationale: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    strategy: Mapped["Strategy"] = relationship(back_populates="signals")
    symbol: Mapped["Symbol"] = relationship(back_populates="signals")
    order: Mapped[Optional["Order"]] = relationship()


class Order(Base):
    """Ордер (открыт/исполнен/отменён)."""
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    exchange_id: Mapped[int] = mapped_column(ForeignKey("exchanges.id"))
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"))
    strategy_id: Mapped[Optional[int]] = mapped_column(ForeignKey("strategies.id"))
    signal_id: Mapped[Optional[int]] = mapped_column(ForeignKey("signals.id"))
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # buy, sell
    order_type: Mapped[str] = mapped_column(String(20))  # market, limit, stop_limit
    amount: Mapped[float] = mapped_column(DECIMAL, nullable=False)
    price: Mapped[Optional[float]] = mapped_column(DECIMAL)
    status: Mapped[str] = mapped_column(
        String(20), default="open"
    )  # open, filled, partial, cancelled, rejected
    filled_amount: Mapped[float] = mapped_column(DECIMAL, default=0)
    filled_price: Mapped[Optional[float]] = mapped_column(DECIMAL)
    fee: Mapped[float] = mapped_column(DECIMAL, default=0)
    stop_loss: Mapped[Optional[float]] = mapped_column(DECIMAL)
    take_profit: Mapped[Optional[float]] = mapped_column(DECIMAL)
    order_id_exchange: Mapped[Optional[str]] = mapped_column(String(100))
    client_order_id: Mapped[Optional[str]] = mapped_column(String(100))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    exchange: Mapped["Exchange"] = relationship(back_populates="orders")
    symbol: Mapped["Symbol"] = relationship(back_populates="orders")
    strategy: Mapped[Optional["Strategy"]] = relationship(back_populates="orders")
    signal: Mapped[Optional["Signal"]] = relationship()
    trade_open: Mapped[Optional["Trade"]] = relationship(
        foreign_keys="Trade.order_open_id"
    )
    trade_close: Mapped[Optional["Trade"]] = relationship(
        foreign_keys="Trade.order_close_id"
    )


class Trade(Base):
    """Сделка (открытие + закрытие = одна сделка)."""
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"))
    strategy_id: Mapped[Optional[int]] = mapped_column(ForeignKey("strategies.id"))
    order_open_id: Mapped[Optional[int]] = mapped_column(ForeignKey("orders.id"))
    order_close_id: Mapped[Optional[int]] = mapped_column(ForeignKey("orders.id"))
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # long, short
    entry_price: Mapped[float] = mapped_column(DECIMAL, nullable=False)
    exit_price: Mapped[Optional[float]] = mapped_column(DECIMAL)
    amount: Mapped[float] = mapped_column(DECIMAL, nullable=False)
    pnl: Mapped[float] = mapped_column(DECIMAL, default=0)
    pnl_pct: Mapped[float] = mapped_column(Float)
    holding_seconds: Mapped[int] = mapped_column(Integer, default=0)
    features_at_open: Mapped[Optional[dict]] = mapped_column(JSON)
    outcome: Mapped[Optional[str]] = mapped_column(
        String(20)
    )  # win, loss, break-even
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    symbol: Mapped["Symbol"] = relationship(back_populates="trades")
    strategy: Mapped[Optional["Strategy"]] = relationship(back_populates="trades")
    order_open: Mapped[Optional["Order"]] = relationship(
        foreign_keys="Trade.order_open_id"
    )
    order_close: Mapped[Optional["Order"]] = relationship(
        foreign_keys="Trade.order_close_id"
    )


class Candle(Base):
    """Свеча (OHLCV)."""
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(primary_key=True)
    exchange_id: Mapped[int] = mapped_column(ForeignKey("exchanges.id"))
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"))
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    open: Mapped[float] = mapped_column(DECIMAL)
    high: Mapped[float] = mapped_column(DECIMAL)
    low: Mapped[float] = mapped_column(DECIMAL)
    close: Mapped[float] = mapped_column(DECIMAL)
    volume: Mapped[float] = mapped_column(DECIMAL)
    close_time: Mapped[Optional[datetime]] = mapped_column(DateTime)

    symbol: Mapped["Symbol"] = relationship(back_populates="candles")
    exchange: Mapped["Exchange"] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "exchange_id", "symbol_id", "timeframe", "open_time",
            name="uq_candle_unique"
        ),
        Index("ix_candles_symbol_tf", "symbol_id", "timeframe"),
    )


class TelegramChannel(Base):
    """Telegram канал для мониторинга сигналов."""
    __tablename__ = "telegram_channels"

    id: Mapped[int] = mapped_column(primary_key=True)
    channel_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    channel_title: Mapped[Optional[str]] = mapped_column(String(200))
    parser_type: Mapped[str] = mapped_column(
        String(20), default="regex"
    )  # regex, llm
    parser_config: Mapped[dict] = mapped_column(JSON, default={})
    quality_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    auto_execute: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    signals: Mapped[list["TelegramSignal"]] = relationship(back_populates="channel")


class TelegramSignal(Base):
    """Полученный сигнал из Telegram."""
    __tablename__ = "telegram_signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("telegram_channels.id"))
    raw_message: Mapped[str] = mapped_column(Text, nullable=False)
    message_date: Mapped[datetime] = mapped_column(DateTime)
    parsed_pair: Mapped[Optional[str]] = mapped_column(String(50))
    parsed_side: Mapped[Optional[str]] = mapped_column(String(10))
    parsed_entry: Mapped[Optional[float]] = mapped_column(DECIMAL)
    parsed_sl: Mapped[Optional[float]] = mapped_column(DECIMAL)
    parsed_tp: Mapped[Optional[float]] = mapped_column(DECIMAL)
    quality_score: Mapped[Optional[float]] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(
        String(30), default="pending"
    )  # executed, rejected, pending
    executed_trade_id: Mapped[Optional[int]] = mapped_column(ForeignKey("trades.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    channel: Mapped["TelegramChannel"] = relationship(back_populates="signals")
    executed_trade: Mapped[Optional["Trade"]] = relationship()


class MLModel(Base):
    """Зарегистрированная ML модель."""
    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # direction_classifier, volatility, quality_scorer, ensemble_weights
    strategy_id: Mapped[Optional[int]] = mapped_column(ForeignKey("strategies.id"))
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    model_path: Mapped[Optional[str]] = mapped_column(String(500))
    params: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_shadow: Mapped[bool] = mapped_column(Boolean, default=False)
    released_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (Index("ix_ml_model_active", "model_type", "is_active"),)


class MLFeature(Base):
    """Сохранённая фича для ML обучения."""
    __tablename__ = "ml_features"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    label_direction: Mapped[Optional[int]] = mapped_column(
        Integer
    )  # -1, 0, +1
    label_volatility: Mapped[Optional[float]] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20))  # live, backtest
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_feature_unique"),
        Index("ix_features_symbol_ts", "symbol", "timestamp"),
    )


class BotConfig(Base):
    """Конфигурация бота (редактируемая из UI)."""
    __tablename__ = "bot_config"

    id: Mapped[int] = mapped_column(primary_key=True)
    config_key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    config_value: Mapped[dict] = mapped_column(JSON, nullable=False)
    source: Mapped[str] = mapped_column(
        String(20), default="default"
    )  # default, ui, api, ml
    updated_by: Mapped[Optional[str]] = mapped_column(String(50))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )


class BotEvent(Base):
    """Журнал событий (логи)."""
    __tablename__ = "bot_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    level: Mapped[str] = mapped_column(String(20), nullable=False)  # INFO, WARNING, ERROR
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_events_created", "created_at"),
        Index("ix_events_level_source", "level", "source"),
    )


class PerformanceSnapshot(Base):
    """Периодический снимок производительности (для графиков)."""
    __tablename__ = "performance_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    snapshot_time: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    total_balance: Mapped[float] = mapped_column(DECIMAL)
    open_pnl: Mapped[float] = mapped_column(DECIMAL, default=0)
    realized_pnl: Mapped[float] = mapped_column(DECIMAL, default=0)
    daily_pnl: Mapped[float] = mapped_column(DECIMAL, default=0)
    weekly_pnl: Mapped[float] = mapped_column(DECIMAL, default=0)
    num_open_positions: Mapped[int] = mapped_column(Integer, default=0)
    num_trades_today: Mapped[int] = mapped_column(Integer, default=0)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    win_rate: Mapped[Optional[float]] = mapped_column(Float)


class TradeDecisionLog(Base):
    """Журнал принятия решения для каждой сделки (decision tree)."""
    __tablename__ = "trade_decision_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[int] = mapped_column(ForeignKey("trades.id"))
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    step_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # market_data, strategy_signal, ml_score, risk_check, execution
    description: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_decision_log_trade", "trade_id"),
    )
