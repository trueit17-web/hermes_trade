"""FastAPI веб-интерфейс — API для управления ботом."""
import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select

from src.config import settings
from src.event_bus import event_bus
from src.risk.risk_manager import risk_manager
from src.execution.executor import execution_engine
from src.ml import model_registry, model_trainer
from src.strategy import strategy_registry
from src.utils.logging import logger
from src.db.session import get_session
from src.db.models import (
    Strategy as StrategyModel,
    Trade,
    Order,
    Signal as SignalModel,
    PerformanceSnapshot,
    BotConfig,
    TelegramChannel,
    TelegramSignal,
)

logger = logging.getLogger(__name__)

# === FastAPI приложение ===

app = FastAPI(
    title="CryptoBot Pro API",
    description="API для управления автономным крипто-трейдер ботом",
    version="1.0.0",
)

# CORS для доступа из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === WebSocket менеджер для real-time обновлений ===

class WebSocketManager:
    """Управление WebSocket подключениями."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Отправить сообщение всем подключенным клиентам."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.debug(f"Ошибка отправки WebSocket сообщения: {e}")
                # Удалить отключившихся клиентов
                try:
                    await connection.close()
                except Exception:
                    pass
                if connection in self.active_connections:
                    self.active_connections.remove(connection)


ws_manager = WebSocketManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket эндпоинт для real-time обновлений."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Ждём сообщение от клиента (опционально)
            data = await websocket.receive_text()
            logger.debug(f"WebSocket сообщение: {data}")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket клиент отключился")
    except Exception as e:
        ws_manager.disconnect(websocket)
        logger.warning(f"WebSocket ошибка: {e}")


# === Pydantic схемы ===

class ConfigUpdateRequest(BaseModel):
    """Запрос на обновление конфигурации."""
    key: str
    value: Any


class RiskConfigUpdate(BaseModel):
    """Обновление риск-конфигурации."""
    daily_loss_limit_usd: Optional[float] = None
    max_open_positions: Optional[int] = None
    max_position_size_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    cooldown_seconds: Optional[int] = None


class StrategyToggleRequest(BaseModel):
    """Включение/выключение стратегии."""
    active: bool


class StrategyUpdateRequest(BaseModel):
    """Обновление параметров стратегии."""
    params: dict = Field(default_factory=dict)


class TelegramChannelCreate(BaseModel):
    """Создание Telegram канала."""
    channel_id: str
    channel_title: Optional[str] = None
    parser_type: str = "regex"
    parser_config: dict = Field(default_factory=dict)
    quality_threshold: float = 0.5
    auto_execute: bool = False


class TelegramSignalConfirm(BaseModel):
    """Подтверждение Telegram сигнала."""
    signal_id: int
    action: str  # execute, reject


class BacktestRequest(BaseModel):
    """Запрос на backtest."""
    strategy_id: str
    symbol: str
    timeframe: str = "1h"
    start_date: str
    end_date: str
    paper_balance: float = 10000.0


# === API эндпоинты ===

@app.get("/")
async def root():
    """Главная страница — статус бота."""
    return {
        "status": "running",
        "mode": settings.trading_mode,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health():
    """Проверка здоровья."""
    return {
        "status": "ok",
        "trading_mode": settings.trading_mode,
        "bot_running": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/status")
async def get_status():
    """Получить текущий статус бота."""
    async with get_session() as session:
        channels = (
            await session.execute(select(TelegramChannel).where(TelegramChannel.active == True))
        ).scalars().all()
        telegram_channels = [
            {"id": c.id, "channel_id": c.channel_id, "channel_title": c.channel_title}
            for c in channels
        ]

    return {
        "trading_mode": settings.trading_mode,
        "is_paper": settings.is_paper,
        "startup_capital": settings.startup_capital_usdt,
        "risk_state": risk_manager.get_state(),
        "paper_balance": execution_engine.get_paper_balance() if settings.is_paper else None,
        "paper_positions": execution_engine.get_paper_positions() if settings.is_paper else None,
        "active_strategies": strategy_registry.list_strategies(),
        "ml_models": await model_registry.list_models(),
        "ml_active_model": await model_registry.get_active_model("direction_classifier"),
        "telegram_channels": telegram_channels,
        "event_bus_history_size": len(event_bus.get_history()),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/risk/state")
async def get_risk_state():
    """Текущее состояние риска."""
    return risk_manager.get_state()


@app.post("/risk/configure")
async def configure_risk(config: RiskConfigUpdate):
    """Обновить риск-конфигурацию."""
    params = {}
    if config.daily_loss_limit_usd is not None:
        params["daily_loss_limit_usd"] = config.daily_loss_limit_usd
    if config.max_open_positions is not None:
        params["max_open_positions"] = config.max_open_positions
    if config.max_position_size_pct is not None:
        params["max_position_size_pct"] = config.max_position_size_pct
    if config.max_drawdown_pct is not None:
        params["max_drawdown_pct"] = config.max_drawdown_pct
    if config.cooldown_seconds is not None:
        params["cooldown_seconds"] = config.cooldown_seconds

    risk_manager.configure(params)

    # Сохранить в БД
    async with get_session() as session:
        for key, value in params.items():
            config_record = (
                await session.execute(select(BotConfig).where(BotConfig.config_key == key))
            ).scalar_one_or_none()
            if config_record:
                config_record.config_value = {"value": value}
            else:
                session.add(BotConfig(
                    config_key=key,
                    config_value={"value": value},
                    source="api",
                    updated_by="api",
                ))
        await session.commit()

    logger.info(f"Risk конфигурация обновлена: {params}")
    return {"success": True, "config": params}


@app.post("/risk/pause")
async def pause_trading():
    """Приостановить торговлю."""
    risk_manager.state.pause()
    return {"success": True, "status": "paused"}


@app.post("/risk/resume")
async def resume_trading():
    """Возобновить торговлю."""
    risk_manager.state.resume()
    return {"success": True, "status": "resumed"}


@app.post("/risk/kill-switch")
async def trigger_kill_switch():
    """Активировать kill switch."""
    risk_manager.state.trigger_kill_switch()
    return {"success": True, "status": "kill_switch_active"}


@app.post("/risk/clear-kill-switch")
async def clear_kill_switch():
    """Сбросить kill switch."""
    risk_manager.state.clear_kill_switch()
    return {"success": True, "status": "kill_switch_cleared"}


@app.get("/strategies")
async def list_strategies():
    """Список стратегий."""
    return {"strategies": strategy_registry.list_strategies()}


@app.post("/strategies/{strategy_id}/toggle")
async def toggle_strategy(strategy_id: str, request: StrategyToggleRequest):
    """Включить/выключить стратегию."""
    strategy = strategy_registry.get(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Стратегия {strategy_id} не найдена")

    strategy.active = request.active
    logger.info(f"Стратегия {strategy_id} {'включена' if request.active else 'выключена'}")

    # Сохранить в БД
    async with get_session() as session:
        db_strategy = None
        if strategy_id.isdigit():
            db_strategy = (
                await session.execute(select(StrategyModel).where(StrategyModel.id == int(strategy_id)))
            ).scalar_one_or_none()
        if db_strategy:
            db_strategy.active = request.active
            await session.commit()

    return {"success": True, "strategy_id": strategy_id, "active": request.active}


@app.post("/strategies/{strategy_id}/update")
async def update_strategy(strategy_id: str, request: StrategyUpdateRequest):
    """Обновить параметры стратегии."""
    strategy = strategy_registry.get(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Стратегия {strategy_id} не найдена")

    strategy.params.update(request.params)
    logger.info(f"Параметры стратегии {strategy_id} обновлены: {request.params}")
    return {"success": True, "strategy_id": strategy_id, "params": strategy.params}


@app.get("/trades")
async def list_trades(limit: int = 100, offset: int = 0):
    """Список сделок."""
    async with get_session() as session:
        trades = (
            await session.execute(
                select(Trade).order_by(Trade.created_at.desc()).offset(offset).limit(limit)
            )
        ).scalars().all()
        return {
            "trades": [
                {
                    "id": t.id,
                    "symbol_id": t.symbol_id,
                    "direction": t.direction,
                    "entry_price": float(t.entry_price),
                    "exit_price": float(t.exit_price) if t.exit_price else None,
                    "amount": float(t.amount),
                    "pnl": float(t.pnl),
                    "pnl_pct": float(t.pnl_pct) if t.pnl_pct else 0,
                    "holding_seconds": t.holding_seconds,
                    "outcome": t.outcome,
                    "is_open": t.is_open,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "closed_at": t.closed_at.isoformat() if t.closed_at else None,
                }
                for t in trades
            ],
            "total": len(trades),
        }


@app.get("/trades/{trade_id}/decision-log")
async def get_trade_decision_log(trade_id: int):
    """Получить decision log для сделки (почему сделка была открыта/закрыта)."""
    async with get_session() as session:
        from src.db.models import TradeDecisionLog
        logs = (
            await session.execute(
                select(TradeDecisionLog)
                .where(TradeDecisionLog.trade_id == trade_id)
                .order_by(TradeDecisionLog.step_order.asc())
            )
        ).scalars().all()

        return {
            "trade_id": trade_id,
            "decision_log": [
                {
                    "step_order": log.step_order,
                    "step_type": log.step_type,
                    "description": log.description,
                    "details": log.details,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log in logs
            ],
        }


@app.get("/orders")
async def list_orders(limit: int = 100):
    """Список ордеров."""
    async with get_session() as session:
        orders = (
            await session.execute(select(Order).order_by(Order.created_at.desc()).limit(limit))
        ).scalars().all()
        return {
            "orders": [
                {
                    "id": o.id,
                    "symbol_id": o.symbol_id,
                    "side": o.side,
                    "order_type": o.order_type,
                    "amount": float(o.amount),
                    "price": float(o.price) if o.price else None,
                    "status": o.status,
                    "filled_amount": float(o.filled_amount),
                    "filled_price": float(o.filled_price) if o.filled_price else None,
                    "fee": float(o.fee),
                    "stop_loss": float(o.stop_loss) if o.stop_loss else None,
                    "take_profit": float(o.take_profit) if o.take_profit else None,
                    "client_order_id": o.client_order_id,
                    "created_at": o.created_at.isoformat() if o.created_at else None,
                    "closed_at": o.closed_at.isoformat() if o.closed_at else None,
                }
                for o in orders
            ],
            "total": len(orders),
        }


@app.get("/ml/models")
async def list_ml_models(model_type: Optional[str] = None):
    """Список ML моделей."""
    return {"models": await model_registry.list_models(model_type)}


@app.get("/ml/models/{model_type}/active")
async def get_active_ml_model(model_type: str):
    """Получить активную модель."""
    model = await model_registry.get_active_model(model_type)
    if not model:
        raise HTTPException(status_code=404, detail=f"Активная модель {model_type} не найдена")
    return model


@app.post("/ml/models/{model_type}/{version}/activate")
async def activate_ml_model(model_type: str, version: int):
    """Активировать модель."""
    success = await model_registry.activate_model(model_type, version)
    if not success:
        raise HTTPException(status_code=404, detail=f"Модель {model_type} v{version} не найдена")
    return {"success": True, "model_type": model_type, "version": version}


@app.post("/ml/retrain")
async def trigger_retrain():
    """Запустить переобучение моделей."""
    logger.info("Запуск переобучения ML моделей...")
    result = await model_trainer.train_direction_classifier()
    if result:
        logger.info(f"Direction classifier переобучен: v{result['version']}")
        await model_registry.activate_model("direction_classifier", result["version"])
    return {
        "success": result is not None,
        "result": result,
    }


@app.get("/config")
async def get_config():
    """Получить конфигурацию бота."""
    async with get_session() as session:
        configs = (await session.execute(select(BotConfig))).scalars().all()
        return {
            "config": [
                {
                    "key": c.config_key,
                    "value": c.config_value,
                    "source": c.source,
                    "updated_by": c.updated_by,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
                for c in configs
            ]
        }


@app.post("/config/{key}")
async def update_config(key: str, value: Any):
    """Обновить конфигурацию."""
    async with get_session() as session:
        config = (
            await session.execute(select(BotConfig).where(BotConfig.config_key == key))
        ).scalar_one_or_none()
        if config:
            config.config_value = {"value": value}
            config.source = "api"
            config.updated_by = "api"
        else:
            session.add(BotConfig(
                config_key=key,
                config_value={"value": value},
                source="api",
                updated_by="api",
            ))
        await session.commit()

    logger.info(f"Конфигурация обновлена: {key} = {value}")
    return {"success": True, "key": key, "value": value}


@app.post("/trading-mode")
async def set_trading_mode(mode: str):
    """Установить режим торговли (paper/real)."""
    if mode not in ["paper", "real"]:
        raise HTTPException(status_code=400, detail="Режим должен быть paper или real")

    settings.trading_mode = mode
    settings.is_paper = (mode == "paper")

    logger.info(f"Режим торговли изменён на: {mode}")

    # Переинициализация execution engine
    if mode == "real" and execution_engine.is_paper:
        await execution_engine.initialize("binance")

    return {"success": True, "mode": mode}


@app.get("/telegram/channels")
async def list_telegram_channels():
    """Список Telegram каналов."""
    async with get_session() as session:
        channels = (await session.execute(select(TelegramChannel))).scalars().all()
        return {
            "channels": [
                {
                    "id": c.id,
                    "channel_id": c.channel_id,
                    "channel_title": c.channel_title,
                    "parser_type": c.parser_type,
                    "auto_execute": c.auto_execute,
                    "active": c.active,
                    "quality_threshold": c.quality_threshold,
                    "signals_count": len(c.signals),
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in channels
            ]
        }


@app.post("/telegram/channels")
async def create_telegram_channel(channel: TelegramChannelCreate):
    """Добавить Telegram канал."""
    async with get_session() as session:
        existing = (
            await session.execute(
                select(TelegramChannel).where(TelegramChannel.channel_id == channel.channel_id)
            )
        ).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=400, detail=f"Канал {channel.channel_id} уже добавлен")

        new_channel = TelegramChannel(
            channel_id=channel.channel_id,
            channel_title=channel.channel_title,
            parser_type=channel.parser_type,
            parser_config=channel.parser_config,
            quality_threshold=channel.quality_threshold,
            auto_execute=channel.auto_execute,
            active=True,
        )
        session.add(new_channel)
        await session.commit()

        logger.info(f"Telegram канал добавлен: {channel.channel_id}")
        return {"success": True, "channel": {
            "id": new_channel.id,
            "channel_id": new_channel.channel_id,
            "channel_title": new_channel.channel_title,
            "parser_type": new_channel.parser_type,
            "auto_execute": new_channel.auto_execute,
        }}


@app.delete("/telegram/channels/{channel_id}")
async def delete_telegram_channel(channel_id: int):
    """Удалить Telegram канал."""
    async with get_session() as session:
        channel = (
            await session.execute(select(TelegramChannel).where(TelegramChannel.id == channel_id))
        ).scalar_one_or_none()
        if not channel:
            raise HTTPException(status_code=404, detail="Канал не найден")
        await session.delete(channel)
        await session.commit()
        logger.info(f"Telegram канал удалён: {channel.channel_id}")
        return {"success": True}


@app.get("/telegram/signals")
async def list_telegram_signals(limit: int = 100):
    """Список Telegram сигналов."""
    async with get_session() as session:
        signals = (
            await session.execute(
                select(TelegramSignal).order_by(TelegramSignal.created_at.desc()).limit(limit)
            )
        ).scalars().all()
        return {
            "signals": [
                {
                    "id": s.id,
                    "channel_id": s.channel_id,
                    "raw_message": s.raw_message[:200] if s.raw_message else "",
                    "parsed_pair": s.parsed_pair,
                    "parsed_side": s.parsed_side,
                    "parsed_entry": float(s.parsed_entry) if s.parsed_entry else None,
                    "parsed_sl": float(s.parsed_sl) if s.parsed_sl else None,
                    "parsed_tp": float(s.parsed_tp) if s.parsed_tp else None,
                    "quality_score": s.quality_score,
                    "decision": s.decision,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                }
                for s in signals
            ],
            "total": len(signals),
        }


@app.get("/performance")
async def get_performance():
    """Производительность бота (последние снимки)."""
    async with get_session() as session:
        snapshots = (
            await session.execute(
                select(PerformanceSnapshot).order_by(PerformanceSnapshot.snapshot_time.desc()).limit(100)
            )
        ).scalars().all()
        return {
            "snapshots": [
                {
                    "time": s.snapshot_time.isoformat() if s.snapshot_time else None,
                    "total_balance": float(s.total_balance),
                    "open_pnl": float(s.open_pnl),
                    "realized_pnl": float(s.realized_pnl),
                    "daily_pnl": float(s.daily_pnl),
                    "weekly_pnl": float(s.weekly_pnl),
                    "num_open_positions": s.num_open_positions,
                    "num_trades_today": s.num_trades_today,
                    "max_drawdown": s.max_drawdown,
                    "sharpe_ratio": s.sharpe_ratio,
                    "win_rate": s.win_rate,
                }
                for s in snapshots
            ]
        }


@app.get("/analytics/decision-log")
async def get_decision_logs(limit: int = 100, trade_id: Optional[int] = None):
    """Decision logs для анализа."""
    async with get_session() as session:
        from src.db.models import TradeDecisionLog
        query = select(TradeDecisionLog)
        if trade_id:
            query = query.where(TradeDecisionLog.trade_id == trade_id)
        logs = (
            await session.execute(query.order_by(TradeDecisionLog.created_at.desc()).limit(limit))
        ).scalars().all()
        return {
            "logs": [
                {
                    "id": log.id,
                    "trade_id": log.trade_id,
                    "step_order": log.step_order,
                    "step_type": log.step_type,
                    "description": log.description,
                    "details": log.details,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log in logs
            ]
        }


# === Запуск приложения ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.web.api:app",
        host=settings.web_host,
        port=settings.web_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
