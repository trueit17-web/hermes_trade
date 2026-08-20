"""FastAPI веб-интерфейс — API для управления ботом."""
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.config import settings
from src.event_bus import event_bus
from src.risk.risk_manager import risk_manager
from src.risk.protections import protection_manager, channel_key
from src.risk import expectancy_sizing
from src.execution.executor import execution_engine
from src.ml import model_registry, model_trainer
from src.strategy import strategy_registry
from src.utils.logging import logger, get_recent_logs, get_logger_families
from src.utils.timeutils import utcnow
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
from src.web import auth
from src.web.settings_store import get_settings_snapshot, apply_settings_update
from src.web.connections_status import get_connections_status
from src.telegram.notifier import send_notification

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

# Пути, доступные без авторизации (страница логина сама, health-check для
# docker-compose, и сам эндпоинт логина).
_PUBLIC_PATHS = {"/login", "/auth/login", "/health"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Простая cookie-авторизация: закрывает всю панель, кроме /login и /health."""
    path = request.url.path
    if path in _PUBLIC_PATHS or path.startswith(("/docs", "/openapi", "/redoc")):
        return await call_next(request)

    token = request.cookies.get(auth.SESSION_COOKIE_NAME)
    if not auth.verify_session(token):
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse(url="/login")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)


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
    token = websocket.cookies.get(auth.SESSION_COOKIE_NAME)
    if not auth.verify_session(token):
        await websocket.close(code=4401)
        return

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

class LoginRequest(BaseModel):
    """Запрос на вход в панель."""
    username: str
    password: str


class SettingsUpdateRequest(BaseModel):
    """Запрос на обновление настроек бота (вкладка «Настройки»)."""
    values: dict[str, Any] = Field(default_factory=dict)


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


class PositionCloseRequest(BaseModel):
    """Запрос на ручное закрытие открытой paper-позиции."""
    symbol: str


# === API эндпоинты ===

@app.get("/")
async def root():
    """Главная страница — статус бота."""
    return {
        "status": "running",
        "mode": settings.trading_mode,
        "timestamp": utcnow().isoformat(),
    }


def _position_source_label(strategy_id: Optional[str]) -> str:
    """Человекочитаемый источник сигнала по строковому strategy_id (см. executor.py/main.py)."""
    if not strategy_id:
        return "—"
    if strategy_id == "telegram_signal":
        return "📲 Telegram"
    strategy = strategy_registry.get(strategy_id)
    return strategy.name if strategy else strategy_id


DASHBOARD_HTML_PATH = Path(__file__).parent / "static" / "dashboard.html"
LOGIN_HTML_PATH = Path(__file__).parent / "static" / "login.html"


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Веб-дашборд управления ботом (статус, позиции, сделки, стратегии, риск-контролы)."""
    return HTMLResponse(DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Страница входа в панель."""
    return HTMLResponse(LOGIN_HTML_PATH.read_text(encoding="utf-8"))


@app.post("/auth/login")
async def login(payload: LoginRequest):
    """Проверить логин/пароль и выдать cookie-сессию."""
    if not auth.authenticate(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

    token = auth.create_session()
    response = JSONResponse(content={"success": True})
    response.set_cookie(
        key=auth.SESSION_COOKIE_NAME,
        value=token,
        max_age=auth.SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.web_cookie_secure,
    )
    return response


@app.post("/auth/logout")
async def logout(request: Request):
    """Завершить сессию."""
    auth.revoke_session(request.cookies.get(auth.SESSION_COOKIE_NAME))
    response = JSONResponse(content={"success": True})
    response.delete_cookie(auth.SESSION_COOKIE_NAME)
    return response


@app.get("/health")
async def health():
    """Проверка здоровья."""
    return {
        "status": "ok",
        "trading_mode": settings.trading_mode,
        "bot_running": True,
        "timestamp": utcnow().isoformat(),
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

    # Ключи "paper_balance"/"paper_positions" сохранены для совместимости с
    # дашбордом, но теперь содержат данные текущего режима (paper или real) —
    # раньше в real-режиме здесь всегда было None, и открытые позиции были
    # попросту невидимы в дашборде.
    open_positions = execution_engine.get_open_positions()
    if open_positions:
        for symbol, pos in open_positions.items():
            pos["source"] = _position_source_label(pos.get("strategy_id"))
            pos["current_price"] = execution_engine.last_prices.get(symbol)

    balance = (
        execution_engine.get_paper_balance()
        if settings.is_paper
        else await execution_engine.get_real_balance()
    )

    return {
        "trading_mode": settings.trading_mode,
        "is_paper": settings.is_paper,
        "startup_capital": settings.startup_capital_usdt,
        "risk_state": risk_manager.get_state(),
        "paper_balance": balance,
        "paper_positions": open_positions,
        "active_strategies": strategy_registry.list_strategies(),
        "ml_models": await model_registry.list_models(),
        "ml_active_model": await model_registry.get_active_model("direction_classifier"),
        "telegram_channels": telegram_channels,
        "event_bus_history_size": len(event_bus.get_history()),
        "timestamp": utcnow().isoformat(),
    }


@app.post("/paper/reset")
async def reset_paper_account():
    """
    Полностью сбросить paper-аккаунт: удалить всю paper-историю ордеров и
    сделок, вернуть баланс к startup_capital_usdt, сбросить просадку/дневной
    PnL и снять паузу (если её причиной была именно эта просадка).
    Необратимо — реальные (real-режим) данные не затрагиваются.
    """
    if not settings.is_paper:
        raise HTTPException(status_code=400, detail="Доступно только в paper-режиме")

    result = await execution_engine.reset_paper_account()
    risk_manager.reset_for_new_paper_account()

    logger.warning(f"🔄 Paper-аккаунт сброшен через веб-панель: {result}")
    return {"success": True, **result}


@app.post("/positions/close")
async def close_position_manually(request: PositionCloseRequest):
    """Закрыть открытую позицию вручную (кнопка в дашборде)."""
    symbol = request.symbol
    tracked = execution_engine.paper_positions if settings.is_paper else execution_engine.real_positions
    position = tracked.get(symbol)
    if not position:
        raise HTTPException(status_code=404, detail=f"Открытая позиция {symbol} не найдена")

    opened_at = position.get("opened_at")
    holding_seconds = int((utcnow() - opened_at).total_seconds()) if opened_at else 0

    if settings.is_paper:
        current_price = execution_engine.last_prices.get(symbol)
        if current_price is None:
            raise HTTPException(
                status_code=400,
                detail="Текущая цена по этому символу ещё не известна — подождите следующей торговой итерации",
            )
        result = await execution_engine.close_paper_position(
            symbol=symbol,
            side=position["side"],
            entry_price=position["entry_price"],
            amount=position["amount"],
            exit_price=current_price,
            reason="manual",
            entry_fee=position.get("entry_fee", 0.0),
            holding_seconds=holding_seconds,
            strategy_id=position.get("strategy_id"),
            order_open_id=position.get("order_id"),
        )
    else:
        # Реальное закрытие исполняется рыночным ордером на бирже —
        # фактическая цена выхода определяется биржей, а не последней
        # известной ценой из основного цикла.
        result = await execution_engine.close_real_position(
            symbol=symbol,
            side=position["side"],
            entry_price=position["entry_price"],
            amount=position["amount"],
            reason="manual",
            entry_fee=position.get("entry_fee", 0.0),
            holding_seconds=holding_seconds,
            strategy_id=position.get("strategy_id"),
            order_open_id=position.get("order_id"),
        )
    if result is None:
        raise HTTPException(status_code=500, detail="Не удалось закрыть позицию")

    risk_manager.on_position_closed(symbol)
    risk_manager.on_trade_closed(result["pnl"])

    if result.get("trade_id"):
        from src.execution.decision_logger import decision_logger
        await decision_logger.flush_for_trade(
            position.get("order_id"), result["trade_id"],
            close_description=f"Позиция закрыта вручную | PnL {result['pnl']:+.2f} ({result['pnl_pct']:+.2f}%)",
            close_details={
                "reason": "manual", "pnl": result["pnl"],
                "pnl_pct": result["pnl_pct"], "outcome": result.get("outcome"),
            },
        )

    # Если позиция была открыта по Telegram-сигналу — довязать исход сделки
    # к сигналу для статистики канала (обычно это делает основной цикл при
    # автозакрытии по SL/TP, здесь закрытие идёт в обход него).
    order_id = position.get("order_id")
    if order_id and result.get("trade_id"):
        async with get_session() as session:
            sig = (
                await session.execute(
                    select(TelegramSignal)
                    .options(selectinload(TelegramSignal.channel))
                    .where(TelegramSignal.executed_order_id == order_id)
                )
            ).scalar_one_or_none()
            if sig:
                sig.executed_trade_id = result["trade_id"]
                channel_id = sig.channel.channel_id if sig.channel else None
                await session.commit()
                if channel_id:
                    from src.telegram.quality_scorer import signal_quality_scorer
                    signal_quality_scorer.update_channel_stats(channel_id, result.get("outcome") == "win")

    emoji = "✅" if result["pnl"] > 0 else "❌"
    logger.info(f"Позиция {symbol} закрыта вручную через веб-панель | PnL: {result['pnl']:+.2f}")
    await send_notification(
        f"{emoji} Закрыта вручную {position['side'].upper()} {symbol}\n"
        f"PnL: {result['pnl']:+.2f} USDT ({result['pnl_pct']:+.2f}%)"
    )

    return {"success": True, **result}


@app.get("/risk/state")
async def get_risk_state():
    """Текущее состояние риска."""
    return risk_manager.get_state()


@app.get("/risk/protections")
async def get_protections():
    """Активные блокировки Protections (cooldown/StoplossGuard/LosingStreak)."""
    return {"enabled": settings.protections_enabled, "locks": await protection_manager.locks.active_locks()}


@app.post("/risk/configure")
async def configure_risk(config: RiskConfigUpdate):
    """Обновить риск-конфигурацию."""
    # Раньше этот endpoint писал изменения в BotConfig под короткими
    # ключами (daily_loss_limit_usd и т.д.), а load_settings_overrides()
    # при старте бота ищет только ключи из SETTINGS_SCHEMA (risk_*) — эти
    # записи никогда не подхватывались и молча пропадали при рестарте.
    # Используем тот же путь сохранения, что и вкладка "Настройки" в
    # дашборде, чтобы изменение реально переживало рестарт.
    field_map = {
        "daily_loss_limit_usd": "risk_daily_loss_limit_usd",
        "max_open_positions": "risk_max_open_positions",
        "max_position_size_pct": "risk_max_position_size_pct",
        "max_drawdown_pct": "risk_max_drawdown_pct",
        "cooldown_seconds": "risk_cooldown_seconds",
    }
    updates = {
        schema_key: getattr(config, short_key)
        for short_key, schema_key in field_map.items()
        if getattr(config, short_key) is not None
    }

    result = await apply_settings_update(updates)
    if result["errors"]:
        raise HTTPException(status_code=400, detail=result["errors"])

    logger.info(f"Risk конфигурация обновлена: {updates}")
    return {"success": True, "config": {k: getattr(settings, k) for k in updates}}


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
    """
    Список сделок, сгруппированных по позиции (order_open_id): частичные
    закрытия одной позиции по уровням TP1/TP2/TP3 показываются одной
    строкой с суммарными объёмом/PnL, а не как отдельные сделки.
    """
    async with get_session() as session:
        # Берём сырые Trade-строки с запасом, чтобы после группировки (до
        # 3 частей на позицию) точно хватило на limit+offset готовых строк.
        raw_trades = (
            await session.execute(
                select(Trade)
                .options(selectinload(Trade.symbol), selectinload(Trade.strategy))
                .order_by(Trade.closed_at.desc(), Trade.created_at.desc())
                .limit((limit + offset) * 3 + 50)
            )
        ).scalars().all()

        groups: dict = {}
        for t in raw_trades:
            key = t.order_open_id if t.order_open_id is not None else f"single-{t.id}"
            groups.setdefault(key, []).append(t)

        aggregated = []
        for group in groups.values():
            group.sort(key=lambda t: t.closed_at or t.created_at)
            first, last = group[0], group[-1]
            total_amount = sum(float(t.amount) for t in group)
            total_pnl = sum(float(t.pnl) for t in group)
            entry_price = float(first.entry_price)
            priced = [t for t in group if t.exit_price is not None]
            exit_price = (
                sum(float(t.exit_price) * float(t.amount) for t in priced) / sum(float(t.amount) for t in priced)
                if priced else None
            )
            pnl_pct = (total_pnl / (entry_price * total_amount) * 100) if entry_price and total_amount else 0.0
            outcome = "win" if total_pnl > 0 else ("loss" if total_pnl < 0 else "break-even")
            aggregated.append({
                "id": last.id,
                "symbol_id": last.symbol_id,
                "symbol": last.symbol.symbol if last.symbol else None,
                "direction": last.direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "amount": total_amount,
                "pnl": total_pnl,
                "pnl_pct": pnl_pct,
                "holding_seconds": last.holding_seconds,
                "outcome": outcome,
                "is_open": last.is_open,
                "parts": len(group),
                "source": _position_source_label(last.strategy.name if last.strategy else None),
                "created_at": first.created_at.isoformat() if first.created_at else None,
                "closed_at": last.closed_at.isoformat() if last.closed_at else None,
                "_sort_key": (last.closed_at or last.created_at).isoformat(),
            })

        aggregated.sort(key=lambda r: r["_sort_key"], reverse=True)
        page = aggregated[offset:offset + limit]
        for row in page:
            del row["_sort_key"]

        return {
            "trades": page,
            "total": len(aggregated),
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
            await session.execute(
                select(Order)
                .options(selectinload(Order.symbol))
                .order_by(Order.created_at.desc())
                .limit(limit)
            )
        ).scalars().all()
        return {
            "orders": [
                {
                    "id": o.id,
                    "symbol_id": o.symbol_id,
                    "symbol": o.symbol.symbol if o.symbol else None,
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


@app.get("/ml/training-readiness")
async def get_ml_training_readiness(symbol: Optional[str] = None):
    """
    Сколько данных сейчас накоплено для обучения ML-моделей и сколько нужно —
    отвечает на вопрос "почему модель не обучилась/не загружена" без
    необходимости смотреть в БД напрямую.
    """
    return await model_trainer.feature_store.get_training_readiness(symbol)


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

    # Раньше это применялось только к живому settings.trading_mode и никогда
    # не сохранялось в BotConfig — после рестарта бот тихо возвращался в
    # режим из .env. apply_settings_update — тот же путь, что и вкладка
    # "Настройки", уже применяет live-эффект (переинициализацию execution
    # engine) и сохраняет для будущих рестартов.
    result = await apply_settings_update({"trading_mode": mode})
    if result["errors"]:
        raise HTTPException(status_code=400, detail=result["errors"])

    logger.info(f"Режим торговли изменён на: {mode}")
    return {"success": True, "mode": mode}


@app.get("/settings")
async def get_settings():
    """Все редактируемые настройки бота (для вкладки «Настройки»), сгруппированные."""
    return {"settings": get_settings_snapshot()}


@app.post("/settings")
async def update_settings(request: SettingsUpdateRequest):
    """Обновить настройки бота — применяется немедленно и сохраняется на будущие перезапуски."""
    result = await apply_settings_update(request.values)
    return {"success": not result["errors"], **result}


@app.get("/logs")
async def get_logs(
    level: Optional[str] = None,
    search: Optional[str] = None,
    loggers: Optional[str] = None,
    limit: int = 200,
):
    """Последние логи процесса (из ring-буфера в памяти) с фильтрами для веб-панели.

    loggers — список выбранных в чекбокс-фильтре 'семейств' логгеров через
    запятую; параметр отсутствует = без фильтра, пустая строка = ничего
    не выбрано (показать пусто).
    """
    logger_families = [x for x in loggers.split(",") if x] if loggers is not None else None
    return {
        "logs": get_recent_logs(
            level=level, search=search, loggers=logger_families, limit=min(limit, 2000)
        )
    }


@app.get("/logs/loggers")
async def get_log_loggers():
    """Список 'семейств' логгеров для чекбокс-фильтра в веб-панели."""
    return {"loggers": get_logger_families()}


@app.get("/connections/status")
async def connections_status():
    """Статусы внешних подключений (БД, биржа, Telegram, CoinGlass)."""
    return {"connections": await get_connections_status()}


@app.post("/system/restart")
async def restart_bot():
    """Перезапустить процесс бота (например, чтобы подхватить добавленные/удалённые
    Telegram-каналы — их список фиксируется при старте). Полагается на
    `restart: unless-stopped` в docker-compose.yml: процесс просто завершается,
    Docker поднимает контейнер заново. Без такой restart-политики (голый
    `python -m src.main` или systemd без Restart=) бот после этого не поднимется сам."""
    logger.warning("🔄 Перезапуск бота запрошен через веб-панель")

    async def _delayed_exit():
        await asyncio.sleep(1)  # даём HTTP-ответу время дойти до клиента
        os._exit(0)

    asyncio.create_task(_delayed_exit())
    return {"success": True, "message": "Бот перезапускается"}


@app.get("/telegram/channels")
async def list_telegram_channels():
    """Список Telegram каналов."""
    async with get_session() as session:
        channels = (
            await session.execute(
                select(TelegramChannel).options(selectinload(TelegramChannel.signals))
            )
        ).scalars().all()
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


@app.get("/telegram/channels/stats")
async def telegram_channels_stats():
    """Статистика по каждому Telegram-каналу: сигналы, исполнение, win rate, PnL."""
    async with get_session() as session:
        channels = (await session.execute(select(TelegramChannel))).scalars().all()

        result = []
        for c in channels:
            signals = (
                await session.execute(
                    select(TelegramSignal)
                    .options(selectinload(TelegramSignal.executed_trade))
                    .where(TelegramSignal.channel_id == c.id)
                )
            ).scalars().all()

            executed = [s for s in signals if s.decision == "executed"]
            closed_trades = [s.executed_trade for s in executed if s.executed_trade is not None]
            wins = sum(1 for t in closed_trades if t.outcome == "win")
            scored = [s.quality_score for s in signals if s.quality_score is not None]

            result.append({
                "channel_id": c.id,
                "total_signals": len(signals),
                "executed": len(executed),
                "closed_trades": len(closed_trades),
                "win_rate": round(wins / len(closed_trades) * 100, 1) if closed_trades else None,
                "total_pnl": round(sum(float(t.pnl) for t in closed_trades), 2) if closed_trades else None,
                "avg_quality": round(sum(scored) / len(scored), 2) if scored else None,
                "size_multiplier": await expectancy_sizing.size_multiplier(channel_key(c.channel_id)),
            })

        return {"channels": result}


@app.get("/telegram/signals")
async def list_telegram_signals(channel_id: Optional[int] = None, limit: int = 100):
    """Список Telegram сигналов (опционально по одному каналу) с данными ордера и исхода сделки."""
    async with get_session() as session:
        query = (
            select(TelegramSignal)
            .options(
                selectinload(TelegramSignal.executed_order),
                selectinload(TelegramSignal.executed_trade),
            )
            .order_by(TelegramSignal.created_at.desc())
            .limit(limit)
        )
        if channel_id is not None:
            query = query.where(TelegramSignal.channel_id == channel_id)
        signals = (await session.execute(query)).scalars().all()

        def _order_data(order):
            if order is None:
                return None
            return {
                "id": order.id,
                "side": order.side,
                "order_type": order.order_type,
                "amount": float(order.amount),
                "price": float(order.price) if order.price else None,
                "status": order.status,
                "filled_price": float(order.filled_price) if order.filled_price else None,
                "fee": float(order.fee),
                "stop_loss": float(order.stop_loss) if order.stop_loss else None,
                "take_profit": float(order.take_profit) if order.take_profit else None,
                "created_at": order.created_at.isoformat() if order.created_at else None,
            }

        def _trade_data(trade):
            if trade is None:
                return None
            return {
                "pnl": float(trade.pnl),
                "pnl_pct": float(trade.pnl_pct) if trade.pnl_pct else 0,
                "outcome": trade.outcome,
                "is_open": trade.is_open,
                "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
            }

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
                    "order": _order_data(s.executed_order),
                    "trade": _trade_data(s.executed_trade),
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
