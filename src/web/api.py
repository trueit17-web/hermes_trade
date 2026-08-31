"""FastAPI веб-интерфейс — API для управления ботом."""
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.config import settings
from src.db.models import (
    BotConfig,
    Order,
    PerformanceSnapshot,
    TelegramChannel,
    TelegramSignal,
    Trade,
)
from src.db.models import (
    Strategy as StrategyModel,
)
from src.db.session import get_session
from src.event_bus import event_bus
from src.execution.executor import execution_engine
from src.ml import model_registry, model_trainer
from src.risk import expectancy_sizing
from src.risk.protections import channel_key, protection_manager
from src.risk.risk_manager import risk_manager
from src.strategy import strategy_registry
from src.telegram.notifier import send_notification
from src.utils.logging import get_logger_families, get_recent_logs
from src.utils.timeutils import utcnow
from src.web import auth
from src.web.connections_status import get_connections_status
from src.web.settings_store import apply_settings_update, get_settings_snapshot

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

    response = await call_next(request)
    # Без явного no-store браузер/промежуточный прокси может закэшировать
    # GET-ответ (напр. /status) по HTTP-эвристике — дашборд тогда выглядит
    # "обновляется" (fetch успешен, таймстамп в шапке меняется — он чисто
    # клиентский), но реально показывает одни и те же старые данные снова
    # и снова, без единой ошибки в консоли.
    response.headers["Cache-Control"] = "no-store"
    return response


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
            # receive_text() без таймаута ждёт сообщение от клиента
            # неограниченно — фронтенд ничего не шлёт (только слушает
            # broadcast), поэтому соединение простаивало полностью в обе
            # стороны, и обратные прокси/браузер тихо рвали "неактивный"
            # WebSocket через некоторое время (клиентский реконнект после
            # этого работал, но с задержкой и разрывом live-обновлений).
            # Периодический ping от сервера держит соединение активным.
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=25.0)
                logger.debug(f"WebSocket сообщение: {data}")
            except TimeoutError:
                await websocket.send_json({"type": "ping"})
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
    daily_loss_limit_usd: float | None = None
    max_open_positions: int | None = None
    max_position_size_pct: float | None = None
    max_drawdown_pct: float | None = None
    cooldown_seconds: int | None = None


class StrategyToggleRequest(BaseModel):
    """Включение/выключение стратегии."""
    active: bool


class StrategyUpdateRequest(BaseModel):
    """Обновление параметров стратегии."""
    params: dict = Field(default_factory=dict)


class TelegramChannelCreate(BaseModel):
    """Создание Telegram канала."""
    channel_id: str
    channel_title: str | None = None
    parser_type: str = "regex"
    parser_config: dict = Field(default_factory=dict)
    quality_threshold: float = 0.5
    auto_execute: bool = False


class TelegramChannelUpdate(BaseModel):
    """Обновление настроек существующего Telegram канала. Незаданные поля не меняются."""
    channel_title: str | None = None
    quality_threshold: float | None = None
    auto_execute: bool | None = None


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


class ManualOrderCreate(BaseModel):
    """Запрос на открытие сделки вручную со вкладки 'Ручная торговля'."""
    symbol: str
    side: str  # "buy" (long) или "sell" (short, только в paper-режиме)
    order_type: str = "market"  # "market" или "limit"
    amount: float | None = None  # объём напрямую в базовой валюте
    amount_usdt: float | None = None  # альтернатива amount — сумма в USDT
    price: float | None = None  # обязателен для limit; для market — референсная цена, если amount_usdt
    stop_loss: float | None = None
    take_profit: float | None = None


class PositionEditRequest(BaseModel):
    """
    Изменение SL/TP уже открытой позиции. Незаданные поля не меняются.
    symbol — в теле запроса, а не в пути (как и в PositionCloseRequest),
    т.к. символы содержат "/" (напр. "BTC/USDT"), который ломает
    сопоставление пути {symbol} по умолчанию в FastAPI/Starlette.
    """
    symbol: str
    stop_loss: float | None = Field(default=None)
    take_profit: float | None = Field(default=None)
    clear_stop_loss: bool = False
    clear_take_profit: bool = False


# === API эндпоинты ===

@app.get("/")
async def root():
    """Главная страница — статус бота."""
    return {
        "status": "running",
        "mode": settings.trading_mode,
        "timestamp": utcnow().isoformat() + "Z",
    }


def _position_source_label(strategy_id: str | None) -> str:
    """Человекочитаемый источник сигнала по строковому strategy_id (см. executor.py/main.py)."""
    if not strategy_id:
        return "—"
    if strategy_id == "telegram_signal":
        return "📲 Telegram"
    if strategy_id == "manual":
        return "🖐 Ручная"
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
        "timestamp": utcnow().isoformat() + "Z",
    }


@app.get("/status")
async def get_status():
    """Получить текущий статус бота."""
    async with get_session() as session:
        channels = (
            await session.execute(select(TelegramChannel).where(TelegramChannel.active == True))  # noqa: E712
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
        order_ids = [pos["order_id"] for pos in open_positions.values() if pos.get("order_id")]
        exchange_ids_by_order: dict[int, str | None] = {}
        if order_ids:
            async with get_session() as session:
                rows = (
                    await session.execute(
                        select(Order.id, Order.order_id_exchange).where(Order.id.in_(order_ids))
                    )
                ).all()
                exchange_ids_by_order = dict(rows)
        for symbol, pos in open_positions.items():
            pos["source"] = _position_source_label(pos.get("strategy_id"))
            pos["current_price"] = execution_engine.last_prices.get(symbol)
            pos["order_id_exchange"] = exchange_ids_by_order.get(pos.get("order_id"))

    balance = (
        execution_engine.get_paper_balance()
        if settings.is_paper
        else await execution_engine.get_real_balance()
    )

    return {
        "trading_mode": settings.trading_mode,
        "active_trading_mode": settings.active_trading_mode,
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
        "timestamp": utcnow().isoformat() + "Z",
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


@app.post("/manual/order")
async def create_manual_order(request: ManualOrderCreate):
    """
    Открыть сделку вручную со вкладки 'Ручная торговля'. Использует тот же
    execution_engine.create_order(), что и стратегии/Telegram-сигналы, с
    strategy_id="manual" — данные ордера (цена, комиссия, ID на бирже)
    заполняются исключительно из реального ответа биржи (или paper-
    симуляции), как и для любого другого источника сигналов.
    """
    symbol = request.symbol.strip().upper()
    side = request.side.strip().lower()
    if side not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail="side должен быть 'buy' или 'sell'")
    if not settings.is_paper and side == "sell":
        raise HTTPException(
            status_code=400,
            detail="На реальном споте нет встроенного шорта — открыть можно только long (buy)",
        )
    order_type = request.order_type.strip().lower()
    if order_type not in ("market", "limit"):
        raise HTTPException(status_code=400, detail="order_type должен быть 'market' или 'limit'")
    if order_type == "limit" and not request.price:
        raise HTTPException(status_code=400, detail="Для лимитного ордера нужна цена (price)")

    existing = execution_engine.get_open_positions().get(symbol)
    if existing:
        raise HTTPException(status_code=400, detail=f"По {symbol} уже есть открытая позиция")

    if request.amount:
        amount = request.amount
    elif request.amount_usdt:
        ref_price = request.price or execution_engine.last_prices.get(symbol)
        if not ref_price:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Текущая цена {symbol} ещё не известна боту — укажите amount "
                    f"(объём напрямую) или price (для расчёта из amount_usdt)"
                ),
            )
        amount = request.amount_usdt / ref_price
    else:
        raise HTTPException(status_code=400, detail="Укажите amount или amount_usdt")
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Объём сделки должен быть положительным")

    order = await execution_engine.create_order(
        symbol=symbol, side=side, amount=amount, price=request.price,
        order_type=order_type, stop_loss=request.stop_loss, take_profit=request.take_profit,
        strategy_id="manual",
    )
    if order is None:
        raise HTTPException(status_code=500, detail="Не удалось создать ордер — см. логи")

    logger.info(
        f"🖐 Ручной ордер: {side.upper()} {amount:.6f} {symbol} @ {order.filled_price} | "
        f"SL={request.stop_loss} TP={request.take_profit} | ID={order.client_order_id}"
    )

    # Без этого позиция была бы видна только в execution_engine (баланс,
    # /status), а основной торговый цикл узнал бы о ней только на
    # следующем плановом обновлении торговой вселенной — SL/TP не
    # проверялись бы и цена не обновлялась бы до тех пор.
    import src.main as main_module
    if main_module.current_bot is not None:
        await main_module.current_bot.register_manual_position(
            symbol=symbol, side="long" if side == "buy" else "short",
            entry_price=float(order.filled_price), amount=float(order.filled_amount),
            order_id=order.id, entry_fee=float(order.fee),
            stop_loss=request.stop_loss, take_profit=request.take_profit,
        )
    else:
        logger.warning(
            f"⚠️ Ручная позиция {symbol} создана, но бот ещё не готов (current_bot=None) — "
            f"SL/TP начнут проверяться после завершения инициализации бота."
        )

    return {
        "success": True,
        "order_id": order.id,
        "order_id_exchange": order.order_id_exchange,
        "symbol": symbol,
        "amount": float(order.filled_amount),
        "price": float(order.filled_price),
    }


@app.post("/positions/edit")
async def edit_position(request: PositionEditRequest):
    """
    Изменить Stop Loss / Take Profit уже открытой позиции (любого
    источника, не только ручных сделок). Обновляет и execution_engine
    (paper_positions/real_positions — источник для /status), и открытые
    позиции основного торгового цикла (main.current_bot.open_positions —
    именно их читает _check_position_exit при проверке SL/TP), иначе
    изменение подействовало бы только на дашборд, но не на сам бот.
    """
    symbol = request.symbol
    tracked = execution_engine.paper_positions if settings.is_paper else execution_engine.real_positions
    position = tracked.get(symbol)
    if position is None:
        raise HTTPException(status_code=404, detail=f"Открытая позиция {symbol} не найдена")

    updates = request.model_dump(exclude_unset=True)
    new_sl = position.get("stop_loss")
    new_tp = position.get("take_profit")
    if request.clear_stop_loss:
        new_sl = None
    elif "stop_loss" in updates:
        new_sl = request.stop_loss
    if request.clear_take_profit:
        new_tp = None
    elif "take_profit" in updates:
        new_tp = request.take_profit

    position["stop_loss"] = new_sl
    position["take_profit"] = new_tp

    if not settings.is_paper:
        # Ручное изменение SL должно сразу отражаться и на бирже — иначе
        # выставленный ранее условный ордер продолжил бы защищать позицию
        # по старой, уже неактуальной цене.
        await execution_engine.sync_stop_loss_order(symbol, position.get("amount") or 0, new_sl)

    import src.main as main_module
    bot_position = (
        main_module.current_bot.open_positions.get(symbol)
        if main_module.current_bot is not None else None
    )
    if bot_position is not None:
        bot_position["sl"] = new_sl
        bot_position["tp"] = new_tp

    logger.info(f"✏️ SL/TP {symbol} изменены вручную: SL={new_sl} TP={new_tp}")
    return {"success": True, "symbol": symbol, "stop_loss": new_sl, "take_profit": new_tp}


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
async def list_trades(limit: int = 100, offset: int = 0, strategy_id: str | None = None):
    """
    Список сделок, сгруппированных по позиции (order_open_id): частичные
    закрытия одной позиции по уровням TP1/TP2/TP3 показываются одной
    строкой с суммарными объёмом/PnL, а не как отдельные сделки.

    strategy_id — опциональный фильтр по строковому идентификатору
    источника (напр. "manual" для вкладки "Ручная торговля",
    "telegram_signal" для Telegram-сигналов) — сверяется с именем
    связанной Strategy, под которым его создаёт
    ExecutionEngine._resolve_strategy_id().
    """
    async with get_session() as session:
        # Берём сырые Trade-строки с запасом, чтобы после группировки (до
        # 3 частей на позицию) точно хватило на limit+offset готовых строк.
        query = (
            select(Trade)
            .options(
                selectinload(Trade.symbol), selectinload(Trade.strategy),
                selectinload(Trade.order_open), selectinload(Trade.order_close),
            )
            .order_by(Trade.closed_at.desc(), Trade.created_at.desc())
            .limit((limit + offset) * 3 + 50)
        )
        if strategy_id:
            query = query.join(StrategyModel, Trade.strategy_id == StrategyModel.id).where(
                StrategyModel.name == strategy_id
            )
        raw_trades = (await session.execute(query)).scalars().all()

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
            # Trade.created_at — это момент вставки строки Trade в БД, а
            # Trade создаётся только при ЗАКРЫТИИ позиции (в
            # close_paper_position/close_real_position) — то есть почти
            # совпадает с closed_at. Настоящее время открытия — это
            # created_at связанного открывающего Order, который создаётся
            # в момент реального входа в позицию.
            opened_at = first.order_open.created_at if first.order_open else first.created_at
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
                "order_id_exchange_open": first.order_open.order_id_exchange if first.order_open else None,
                "order_id_exchange_close": last.order_close.order_id_exchange if last.order_close else None,
                "created_at": opened_at.isoformat() + "Z" if opened_at else None,
                "closed_at": last.closed_at.isoformat() + "Z" if last.closed_at else None,
                "_sort_key": (last.closed_at or last.created_at).isoformat() + "Z",
            })

        aggregated.sort(key=lambda r: r["_sort_key"], reverse=True)
        page = aggregated[offset:offset + limit]
        for row in page:
            del row["_sort_key"]

        return {
            "trades": page,
            "total": len(aggregated),
        }


@app.get("/trades/{trade_id}/detail")
async def get_trade_detail(trade_id: int):
    """
    Подробности по сделке для разворачиваемой строки на дашборде: сама
    сделка — это последняя (закрывающая) часть позиции, возможно
    частично закрытой по уровням TP1/TP2/TP3 (см. GET /trades); здесь
    подтягиваем ВСЕ части той же позиции (по order_open_id) и decision
    log по каждой из них — каждое частичное закрытие пишет свой decision
    log под своим Trade.id, поэтому лог по последней части в одиночку
    не показал бы, почему сработали более ранние уровни TP.
    """
    async with get_session() as session:
        from src.db.models import TradeDecisionLog

        trade = (
            await session.execute(
                select(Trade)
                .options(
                    selectinload(Trade.symbol), selectinload(Trade.strategy),
                    selectinload(Trade.order_open), selectinload(Trade.order_close),
                )
                .where(Trade.id == trade_id)
            )
        ).scalar_one_or_none()
        if trade is None:
            raise HTTPException(status_code=404, detail="Сделка не найдена")

        def _order_card(o):
            """
            Все доступные поля ордера/транзакции с биржи — для карточки
            "открывающий/закрывающий ордер" в развёрнутой сделке на
            дашборде. filled_price/filled_amount — то, что реально
            произошло на бирже (после сегодняшнего фикса подтверждения
            через историю сделок — реальная цена и комиссия, а не
            запрошенная цена/0 как раньше); price/amount — то, что было
            запрошено ботом.
            """
            if o is None:
                return None
            return {
                "id": o.id,
                "order_id_exchange": o.order_id_exchange,
                "client_order_id": o.client_order_id,
                "side": o.side,
                "order_type": o.order_type,
                "status": o.status,
                "amount_requested": float(o.amount) if o.amount is not None else None,
                "price_requested": float(o.price) if o.price is not None else None,
                "filled_amount": float(o.filled_amount) if o.filled_amount is not None else None,
                "filled_price": float(o.filled_price) if o.filled_price is not None else None,
                "fee": float(o.fee) if o.fee is not None else None,
                "fee_currency": o.fee_currency,
                "stop_loss": float(o.stop_loss) if o.stop_loss is not None else None,
                "take_profit": float(o.take_profit) if o.take_profit is not None else None,
                "notes": o.notes,
                "created_at": o.created_at.isoformat() + "Z" if o.created_at else None,
            }

        if trade.order_open_id is not None:
            group = (
                await session.execute(
                    select(Trade)
                    .options(selectinload(Trade.order_close))
                    .where(Trade.order_open_id == trade.order_open_id)
                    .order_by(Trade.closed_at.asc(), Trade.created_at.asc())
                )
            ).scalars().all()
        else:
            group = [trade]

        leg_ids = [t.id for t in group]
        logs = (
            await session.execute(
                select(TradeDecisionLog)
                .where(TradeDecisionLog.trade_id.in_(leg_ids))
                .order_by(TradeDecisionLog.created_at.asc(), TradeDecisionLog.step_order.asc())
            )
        ).scalars().all()

        total_amount = sum(float(t.amount) for t in group)
        total_pnl = sum(float(t.pnl) for t in group)
        entry_price = float(group[0].entry_price)
        # Trade.created_at — момент вставки строки Trade в БД (при
        # ЗАКРЫТИИ позиции), а не настоящее время открытия — см. тот же
        # комментарий в GET /trades. Настоящее время входа — created_at
        # связанного открывающего Order.
        opened_at = trade.order_open.created_at if trade.order_open else group[0].created_at

        return {
            "trade_id": trade_id,
            "symbol": trade.symbol.symbol if trade.symbol else None,
            "direction": trade.direction,
            "source": _position_source_label(trade.strategy.name if trade.strategy else None),
            "entry_price": entry_price,
            "amount": total_amount,
            "pnl": total_pnl,
            "pnl_pct": (total_pnl / (entry_price * total_amount) * 100) if entry_price and total_amount else 0.0,
            "outcome": trade.outcome,
            "is_open": trade.is_open,
            "order_id_exchange_open": trade.order_open.order_id_exchange if trade.order_open else None,
            "created_at": opened_at.isoformat() + "Z" if opened_at else None,
            "closed_at": trade.closed_at.isoformat() + "Z" if trade.closed_at else None,
            "opening_order": _order_card(trade.order_open),
            "legs": [
                {
                    "id": t.id,
                    "exit_price": float(t.exit_price) if t.exit_price is not None else None,
                    "amount": float(t.amount),
                    "pnl": float(t.pnl),
                    "pnl_pct": t.pnl_pct,
                    "outcome": t.outcome,
                    "holding_seconds": t.holding_seconds,
                    "order_id_exchange_close": t.order_close.order_id_exchange if t.order_close else None,
                    "closed_at": t.closed_at.isoformat() + "Z" if t.closed_at else None,
                    "closing_order": _order_card(t.order_close),
                }
                for t in group
            ],
            "decision_log": [
                {
                    "trade_id": log.trade_id,
                    "step_order": log.step_order,
                    "step_type": log.step_type,
                    "description": log.description,
                    "details": log.details,
                    "created_at": log.created_at.isoformat() + "Z" if log.created_at else None,
                }
                for log in logs
            ],
        }


@app.post("/trades/{trade_id}/recalculate")
async def recalculate_trade(trade_id: int):
    """
    Перезапросить у биржи реальные цену/объём/комиссию для ордеров этой
    (уже закрытой) сделки и пересчитать PnL — ручной способ подтянуть
    точные данные постфактум, если изначально они были записаны по оценке
    (см. ExecutionEngine.recalculate_closed_trade). Только для real-режима.
    """
    if settings.is_paper:
        raise HTTPException(status_code=400, detail="Доступно только в real-режиме — в paper реальных данных с биржи нет")
    result = await execution_engine.recalculate_closed_trade(trade_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Сделка не найдена или у неё нет обоих ордеров")
    return result


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
                    "created_at": log.created_at.isoformat() + "Z" if log.created_at else None,
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
                    "created_at": o.created_at.isoformat() + "Z" if o.created_at else None,
                    "closed_at": o.closed_at.isoformat() + "Z" if o.closed_at else None,
                }
                for o in orders
            ],
            "total": len(orders),
        }


@app.get("/ml/models")
async def list_ml_models(model_type: str | None = None):
    """Список ML моделей."""
    return {"models": await model_registry.list_models(model_type)}


@app.get("/ml/training-readiness")
async def get_ml_training_readiness(symbol: str | None = None):
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
    """Запустить переобучение моделей (direction_classifier + volatility_predictor)."""
    logger.info("Запуск переобучения ML моделей...")
    result = await model_trainer.train_direction_classifier()
    if result:
        logger.info(f"Direction classifier переобучен: v{result['version']}")
        await model_registry.activate_model("direction_classifier", result["version"])

    vol_result = await model_trainer.train_volatility_predictor()
    if vol_result:
        logger.info(f"Volatility predictor переобучен: v{vol_result['version']}")
        await model_registry.activate_model("volatility_predictor", vol_result["version"])

    return {
        "success": result is not None or vol_result is not None,
        "result": result,
        "volatility_result": vol_result,
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
                    "updated_at": c.updated_at.isoformat() + "Z" if c.updated_at else None,
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


@app.post("/trading-source-mode")
async def set_trading_source_mode(mode: str):
    """
    Переключить источник новых торговых сигналов: signals (только
    Telegram-каналы) или algo (только встроенные ML/Ensemble/BB-стратегии).
    Уже открытые позиции обоих источников продолжают отслеживаться (SL/TP)
    независимо от режима — переключатель гейтит только открытие НОВЫХ
    позиций (см. TradingBot._process_symbol/_on_telegram_signal в main.py).
    Тот же apply_settings_update, что и вкладка "Настройки" — применяется
    немедленно и сохраняется на будущие перезапуски.
    """
    if mode not in ("signals", "algo"):
        raise HTTPException(status_code=400, detail="Режим должен быть signals или algo")

    result = await apply_settings_update({"active_trading_mode": mode})
    if result["errors"]:
        raise HTTPException(status_code=400, detail=result["errors"])

    logger.info(f"Источник торговых сигналов изменён на: {mode}")
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
    level: str | None = None,
    search: str | None = None,
    loggers: str | None = None,
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


@app.post("/system/redeploy")
async def redeploy_bot():
    """
    Запросить редеплой (git pull + docker compose build/up + alembic
    upgrade) через отдельный деплой-агент, который работает ВНЕ контейнера
    бота (см. scripts/deploy_agent.py, docker-compose.yml). Сам бот
    намеренно не получает доступа к docker.sock хоста — только шлёт
    HTTP-запрос агенту с общим секретом; агент запускает деплой в фоне и
    отвечает сразу же, не дожидаясь окончания (сборка образа может занять
    заметное время) — прогресс можно смотреть через /system/redeploy/status.
    """
    if not settings.deploy_agent_url:
        raise HTTPException(
            status_code=503,
            detail="Деплой-агент не настроен (DEPLOY_AGENT_URL) — см. scripts/deploy_agent.py и docker-compose.yml",
        )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.deploy_agent_url}/deploy",
                headers={"Authorization": f"Bearer {settings.deploy_agent_token or ''}"},
            )
        if resp.status_code == 409:
            return {"success": False, "message": "Деплой уже выполняется"}
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Деплой-агент отклонил запрос: {e.response.status_code} {e.response.text}",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с деплой-агентом: {e}") from e


@app.get("/system/redeploy/status")
async def redeploy_status():
    """Текущее состояние деплой-агента (идёт ли деплой сейчас, код выхода
    последнего, хвост лога) — для прогресса кнопки "Редеплой" в дашборде."""
    if not settings.deploy_agent_url:
        raise HTTPException(status_code=503, detail="Деплой-агент не настроен (DEPLOY_AGENT_URL)")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.deploy_agent_url}/status",
                headers={"Authorization": f"Bearer {settings.deploy_agent_token or ''}"},
            )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с деплой-агентом: {e}") from e


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
                    "created_at": c.created_at.isoformat() + "Z" if c.created_at else None,
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


@app.patch("/telegram/channels/{channel_id}")
async def update_telegram_channel(channel_id: int, update: TelegramChannelUpdate):
    """
    Изменить настройки существующего канала (порог качества, автоисполнение,
    название). quality_threshold/auto_execute читаются "вживую" из БД при
    обработке каждого Telegram-сигнала (см. TradingBot._get_channel_quality_threshold
    в main.py), поэтому применяются сразу, без перезапуска бота — в отличие
    от списка отслеживаемых каналов, который фиксируется при старте.
    """
    async with get_session() as session:
        channel = await session.get(TelegramChannel, channel_id)
        if channel is None:
            raise HTTPException(status_code=404, detail="Канал не найден")

        updates = update.model_dump(exclude_unset=True)
        for key, value in updates.items():
            setattr(channel, key, value)
        await session.commit()

    logger.info(f"Telegram канал {channel_id} обновлён: {updates}")
    return {"success": True, "updated": list(updates.keys())}


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
async def list_telegram_signals(channel_id: int | None = None, limit: int = 100):
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
                "created_at": order.created_at.isoformat() + "Z" if order.created_at else None,
            }

        def _trade_data(trade):
            if trade is None:
                return None
            return {
                "pnl": float(trade.pnl),
                "pnl_pct": float(trade.pnl_pct) if trade.pnl_pct else 0,
                "outcome": trade.outcome,
                "is_open": trade.is_open,
                "closed_at": trade.closed_at.isoformat() + "Z" if trade.closed_at else None,
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
                    "created_at": s.created_at.isoformat() + "Z" if s.created_at else None,
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
                    "time": s.snapshot_time.isoformat() + "Z" if s.snapshot_time else None,
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
async def get_decision_logs(limit: int = 100, trade_id: int | None = None):
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
                    "created_at": log.created_at.isoformat() + "Z" if log.created_at else None,
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
