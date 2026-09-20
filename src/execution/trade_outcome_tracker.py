"""
Отслеживание движения цены ПОСЛЕ закрытия сделки — по запросу пользователя:
понять постфактум, как нужно было бы выставить вход/SL/TP, чтобы поймать
максимум доступной прибыли (а не то, что реально зафиксировано), и
использовать эти данные для (1) дашборда с разбором сделок и (2) контекста
LLM-фолбэка парсера сигналов (см. channel_outcome_context.py) — какие
каналы систематически выходят слишком рано или со слишком тесным SL.

Одна строка TradeOutcomeTracking на позицию (её последний/закрывающий
Trade — см. docstring модели в src/db/models.py). Тречинг идёт, пока не
будет найден локальный экстремум цены (см. _advance_one) — то есть
неопределённое время вперёд, ограниченное только outcome_tracking_max_days
на случай, если цена вообще не разворачивается заметно ни в одну сторону.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.config import settings
from src.db.models import Trade, TradeOutcomeTracking
from src.db.session import get_session
from src.utils.logging import logger
from src.utils.timeutils import utcnow

# Насколько далеко в прошлое искать НОВЫЕ полностью закрытые позиции —
# независимо от outcome_tracking_max_days (тот ограничивает, сколько
# тречить уже найденную сделку, а не окно поиска новых). Позиция должна
# обнаруживаться в течение часов после последнего частичного закрытия
# (задача запускается каждые outcome_tracking_check_interval_minutes) —
# 30 дней это большой запас на случай пропущенных циклов/простоя бота.
_NEW_TRADE_LOOKBACK_DAYS = 30

# Порог "малозначимой" упущенной прибыли — ниже него сделка считается
# оптимальной, даже если формально missed_pnl_pct положительный (обычный
# шум между ценой закрытия и последующим локальным пиком).
_NEGLIGIBLE_MISSED_PNL_PCT = 1.0


async def track_closed_trade_outcomes() -> None:
    """Точка входа для периодической задачи (см. TradingBot._track_closed_trade_outcomes)."""
    async with get_session() as session:
        await _register_newly_closed_trades(session)
    async with get_session() as session:
        await _advance_active_tracking(session)


async def _register_newly_closed_trades(session) -> None:
    """
    Найти позиции, которые закрылись ПОЛНОСТЬЮ (суммарный закрытый объём
    догнал изначально исполненный — иначе последняя по времени Trade-строка
    группы это ещё промежуточное частичное закрытие TP1/TP2, а не финал
    позиции), и завести для них строку тречинга.
    """
    since = utcnow() - timedelta(days=_NEW_TRADE_LOOKBACK_DAYS)
    existing_trade_ids = set(
        (await session.execute(select(TradeOutcomeTracking.trade_id))).scalars().all()
    )
    rows = (
        await session.execute(
            select(Trade)
            .options(selectinload(Trade.order_open))
            .where(Trade.order_open_id.isnot(None), Trade.closed_at >= since)
            .order_by(Trade.order_open_id, Trade.closed_at)
        )
    ).scalars().all()

    groups: dict[int, list[Trade]] = {}
    for t in rows:
        groups.setdefault(t.order_open_id, []).append(t)

    for group in groups.values():
        last = group[-1]
        if last.id in existing_trade_ids:
            continue
        opening_order = last.order_open
        if opening_order is None or not opening_order.filled_amount:
            continue
        total_closed = sum(float(t.amount) for t in group)
        if total_closed < float(opening_order.filled_amount) * 0.999:
            continue

        priced_legs = [t for t in group if t.exit_price is not None]
        if not priced_legs:
            continue
        close_price = (
            sum(float(t.exit_price) * float(t.amount) for t in priced_legs)
            / sum(float(t.amount) for t in priced_legs)
        )
        total_pnl = sum(float(t.pnl) for t in group)
        entry_price = float(last.entry_price)
        baseline_pnl_pct = (
            (total_pnl / (entry_price * total_closed) * 100) if entry_price and total_closed else 0.0
        )

        session.add(TradeOutcomeTracking(
            trade_id=last.id,
            symbol_id=last.symbol_id,
            direction=last.direction,
            entry_price=entry_price,
            close_price=close_price,
            close_time=last.closed_at or utcnow(),
            baseline_pnl_pct=baseline_pnl_pct,
        ))
    await session.commit()


async def _advance_active_tracking(session) -> None:
    rows = (
        await session.execute(
            select(TradeOutcomeTracking)
            .options(
                selectinload(TradeOutcomeTracking.symbol),
                selectinload(TradeOutcomeTracking.trade).selectinload(Trade.order_open),
            )
            .where(TradeOutcomeTracking.status == "tracking")
        )
    ).scalars().all()

    for row in rows:
        try:
            await _advance_one(row)
        except Exception as e:
            logger.debug(f"Не удалось обновить тречинг исхода сделки #{row.trade_id}: {e}")
    await session.commit()


async def _advance_one(row: TradeOutcomeTracking) -> None:
    from src.execution.executor import execution_engine

    symbol = row.symbol.symbol if row.symbol else None
    if not symbol:
        return
    market_type = row.trade.order_open.market_type if row.trade and row.trade.order_open else None
    current_price = await execution_engine.get_reference_price(symbol, market_type)
    if current_price is None:
        return

    now = utcnow()
    direction = row.direction
    close_price = float(row.close_price)

    if direction == "long":
        if row.best_price is None or current_price > float(row.best_price):
            row.best_price = current_price
            row.best_price_at = now
        if row.worst_price is None or current_price < float(row.worst_price):
            row.worst_price = current_price
            row.worst_price_at = now
    else:
        if row.best_price is None or current_price < float(row.best_price):
            row.best_price = current_price
            row.best_price_at = now
        if row.worst_price is None or current_price > float(row.worst_price):
            row.worst_price = current_price
            row.worst_price_at = now

    row.last_price = current_price
    row.last_checked_at = now

    best_price = float(row.best_price)
    excursion = abs(best_price - close_price)
    excursion_pct = (excursion / close_price * 100) if close_price else 0.0
    age_days = (now - row.close_time).total_seconds() / 86400.0

    should_finalize = False
    if excursion_pct >= settings.outcome_tracking_min_excursion_pct:
        retracement = (best_price - current_price) if direction == "long" else (current_price - best_price)
        if retracement >= excursion * (settings.outcome_tracking_retracement_pct / 100.0):
            should_finalize = True

    if age_days >= settings.outcome_tracking_max_days:
        row.status = "stopped_max_horizon"
        _finalize_verdict(row)
    elif should_finalize:
        row.status = "done"
        _finalize_verdict(row)


def _finalize_verdict(row: TradeOutcomeTracking) -> None:
    entry_price = float(row.entry_price)
    best_price = float(row.best_price)
    if row.direction == "long":
        optimal_pnl_pct = (best_price - entry_price) / entry_price * 100
    else:
        optimal_pnl_pct = (entry_price - best_price) / entry_price * 100

    row.optimal_pnl_pct = optimal_pnl_pct
    missed = optimal_pnl_pct - row.baseline_pnl_pct
    row.missed_pnl_pct = missed

    if missed <= _NEGLIGIBLE_MISSED_PNL_PCT:
        row.verdict = "optimal"
    elif row.baseline_pnl_pct < 0 and optimal_pnl_pct > 0:
        # Закрылись в минус (скорее всего по SL), а цена потом всё же ушла
        # в плюс от входа — SL был слишком тесным для движения сигнала.
        row.verdict = "sl_too_tight"
    elif row.baseline_pnl_pct > 0 and missed >= max(row.baseline_pnl_pct, 2.0):
        # Закрылись в плюс, но цена ушла ощутимо (минимум вдвое, либо на
        # заметные +2%) дальше — TP был слишком консервативным.
        row.verdict = "tp_too_conservative"
    else:
        row.verdict = "premature_exit"
