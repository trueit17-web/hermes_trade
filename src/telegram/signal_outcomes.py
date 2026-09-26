"""
Исход Telegram-сигнала по ВСЕМ Trade-частям его позиции.

Раньше исход сигнала определялся только по ссылке TelegramSignal.
executed_trade_id, которую ставил лишь обычный путь закрытия в main.py.
Позиции, закрытые вне цикла бота (биржевой SL/TP сработал сам — сейчас это
самый частый путь, см. ExecutionEngine._record_external_close), ссылку не
получали: статистика канала на дашборде (закрыто/win rate) и
historical-accuracy компонент quality_scorer переставали учитывать такие
сделки — "перестало считать историю каналов" (прод, 2026-09-26).
"""
import logging

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.db.models import Order, TelegramSignal, Trade
from src.db.session import get_session

logger = logging.getLogger(__name__)


async def order_outcomes(session, order_ids) -> dict[int, tuple[float, bool]]:
    """
    order_open_id -> (суммарный PnL всех закрытых частей, закрыта ли позиция
    целиком). Целиком — если закрытые части покрывают исполненный объём
    ордера открытия (та же проверка, что в GET /telegram/signals). Ордера
    без единой закрытой части в результат не попадают.
    """
    order_ids = {oid for oid in order_ids if oid is not None}
    if not order_ids:
        return {}
    legs: dict[int, list[tuple[float, float]]] = {}
    for order_open_id, pnl, amount in (
        await session.execute(
            select(Trade.order_open_id, Trade.pnl, Trade.amount).where(Trade.order_open_id.in_(order_ids))
        )
    ).all():
        legs.setdefault(order_open_id, []).append((float(pnl or 0.0), float(amount or 0.0)))
    if not legs:
        return {}
    order_amounts = {
        oid: float(filled if filled else amount or 0.0)
        for oid, filled, amount in (
            await session.execute(
                select(Order.id, Order.filled_amount, Order.amount).where(Order.id.in_(legs.keys()))
            )
        ).all()
    }
    result: dict[int, tuple[float, bool]] = {}
    for oid, parts in legs.items():
        closed_amount = sum(a for _, a in parts)
        order_amount = order_amounts.get(oid, 0.0)
        fully_closed = order_amount - closed_amount <= max(order_amount * 0.01, 1e-9)
        result[oid] = (sum(p for p, _ in parts), fully_closed)
    return result


async def link_signal_to_closed_trade(order_id: int | None, trade_id: int) -> None:
    """
    Проставить executed_trade_id у сигнала, чей ордер открытия только что
    закрылся целиком, и передать исход канала в quality_scorer. Исход — по
    сумме PnL всех частей (частичные TP + финальное закрытие), а не по
    одной последней части: после TP1/TP2 остаток часто закрывается по
    стопу в безубытке/небольшом минусе, хотя сигнал в целом прибыльный.
    """
    if order_id is None:
        return
    try:
        async with get_session() as session:
            signal = (
                await session.execute(
                    select(TelegramSignal)
                    .options(selectinload(TelegramSignal.channel))
                    .where(TelegramSignal.executed_order_id == order_id)
                )
            ).scalar_one_or_none()
            if signal is None:
                return
            already_linked = signal.executed_trade_id is not None
            signal.executed_trade_id = trade_id
            channel_id = signal.channel.channel_id if signal.channel else None
            outcome = (await order_outcomes(session, [order_id])).get(order_id)
            await session.commit()

        if channel_id and outcome is not None and not already_linked:
            from src.telegram.quality_scorer import signal_quality_scorer

            signal_quality_scorer.update_channel_stats(channel_id, outcome[0] > 0)
    except Exception as e:
        logger.warning(f"Не удалось связать Telegram-сигнал со сделкой #{trade_id}: {e}")
