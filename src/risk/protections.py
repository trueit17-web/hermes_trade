"""Protections — freqtrade-style автопаузы после плохой серии сделок.

Портировано из наработок предыдущего бота (clonerbot: risk/protections.py) и
адаптировано под схему hermes_trade: там сделка — это единая мутируемая
Position-строка с полем close_reason, здесь — audit-trail Order+Trade без
хранимой причины закрытия, поэтому свою бухгалтерию Protections ведёт в
собственных таблицах (risk_locks/risk_close_events), не трогая Trade/Order.

Реализовано:
  * Cooldown — после ЛЮБОГО полного закрытия сделки источник (Telegram-канал
    или ML-стратегия, из которого пришёл сигнал) на время перестаёт
    открывать новые позиции. Это отдельный, более узкий механизм, чем
    risk_manager.state.cooldown_active — тот блокирует ВСЮ торговлю после
    любой сделки, этот — только источник, из которого пришла именно эта
    сделка.
  * StoplossGuard — если за окно времени накопилось N закрытий по стопу
    (по всем источникам сразу) — глобальная пауза всей торговли: явный
    признак того, что рынок сейчас не подходит под текущую стратегию/сигналы.
  * LosingStreak — N убыточных закрытий подряд у одного источника ->
    блокировка только этого источника (например, канал стабильно сливает,
    но остальные каналы работают нормально).

Блокировки — записи в risk_locks с TTL (until), переживают рестарт бота.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from sqlalchemy import delete, select

from src.config import settings
from src.db.models import RiskCloseEvent, RiskLock
from src.db.session import get_session
from src.utils.logging import logger
from src.utils.timeutils import utcnow

GLOBAL_KEY = "global"


def channel_key(channel_id: str) -> str:
    return f"telegram:{channel_id}"


def strategy_key(strategy_id: str) -> str:
    return f"strategy:{strategy_id}"


class LockStore:
    async def add(self, scope_key: str, minutes: int, reason: str) -> None:
        if minutes <= 0:
            return
        async with get_session() as session:
            session.add(RiskLock(
                scope_key=scope_key, reason=reason,
                until=utcnow() + timedelta(minutes=minutes),
            ))
            await session.commit()
        logger.warning(f"🔒 Protections: '{scope_key}' заблокирован на {minutes} мин — {reason}")

    async def active_reason(self, keys: list[str]) -> Optional[str]:
        """Причина активной блокировки по любому из ключей, иначе None."""
        async with get_session() as session:
            row = (
                await session.execute(
                    select(RiskLock)
                    .where(RiskLock.scope_key.in_(keys), RiskLock.until > utcnow())
                    .order_by(RiskLock.until.desc())
                )
            ).scalars().first()
            return row.reason if row else None

    async def clear_expired(self) -> int:
        async with get_session() as session:
            result = await session.execute(delete(RiskLock).where(RiskLock.until <= utcnow()))
            await session.commit()
            return result.rowcount or 0

    async def active_locks(self) -> list[dict]:
        """Все сейчас активные блокировки — для отображения в дашборде."""
        async with get_session() as session:
            rows = (
                await session.execute(
                    select(RiskLock).where(RiskLock.until > utcnow()).order_by(RiskLock.until.desc())
                )
            ).scalars().all()
            return [
                {"scope": r.scope_key, "reason": r.reason, "until": r.until.isoformat() + "Z"}
                for r in rows
            ]


class ProtectionManager:
    def __init__(self) -> None:
        self.locks = LockStore()

    async def locked_reason(self, keys: list[str]) -> Optional[str]:
        return await self.locks.active_reason(keys)

    async def on_close(
        self, source_key: str, symbol: str, pnl: float, reason: str, pnl_pct: float = 0.0,
    ) -> None:
        """Вызывается ПОСЛЕ каждого ПОЛНОГО (не частичного) закрытия позиции.

        Пишет событие в risk_close_events независимо от protections_enabled —
        expectancy-based sizing (src/risk/expectancy_sizing.py) читает тот же
        журнал и не должна замолкать вместе с Protections."""
        async with get_session() as session:
            session.add(RiskCloseEvent(
                scope_key=source_key, symbol=symbol, reason=reason, pnl=pnl, pnl_pct=pnl_pct,
            ))
            await session.commit()

        if not settings.protections_enabled:
            return

        await self.locks.add(
            source_key, settings.protections_channel_cooldown_minutes,
            "cooldown после закрытия сделки",
        )

        if reason == "stop_loss":
            await self._maybe_stoploss_guard()
        if pnl < 0:
            await self._maybe_losing_streak(source_key)

    async def _maybe_stoploss_guard(self) -> None:
        s = settings
        since = utcnow() - timedelta(minutes=s.protections_stoploss_guard_window_min)
        async with get_session() as session:
            n = len((
                await session.execute(
                    select(RiskCloseEvent.id).where(
                        RiskCloseEvent.reason == "stop_loss",
                        RiskCloseEvent.closed_at >= since,
                    )
                )
            ).scalars().all())
        if n >= s.protections_stoploss_guard_count:
            await self.locks.add(
                GLOBAL_KEY, s.protections_stoploss_guard_lock_min,
                f"stoploss guard: {n} стопов за {s.protections_stoploss_guard_window_min} мин",
            )

    async def _maybe_losing_streak(self, source_key: str) -> None:
        s = settings
        n_needed = s.protections_losing_streak_count
        async with get_session() as session:
            rows = (
                await session.execute(
                    select(RiskCloseEvent.pnl)
                    .where(RiskCloseEvent.scope_key == source_key)
                    .order_by(RiskCloseEvent.closed_at.desc())
                    .limit(n_needed)
                )
            ).scalars().all()
        if len(rows) >= n_needed and all((p or 0) < 0 for p in rows):
            await self.locks.add(
                source_key, s.protections_losing_streak_lock_min,
                f"losing streak: {n_needed} убыточных сделок подряд",
            )


# Глобальный экземпляр
protection_manager = ProtectionManager()
