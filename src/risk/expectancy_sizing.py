"""Expectancy-based sizing по источнику сигнала (Telegram-канал / ML-стратегия).

Портировано из наработок предыдущего бота (clonerbot: scoring/channel_scorer.py),
адаптировано под источники обоих типов через тот же ключ scope_key, что и
Protections (telegram:<channel_id> / strategy:<strategy_id>), и под общий
журнал закрытий risk_close_events вместо отдельной таблицы каналов.

Идея: размер новой позиции масштабируется множителем 0.5x–1.5x по
фактическому среднему % доходности на сделку у источника, из которого пришёл
сигнал — источники с положительным математическим ожиданием получают больше
капитала, источники с неположительным — пропускаются полностью (0x).
Источник без достаточной статистики (мало закрытых сделок) торгует
уменьшённым размером (0.5x), пока не наберёт историю — это тот самый
"доказывай результатом прежде, чем получишь больше капитала" подход.

Выключено по умолчанию (expectancy_sizing_enabled=False) — включение меняет
фактический размер позиций, это осознанное решение пользователя, а не
поведение по умолчанию для уже работающего бота.
"""
from __future__ import annotations

from sqlalchemy import select

from src.config import settings
from src.db.models import RiskCloseEvent
from src.db.session import get_session

MIN_MULTIPLIER = 0.5
MAX_MULTIPLIER = 1.5
# Средний % доходности на сделку, при котором источник получает полный
# (MAX) размер. Ниже — линейно меньше; при/ниже min_expectancy — 0 (skip).
_EXPECTANCY_TARGET_PCT = 2.0


async def size_multiplier(source_key: str) -> float:
    """Множитель размера позиции для source_key, обученный на его истории."""
    if not settings.expectancy_sizing_enabled:
        return 1.0

    async with get_session() as session:
        rows = (
            await session.execute(
                select(RiskCloseEvent.pnl_pct)
                .where(RiskCloseEvent.scope_key == source_key)
                .order_by(RiskCloseEvent.closed_at.desc())
                .limit(settings.expectancy_sizing_max_trades)
            )
        ).scalars().all()

    n = len(rows)
    if n < settings.expectancy_sizing_min_trades:
        # Недостаточно доказанной статистики — торгуем осторожно, не по нулю
        # (иначе новый источник никогда не набрал бы сделок для оценки).
        return MIN_MULTIPLIER

    mean_pct = sum(rows) / n
    if mean_pct <= settings.expectancy_sizing_min_expectancy_pct:
        return 0.0

    frac = min(1.0, mean_pct / _EXPECTANCY_TARGET_PCT)
    mult = MIN_MULTIPLIER + (MAX_MULTIPLIER - MIN_MULTIPLIER) * frac
    return round(max(0.0, min(MAX_MULTIPLIER, mult)), 3)
