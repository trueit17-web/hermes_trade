"""
Краткий контекст по фактическим исходам СДЕЛОК канала (после закрытия —
см. src/execution/trade_outcome_tracker.py) для LLM-фолбэков парсера
сигналов (llm_parser.py/gemini_parser.py/groq_parser.py/cerebras_parser.py)
— по запросу пользователя: "обучать LLM" на данных отслеживания цены
после закрытия сделки.

Строго информационно и ТОЛЬКО для влияния на confidence/is_signal при
неоднозначном сообщении — ни один из парсеров не должен использовать этот
контекст, чтобы менять или придумывать явно указанные в тексте числа
(entry/SL/TP): это нарушило бы их базовый принцип "never invent, guess,
or extrapolate prices" (см. _SYSTEM в каждом из них). Формулировка контекста
ниже поэтому явно это оговаривает.

Кэшируется в памяти на _CACHE_TTL_SECONDS, чтобы не делать отдельный
запрос к БД на каждое сообщение канала — в парсер попадают только
сообщения, не распознанные регулярками, но это всё равно может быть
заметная доля потока канала.
"""
from __future__ import annotations

import time

from sqlalchemy import select

from src.config import settings
from src.db.models import TelegramChannel, TelegramSignal, TradeOutcomeTracking
from src.db.session import get_session

_CACHE_TTL_SECONDS = 3600
_cache: dict[str, tuple[float, str | None]] = {}


async def get_channel_outcome_context(channel_id: str | None) -> str | None:
    if not channel_id or not settings.llm_channel_outcome_context_enabled:
        return None

    cached = _cache.get(channel_id)
    if cached and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        text = await _build_context(channel_id)
    except Exception:
        # Контекст — необязательное обогащение промпта, а не критичный
        # путь: сбой БД здесь не должен ронять сам разбор сигнала.
        text = None
    _cache[channel_id] = (time.monotonic(), text)
    return text


async def _build_context(channel_id: str) -> str | None:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(TradeOutcomeTracking.verdict)
                .join(TelegramSignal, TelegramSignal.executed_trade_id == TradeOutcomeTracking.trade_id)
                .join(TelegramChannel, TelegramChannel.id == TelegramSignal.channel_id)
                .where(
                    TelegramChannel.channel_id == channel_id,
                    TradeOutcomeTracking.status.in_(("done", "stopped_max_horizon")),
                    TradeOutcomeTracking.verdict.in_(("sl_too_tight", "tp_too_conservative")),
                )
                .order_by(TradeOutcomeTracking.missed_pnl_pct.desc())
                .limit(10)
            )
        ).scalars().all()

    if not rows:
        return None

    sl_too_tight = sum(1 for v in rows if v == "sl_too_tight")
    tp_too_conservative = sum(1 for v in rows if v == "tp_too_conservative")
    parts = []
    if sl_too_tight:
        parts.append(
            f"{sl_too_tight} past executed trade(s) from this channel hit stop-loss, "
            "then price later reversed back favorably"
        )
    if tp_too_conservative:
        parts.append(
            f"{tp_too_conservative} past executed trade(s) from this channel hit take-profit, "
            "then price kept moving favorably well beyond it"
        )
    if not parts:
        return None

    return (
        "Historical note about this channel's past executed trades (informational only — "
        "do NOT use this to alter, invent, or adjust any explicit price stated in the message; "
        "use it only to inform your confidence/is_signal judgement when the message itself is "
        "ambiguous): " + "; ".join(parts) + "."
    )
