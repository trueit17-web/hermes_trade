"""Anthropic LLM-фолбэк для сигналов, которые не распознали регулярки.

Перенесено из наработок предыдущего бота (clonerbot: parser/llm_parser.py),
адаптировано под возвращаемый формат hermes_trade (dict pair/side/entry/sl/tp/raw
из channel_monitor.parse_with_regex, а не отдельный dataclass под futures-поля).

LLM возвращает строгий JSON через forced tool-call (никогда не выдумывает
цены — извлекает только то, что явно указано в сообщении). Любая ошибка
(нет ключа, сетевой сбой, сообщение — не сигнал) -> None, вызывающий код
просто не находит сигнал, как и при неудаче регулярок.
"""
from __future__ import annotations

import json

from src.config import settings
from src.telegram.channel_monitor import normalize_pair
from src.utils.logging import logger

_SYSTEM = """You extract crypto trading signals from noisy Telegram messages.

Return ONLY the trade the message explicitly states. NEVER invent, guess, or \
extrapolate prices. If a field is not stated, leave it null/empty.

A message is NOT a signal (return "is_signal": false) when it is commentary, \
news, a meme, a question, an update about an already-open trade, an ad, or too \
ambiguous to trade safely.

Rules:
- side: "long" for long/buy, "short" for short/sell.
- base: the coin ticker, uppercase (e.g. BTC). quote: default USDT if unstated.
- entry: the entry price (a zone -> pick the near/first number). null = unstated \
OR the message says to enter at the current market price ("по рынку", "at market", \
no fixed price given).
- is_market_entry: true when the message explicitly asks for market execution \
(no fixed entry price) rather than just omitting the entry by accident/ambiguity.
- take_profits: list of target prices, ascending.
- stop_loss: single number or null.
- leverage: the leverage multiplier if explicitly stated (e.g. "Leverage: 20x", \
"плечо x10" -> 20, 10; a range like "25-30x" -> the lower bound, 25). null if unstated.
- confidence: your 0..1 confidence this is a clean, tradeable signal."""

_TOOL = {
    "name": "emit_signal",
    "description": "Emit the structured trading signal extracted from the message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_signal": {"type": "boolean"},
            "base": {"type": ["string", "null"]},
            "quote": {"type": ["string", "null"]},
            "side": {"type": ["string", "null"], "enum": ["long", "short", None]},
            "entry": {"type": ["number", "null"]},
            "is_market_entry": {"type": "boolean"},
            "take_profits": {"type": "array", "items": {"type": "number"}},
            "stop_loss": {"type": ["number", "null"]},
            "leverage": {"type": ["number", "null"]},
            "confidence": {"type": "number"},
        },
        "required": ["is_signal", "confidence"],
    },
}

_MIN_CONFIDENCE = 0.5

_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY не настроен")
        import anthropic

        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def parse_with_llm(text: str, channel_config: dict | None = None) -> dict | None:
    """Распарсить сигнал через LLM. Возвращает тот же формат, что и
    parse_with_regex (pair/side/entry/sl/tp/raw), или None."""
    if not settings.telegram_llm_fallback_enabled or not settings.anthropic_api_key:
        return None

    try:
        client = _get_client()
        resp = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=512,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "emit_signal"},
            messages=[{"role": "user", "content": text[:4000]}],
        )
    except Exception as e:
        logger.warning(f"LLM-парсер сигнала: ошибка запроса — {e}")
        return None

    data = None
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "emit_signal":
            data = block.input if isinstance(block.input, dict) else json.loads(block.input)
            break
    if data is None:
        logger.warning("LLM-парсер сигнала: модель не вызвала emit_signal")
        return None

    if not data.get("is_signal") or float(data.get("confidence", 0.0)) < _MIN_CONFIDENCE:
        return None

    base = data.get("base")
    side = data.get("side")
    entry = data.get("entry")
    # entry=None допустим ТОЛЬКО как явный маркет-вход (see is_market_entry
    # в промпте/схеме) — иначе (просто не смогла определить цену) сигнал
    # остаётся отклонённым, как и раньше.
    if not base or side not in ("long", "short") or (entry is None and not data.get("is_market_entry")):
        return None

    quote = data.get("quote") or "USDT"
    pair = normalize_pair(f"{base}{quote}")

    # Промпт просит цели по возрастанию ЦЕНЫ ("ascending"), а _tp_levels()
    # в main.py ожидает порядок по возрастанию РАССТОЯНИЯ ОТ ВХОДА В
    # ПРИБЫЛЬНУЮ СТОРОНУ (ближайшая цель первая) — для long это то же
    # самое (цена растёт), а для short ровно наоборот (профит растёт при
    # падении цены, значит ближайшая цель — самая ВЫСОКАЯ из тех, что ниже
    # входа) — разворачиваем список для short.
    take_profits = sorted(float(x) for x in (data.get("take_profits") or []))
    if side == "short":
        take_profits.reverse()
    tp = take_profits[-1] if take_profits else None
    sl = data.get("stop_loss")
    leverage = data.get("leverage")

    return {
        "pair": pair,
        "side": side,
        "entry": float(entry) if entry is not None else None,
        "sl": float(sl) if sl is not None else None,
        "tp": float(tp) if tp is not None else None,
        "take_profits": take_profits,
        "leverage": float(leverage) if leverage else None,
        "confidence": float(data.get("confidence", _MIN_CONFIDENCE)),
        "raw": text,
    }
