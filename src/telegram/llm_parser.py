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
from typing import Optional

from src.config import settings
from src.utils.logging import logger
from src.telegram.channel_monitor import normalize_pair

_SYSTEM = """You extract crypto trading signals from noisy Telegram messages.

Return ONLY the trade the message explicitly states. NEVER invent, guess, or \
extrapolate prices. If a field is not stated, leave it null/empty.

A message is NOT a signal (return "is_signal": false) when it is commentary, \
news, a meme, a question, an update about an already-open trade, an ad, or too \
ambiguous to trade safely.

Rules:
- side: "long" for long/buy, "short" for short/sell.
- base: the coin ticker, uppercase (e.g. BTC). quote: default USDT if unstated.
- entry: the entry price (a zone -> pick the near/first number). null = unstated.
- take_profits: list of target prices, ascending.
- stop_loss: single number or null.
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
            "take_profits": {"type": "array", "items": {"type": "number"}},
            "stop_loss": {"type": ["number", "null"]},
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


async def parse_with_llm(text: str, channel_config: Optional[dict] = None) -> Optional[dict]:
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
    if not base or side not in ("long", "short") or entry is None:
        return None

    quote = data.get("quote") or "USDT"
    pair = normalize_pair(f"{base}{quote}")

    take_profits = [float(x) for x in (data.get("take_profits") or [])]
    tp = max(take_profits) if side == "long" else (min(take_profits) if take_profits else None)
    sl = data.get("stop_loss")

    return {
        "pair": pair,
        "side": side,
        "entry": float(entry),
        "sl": float(sl) if sl is not None else None,
        "tp": float(tp) if tp is not None else None,
        "raw": text,
    }
