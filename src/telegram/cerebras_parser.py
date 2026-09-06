"""Cerebras LLM-фолбэк для сигналов — четвёртый уровень, после Anthropic,
Groq и Gemini (см. src/telegram/llm_parser.py, groq_parser.py,
gemini_parser.py), пробуется когда регулярки не распознали сообщение И все
предыдущие либо не настроены, либо тоже не смогли. Cerebras хостит открытые
модели (Llama и т.п.) с бесплатным API — ещё один независимый бесплатный
источник на случай, если у Groq/Gemini кончилась квота или недоступна
конкретная модель на конкретном ключе (реальный инцидент: оба этих кейса
уже случались одновременно, оставляя сигнал вообще без LLM-разбора).

Тот же контракт возврата (dict pair/side/entry/sl/tp/raw или None) и тот же
принцип "никогда не придумывать цену" — промпт и схема скопированы из
groq_parser.py почти дословно (тот же OpenAI-совместимый chat.completions с
forced tool-call — Cerebras Cloud SDK повторяет этот же интерфейс), чтобы
поведение фолбэков не расходилось.
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
    "type": "function",
    "function": {
        "name": "emit_signal",
        "description": "Emit the structured trading signal extracted from the message.",
        "parameters": {
            "type": "object",
            "properties": {
                "is_signal": {"type": "boolean"},
                "base": {"type": ["string", "null"]},
                "quote": {"type": ["string", "null"]},
                "side": {"type": ["string", "null"], "enum": ["long", "short", None]},
                "entry": {"type": ["number", "null"]},
                "is_market_entry": {"type": ["boolean", "null"]},
                "take_profits": {"type": "array", "items": {"type": "number"}},
                "stop_loss": {"type": ["number", "null"]},
                "leverage": {"type": ["number", "null"]},
                "confidence": {"type": "number"},
            },
            "required": ["is_signal", "confidence"],
        },
    },
}

_MIN_CONFIDENCE = 0.5

_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.cerebras_api_key:
            raise RuntimeError("CEREBRAS_API_KEY не настроен")
        from cerebras.cloud.sdk import AsyncCerebras

        _client = AsyncCerebras(api_key=settings.cerebras_api_key)
    return _client


async def parse_with_cerebras(text: str, channel_config: dict | None = None) -> dict | None:
    """Распарсить сигнал через Cerebras. Возвращает тот же формат, что и
    parse_with_regex/parse_with_llm/parse_with_groq/parse_with_gemini
    (pair/side/entry/sl/tp/raw), или None."""
    if not settings.telegram_llm_fallback_enabled or not settings.cerebras_api_key:
        return None

    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model=settings.cerebras_model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": text[:4000]},
            ],
            tools=[_TOOL],
            tool_choice={"type": "function", "function": {"name": "emit_signal"}},
        )
    except Exception as e:
        logger.warning(f"Cerebras-парсер сигнала: ошибка запроса — {e}")
        return None

    try:
        tool_calls = resp.choices[0].message.tool_calls
        raw_args = tool_calls[0].function.arguments
    except Exception as e:
        logger.warning(f"Cerebras-парсер сигнала: не удалось прочитать tool-call в ответе — {e}")
        return None

    try:
        data = json.loads(raw_args)
    except Exception as e:
        logger.warning(
            f"Cerebras-парсер сигнала: не удалось разобрать JSON-аргументы — {e} | "
            f"сырые аргументы: {raw_args[:500]!r}"
        )
        return None

    if not isinstance(data, dict):
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

    # См. комментарий в llm_parser.py/groq_parser.py/gemini_parser.py:
    # промпт отдаёт цели по возрастанию цены, а _tp_levels() в main.py
    # ожидает порядок по возрастанию расстояния от входа в прибыльную
    # сторону (ближайшая первая) — для short это обратный порядок цены.
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
