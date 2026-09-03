"""Google Gemini LLM-фолбэк для сигналов — второй уровень, после
Anthropic (см. src/telegram/llm_parser.py), пробуется когда регулярки не
распознали сообщение И Anthropic либо не настроен, либо тоже не смог.
Gemini выбран как бесплатный по тарифу вариант (щедрый free tier).

Тот же контракт возврата (dict pair/side/entry/sl/tp/raw или None) и тот
же принцип "никогда не придумывать цену" — промпт и схема скопированы из
llm_parser.py почти дословно, чтобы поведение фолбэков не расходилось.
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

_RESPONSE_SCHEMA = {
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
}

_MIN_CONFIDENCE = 0.5

_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY не настроен")
        from google import genai

        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


async def parse_with_gemini(text: str, channel_config: dict | None = None) -> dict | None:
    """Распарсить сигнал через Gemini. Возвращает тот же формат, что и
    parse_with_regex/parse_with_llm (pair/side/entry/sl/tp/raw), или None."""
    if not settings.telegram_llm_fallback_enabled or not settings.gemini_api_key:
        return None

    try:
        client = _get_client()
        resp = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=text[:4000],
            config={
                "system_instruction": _SYSTEM,
                "response_mime_type": "application/json",
                "response_json_schema": _RESPONSE_SCHEMA,
                # 512 оказалось мало на практике (реальный инцидент, прод):
                # "thinking"-модели (в т.ч. flash-варианты) тратят часть
                # max_output_tokens на внутренние рассуждения ДО собственно
                # JSON-ответа — бюджет выбивался ПОСЕРЕДИНЕ JSON, и
                # json.loads падал с "Unterminated string starting at:",
                # сигнал тихо терялся (Anthropic-фолбэк не настроен —
                # единственный работающий уровень LLM-парсинга отваливался
                # на каждом таком сообщении). Схема ответа компактная
                # (несколько чисел/строк) — 2048 с запасом покрывает и
                # рассуждения, и сам JSON.
                "max_output_tokens": 2048,
            },
        )
    except Exception as e:
        logger.warning(f"Gemini-парсер сигнала: ошибка запроса — {e}")
        return None

    try:
        raw_text = resp.text or ""
    except Exception as e:
        logger.warning(f"Gemini-парсер сигнала: не удалось прочитать текст ответа — {e}")
        return None

    try:
        data = json.loads(raw_text)
    except Exception as e:
        logger.warning(
            f"Gemini-парсер сигнала: не удалось разобрать JSON-ответ — {e} | "
            f"сырой ответ: {raw_text[:500]!r}"
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

    # См. комментарий в llm_parser.py: промпт отдаёт цели по возрастанию
    # цены, а _tp_levels() в main.py ожидает порядок по возрастанию
    # расстояния от входа в прибыльную сторону (ближайшая первая) — для
    # short это обратный порядок цены.
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
