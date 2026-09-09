"""Проверка статусов внешних подключений бота для веб-панели."""
from sqlalchemy import text

from src.config import settings
from src.db.session import get_session

# Все биржи, поддержанные ExecutionEngine (см. _exchange_credentials_present
# в executor.py) — используется и здесь (по одной статус-записи на биржу,
# не только на активную), и как источник правды для UI-дропдауна в
# settings_store.py (SETTINGS_SCHEMA.active_exchange/credentials_exchange_ui).
SUPPORTED_EXCHANGES = ("binance", "bybit", "okx", "kucoin", "bingx", "bitget", "bitmex", "hyperliquid")

_EXCHANGE_LABELS = {
    "binance": "Binance",
    "bybit": "Bybit",
    "okx": "OKX",
    "kucoin": "KuCoin",
    "bingx": "BingX",
    "bitget": "Bitget",
    "bitmex": "BitMEX",
    "hyperliquid": "HyperLiquid",
}


async def get_connections_status() -> list[dict]:
    """Собрать статус каждого внешнего подключения (дёшево, без лишних сетевых вызовов)."""
    statuses = []

    # База данных
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        db_status, db_detail = "connected", ""
    except Exception as e:
        db_status, db_detail = "error", str(e)
    statuses.append({"key": "database", "name": "База данных", "status": db_status, "detail": db_detail})

    # Биржи — раньше здесь была ОДНА запись "Биржа (исполнение)" только про
    # активную биржу (settings.active_exchange): подключив ключи сразу
    # нескольких бирж заранее (чтобы потом переключаться между ними без
    # похода в настройки), остальные были попросту не видны в дашборде —
    # только текущая. Теперь показывается статус КАЖДОЙ поддержанной
    # биржи отдельной записью (см. group="exchange" — дашборд рисует их
    # отдельным блоком карточек, а не общим списком).
    from src.execution.executor import execution_engine
    for exchange_id in SUPPORTED_EXCHANGES:
        is_active = exchange_id == settings.active_exchange
        credentials_present = execution_engine._exchange_credentials_present(exchange_id)
        if is_active and execution_engine.exchange is not None and not execution_engine.is_paper:
            sandbox_suffix = " (демо)" if settings.use_exchange_sandbox else ""
            ex_status, ex_detail = "connected", f"активная{sandbox_suffix}"
        elif is_active and settings.is_paper:
            ex_status, ex_detail = "paper_mode", "активная, но бот в paper-режиме"
        elif not credentials_present:
            ex_status, ex_detail = "not_configured", "ключи не заданы"
        else:
            ex_status, ex_detail = "configured", "ключи заданы, сейчас не активна"
        statuses.append({
            "key": f"exchange_{exchange_id}",
            "name": _EXCHANGE_LABELS[exchange_id],
            "status": ex_status,
            "detail": ex_detail,
            "group": "exchange",
        })

    # Telegram — мониторинг каналов (Telethon)
    from src.telegram.channel_monitor import get_telegram_client
    client = get_telegram_client()
    if client is not None:
        try:
            connected = client.is_connected()
        except Exception:
            connected = False
        tg_status, tg_detail = ("connected", "") if connected else ("error", "клиент создан, но не подключён")
    elif settings.telegram_api_id and settings.telegram_api_hash:
        tg_status, tg_detail = "not_connected", "клиент не инициализирован, см. логи запуска"
    else:
        tg_status, tg_detail = "not_configured", "TELEGRAM_API_ID/HASH не заданы"
    statuses.append({"key": "telegram_monitor", "name": "Telegram (мониторинг сигналов)", "status": tg_status, "detail": tg_detail})

    # Telegram — исходящие уведомления (Bot API)
    if settings.telegram_bot_token and settings.telegram_chat_id:
        notif_status, notif_detail = "configured", ""
    else:
        notif_status, notif_detail = "not_configured", "TELEGRAM_BOT_TOKEN/CHAT_ID не заданы"
    statuses.append({"key": "telegram_notify", "name": "Telegram (уведомления)", "status": notif_status, "detail": notif_detail})

    # CoinGlass
    if settings.coinglass_api_key:
        cg_status, cg_detail = "configured", ""
    else:
        cg_status, cg_detail = "not_configured", "COINGLASS_API_KEY не задан (публичные лимиты)"
    statuses.append({"key": "coinglass", "name": "CoinGlass API", "status": cg_status, "detail": cg_detail})

    # LLM-фолбэк парсинга Telegram-сигналов (Anthropic/Groq/Gemini/Cerebras) —
    # уже давно поддержаны (см. src/telegram/*_parser.py, settings.
    # telegram_llm_fallback_enabled), но статус их ключей нигде не был виден
    # в дашборде — единственным способом узнать, что, например, у Anthropic
    # закончился баланс, а у Gemini исчерпана квота, было читать /logs.
    llm_providers = (
        ("anthropic", "Anthropic (LLM-фолбэк)", settings.anthropic_api_key),
        ("groq", "Groq (LLM-фолбэк)", settings.groq_api_key),
        ("gemini", "Gemini (LLM-фолбэк)", settings.gemini_api_key),
        ("cerebras", "Cerebras (LLM-фолбэк)", settings.cerebras_api_key),
    )
    for key, name, api_key in llm_providers:
        if not settings.telegram_llm_fallback_enabled:
            llm_status, llm_detail = "disabled", "LLM-фолбэк парсинга выключен в настройках"
        elif not api_key:
            llm_status, llm_detail = "not_configured", "API ключ не задан"
        else:
            llm_status, llm_detail = "configured", ""
        statuses.append({"key": f"llm_{key}", "name": name, "status": llm_status, "detail": llm_detail})

    return statuses
