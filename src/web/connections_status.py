"""Проверка статусов внешних подключений бота для веб-панели."""
from sqlalchemy import text

from src.config import settings
from src.db.session import get_session


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

    # Биржа (исполнение ордеров)
    from src.execution.executor import execution_engine
    if execution_engine.exchange is not None:
        ex_status, ex_detail = "connected", execution_engine.exchange_id or ""
    elif settings.is_paper:
        ex_status, ex_detail = "paper_mode", "живое подключение не требуется"
    elif not (settings.binance_api_key and settings.binance_api_secret):
        ex_status, ex_detail = "not_configured", "API ключи не заданы"
    else:
        ex_status, ex_detail = "error", "инициализация не удалась, см. логи"
    statuses.append({"key": "exchange", "name": "Биржа (исполнение)", "status": ex_status, "detail": ex_detail})

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

    return statuses
