"""Telegram-уведомления через Bot API — только исходящие алерты, без приёма команд."""
import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


async def send_notification(text: str) -> bool:
    """
    Отправить уведомление в Telegram-чат (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID).
    Если не настроено — тихо пропускает (возвращает False).
    """
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return False

    url = TELEGRAM_API_URL.format(token=settings.telegram_bot_token)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                json={
                    "chat_id": settings.telegram_chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            response.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Не удалось отправить Telegram-уведомление: {e}")
        return False
