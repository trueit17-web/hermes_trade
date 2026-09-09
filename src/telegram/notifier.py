"""Telegram-уведомления через Bot API — только исходящие алерты, без приёма команд."""
import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_EDIT_API_URL = "https://api.telegram.org/bot{token}/editMessageText"


async def send_notification(text: str, reply_to_message_id: int | None = None) -> int | None:
    """
    Отправить уведомление в Telegram-чат (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID).
    Если не настроено — тихо пропускает (возвращает None).

    reply_to_message_id — id более раннего уведомления (обычно об открытии
    позиции), на которое нужно ответить, чтобы последующие уведомления по
    той же сделке были видны в чате как единая цепочка, а не разрозненные
    сообщения. Если это сообщение с тех пор удалено — Telegram отвечает
    ошибкой на весь запрос ("message to be replied not found"), а не просто
    игнорирует reply_to_message_id — без фолбэка уведомление терялось бы
    целиком из-за одной лишь удалённой родительской записи; при такой
    ошибке повторяем отправку без него.

    Возвращает message_id отправленного сообщения (нужен вызывающему коду,
    чтобы на НЕГО могли ответить более поздние уведомления), либо None при
    сбое/отсутствии настройки.
    """
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return None

    url = TELEGRAM_API_URL.format(token=settings.telegram_bot_token)
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code >= 400 and reply_to_message_id is not None:
                logger.debug(
                    f"Ответ на сообщение {reply_to_message_id} не удался (вероятно, оно удалено) — "
                    f"повтор без reply_to_message_id: {response.status_code} {response.text[:200]}"
                )
                # Отдельный словарь для повтора, а не мутация payload на
                # месте — иначе он совпадал бы с уже отправленным (и
                # залогированным/проверяемым в тестах) телом первого запроса.
                retry_payload = {k: v for k, v in payload.items() if k != "reply_to_message_id"}
                response = await client.post(url, json=retry_payload)
            response.raise_for_status()
            data = response.json()
        return data.get("result", {}).get("message_id")
    except Exception as e:
        logger.warning(f"Не удалось отправить Telegram-уведомление: {e}")
        return None


async def edit_notification(message_id: int, text: str) -> bool:
    """
    Отредактировать уже отправленное уведомление (editMessageText) вместо
    отправки нового сообщения-ответа — по запросу пользователя закрытие
    позиции по TP/SL обновляет исходное сообщение об открытии на месте
    (зачёркнутые уровни + PnL), а не плодит цепочку ответов.

    Возвращает False при любом сбое (сообщение удалено пользователем,
    старше ~48ч — Telegram запрещает редактирование, текст не изменился —
    "message is not modified", токен/чат не настроены) — вызывающий код
    сам решает, откатываться ли на send_notification(reply_to_message_id=...).
    """
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return False

    url = TELEGRAM_EDIT_API_URL.format(token=settings.telegram_bot_token)
    payload = {
        "chat_id": settings.telegram_chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code >= 400:
                logger.debug(
                    f"Не удалось отредактировать сообщение {message_id} (вероятно, оно "
                    f"удалено/устарело для редактирования): {response.status_code} {response.text[:200]}"
                )
                return False
            return True
    except Exception as e:
        logger.warning(f"Не удалось отредактировать Telegram-уведомление {message_id}: {e}")
        return False
