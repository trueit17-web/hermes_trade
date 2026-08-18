"""Telegram канал мониторинг — получение сигналов из Telegram каналов."""
import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Optional

from telethon import TelegramClient, events
from telethon.tl.types import Channel, Message
from telethon.utils import get_peer_id

from src.config import settings
from src.utils.logging import logger
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Файл сессии Telethon должен жить в примонтированном volume (data/),
# иначе при пересоздании контейнера он теряется и авторизацию нужно
# проходить заново при каждом запуске.
SESSION_DIR = Path(__file__).parent.parent.parent / "data"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SESSION_PATH = str(SESSION_DIR / "crypto_bot_sessions")

# Глобальный экземпляр клиента
_telegram_client: Optional[TelegramClient] = None
_subscribers: list[callable] = []


async def init_telegram():
    """Инициализировать Telegram клиент."""
    global _telegram_client

    api_id = settings.telegram_api_id
    api_hash = settings.telegram_api_hash
    user_id = settings.telegram_user_id

    if not api_id or not api_hash:
        logger.warning("Telegram API credentials not configured — monitoring disabled")
        return None

    _telegram_client = TelegramClient(SESSION_PATH, api_id, api_hash)

    try:
        await _telegram_client.start()
        me = await _telegram_client.get_me()
        logger.info(f"✅ Telegram клиент инициализирован: {me.username} ({me.id})")
        return _telegram_client
    except Exception as e:
        logger.error(f"Ошибка инициализации Telegram клиента: {e}")
        _telegram_client = None
        return None


async def close_telegram():
    """Закрыть Telegram клиент."""
    global _telegram_client
    if _telegram_client:
        await _telegram_client.disconnect()
        logger.info("Telegram клиент закрыт")
        _telegram_client = None


def get_telegram_client() -> Optional[TelegramClient]:
    """Получить Telegram клиент."""
    return _telegram_client


async def monitor_channels(channels: list[dict]):
    """
    Запустить мониторинг каналов.
    channels: [{"channel_id": "@channelname", "channel_title": "Channel Name", "parser_config": {...}}]
    """
    global _telegram_client, _subscribers

    if not _telegram_client:
        logger.warning("Telegram клиент не инициализирован")
        return

    client = _telegram_client

    logger.info(f"Запуск мониторинга каналов: {[c['channel_id'] for c in channels]}")

    # event.chat_id из Telethon — это всегда нормализованный числовой ID
    # (напр. -1001234567890), а channel_id в БД чаще всего — то, что ввёл
    # пользователь в дашборде ("@channelname"). Раньше матчинг делался как
    # `c["channel_id"] == str(event.chat_id)`, что для username-каналов не
    # совпадало НИКОГДА — все сообщения из таких каналов тихо отбрасывались
    # без единой ошибки в логах. Резолвим каждый канал в реальную entity
    # заранее и матчим по числовому id, а не по исходной строке.
    resolved: dict[int, dict] = {}
    entities = []
    for c in channels:
        try:
            entity = await client.get_entity(c["channel_id"])
            chat_id = get_peer_id(entity)
            resolved[chat_id] = c
            entities.append(entity)
        except Exception as e:
            logger.error(f"Telegram: не удалось найти канал {c['channel_id']}: {e}")

    if not resolved:
        logger.warning("Telegram: ни один канал не удалось резолвить — мониторинг не запущен")
        return

    @client.on(events.NewMessage(chats=entities))
    async def handler(event: events.NewMessage.Event):
        message: Message = event.message
        channel = resolved.get(event.chat_id)

        if channel is None:
            return

        raw_text = message.text or ""
        logger.debug(f"[TG] Получено сообщение из {channel['channel_id']}: {raw_text[:100]}...")

        # Парсинг сигнала
        parsed = parse_telegram_signal(raw_text, channel)

        if parsed:
            signal_event = {
                "type": "telegram_signal",
                "channel_id": channel["channel_id"],
                "channel_title": channel.get("channel_title", ""),
                "raw_message": raw_text,
                "message_date": utcnow(),
                "parsed_pair": parsed.get("pair"),
                "parsed_side": parsed.get("side"),
                "parsed_entry": parsed.get("entry"),
                "parsed_sl": parsed.get("sl"),
                "parsed_tp": parsed.get("tp"),
                "parsed_rationale": parsed.get("rationale", ""),
                "parsed_raw": parsed.get("raw", ""),
            }
            logger.info(
                f"[TG SIGNAL] {signal_event['parsed_pair']} {signal_event['parsed_side'].upper()} | "
                f"Entry: {signal_event['parsed_entry']} | SL: {signal_event['parsed_sl']} | TP: {signal_event['parsed_tp']}"
            )

            # Уведомить подписчиков
            for cb in _subscribers:
                try:
                    await cb(signal_event)
                except Exception as e:
                    logger.error(f"Ошибка уведомления подписчика Telegram сигнала: {e}")

    logger.info(f"Мониторинг каналов запущен: {[c['channel_id'] for c in resolved.values()]}")


def subscribe_telegram_signal(callback: callable):
    """Подписаться на Telegram сигналы."""
    _subscribers.append(callback)


def unsubscribe_telegram_signal(callback: callable):
    """Отписаться от Telegram сигналов."""
    if callback in _subscribers:
        _subscribers.remove(callback)


# === Парсер сигналов ===

def parse_telegram_signal(text: str, channel_config: Optional[dict] = None) -> Optional[dict]:
    """
    Попытаться распарсить торговый сигнал из текста сообщения.
    Возвращает dict с полями: pair, side, entry, sl, tp, rationale, raw.
    Или None если сигнал не распознан.
    """
    if not text:
        return None

    # Попытка 1: регулярки для стандартных форматов
    parsed = parse_with_regex(text)
    if parsed:
        return parsed

    # Попытка 2: LLM/extractor (заглушка — в реальном коде интеграция с LLM API)
    # parsed = parse_with_llm(text, channel_config)
    # if parsed:
    #     return parsed

    return None


def parse_with_regex(text: str) -> Optional[dict]:
    """
    Парсинг сигнала с помощью регулярных выражений.
    Поддерживает форматы:
    - "BTC/USDT Long 69000 SL 68000 TP 72000"
    - "ETH/USDT - Short | Entry: 3500 | Stop: 3600 | Target: 3200"
    - "XRPUSDT LONG 1.85 SL 1.70 TP 2.10"
    - "BTCUSDT short 68500 sl 67500 tp 65000"
    """
    text = text.strip()

    # Нормализуем: убираем лишние символы
    clean = re.sub(r"[•·•]", "", text)
    clean = re.sub(r"\s+", " ", clean)

    # Ищем пару (BTC/USDT, ETH/USDT и т.д.)
    pair_patterns = [
        r"([A-Z]{2,10}/[A-Z]{2,10})",  # BTC/USDT
        r"([A-Z]{2,10}USDT)",  # BTCUSDT
        r"([A-Z]{2,10}USD)",  # BTCUSD
        r"([A-Z]{2,10}/USDT)",  # BTC/USDT
    ]

    pair = None
    for pattern in pair_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pair = match.group(1).upper()
            break

    if not pair:
        return None

    # Ищем направление
    side = None
    side_patterns = [
        r"\b(LONG|LONG|long|buy|BUY)\b",
        r"\b(SHORT|short|sell|SELL)\b",
        r"\b(-)\s*(Short|short|SELL|sell)\b",
    ]

    for pattern in side_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if "long" in match.group(0).lower() or "buy" in match.group(0).lower():
                side = "long"
            elif "short" in match.group(0).lower() or "sell" in match.group(0).lower():
                side = "short"
            break

    if not side:
        # Если нет явного направления, пробуем по контексту
        if "buy" in text.lower() or "long" in text.lower():
            side = "long"
        elif "sell" in text.lower() or "short" in text.lower():
            side = "short"

    if not side:
        return None

    # Ищем entry, SL, TP
    entry = extract_price(text, side, "entry", "price", "entry_price")
    sl = extract_price(text, side, "sl", "stop", "stop_loss", "stop loss")
    tp = extract_price(text, side, "tp", "target", "take_profit", "take profit", "profit")

    if entry is None:
        # Формат без ключевого слова: "BTCUSDT LONG 1.85 SL 1.70 TP 2.10" —
        # цена входа идёт сразу после направления
        match = re.search(r"\b(?:long|short|buy|sell)\b\s*[:\-–—]?\s*([\d.]+)", text, re.IGNORECASE)
        if match:
            try:
                entry = float(match.group(1))
            except ValueError:
                entry = None

    if entry is None:
        return None

    return {
        "pair": pair,
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "raw": text,
    }


def extract_price(
    text: str,
    side: str,
    *keywords,
) -> Optional[float]:
    """
    Извлечь цену из текста по ключевым словам.
    """
    for keyword in keywords:
        # Ищем ключевое слово, за которым следует число
        patterns = [
            rf"{keyword}\s*[:\-–—]?\s*([\d.]+)",  # "SL: 68000" или "SL 68000"
            rf"{keyword}\s+([\d.]+)",  # "SL 68000"
            rf"[\(]+\s*{keyword}\s*[:\-–—]?\s*([\d.]+)",  # "(SL: 68000)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

    return None
