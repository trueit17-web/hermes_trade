"""Telegram-мониторинг и парсинг сигналов.

Модуль позволяет:
- Мониторить Telegram-каналы на торговые сигналы
- Парсить сообщения регулярками + LLM fallback
- Оценивать качество сигналов (quality scorer)
- Хранить историю сигналов в БД
- Передавать сигналы в основной цикл бота через event_bus
"""
from src.telegram.channel_monitor import (
    close_telegram,
    init_telegram,
    parse_telegram_signal,
    subscribe_telegram_signal,
    unsubscribe_telegram_signal,
)
from src.telegram.notifier import (
    send_notification,
)
from src.telegram.quality_scorer import (
    signal_quality_scorer,
)

__all__ = [
    "close_telegram",
    "init_telegram",
    "parse_telegram_signal",
    "send_notification",
    "signal_quality_scorer",
    "subscribe_telegram_signal",
    "unsubscribe_telegram_signal",
]
