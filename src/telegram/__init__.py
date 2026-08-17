"""Telegram-мониторинг и парсинг сигналов.

Модуль позволяет:
- Мониторить Telegram-каналы на торговые сигналы
- Парсить сообщения регулярками + LLM fallback
- Оценивать качество сигналов (quality scorer)
- Хранить историю сигналов в БД
- Передавать сигналы в основной цикл бота через event_bus
"""
from src.telegram.channel_monitor import (
    init_telegram,
    close_telegram,
    subscribe_telegram_signal,
    unsubscribe_telegram_signal,
    parse_telegram_signal,
)
from src.telegram.signal_parser import (
    telegram_signal_parser,
)
from src.telegram.quality_scorer import (
    signal_quality_scorer,
)
