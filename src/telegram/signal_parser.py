"""Telegram signal parser — parsing signals from messages with regex + LLM fallback."""
import logging
from typing import Optional

from src.telegram.channel_monitor import parse_with_regex

logger = logging.getLogger(__name__)


class TelegramSignalParser:
    """
    Парсер Telegram сигналов.
    Поддерживает:
    - Regex парсинг для стандартных форматов
    - LLM-based парсинг (заглушка, интеграция с API)
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def parse(self, text: str, channel_config: Optional[dict] = None) -> Optional[dict]:
        """
        Попытаться распарсить сигнал.
        Сначала regex, потом LLM если включён.
        """
        # 1. Regex
        parsed = parse_with_regex(text)
        if parsed:
            parsed["parser"] = "regex"
            return parsed

        # 2. LLM (опционально)
        if self.use_llm and channel_config:
            parsed = self._parse_with_llm(text, channel_config)
            if parsed:
                parsed["parser"] = "llm"
                return parsed

        return None

    def _parse_with_llm(self, text: str, channel_config: dict) -> Optional[dict]:
        """
        LLM-based парсинг (заглушка).
        В реальной реализации — вызов LLM API (OpenAI,本地 모델).
        """
        # Заглушка — в реальном коде интеграция с LLM API
        # Пример интеграции OpenAI:
        # import openai
        # response = openai.chat.completions.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "Extract trading signal from message. Return JSON: {'pair': 'BTC/USDT', 'side': 'long|short', 'entry': float, 'sl': float, 'tp': float, 'rationale': '...'}"},
        #         {"role": "user", "content": text},
        #     ],
        # )
        # result = response.choices[0].message.content
        # parsed = json.loads(result)

        logger.warning(f"LLM парсинг не реализован для текста: {text[:100]}")
        return None

    def validate_signal(self, signal: dict) -> bool:
        """Проверить валидность распарсенного сигнала."""
        if not signal:
            return False

        required = ["pair", "side", "entry"]
        for field in required:
            if field not in signal or signal[field] is None:
                return False

        if signal["side"] not in ["long", "short"]:
            return False

        if signal["entry"] <= 0:
            return False

        return True


# Глобальный экземпляр
telegram_signal_parser = TelegramSignalParser(use_llm=False)
