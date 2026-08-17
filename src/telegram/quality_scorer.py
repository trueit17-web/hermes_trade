"""Telegram signal quality scorer — оценка качества сигналов."""
import logging
from typing import Optional

import numpy as np

from src.utils.logging import logger

logger = logging.getLogger(__name__)


class SignalQualityScorer:
    """
    Оценка качества Telegram сигналов.
    Используется для фильтрации сигналов перед исполнением.
    """

    def __init__(self):
        self.channel_stats: dict[str, dict] = {}

    def update_channel_stats(self, channel_id: str, signal_was_good: bool):
        """Обновить статистику канала."""
        if channel_id not in self.channel_stats:
            self.channel_stats[channel_id] = {
                "signals_count": 0,
                "good_signals": 0,
                "bad_signals": 0,
                "avg_entry_delay": 0,
                "win_rate": 0.5,
            }

        stats = self.channel_stats[channel_id]
        stats["signals_count"] += 1
        if signal_was_good:
            stats["good_signals"] += 1
        else:
            stats["bad_signals"] += 1

        # Пересчёт win rate
        total = stats["good_signals"] + stats["bad_signals"]
        stats["win_rate"] = stats["good_signals"] / total if total > 0 else 0.5

        logger.debug(f"Канал {channel_id}: win_rate={stats['win_rate']:.2%}, signals={total}")

    def score_signal(
        self,
        signal: dict,
        channel_id: str,
        market_context: Optional[dict] = None,
    ) -> float:
        """
        Оценить качество сигнала (0.0 - 1.0).
        Чем выше — тем качественнее.
        """
        score = 0.5  # базовый балл

        # 1. Историческая точность канала
        stats = self.channel_stats.get(channel_id, {})
        channel_win_rate = stats.get("win_rate", 0.5)
        score *= (0.5 + channel_win_rate * 0.5)  # вклад 50%

        # 2. Уверенность сигнала (если есть)
        confidence = signal.get("confidence", 0.5)
        score *= (0.3 + confidence * 0.7)  # вклад 30%

        # 3. Совпадение с рыночным контекстом
        if market_context:
            trend = market_context.get("trend", "neutral")
            signal_side = signal.get("side", "")

            if trend == "bull" and signal_side == "long":
                score *= 1.1
            elif trend == "bear" and signal_side == "short":
                score *= 1.1
            elif trend == "neutral":
                score *= 1.0
            else:
                score *= 0.9  # против тренда — снижаем

        # 4. Наличие SL/TP (риск-менеджмент)
        has_sl = signal.get("sl") is not None
        has_tp = signal.get("tp") is not None
        if has_sl and has_tp:
            score *= 1.1
        elif has_sl:
            score *= 1.0
        else:
            score *= 0.7  # нет SL — рискованно

        # 5. Справедливый RR (risk/reward)
        entry = signal.get("entry", 0)
        sl = signal.get("sl", 0)
        tp = signal.get("tp", 0)
        if entry > 0 and sl > 0 and tp > 0:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio >= 2.0:
                    score *= 1.05
                elif rr_ratio >= 1.5:
                    score *= 1.0
                elif rr_ratio >= 1.0:
                    score *= 0.95
                else:
                    score *= 0.8

        # Ограничиваем 0-1
        score = max(0.0, min(1.0, score))

        return round(score, 3)

    def get_threshold(self, channel_id: str) -> float:
        """Получить порог качества для канала."""
        config = self.channel_stats.get(channel_id, {})
        return config.get("quality_threshold", 0.5)

    def set_threshold(self, channel_id: str, threshold: float):
        """Установить порог качества для канала."""
        if channel_id not in self.channel_stats:
            self.channel_stats[channel_id] = {}
        self.channel_stats[channel_id]["quality_threshold"] = threshold
        logger.info(f"Порог качества канала {channel_id} установлен: {threshold:.2f}")


# Глобальный экземпляр
signal_quality_scorer = SignalQualityScorer()
