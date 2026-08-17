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
        Чем выше — тем качественнее. Аддитивная модель: каждый фактор
        вносит свой вклад независимо, что даёт предсказуемый диапазон
        (мультипликативная модель с базой 0.5 не может превысить ~0.64
        даже для идеального сигнала).
        """
        score = 0.0

        # 1. Историческая точность канала — до 35%
        stats = self.channel_stats.get(channel_id, {})
        channel_win_rate = stats.get("win_rate", 0.5)
        score += channel_win_rate * 0.35

        # 2. Уверенность сигнала (если есть) — до 35%
        confidence = signal.get("confidence", 0.5)
        score += confidence * 0.35

        # 3. Совпадение с рыночным контекстом — до 10%
        if market_context:
            trend = market_context.get("trend", "neutral")
            signal_side = signal.get("side", "")

            if trend == "bull" and signal_side == "long":
                score += 0.1
            elif trend == "bear" and signal_side == "short":
                score += 0.1
            elif trend != "neutral":
                score -= 0.1  # против тренда — штраф

        # 4. Наличие SL/TP (риск-менеджмент) — до 10%
        has_sl = signal.get("sl") is not None
        has_tp = signal.get("tp") is not None
        if has_sl and has_tp:
            score += 0.1
        elif has_sl:
            score += 0.03
        else:
            score -= 0.15  # нет SL — рискованно

        # 5. Справедливый RR (risk/reward) — до 10%
        entry = signal.get("entry", 0)
        sl = signal.get("sl", 0)
        tp = signal.get("tp", 0)
        if entry > 0 and sl > 0 and tp > 0:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio >= 2.0:
                    score += 0.1
                elif rr_ratio >= 1.5:
                    score += 0.05
                elif rr_ratio >= 1.0:
                    score += 0.02

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
