"""Telegram signal quality scorer — оценка качества сигналов."""
import logging

logger = logging.getLogger(__name__)

_PRIOR_WINS = 2.0
_PRIOR_LOSSES = 2.0


class SignalQualityScorer:
    """
    Оценка качества Telegram сигналов.
    Используется для фильтрации сигналов перед исполнением.
    """

    def __init__(self):
        self.channel_stats: dict[str, dict] = {}

    async def restore_channel_stats_from_db(self):
        """
        Восстановить channel_stats из истории Telegram-сигналов в БД при
        старте бота. channel_stats существует только в памяти — без этого
        каждый рестарт бота (в т.ч. кнопкой в дашборде) обнулял бы
        накопленную историческую точность канала обратно к нейтральным
        50%, хотя вся история решений и исходов сделок хранится в БД
        (TelegramSignal.executed_trade -> Trade.outcome).
        """
        from sqlalchemy import select

        from src.db.models import TelegramChannel, TelegramSignal, Trade
        from src.db.session import get_session

        # Один JOIN-запрос вместо запроса на каждый канал (N+1).
        try:
            async with get_session() as session:
                rows = (
                    await session.execute(
                        select(TelegramChannel.channel_id, Trade.outcome)
                        .join(TelegramSignal, TelegramSignal.channel_id == TelegramChannel.id)
                        .join(Trade, TelegramSignal.executed_trade_id == Trade.id)
                        .where(Trade.outcome.is_not(None))
                        .order_by(TelegramSignal.id)
                    )
                ).all()
        except Exception as e:
            logger.warning(f"Не удалось восстановить channel_stats из БД: {e}")
            return

        for channel_id, outcome in rows:
            self.update_channel_stats(channel_id, outcome == "win")

        if self.channel_stats:
            logger.info(
                f"♻️ Восстановлена статистика качества по {len(self.channel_stats)} Telegram-каналам из БД"
            )

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

    def smoothed_win_rate(self, channel_id: str) -> float:
        """
        Win-rate канала с байесовским сглаживанием (априорное Beta(2, 2)):
        канал с одной выигрышной сделкой раньше получал win_rate=1.0 и
        максимальный бонус к качеству; теперь оценка растёт по мере
        накопления истории: 1/1 -> 0.6, 8/10 -> 0.71, 80/100 -> 0.79.
        """
        stats = self.channel_stats.get(channel_id)
        if not stats:
            return 0.5
        good = stats.get("good_signals")
        bad = stats.get("bad_signals")
        if good is None or bad is None:
            return stats.get("win_rate", 0.5)
        return (good + _PRIOR_WINS) / (good + bad + _PRIOR_WINS + _PRIOR_LOSSES)

    def score_signal(
        self,
        signal: dict,
        channel_id: str,
        market_context: dict | None = None,
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
        score += self.smoothed_win_rate(channel_id) * 0.35

        # 2. Уверенность сигнала (если есть) — до 35%
        confidence = signal.get("confidence", 0.5)
        score += confidence * 0.35

        # 3. Совпадение с рыночным контекстом — до 10%
        if market_context:
            trend = market_context.get("trend", "neutral")
            signal_side = signal.get("side", "")

            if trend == "bull" and signal_side == "long" or trend == "bear" and signal_side == "short":
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
        # "or 0" — не просто fallback на отсутствующий ключ: signal.get(...)
        # с явным ключом, чьё значение None (SL/TP не распознаны в
        # сообщении канала), возвращает именно None, а не default 0 —
        # .get(key, default) подставляет default только когда ключа нет
        # вообще. Без "or 0" сравнение "None > 0" ниже падало бы с
        # TypeError на любом сигнале без SL/TP.
        entry = signal.get("entry", 0) or 0
        sl = signal.get("sl", 0) or 0
        tp = signal.get("tp", 0) or 0
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

        # 6. ML-модель качества сигнала (если обучена и лучше базовой линии):
        # сдвиг относительно средней доли убыточных сигналов, так что в
        # среднем по потоку сигналов поправка ~0 и не смещает пороги каналов.
        p_loss = signal.get("ml_loss_proba")
        if p_loss is not None:
            from src.config import settings
            base_rate = signal.get("ml_base_loss_rate", 0.5)
            score += settings.ml_signal_quality_weight * (base_rate - p_loss)

        # Ограничиваем 0-1
        score = max(0.0, min(1.0, score))

        return round(score, 3)


# Глобальный экземпляр
signal_quality_scorer = SignalQualityScorer()
