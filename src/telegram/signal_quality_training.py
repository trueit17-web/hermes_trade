"""
Обучающий датасет для ML-модели качества сигнала — третий этап плана
(после history_backfill.py и signal_outcome_simulation.py): превращает
уже размеченные (simulated_outcome не NULL) строки HistoricalSignal в
таблицу признаков + бинарную метку "убыточный сигнал", готовую для
ModelTrainer.train_signal_quality_classifier (src/ml/__init__.py).

Признаки посчитаны ТОЛЬКО из того, что известно В МОМЕНТ публикации
сигнала (геометрия SL/TP канала, плечо, время) — специально без цены
исполнения/результата сделки, иначе модель не сможет использоваться на
ещё не отыгранных сигналах в реальном времени.
"""
from datetime import datetime

import pandas as pd
from sqlalchemy import select

from src.db.models import HistoricalSignal
from src.db.session import get_session
from src.telegram.signal_outcome_simulation import _tp_levels

SIGNAL_QUALITY_FEATURE_COLS = [
    "sl_distance_pct",
    "first_tp_distance_pct",
    "final_tp_distance_pct",
    "risk_reward_ratio",
    "num_tp_levels",
    "leverage",
    "is_long",
    "hour",
    "day_of_week",
]

# Метка (простановка "loss") считается только по этим трём разрешённым
# исходам — "unresolved"/NULL сигналы не участвуют в обучении вообще
# (см. отбор в build_signal_quality_training_data).
_RESOLVED_OUTCOMES = ("win", "loss", "break-even")


def extract_signal_features(
    side: str,
    entry: float,
    sl: float,
    tp: float | None,
    take_profits: list[float] | None,
    leverage: float | None,
    message_date: datetime,
) -> dict[str, float] | None:
    """
    Чистая функция без БД/сети — та же геометрия TP-уровней, что и у
    симуляции исхода (_tp_levels), но признаки считаются ДО срабатывания
    любого уровня, только по объявленным ценам сигнала.

    None, если данных недостаточно, чтобы посчитать признаки (нет entry
    или sl_distance_pct вышел нулевым — деление на entry_distance ниже
    было бы неопределено).
    """
    tp_levels = _tp_levels(entry, tp, take_profits)
    if not tp_levels or not sl or not entry:
        return None

    sl_distance_pct = abs(entry - sl) / entry * 100
    if sl_distance_pct == 0:
        return None

    final_tp_distance_pct = abs(tp_levels[-1] - entry) / entry * 100

    return {
        "sl_distance_pct": sl_distance_pct,
        "first_tp_distance_pct": abs(tp_levels[0] - entry) / entry * 100,
        "final_tp_distance_pct": final_tp_distance_pct,
        "risk_reward_ratio": final_tp_distance_pct / sl_distance_pct,
        "num_tp_levels": float(len(tp_levels)),
        "leverage": float(leverage) if leverage else 0.0,
        "is_long": 1.0 if side == "long" else 0.0,
        "hour": float(message_date.hour),
        "day_of_week": float(message_date.weekday()),
    }


async def build_signal_quality_training_data() -> pd.DataFrame | None:
    """
    Тянет ВСЕ уже просимулированные (win/loss/break-even, любой канал)
    исторические сигналы и строит из них датафрейм признаков + колонку
    "target" (1.0 — сигнал закрылся в убыток, 0.0 — иначе). None, если
    подходящих строк нет вообще.
    """
    async with get_session() as session:
        rows = (
            await session.execute(
                select(HistoricalSignal).where(
                    HistoricalSignal.simulated_outcome.in_(_RESOLVED_OUTCOMES),
                    HistoricalSignal.parsed_entry.is_not(None),
                    HistoricalSignal.parsed_sl.is_not(None),
                    HistoricalSignal.parsed_side.is_not(None),
                )
            )
        ).scalars().all()

    data = []
    for row in rows:
        features = extract_signal_features(
            row.parsed_side,
            float(row.parsed_entry),
            float(row.parsed_sl),
            float(row.parsed_tp) if row.parsed_tp else None,
            [float(x) for x in row.parsed_take_profits] if row.parsed_take_profits else None,
            float(row.parsed_leverage) if row.parsed_leverage else None,
            row.message_date,
        )
        if features is None:
            continue
        features["target"] = 1.0 if row.simulated_outcome == "loss" else 0.0
        data.append(features)

    return pd.DataFrame(data) if data else None


async def get_signal_quality_dataset_summary() -> dict:
    """Сколько сейчас размеченных строк доступно под обучение — для
    отображения во вкладке "Обучение" дашборда без реального запуска
    тренировки."""
    async with get_session() as session:
        rows = (
            await session.execute(
                select(HistoricalSignal.simulated_outcome).where(
                    HistoricalSignal.simulated_outcome.in_(_RESOLVED_OUTCOMES),
                )
            )
        ).scalars().all()

    counts = {"win": 0, "loss": 0, "break-even": 0}
    for outcome in rows:
        counts[outcome] += 1

    from src.ml import MIN_TRAINING_SAMPLES

    total = len(rows)
    return {
        "total_resolved": total,
        "win": counts["win"],
        "loss": counts["loss"],
        "break_even": counts["break-even"],
        "min_training_samples": MIN_TRAINING_SAMPLES,
        "ready": total >= MIN_TRAINING_SAMPLES,
    }
