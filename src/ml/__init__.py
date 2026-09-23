"""ML компоненты: Feature Store, Trainer, Model Registry, Inference."""
import asyncio
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sqlalchemy import func, select, update

from src.config import settings
from src.db.models import MLFeature, MLModel, Trade
from src.db.session import get_session
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_CLASSIFIER_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

DEFAULT_REGRESSOR_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "min_child_samples": 20,
}

MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Ниже этого числа валидных (после dropna по признакам) строк ML-модель не
# обучается вообще — train_direction_classifier/train_volatility_predictor
# просто возвращают None. Вынесено в константу, т.к. раньше было
# продублированным магическим числом в обеих функциях.
MIN_TRAINING_SAMPLES = 100

# _retrain_ml() в main.py не пытается переобучать модель, пока сделок
# меньше этого числа — отдельный, более ранний гейт, чем MIN_TRAINING_SAMPLES
# (сделки и ml_features — разные таблицы с разным темпом накопления).
MIN_TRADES_FOR_RETRAIN_ATTEMPT = 50

# Горизонт метки в часах — совпадает с FeatureEngine.extract_features_for_ml
# (horizon=5 на 1h-свечах); используется как зазор при хронологическом
# разбиении, чтобы метки train не перекрывались с периодом val/test.
LABEL_HORIZON_HOURS = 5

# Минимум свежих строк test, на которых сравниваются активная модель и
# претендент — на меньшем числе сравнение слишком шумное.
MIN_CHAMPION_EVAL_ROWS = 30

# Только масштабно-независимые признаки: модель обучается сразу по всем
# символам, и абсолютные величины (obv, atr_14, macd, лаги цены) у BTC и PEPE
# различаются на порядки — модель запоминала бы "какой это символ", а не
# рыночную ситуацию (natr_14 — нормированный аналог atr_14).
DIRECTION_FEATURE_COLS = [
    "rsi_14", "rsi_7", "rsi_21",
    "bb_pct", "bb_width",
    "ema_20_slope", "ema_50_slope",
    "price_above_ema20", "price_above_ema50",
    "natr_14",
    "realized_vol_20", "realized_vol_60",
    "volume_ratio",
    "return_1", "return_3", "return_5", "return_10", "return_20",
    "log_return", "momentum_10", "momentum_20",
    "dist_from_ema20", "dist_from_ema50",
    "high_low_range", "range_ratio", "close_position",
    "stoch_k", "stoch_d", "wr_14", "mfi_14",
    "hour", "day_of_week", "is_weekend",
    "roc_5", "roc_10", "roc_20",
]

VOLATILITY_FEATURE_COLS = [
    "natr_14", "realized_vol_20", "realized_vol_60",
    "bb_width", "high_low_range", "range_ratio",
    "volume_ratio", "return_1", "return_3", "rsi_14",
    "hour", "day_of_week",
]

ML_FEATURE_COLS = list(dict.fromkeys(DIRECTION_FEATURE_COLS + VOLATILITY_FEATURE_COLS))


class FeatureStore:
    """Хранилище фичей для ML."""

    def __init__(self):
        self._online_features: dict[str, dict] = {}

    async def add_features(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        features: dict[str, float],
        labels: dict | None = None,
    ):
        """Добавить фичи в онлайн-хранилище и сохранить в БД (идемпотентно по symbol+timeframe+timestamp)."""
        key = f"{symbol}:{timeframe}:{timestamp.isoformat()}"
        self._online_features[key] = features

        try:
            async with get_session() as session:
                # Защита от дублей в main.py (_last_ml_feature_ts) — только
                # in-memory и слетает при рестарте бота; проверяем по БД
                # напрямую, иначе после каждого рестарта первая попытка на
                # символ бьётся об uq_feature_unique и логирует ERROR.
                exists = (
                    await session.execute(
                        select(MLFeature.id).where(
                            MLFeature.symbol == symbol,
                            MLFeature.timeframe == timeframe,
                            MLFeature.timestamp == timestamp,
                        )
                    )
                ).scalar_one_or_none()
                if exists is not None:
                    return

                feature = MLFeature(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    features=features,
                    label_direction=labels.get("direction") if labels else None,
                    label_volatility=labels.get("volatility") if labels else None,
                    source="live",
                )
                session.add(feature)
        except Exception as e:
            logger.debug(f"Не удалось сохранить фичи в БД: {e}")

    async def get_latest_features(self, symbol: str, timeframe: str = "1h") -> dict | None:
        """Получить последние фичи для символа (из онлайн-кэша или БД)."""
        cached = self._online_features.get(f"{symbol}:{timeframe}:latest")
        if cached is not None:
            return cached

        try:
            async with get_session() as session:
                from src.db.models import MLFeature as MF
                result = await session.execute(
                    select(MF)
                    .where(MF.symbol == symbol, MF.timeframe == timeframe)
                    .order_by(MF.timestamp.desc())
                    .limit(1)
                )
                feature = result.scalar_one_or_none()
                return dict(feature.features) if feature else None
        except Exception as e:
            logger.error(f"Ошибка получения последних фичей для {symbol}: {e}")
            return None

    async def get_features_for_training(
        self,
        symbol: str | None = None,
        limit: int = 10000,
    ) -> pd.DataFrame | None:
        """Получить фичи для обучения из БД."""
        try:
            async with get_session() as session:
                from src.db.models import MLFeature as MF
                query = select(MF).order_by(MF.timestamp.desc())
                if symbol:
                    query = query.where(MF.symbol == symbol)
                result = await session.execute(query.limit(limit))
                features = result.scalars().all()[::-1]
                data = []
                for f in features:
                    row = dict(f.features)
                    row["symbol"] = f.symbol
                    row["timeframe"] = f.timeframe
                    row["timestamp"] = f.timestamp
                    row["label_direction"] = f.label_direction
                    row["label_volatility"] = f.label_volatility
                    data.append(row)
                return pd.DataFrame(data) if data else None
        except Exception as e:
            logger.error(f"Ошибка получения фичей для обучения: {e}")
            return None

    def clear_online_cache(self):
        """Очистить онлайн кэш."""
        self._online_features.clear()

    async def get_training_readiness(self, symbol: str | None = None) -> dict:
        """
        Сколько данных сейчас доступно для обучения и сколько нужно —
        чтобы ответить на вопрос "почему модель не обучилась/не загружена"
        без необходимости лезть в БД руками. Два независимых порога:
        сделки (Trade) для _retrain_ml() (решает, пытаться ли вообще
        переобучать) и размеченные строки ml_features для самого обучения
        (решает, получится ли обучение, если оно запустится).
        """
        async with get_session() as session:
            feature_filter = [MLFeature.symbol == symbol] if symbol else []

            total_features = (
                await session.execute(
                    select(func.count()).select_from(MLFeature).where(*feature_filter)
                )
            ).scalar_one()
            labeled_direction = (
                await session.execute(
                    select(func.count()).select_from(MLFeature)
                    .where(*feature_filter, MLFeature.label_direction.is_not(None))
                )
            ).scalar_one()
            labeled_volatility = (
                await session.execute(
                    select(func.count()).select_from(MLFeature)
                    .where(*feature_filter, MLFeature.label_volatility.is_not(None))
                )
            ).scalar_one()
            by_symbol = (
                await session.execute(
                    select(MLFeature.symbol, func.count())
                    .where(*feature_filter)
                    .group_by(MLFeature.symbol)
                    .order_by(func.count().desc())
                )
            ).all()
            trades_count = (
                await session.execute(select(func.count()).select_from(Trade))
            ).scalar_one()

        return {
            "total_features": total_features,
            "labeled_direction": labeled_direction,
            "labeled_volatility": labeled_volatility,
            "min_training_samples": MIN_TRAINING_SAMPLES,
            "direction_ready": labeled_direction >= MIN_TRAINING_SAMPLES,
            "volatility_ready": labeled_volatility >= MIN_TRAINING_SAMPLES,
            "trades_count": trades_count,
            "min_trades_for_retrain_attempt": MIN_TRADES_FOR_RETRAIN_ATTEMPT,
            "by_symbol": [{"symbol": s, "count": c} for s, c in by_symbol],
        }


def _to_naive_utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True).dt.tz_localize(None)


def chronological_split(
    frame: pd.DataFrame,
    time_col: str | None,
    val_frac: float,
    test_frac: float,
    purge: pd.Timedelta | None = None,
) -> tuple[pd.Index, pd.Index, pd.Index] | None:
    """
    Разбиение train < val < test строго по времени. Случайное разбиение на
    автокоррелированных свечах давало утечку будущего (соседние часы с почти
    одинаковыми признаками попадали и в train, и в val) и сильно завышенные
    метрики. purge — зазор между частями: метка строки смотрит на horizon
    вперёд, без зазора хвост train "знает" исход начала val.
    """
    if time_col and time_col in frame.columns:
        ordered = frame.sort_values(time_col, kind="stable")
        times = _to_naive_utc(ordered[time_col].to_numpy())
    else:
        ordered, times = frame, None

    n = len(ordered)
    n_test = max(int(round(n * test_frac)), 1)
    n_val = max(int(round(n * val_frac)), 1)
    n_train = n - n_test - n_val
    if n_train < 2:
        return None

    idx = ordered.index
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    if times is not None and purge is not None and purge > pd.Timedelta(0):
        val_start, test_start = times.iloc[n_train], times.iloc[n_train + n_val]
        train_idx = train_idx[(times.iloc[:n_train] < val_start - purge).to_numpy()]
        val_idx = val_idx[(times.iloc[n_train:n_train + n_val] < test_start - purge).to_numpy()]
        if len(train_idx) < 2 or len(val_idx) == 0:
            return None

    return train_idx, val_idx, test_idx


def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _dump_pickle(path: Path, payload: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _fit_lgbm(kind: str, params: dict, X_tr, y_tr, X_val, y_val):
    estimator = lgb.LGBMClassifier if kind == "classifier" else lgb.LGBMRegressor
    model = estimator(**params, random_state=42, verbose=-1)
    with warnings.catch_warnings():
        # lightgbm>=4.7 объявил eval_set устаревшим в пользу eval_X/eval_y,
        # которых нет в более ранних версиях (requirements: >=4.1).
        warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
    return model


def _classifier_loss(model, X, y) -> float:
    return float(log_loss(y, model.predict_proba(X), labels=model.classes_))


def _regressor_loss(model, X, y) -> float:
    return float(mean_absolute_error(y, model.predict(X)))


def _tune_params(kind: str, X_tr, y_tr, X_val, y_val, n_trials: int) -> dict:
    """Optuna (TPE) по proper scoring rule на val: log-loss для классификатора
    (нужны откалиброванные вероятности — стратегия сравнивает их с порогом),
    MAE для регрессора."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = _fit_lgbm(kind, params, X_tr, y_tr, X_val, y_val)
        loss_fn = _classifier_loss if kind == "classifier" else _regressor_loss
        return loss_fn(model, X_val, y_val)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, "subsample_freq": 1}


def _classifier_metrics(model, X, y, y_train) -> dict:
    classes = model.classes_
    proba = model.predict_proba(X)
    pred = classes[np.argmax(proba, axis=1)]
    prior = y_train.value_counts(normalize=True).reindex(classes, fill_value=0.0).to_numpy(dtype=float)
    prior = np.clip(prior, 1e-6, None)
    prior = prior / prior.sum()
    ll = log_loss(y, proba, labels=classes)
    base_ll = log_loss(y, np.tile(prior, (len(y), 1)), labels=classes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced = balanced_accuracy_score(y, pred)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced), 4),
        "precision": round(float(precision_score(y, pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "log_loss": round(float(ll), 4),
        "baseline_log_loss": round(float(base_ll), 4),
        "baseline_accuracy": round(float(np.mean(y.to_numpy() == classes[np.argmax(prior)])), 4),
        "skill": round(float(1 - ll / base_ll) if base_ll > 0 else 0.0, 4),
        "class_prior": {str(int(c)): round(float(p), 4) for c, p in zip(classes, prior)},
    }


def _regressor_metrics(model, X, y, y_train) -> dict:
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    base_mae = mean_absolute_error(y, np.full(len(y), float(y_train.mean())))
    return {
        "mae": round(float(mae), 6),
        "mse": round(float(mean_squared_error(y, pred)), 6),
        "baseline_mae": round(float(base_mae), 6),
        "skill": round(float(1 - mae / base_mae) if base_mae > 0 else 0.0, 4),
    }


def _feature_importance(model, feature_cols: list[str]) -> dict[str, float]:
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        return {}
    total = float(np.sum(raw)) or 1.0
    return {c: round(float(v) / total, 4) for c, v in zip(feature_cols, raw)}


class ModelTrainer:
    """
    Тренер ML моделей (LightGBM). Общий пайплайн для всех моделей:
    хронологическое разбиение train/val/test с зазором -> Optuna на val ->
    early stopping на val -> честная оценка на отложенном test (с базовой
    линией и skill) -> сравнение с текущей активной моделью на тех же
    свежих строках -> активация только если претендент лучше.
    CPU-тяжёлые шаги выполняются в отдельном потоке, чтобы не блокировать
    event loop торгового цикла/веб-панели.
    """

    def __init__(self):
        self.feature_store = FeatureStore()
        self.last_training_time: datetime | None = None

    async def train_direction_classifier(
        self,
        symbol: str | None = None,
        training_data: pd.DataFrame | None = None,
    ) -> dict | None:
        """Классификатор направления цены через LABEL_HORIZON_HOURS свечей (-1/0/+1)."""
        logger.info("Начато обучение direction classifier" + (f" для {symbol}" if symbol else ""))
        if training_data is None:
            training_data = await self.feature_store.get_features_for_training(symbol)
        if training_data is None or training_data.empty or "label_direction" not in training_data.columns:
            logger.warning("Нет данных для обучения direction classifier")
            return None
        return await self._train(
            "direction_classifier", "classifier", training_data, DIRECTION_FEATURE_COLS,
            "label_direction", time_col="timestamp", purge=pd.Timedelta(hours=LABEL_HORIZON_HOURS),
        )

    async def train_volatility_predictor(
        self,
        symbol: str | None = None,
        training_data: pd.DataFrame | None = None,
    ) -> dict | None:
        """Регрессор будущей реализованной волатильности."""
        logger.info("Начато обучение volatility predictor" + (f" для {symbol}" if symbol else ""))
        if training_data is None:
            training_data = await self.feature_store.get_features_for_training(symbol)
        if training_data is None or training_data.empty or "label_volatility" not in training_data.columns:
            logger.warning("Нет данных для обучения volatility predictor")
            return None
        return await self._train(
            "volatility_predictor", "regressor", training_data, VOLATILITY_FEATURE_COLS,
            "label_volatility", time_col="timestamp", purge=pd.Timedelta(hours=LABEL_HORIZON_HOURS),
        )

    async def train_signal_quality_classifier(
        self,
        training_data: pd.DataFrame | None = None,
    ) -> dict | None:
        """
        Классификатор "убыточный ли сигнал канала" по признакам, известным ДО
        исполнения (геометрия SL/TP, плечо, время); метка — simulated_outcome
        HistoricalSignal (см. src/telegram/signal_quality_training.py).
        """
        logger.info("Начато обучение signal quality classifier")
        from src.telegram.signal_quality_training import (
            SIGNAL_QUALITY_FEATURE_COLS,
            build_signal_quality_training_data,
        )

        if training_data is None:
            training_data = await build_signal_quality_training_data()
        if training_data is None or training_data.empty or "target" not in training_data.columns:
            logger.warning("Нет данных для обучения signal quality classifier")
            return None
        return await self._train(
            "signal_quality_classifier", "classifier", training_data, SIGNAL_QUALITY_FEATURE_COLS,
            "target", time_col="message_date", purge=None,
        )

    async def _train(
        self,
        model_type: str,
        kind: str,
        data: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        time_col: str | None,
        purge: pd.Timedelta | None,
    ) -> dict | None:
        available_cols = [c for c in feature_cols if c in data.columns]
        if not available_cols:
            logger.warning(f"Нет доступных признаков для обучения {model_type}")
            return None

        frame = data[data[target_col].notna()]
        frame = frame[frame[available_cols].notna().any(axis=1)]
        if len(frame) < MIN_TRAINING_SAMPLES:
            logger.warning(
                f"Слишком мало данных для обучения {model_type}: {len(frame)} (нужно ≥{MIN_TRAINING_SAMPLES})"
            )
            return None
        frame = frame.reset_index(drop=True)

        y_all = frame[target_col].astype(float)
        if kind == "classifier":
            y_all = y_all.round().astype(int)

        split = chronological_split(
            frame, time_col, settings.ml_val_fraction, settings.ml_test_fraction, purge,
        )
        if split is None:
            logger.warning(f"{model_type}: не удалось построить хронологическое разбиение")
            return None
        train_idx, val_idx, test_idx = split

        if kind == "classifier":
            train_classes = set(y_all.loc[train_idx].unique())
            if len(train_classes) < 2:
                logger.warning(f"{model_type}: в обучающей выборке только один класс — обучение невозможно")
                return None
            val_idx = val_idx[y_all.loc[val_idx].isin(train_classes).to_numpy()]
            test_idx = test_idx[y_all.loc[test_idx].isin(train_classes).to_numpy()]
            if len(val_idx) == 0 or len(test_idx) == 0:
                logger.warning(f"{model_type}: val/test не содержат классов из train")
                return None

        X = frame[available_cols].astype(float)
        X_tr, y_tr = X.loc[train_idx], y_all.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y_all.loc[val_idx]
        X_te, y_te = X.loc[test_idx], y_all.loc[test_idx]

        def fit_and_evaluate():
            if len(X_tr) >= settings.ml_optuna_min_samples:
                params = _tune_params(kind, X_tr, y_tr, X_val, y_val, settings.ml_optuna_trials)
            else:
                params = dict(DEFAULT_CLASSIFIER_PARAMS if kind == "classifier" else DEFAULT_REGRESSOR_PARAMS)
            model = _fit_lgbm(kind, params, X_tr, y_tr, X_val, y_val)
            metrics_fn = _classifier_metrics if kind == "classifier" else _regressor_metrics
            return model, params, metrics_fn(model, X_te, y_te, y_tr)

        model, params, metrics = await asyncio.to_thread(fit_and_evaluate)
        metrics.update({
            "train_samples": len(X_tr), "val_samples": len(X_val), "test_samples": len(X_te),
            "split": "chronological",
        })
        if hasattr(model, "best_iteration_") and model.best_iteration_:
            params = {**params, "n_estimators": int(model.best_iteration_)}

        comparison = await self._compare_with_champion(
            model_type, kind, model, frame.loc[test_idx], y_te, available_cols, time_col,
        )
        promoted, reason = self._promotion_decision(comparison, metrics)
        metrics["promotion"] = {"promoted": promoted, "reason": reason, **(comparison or {})}

        version = await self._get_next_version(model_type)
        model_path = MODELS_DIR / f"{model_type}_v{version}.pkl"
        trained_at = utcnow().isoformat()
        await asyncio.to_thread(_dump_pickle, model_path, {
            "model": model,
            "feature_cols": available_cols,
            "version": version,
            "trained_at": trained_at,
            "metrics": metrics,
            "feature_importance": _feature_importance(model, available_cols),
        })

        registered = False
        try:
            async with get_session() as session:
                session.add(MLModel(
                    model_type=model_type,
                    version=version,
                    model_path=str(model_path),
                    params={**params, "feature_cols": available_cols},
                    metrics=metrics,
                    is_active=False,
                    is_shadow=not promoted,
                    released_at=utcnow(),
                ))
                await session.commit()
            registered = True
        except Exception as e:
            logger.warning(f"Не удалось зарегистрировать модель в БД: {e}")

        if promoted and registered:
            await model_registry.activate_model(model_type, version)
        await self._prune_old_model_files(model_type)

        self.last_training_time = utcnow()
        summary = ", ".join(
            f"{k}={metrics[k]}" for k in ("accuracy", "balanced_accuracy", "log_loss", "mae", "skill")
            if k in metrics
        )
        logger.info(
            f"✅ {model_type} обучен: v{version} | {summary} | "
            f"{'активирован' if promoted and registered else 'НЕ активирован'} ({reason})"
        )
        return {
            "version": version,
            "model_path": str(model_path),
            "metrics": metrics,
            "feature_cols": available_cols,
            "promoted": promoted and registered,
            "trained_at": trained_at,
        }

    async def _compare_with_champion(
        self, model_type, kind, challenger, frame_test, y_test, feature_cols, time_col,
    ) -> dict | None:
        """
        Оценить текущую активную модель и претендента на одних и тех же строках
        отложенного test. Только на строках ПОСЛЕ обучения чемпиона — иначе он
        мог видеть их при своём обучении и выглядел бы нечестно хорошо.
        """
        active = await model_registry.load_active_model(model_type)
        if not active or not active.get("model_path"):
            return None
        try:
            champion_data = await asyncio.to_thread(_load_pickle, active["model_path"])
        except Exception as e:
            logger.warning(f"{model_type}: не удалось загрузить активную модель для сравнения: {e}")
            return {"champion_version": active.get("version"), "comparable": False}

        frame, y = frame_test, y_test
        trained_at = champion_data.get("trained_at")
        if trained_at and time_col and time_col in frame.columns:
            fresh = (_to_naive_utc(frame[time_col].to_numpy()) > pd.Timestamp(trained_at)).to_numpy()
            frame, y = frame[fresh], y[fresh]

        result = {"champion_version": active.get("version"), "eval_rows": int(len(frame)), "comparable": False}
        if len(frame) < MIN_CHAMPION_EVAL_ROWS:
            return result

        champion = champion_data["model"]
        champion_X = frame.reindex(columns=champion_data.get("feature_cols", [])).astype(float)
        challenger_X = frame[feature_cols].astype(float)
        try:
            if kind == "classifier":
                if not set(y.unique()) <= set(champion.classes_):
                    return result
                champion_loss = _classifier_loss(champion, champion_X, y)
                challenger_loss = _classifier_loss(challenger, challenger_X, y)
            else:
                champion_loss = _regressor_loss(champion, champion_X, y)
                challenger_loss = _regressor_loss(challenger, challenger_X, y)
        except Exception as e:
            logger.warning(f"{model_type}: сравнение с активной моделью не удалось: {e}")
            return result

        return {
            **result,
            "comparable": True,
            "champion_loss": round(champion_loss, 6),
            "challenger_loss": round(challenger_loss, 6),
        }

    @staticmethod
    def _promotion_decision(comparison: dict | None, metrics: dict) -> tuple[bool, str]:
        if comparison is None:
            return True, "активной модели нет"
        if comparison.get("comparable"):
            if comparison["challenger_loss"] < comparison["champion_loss"]:
                return True, "лучше активной модели на свежих данных"
            return False, "не лучше активной модели на свежих данных"
        if metrics.get("skill", 0) > 0:
            return True, "активная модель несравнима, претендент лучше базовой линии"
        return False, "активная модель несравнима, претендент не лучше базовой линии"

    async def _prune_old_model_files(self, model_type: str) -> None:
        """Удалить с диска pickle-файлы старых неактивных версий (строки в БД остаются)."""
        keep = max(settings.ml_keep_model_files, 1)
        try:
            async with get_session() as session:
                rows = (
                    await session.execute(
                        select(MLModel.model_path, MLModel.is_active)
                        .where(MLModel.model_type == model_type)
                        .order_by(MLModel.version.desc())
                    )
                ).all()
        except Exception as e:
            logger.debug(f"Не удалось получить список версий {model_type} для очистки: {e}")
            return
        inactive = [path for path, is_active in rows if path and not is_active]
        for path in inactive[keep:]:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as e:
                logger.debug(f"Не удалось удалить старую модель {path}: {e}")

    async def _get_next_version(self, model_type: str) -> int:
        """Получить следующую версию модели."""
        try:
            async with get_session() as session:
                last = (
                    await session.execute(
                        select(func.max(MLModel.version)).where(MLModel.model_type == model_type)
                    )
                ).scalar_one_or_none()
                return (last or 0) + 1
        except Exception:
            return 1

    def get_training_schedule(self) -> dict:
        """Получить конфигурацию расписания обучения."""
        return {
            "interval_hours": settings.ml_retraining_interval_hours,
            "max_trades": settings.ml_max_trades_for_retrain,
            "last_training": self.last_training_time.isoformat() if self.last_training_time else None,
        }


class ModelRegistry:
    """Реестр ML моделей."""

    def __init__(self):
        self._active_models: dict[str, dict] = {}

    @staticmethod
    def _describe(model: MLModel) -> dict:
        return {
            "id": model.id,
            "model_type": model.model_type,
            "version": model.version,
            "model_path": model.model_path,
            "params": model.params,
            "metrics": model.metrics,
            "released_at": model.released_at.isoformat() if model.released_at else None,
            "is_active": model.is_active,
        }

    async def load_active_model(self, model_type: str) -> dict | None:
        """Загрузить активную модель из БД."""
        try:
            async with get_session() as session:
                model = (
                    await session.execute(
                        select(MLModel)
                        .where(MLModel.model_type == model_type, MLModel.is_active == True)  # noqa: E712
                        .order_by(MLModel.version.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
            if model is None:
                self._active_models.pop(model_type, None)
                return None
            loaded = self._describe(model)
            self._active_models[model_type] = loaded
            return loaded
        except Exception as e:
            logger.error(f"Ошибка загрузки активной модели {model_type}: {e}")
            return None

    async def get_model_version(self, model_type: str, version: int) -> dict | None:
        """Получить конкретную версию модели."""
        try:
            async with get_session() as session:
                model = (
                    await session.execute(
                        select(MLModel).where(MLModel.model_type == model_type, MLModel.version == version)
                    )
                ).scalar_one_or_none()
            return self._describe(model) if model else None
        except Exception as e:
            logger.error(f"Ошибка получения версии модели {model_type} v{version}: {e}")
            return None

    async def list_models(self, model_type: str | None = None) -> list[dict]:
        """Список моделей."""
        try:
            async with get_session() as session:
                query = select(MLModel)
                if model_type:
                    query = query.where(MLModel.model_type == model_type)
                models = (await session.execute(query.order_by(MLModel.created_at.desc()))).scalars().all()
            return [
                {
                    "id": m.id,
                    "model_type": m.model_type,
                    "version": m.version,
                    "is_active": m.is_active,
                    "is_shadow": m.is_shadow,
                    "metrics": m.metrics,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in models
            ]
        except Exception as e:
            logger.error(f"Ошибка получения списка моделей: {e}")
            return []

    async def activate_model(self, model_type: str, version: int) -> bool:
        """Активировать модель определённой версии (остальные версии деактивируются)."""
        try:
            async with get_session() as session:
                exists = (
                    await session.execute(
                        select(MLModel.id).where(MLModel.model_type == model_type, MLModel.version == version)
                    )
                ).scalar_one_or_none()
                if exists is None:
                    return False
                await session.execute(
                    update(MLModel).where(MLModel.model_type == model_type).values(is_active=False)
                )
                await session.execute(
                    update(MLModel)
                    .where(MLModel.model_type == model_type, MLModel.version == version)
                    .values(is_active=True, is_shadow=False)
                )
                await session.commit()
        except Exception as e:
            logger.error(f"Ошибка активации модели {model_type} v{version}: {e}")
            return False
        logger.info(f"✅ Модель {model_type} v{version} активирована")
        await self.load_active_model(model_type)
        ml_inference.invalidate(model_type)
        return True

    async def get_active_model(self, model_type: str) -> dict | None:
        """Получить активную модель (из кэша или загрузить)."""
        if model_type in self._active_models:
            return self._active_models[model_type]
        return await self.load_active_model(model_type)


class MLInference:
    """Инференс ML моделей (предсказания)."""

    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry if registry is not None else model_registry
        self._models: dict[str, Any] = {}

    def invalidate(self, model_type: str | None = None) -> None:
        """Сбросить загруженную модель — следующий инференс подтянет активную версию из реестра."""
        if model_type is None:
            self._models.clear()
        else:
            self._models.pop(model_type, None)

    def load_model(self, model_type: str, model_path: str) -> bool:
        """Загрузить модель из файла."""
        try:
            self._models[model_type] = _load_pickle(model_path)
            logger.info(f"ML модель загружена: {model_type} из {model_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_type}: {e}")
            return False

    async def _get(self, model_type: str) -> dict | None:
        model_data = self._models.get(model_type)
        if model_data is None:
            active = await self.registry.get_active_model(model_type)
            if active and active.get("model_path") and self.load_model(model_type, active["model_path"]):
                model_data = self._models.get(model_type)
        return model_data

    @staticmethod
    def _frame(features: dict[str, float], feature_cols: list[str]) -> pd.DataFrame:
        # Отсутствующий признак — NaN, а не 0: LightGBM обрабатывает пропуски
        # нативно, а 0 для, скажем, RSI — это экстремальное значение.
        row = [features.get(col) for col in feature_cols]
        return pd.DataFrame([[np.nan if v is None else float(v) for v in row]], columns=feature_cols)

    @staticmethod
    def _has_no_skill(model_data: dict) -> bool:
        skill = (model_data.get("metrics") or {}).get("skill")
        return skill is not None and skill <= 0

    async def predict_direction(
        self,
        features: dict[str, float],
        model_type: str = "direction_classifier",
    ) -> dict | None:
        """
        Предсказать направление. Вероятности сопоставляются с классами по
        model.classes_ (-1/0/+1), а не по позиции в массиве predict_proba —
        раньше proba[1] при классах [-1, 0, 1] был P(нейтрально), а proba[-1]
        — P(роста), то есть "P(up)" и "P(down)" были перепутаны.
        """
        model_data = await self._get(model_type)
        if model_data is None:
            logger.warning(f"ML модель {model_type} не загружена, предсказание невозможно")
            return None
        if self._has_no_skill(model_data):
            logger.debug(f"ML модель {model_type} не лучше базовой линии — предсказание не используется")
            return None

        model = model_data["model"]
        feature_cols = model_data.get("feature_cols", [])
        try:
            proba = model.predict_proba(self._frame(features, feature_cols))[0]
        except Exception as e:
            logger.error(f"Ошибка инференса модели {model_type}: {e}")
            return None

        by_class = {int(round(float(c))): float(p) for c, p in zip(model.classes_, proba)}
        result = {
            "proba_up": by_class.get(1, 0.0),
            "proba_down": by_class.get(-1, 0.0),
            "proba_neutral": by_class.get(0, 0.0),
        }
        importance = model_data.get("feature_importance")
        if importance is None:
            importance = _feature_importance(model, feature_cols)
            model_data["feature_importance"] = importance
        result["feature_importance"] = importance

        logger.debug(f"ML Inference: P(up)={result['proba_up']:.2f} P(down)={result['proba_down']:.2f}")
        return result

    async def predict_volatility(self, features: dict[str, float]) -> float | None:
        """Предсказать волатильность."""
        model_data = await self._get("volatility_predictor")
        if model_data is None or self._has_no_skill(model_data):
            return None
        try:
            pred = model_data["model"].predict(self._frame(features, model_data.get("feature_cols", [])))[0]
            return float(pred)
        except Exception as e:
            logger.error(f"Ошибка инференса volatility: {e}")
            return None

    async def predict_signal_loss_proba(self, features: dict[str, float]) -> dict | None:
        """
        P(убыток) для Telegram-сигнала + базовая доля убыточных в обучении.
        None, если модели нет или у неё нет доказанного преимущества над
        базовой линией на отложенных данных (в т.ч. модели, обученные до
        появления метрики skill) — такая модель не должна влиять на торговлю.
        """
        model_data = await self._get("signal_quality_classifier")
        if model_data is None:
            return None
        metrics = model_data.get("metrics") or {}
        skill = metrics.get("skill")
        if skill is None or skill <= 0:
            return None

        model = model_data["model"]
        try:
            proba = model.predict_proba(self._frame(features, model_data.get("feature_cols", [])))[0]
        except Exception as e:
            logger.error(f"Ошибка инференса signal quality: {e}")
            return None
        by_class = {int(round(float(c))): float(p) for c, p in zip(model.classes_, proba)}
        base_rate = float((metrics.get("class_prior") or {}).get("1", 0.5))
        return {
            "p_loss": by_class.get(1, 0.0),
            "base_loss_rate": base_rate,
            "skill": float(skill),
            "version": model_data.get("version"),
        }


# Глобальные экземпляры
feature_store = FeatureStore()
model_trainer = ModelTrainer()
model_registry = ModelRegistry()
ml_inference = MLInference(model_registry)
