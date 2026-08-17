"""ML компоненты: Feature Store, Trainer, Model Registry, Inference."""
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sqlalchemy import select, update

from src.config import settings
from src.db.session import get_session
from src.db.models import MLModel, MLFeature, Strategy
from src.utils.logging import logger
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_CLASSIFIER_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

DEFAULT_REGRESSOR_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.05,
}

MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
        labels: Optional[dict] = None,
    ):
        """Добавить фичи в онлайн-хранилище и сохранить в БД."""
        key = f"{symbol}:{timeframe}:{timestamp.isoformat()}"
        self._online_features[key] = features

        try:
            async with get_session() as session:
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
                await session.commit()
        except Exception as e:
            logger.debug(f"Не удалось сохранить фичи в БД: {e}")

    async def get_latest_features(self, symbol: str, timeframe: str = "1h") -> Optional[dict]:
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
        symbol: Optional[str] = None,
        limit: int = 10000,
    ) -> Optional[pd.DataFrame]:
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


class ModelTrainer:
    """Тренер ML моделей (LightGBM)."""

    def __init__(self):
        self.feature_store = FeatureStore()
        self.last_training_time: Optional[datetime] = None

    async def train_direction_classifier(
        self,
        symbol: Optional[str] = None,
        training_data: Optional[pd.DataFrame] = None,
    ) -> Optional[dict]:
        """
        Обучить модель классификации направления.
        Возвращает dict с результатами обучения и метриками.
        """
        logger.info(f"Начато обучение direction classifier" + (f" для {symbol}" if symbol else ""))

        if training_data is None:
            training_data = await self.feature_store.get_features_for_training(symbol)

        if training_data is None or training_data.empty:
            logger.warning("Нет данных для обучения direction classifier")
            return None

        # Подготовка данных
        feature_cols = [
            "rsi_14", "rsi_7", "rsi_21",
            "macd", "macd_signal", "macd_hist",
            "bb_pct", "bb_width",
            "ema_20_slope", "ema_50_slope",
            "price_above_ema20", "price_above_ema50",
            "atr_14", "natr_14",
            "realized_vol_20", "realized_vol_60",
            "volume_ratio", "obv",
            "return_1", "return_3", "return_5", "return_10",
            "log_return", "momentum_10", "momentum_20",
            "dist_from_ema20", "dist_from_ema50",
            "high_low_range", "range_ratio",
            "stoch_k", "stoch_d", "wr_14", "mfi_14",
            "hour", "day_of_week", "is_weekend",
            "roc_5", "roc_10", "roc_20",
        ]
        available_cols = [c for c in feature_cols if c in training_data.columns]

        if not available_cols:
            logger.warning("Нет доступных признаков для обучения")
            return None

        X = training_data[available_cols].dropna()
        y = training_data.loc[X.index, "label_direction"]

        if len(X) < 100:
            logger.warning(f"Слишком мало данных для обучения: {len(X)}")
            return None

        # Разделение на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        # Подбор гиперпараметров (Optuna) при достаточном объёме данных
        if len(X_train) >= settings.ml_optuna_min_samples:
            best_params = self._tune_classifier_params(
                X_train, y_train, X_val, y_val, settings.ml_optuna_trials
            )
            logger.info(f"Optuna: лучшие параметры direction classifier: {best_params}")
        else:
            best_params = dict(DEFAULT_CLASSIFIER_PARAMS)

        # Тренировка LightGBM
        model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
            ],
        )

        # Метрики
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        # Сохранение модели
        version = await self._get_next_version("direction_classifier")
        model_path = MODELS_DIR / f"direction_classifier_v{version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_cols": available_cols,
                "version": version,
                "trained_at": utcnow().isoformat(),
            }, f)

        logger.info(
            f"✅ Direction classifier обучен: v{version} | "
            f"accuracy={accuracy:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}"
        )

        # Регистрация в БД
        try:
            async with get_session() as session:
                model_record = MLModel(
                    model_type="direction_classifier",
                    version=version,
                    model_path=str(model_path),
                    params={
                        **best_params,
                        "feature_cols": available_cols,
                    },
                    metrics={
                        "accuracy": round(accuracy, 4),
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1": round(f1, 4),
                        "train_samples": len(X_train),
                        "val_samples": len(X_val),
                    },
                    is_active=True,
                    released_at=utcnow(),
                )
                session.add(model_record)
                await session.commit()
        except Exception as e:
            logger.warning(f"Не удалось зарегистрировать модель в БД: {e}")

        return {
            "version": version,
            "model_path": str(model_path),
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            },
            "feature_cols": available_cols,
            "trained_at": utcnow().isoformat(),
        }

    async def train_volatility_predictor(
        self,
        symbol: Optional[str] = None,
        training_data: Optional[pd.DataFrame] = None,
    ) -> Optional[dict]:
        """Обучить модель предсказания волатильности."""
        logger.info(f"Начато обучение volatility predictor" + (f" для {symbol}" if symbol else ""))

        if training_data is None:
            training_data = await self.feature_store.get_features_for_training(symbol)

        if training_data is None or training_data.empty or "label_volatility" not in training_data.columns:
            logger.warning("Нет данных для обучения volatility predictor")
            return None

        feature_cols = [
            "rsi_14", "natr_14", "realized_vol_20", "realized_vol_60",
            "volume_ratio", "atr_14", "return_1", "return_3",
        ]
        available_cols = [c for c in feature_cols if c in training_data.columns]

        if not available_cols:
            return None

        X = training_data[available_cols].dropna()
        y = training_data.loc[X.index, "label_volatility"]

        if len(X) < 100:
            return None

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if len(X_train) >= settings.ml_optuna_min_samples:
            best_params = self._tune_regressor_params(
                X_train, y_train, X_val, y_val, settings.ml_optuna_trials
            )
            logger.info(f"Optuna: лучшие параметры volatility predictor: {best_params}")
        else:
            best_params = dict(DEFAULT_REGRESSOR_PARAMS)

        model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        y_pred = model.predict(X_val)
        mse = np.mean((y_val - y_pred) ** 2)
        mae = np.mean(np.abs(y_val - y_pred))

        version = await self._get_next_version("volatility_predictor")
        model_path = MODELS_DIR / f"volatility_predictor_v{version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_cols": available_cols,
                "version": version,
                "trained_at": utcnow().isoformat(),
            }, f)

        logger.info(f"✅ Volatility predictor обучен: v{version} | MSE={mse:.6f} MAE={mae:.6f}")

        return {
            "version": version,
            "model_path": str(model_path),
            "metrics": {"mse": round(mse, 6), "mae": round(mae, 6)},
            "trained_at": utcnow().isoformat(),
        }

    def _tune_classifier_params(self, X_train, y_train, X_val, y_val, n_trials: int) -> dict:
        """Подобрать гиперпараметры LightGBM-классификатора через Optuna."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            preds = model.predict(X_val)
            return f1_score(y_val, preds, average="weighted", zero_division=0)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params

    def _tune_regressor_params(self, X_train, y_train, X_val, y_val, n_trials: int) -> dict:
        """Подобрать гиперпараметры LightGBM-регрессора через Optuna."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            preds = model.predict(X_val)
            return float(np.mean((y_val - preds) ** 2))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params

    async def _get_next_version(self, model_type: str) -> int:
        """Получить следующую версию модели."""
        try:
            async with get_session() as session:
                from src.db.models import MLModel as M
                result = await session.execute(
                    select(M).where(M.model_type == model_type).order_by(M.version.desc()).limit(1)
                )
                last = result.scalar_one_or_none()
                return (last.version if last else 0) + 1
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

    async def load_active_model(self, model_type: str) -> Optional[dict]:
        """Загрузить активную модель из БД."""
        try:
            async with get_session() as session:
                from src.db.models import MLModel as M
                result = await session.execute(
                    select(M)
                    .where(M.model_type == model_type, M.is_active == True)
                    .order_by(M.version.desc())
                    .limit(1)
                )
                model = result.scalar_one_or_none()
                if model:
                    loaded = {
                        "id": model.id,
                        "model_type": model_type,
                        "version": model.version,
                        "model_path": model.model_path,
                        "params": model.params,
                        "metrics": model.metrics,
                        "released_at": model.released_at.isoformat() if model.released_at else None,
                    }
                    self._active_models[model_type] = loaded
                    return loaded
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки активной модели {model_type}: {e}")
            return None

    async def get_model_version(self, model_type: str, version: int) -> Optional[dict]:
        """Получить конкретную версию модели."""
        try:
            async with get_session() as session:
                from src.db.models import MLModel as M
                result = await session.execute(
                    select(M).where(M.model_type == model_type, M.version == version)
                )
                model = result.scalar_one_or_none()
                if model:
                    return {
                        "id": model.id,
                        "model_type": model_type,
                        "version": model.version,
                        "model_path": model.model_path,
                        "params": model.params,
                        "metrics": model.metrics,
                        "released_at": model.released_at.isoformat() if model.released_at else None,
                        "is_active": model.is_active,
                    }
            return None
        except Exception as e:
            logger.error(f"Ошибка получения версии модели {model_type} v{version}: {e}")
            return None

    async def list_models(self, model_type: Optional[str] = None) -> list[dict]:
        """Список моделей."""
        try:
            async with get_session() as session:
                from src.db.models import MLModel as M
                query = select(M)
                if model_type:
                    query = query.where(M.model_type == model_type)
                result = await session.execute(query.order_by(M.created_at.desc()))
                models = result.scalars().all()
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
        """Активировать модель определённой версии."""
        try:
            async with get_session() as session:
                from src.db.models import MLModel as M
                # Deactivate all
                await session.execute(
                    update(M).where(M.model_type == model_type).values(is_active=False)
                )
                # Activate specific
                result = await session.execute(
                    update(M)
                    .where(M.model_type == model_type, M.version == version)
                    .values(is_active=True)
                )
                await session.commit()
                success = result.rowcount > 0
            if success:
                logger.info(f"✅ Модель {model_type} v{version} активирована")
                await self.load_active_model(model_type)
            return success
        except Exception as e:
            logger.error(f"Ошибка активации модели {model_type} v{version}: {e}")
            return False

    async def get_active_model(self, model_type: str) -> Optional[dict]:
        """Получить активную модель (из кэша или загрузить)."""
        if model_type in self._active_models:
            return self._active_models[model_type]
        return await self.load_active_model(model_type)


class MLInference:
    """Инференс ML моделей (предсказания)."""

    def __init__(self):
        self.registry = ModelRegistry()
        self._models: dict[str, Any] = {}

    def load_model(self, model_type: str, model_path: str) -> bool:
        """Загрузить модель из файла."""
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self._models[model_type] = data
            logger.info(f"ML модель загружена: {model_type} из {model_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_type}: {e}")
            return False

    async def predict_direction(
        self,
        features: dict[str, float],
        model_type: str = "direction_classifier",
    ) -> Optional[dict]:
        """
        Предсказать направление на основе фичей.
        Возвращает {proba_up, proba_down, proba_neutral, feature_importance}
        """
        model_data = self._models.get(model_type)
        if model_data is None:
            # Попробовать загрузить из реестра
            active = await self.registry.get_active_model(model_type)
            if active and active.get("model_path"):
                if self.load_model(model_type, active["model_path"]):
                    model_data = self._models.get(model_type)

        if model_data is None:
            logger.warning(f"ML модель {model_type} не загружена, предсказание невозможно")
            return None

        model = model_data["model"]
        feature_cols = model_data.get("feature_cols", [])

        # Подготовка признаков
        X = []
        for col in feature_cols:
            val = features.get(col, 0)
            X.append(val)

        if len(X) != len(feature_cols):
            logger.warning(f"Не все признаки предоставлены: {len(X)} vs {len(feature_cols)}")
            return None

        # Предсказание
        try:
            proba = model.predict_proba([X])[0]
            classes = model.classes_

            # Составляем результат
            result = {
                "proba_up": float(proba[1]) if 1 in classes else 0.0,
                "proba_down": float(proba[-1]) if -1 in classes else 0.0,
                "proba_neutral": float(proba[0]) if 0 in classes else 1.0 - float(proba[1]) - float(proba[-1]),
            }

            # Feature importance (если доступно)
            if hasattr(model, "feature_importances_"):
                result["feature_importance"] = dict(zip(feature_cols, model.feature_importances_))

            logger.debug(f"ML Inference: P(up)={result['proba_up']:.2f} P(down)={result['proba_down']:.2f}")
            return result

        except Exception as e:
            logger.error(f"Ошибка инференса модели {model_type}: {e}")
            return None

    async def predict_volatility(
        self,
        features: dict[str, float],
    ) -> Optional[float]:
        """Предсказать волатильность."""
        model_data = self._models.get("volatility_predictor")
        if model_data is None:
            active = await self.registry.get_active_model("volatility_predictor")
            if active and active.get("model_path"):
                if self.load_model("volatility_predictor", active["model_path"]):
                    model_data = self._models.get("volatility_predictor")

        if model_data is None:
            return None

        model = model_data["model"]
        feature_cols = model_data.get("feature_cols", [])

        X = []
        for col in feature_cols:
            val = features.get(col, 0)
            X.append(val)

        try:
            pred = model.predict([X])[0]
            return float(pred)
        except Exception as e:
            logger.error(f"Ошибка инференса volatility: {e}")
            return None


# Глобальные экземпляры
feature_store = FeatureStore()
model_trainer = ModelTrainer()
model_registry = ModelRegistry()
ml_inference = MLInference()
