"""Feature Engineering — построение признаков для стратегий и ML."""
import logging
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Инженер признаков: вычисляет индикаторы, создаёт фичи для ML.
    Работает с pandas DataFrame из OHLCV данных.
    """

    def __init__(self):
        self._indicators_cache: dict[str, pd.DataFrame] = {}

    def compute_all_indicators(
        self,
        df: pd.DataFrame,
        tf_name: str = "1h",
    ) -> pd.DataFrame:
        """
        Вычислить все индикаторы для DataFrame.
        На входе: OHLCV свечи (columns: open, high, low, close, volume).
        На выходе: DataFrame с добавленными индикаторами.
        """
        if df.empty:
            return df

        result = df.copy()

        # === Price-based indicators ===
        # RSI (14, 7, 21)
        result["rsi_14"] = ta.rsi(result["close"], length=14)
        result["rsi_7"] = ta.rsi(result["close"], length=7)
        result["rsi_21"] = ta.rsi(result["close"], length=21)

        # MACD
        macd = ta.macd(result["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            result["macd"] = macd.iloc[:, 0]
            result["macd_signal"] = macd.iloc[:, 1]
            result["macd_hist"] = macd.iloc[:, 2]

        # Bollinger Bands
        bb = ta.bbands(result["close"], length=20, std=2.0)
        if bb is not None:
            result["bb_upper"] = bb.iloc[:, 0]
            result["bb_mid"] = bb.iloc[:, 1]
            result["bb_lower"] = bb.iloc[:, 2]
            result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_mid"]
            result["bb_pct"] = (result["close"] - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"])

        # Moving Averages
        result["ema_20"] = ta.ema(result["close"], length=20)
        result["ema_50"] = ta.ema(result["close"], length=50)
        result["ema_200"] = ta.ema(result["close"], length=200)
        result["sma_20"] = ta.sma(result["close"], length=20)
        result["sma_50"] = ta.sma(result["close"], length=50)

        # EMA Slope (тренд)
        result["ema_20_slope"] = result["ema_20"].diff()
        result["ema_50_slope"] = result["ema_50"].diff()
        result["price_above_ema20"] = (result["close"] > result["ema_20"]).astype(float)
        result["price_above_ema50"] = (result["close"] > result["ema_50"]).astype(float)

        # === Volatility ===
        result["atr_14"] = ta.atr(result["high"], result["low"], result["close"], length=14)
        result["atr_7"] = ta.atr(result["high"], result["low"], result["close"], length=7)
        result["natr_14"] = result["atr_14"] / result["close"]  # Normalised ATR
        result["realized_vol_20"] = result["close"].pct_change().rolling(20).std() * np.sqrt(20)
        result["realized_vol_60"] = result["close"].pct_change().rolling(60).std() * np.sqrt(60)

        # === Volume ===
        result["volume_sma_20"] = ta.sma(result["volume"], length=20)
        result["volume_ratio"] = result["volume"] / (result["volume_sma_20"] + 1e-10)
        result["obv"] = ta.obv(result["close"], result["volume"])

        # Volume profile (gamma, delta, cumulative)
        result["volume_delta"] = result["volume"].diff()
        result["cum_volume_20"] = result["volume"].rolling(20).sum()

        # === Price-based returns ===
        for period in [1, 3, 5, 10, 20, 50]:
            result[f"return_{period}"] = result["close"].pct_change(periods=period)

        result["log_return"] = np.log(result["close"] / result["close"].shift(1))
        result["momentum_10"] = result["close"] / result["close"].shift(10) - 1
        result["momentum_20"] = result["close"] / result["close"].shift(20) - 1

        # Distance from EMA (mean reversion score)
        result["dist_from_ema20"] = (result["close"] - result["ema_20"]) / result["ema_20"]
        result["dist_from_ema50"] = (result["close"] - result["ema_50"]) / result["ema_50"]

        # High-Low range
        result["high_low_range"] = (result["high"] - result["low"]) / result["low"]
        result["range_ratio"] = result["high_low_range"] / (result["atr_14"] / result["close"] + 1e-10)

        # Close position within day range
        result["close_position"] = (result["close"] - result["low"]) / (result["high"] - result["low"] + 1e-10)

        # === Trend strength ===
        adx = ta.adx(result["high"], result["low"], result["close"], length=14)
        if adx is not None:
            result["adx_14"] = adx.iloc[:, 0]

        # Money Flow Index (MFI)
        mfi = ta.mfi(result["high"], result["low"], result["close"], result["volume"], length=14)
        if mfi is not None:
            result["mfi_14"] = mfi

        # Stochastic
        stoch = ta.stoch(result["high"], result["low"], result["close"], k=14, d=3)
        if stoch is not None:
            result["stoch_k"] = stoch.iloc[:, 0]
            result["stoch_d"] = stoch.iloc[:, 1]

        # Williams %R
        result["wr_14"] = ta.willr(result["high"], result["low"], result["close"], length=14)

        # === Session / Calendar features ===
        result["hour"] = result.index.hour if hasattr(result.index, "hour") else 0
        result["day_of_week"] = result.index.dayofweek if hasattr(result.index, "dayofweek") else 0
        result["is_weekend"] = (result.index.dayofweek >= 5).astype(float) if hasattr(result.index, "dayofweek") else 0.0

        # === Lag features (для временных рядов) ===
        for col in ["close", "volume", "return_1"]:
            for lag in [1, 2, 3, 5]:
                result[f"{col}_lag_{lag}"] = result[col].shift(lag)

        # === Rate of change ===
        for period in [5, 10, 20]:
            result[f"roc_{period}"] = ta.roc(result["close"], length=period)

        return result

    def extract_features_for_ml(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
        horizon: int = 5,
        threshold: float = 0.002,
    ) -> pd.DataFrame:
        """
        Извлечь признаки для ML модели.
        horizon: количество свечей вперёд для прогноза
        threshold: минимальное движение цены для считать сигнал значимым
        """
        if df.empty or len(df) < horizon + 20:
            return pd.DataFrame()

        features = df.copy()

        # Вычисляем индикаторы (если ещё не вычислены)
        if "rsi_14" not in features.columns:
            features = self.compute_all_indicators(features)

        # Выбираем колонки-признаки (исключаем сырые цены, которые могут вызвать leakage)
        feature_cols = [
            "rsi_14", "rsi_7", "rsi_21",
            "macd", "macd_signal", "macd_hist",
            "bb_pct", "bb_width",
            "ema_20_slope", "ema_50_slope",
            "price_above_ema20", "price_above_ema50",
            "atr_14", "natr_14",
            "realized_vol_20", "realized_vol_60",
            "volume_ratio", "obv",
            "return_1", "return_3", "return_5", "return_10", "return_20",
            "log_return", "momentum_10", "momentum_20",
            "dist_from_ema20", "dist_from_ema50",
            "high_low_range", "range_ratio",
            "close_position",
            "stoch_k", "stoch_d", "wr_14", "mfi_14",
            "hour", "day_of_week", "is_weekend",
            "roc_5", "roc_10", "roc_20",
        ]

        # Добавляем лаги
        for col in ["close", "volume"]:
            for lag in [1, 2, 3]:
                feature_cols.append(f"{col}_lag_{lag}")

        available_cols = [c for c in feature_cols if c in features.columns]
        feature_df = features[available_cols].copy()

        # Target: направление цены через horizon свечей
        if include_target:
            future_return = features["close"].shift(-horizon) / features["close"] - 1
            feature_df["target_direction"] = (future_return > threshold).astype(float) - (
                future_return < -threshold
            ).astype(float)
            # 1 = up, -1 = down, 0 = neutral

            # Также volatility target (для regression)
            future_vol = features["close"].shift(-horizon).pct_change().rolling(horizon).std()
            feature_df["target_volatility"] = future_vol

        # Удаляем NaN
        feature_df = feature_df.dropna()

        return feature_df

    def compute_features_for_symbol(
        self,
        candles: list[dict],
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """
        Создать DataFrame с фичами из сырых свечей.
        candles: список dict с {open_time, open, high, low, close, volume}
        """
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        df.sort_index(inplace=True)

        return self.compute_all_indicators(df, timeframe)

    def get_latest_features(self, df: pd.DataFrame) -> dict[str, float]:
        """Получить последнюю строку фичей как dict."""
        if df.empty:
            return {}
        latest = df.iloc[-1]
        features = {}
        for col in df.columns:
            if col in ["target_direction", "target_volatility"]:
                continue
            val = latest[col]
            if isinstance(val, (int, float, np.integer, np.floating)):
                if not np.isnan(val):
                    features[col] = float(val)
        return features

    def compute_market_regime(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Определить текущий рыночный режим на основе данных.
        Возвращает: trend (bull/bear/neutral), volatility (low/med/high),
        volume (low/normal/high), regime_score.
        """
        if df.empty or len(df) < 20:
            return {"trend": "unknown", "volatility": "unknown", "volume": "unknown"}

        latest = df.iloc[-1]

        # Trend
        ema20 = latest.get("ema_20", 0)
        ema50 = latest.get("ema_50", 0)
        latest.get("close", 0)
        ema_slope = latest.get("ema_20_slope", 0)

        if ema20 > ema50 * 1.02 and ema_slope > 0:
            trend = "bull"
        elif ema20 < ema50 * 0.98 and ema_slope < 0:
            trend = "bear"
        else:
            trend = "neutral"

        # Volatility (по natr)
        natr = latest.get("natr_14", 0)
        if natr < 0.01:
            volatility = "low"
        elif natr < 0.03:
            volatility = "medium"
        else:
            volatility = "high"

        # Volume
        vol_ratio = latest.get("volume_ratio", 1)
        if vol_ratio < 0.7:
            volume_regime = "low"
        elif vol_ratio < 1.5:
            volume_regime = "normal"
        else:
            volume_regime = "high"

        # Regime score (для ML)
        regime_score = 0.0
        regime_score += 0.3 if trend == "bull" else -0.3 if trend == "bear" else 0.0
        regime_score += 0.2 if volatility == "low" else -0.2 if volatility == "high" else 0.0
        regime_score += 0.1 if volume_regime == "high" else -0.1 if volume_regime == "low" else 0.0

        return {
            "trend": trend,
            "volatility": volatility,
            "volume": volume_regime,
            "regime_score": regime_score,
            "atr": float(latest.get("atr_14", 0)),
            "natr": float(natr),
            "volume_ratio": float(vol_ratio),
        }


# Глобальный экземпляр
_feature_engine: FeatureEngine | None = None


def get_feature_engine() -> FeatureEngine:
    global _feature_engine
    if _feature_engine is None:
        _feature_engine = FeatureEngine()
    return _feature_engine
