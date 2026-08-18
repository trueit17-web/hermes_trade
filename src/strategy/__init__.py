"""Стратегии торговли — правиловые и ML-стратегии."""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
import numpy as np

from src.config import settings
from src.event_bus import (
    event_bus,
    SignalGeneratedEvent,
    MarketDataEvent,
    TradeEvent,
)
from src.utils.logging import logger
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def min_profitable_margin_pct() -> float:
    """
    Минимальная дистанция TP от цены входа (в долях), при которой номинально
    прибыльный выход (reason=take_profit) остаётся прибыльным и после
    комиссий/слиппеджа paper-режима. Используется там, где TP считается не
    от %, а от произвольного ценового уровня (напр. средняя полоса Боллинджера),
    который может оказаться ближе комиссий — тогда выход по цене TP на деле
    закрывает сделку в минус.
    """
    round_trip_fees = settings.paper_fee_pct * 2 / 100  # открытие + закрытие
    entry_slippage = settings.paper_slippage_pct / 100
    return round_trip_fees + entry_slippage + 0.002  # + запас 0.2%


# === Сигнал от стратегии ===

class StrategySignal:
    """Сигнал, генерируемый стратегией."""
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        side: str,  # long, short
        confidence: float = 0.5,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_size_pct: float = 5.0,
        timeframe: str = "1h",
        rationale: str = "",
    ):
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.side = side
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size_pct = position_size_pct
        self.timeframe = timeframe
        self.rationale = rationale
        self.timestamp = utcnow()


# === Интерфейс стратегии ===

class BaseStrategy(ABC):
    """Базовый класс стратегии."""

    def __init__(self, strategy_id: str, name: str, params: Optional[dict] = None):
        self.strategy_id = strategy_id
        self.name = name
        self.params = params or {}
        self.active = True
        self.weight = 1.0

    @abstractmethod
    def generate_signal(self, data: dict[str, Any]) -> Optional[StrategySignal]:
        """Сгенерировать сигнал на основе данных. Возвращает None если сигнала нет."""
        pass

    def get_config_schema(self) -> dict:
        """JSON schema для настроек стратегии."""
        return {"type": "object", "properties": {}}

    def __repr__(self):
        return f"<Strategy {self.name}({self.strategy_id}) active={self.active}>"


# === RSI Strategy ===

class RSIMeanReversionStrategy(BaseStrategy):
    """
    Стратегия mean-reversion на основе RSI.
    Long: когда RSI < oversold_level (например, 30)
    Short: когда RSI > overbought_level (например, 70)
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(
            strategy_id="rsi_mr",
            name="RSI Mean Reversion",
            params=params or {},
        )

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "rsi_period": {"type": "integer", "default": 14, "min": 2, "max": 50},
                "oversold_level": {"type": "number", "default": 30, "min": 10, "max": 50},
                "overbought_level": {"type": "number", "default": 70, "min": 50, "max": 90},
                "cooldown_period": {"type": "integer", "default": 3, "min": 1, "max": 20},
                "position_size_pct": {"type": "number", "default": 5.0, "min": 0.5, "max": 20},
                "use_ml_confidence": {"type": "boolean", "default": True},
            },
        }

    def generate_signal(self, data: dict[str, Any]) -> Optional[StrategySignal]:
        rsi = data.get("rsi_14")
        close = data.get("close")
        ema20 = data.get("ema_20")
        symbol = data.get("symbol", "")
        timeframe = data.get("timeframe", "1h")

        if rsi is None or close is None:
            return None

        rsi_period = self.params.get("rsi_period", 14)
        oversold = self.params.get("oversold_level", 30)
        overbought = self.params.get("overbought_level", 70)
        cooldown = self.params.get("cooldown_period", 3)
        size_pct = self.params.get("position_size_pct", 5.0)
        use_ml = self.params.get("use_ml_confidence", True)

        # Проверка cooldown (можно добавить проверку последней сделки по символу)
        signal = None

        if rsi < oversold and close > ema20:
            # Long: RSI oversold + цена выше EMA20 (подтверждение тренда)
            confidence = min(1.0, (oversold - rsi) / oversold)
            if use_ml:
                confidence *= data.get("ml_confidence_long", 1.0)

            sl = close * (1 - self.params.get("sl_pct", 0.02))
            tp = close * (1 + self.params.get("tp_pct", 0.04))

            signal = StrategySignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="long",
                confidence=confidence,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                position_size_pct=size_pct,
                timeframe=timeframe,
                rationale=f"RSI={rsi:.2f} < {oversold} + цена выше EMA20",
            )

        elif rsi > overbought and close < ema20:
            # Short: RSI overbought + цена ниже EMA20
            confidence = min(1.0, (rsi - overbought) / (100 - overbought))
            if use_ml:
                confidence *= data.get("ml_confidence_short", 1.0)

            sl = close * (1 + self.params.get("sl_pct", 0.02))
            tp = close * (1 - self.params.get("tp_pct", 0.04))

            signal = StrategySignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="short",
                confidence=confidence,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                position_size_pct=size_pct,
                timeframe=timeframe,
                rationale=f"RSI={rsi:.2f} > {overbought} + цена ниже EMA20",
            )

        if signal:
            logger.info(
                f"[RSI] {signal.side.upper()} {signal.symbol} | "
                f"conf={signal.confidence:.2f} | SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} | "
                f"{signal.rationale}"
            )

        return signal


# === EMA Crossover Strategy ===

class EMACrossoverStrategy(BaseStrategy):
    """
    Стратегия на основе пересечения EMA.
    Long: когда быстрая EMA пересекает медленную снизу вверх
    Short: когда быстрая EMA пересекает медленную сверху вниз
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(
            strategy_id="ema_cross",
            name="EMA Crossover",
            params=params or {},
        )

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "fast_ema_period": {"type": "integer", "default": 9, "min": 2, "max": 50},
                "slow_ema_period": {"type": "integer", "default": 21, "min": 5, "max": 100},
                "confirm_hourly_ema": {"type": "boolean", "default": True},
                "position_size_pct": {"type": "number", "default": 5.0},
                "sl_pct": {"type": "number", "default": 0.02},
                "tp_pct": {"type": "number", "default": 0.04},
            },
        }

    def generate_signal(self, data: dict[str, Any]) -> Optional[StrategySignal]:
        ema_fast = data.get(f"ema_{self.params.get('fast_ema_period', 9)}")
        ema_slow = data.get(f"ema_{self.params.get('slow_ema_period', 21)}")
        close = data.get("close")
        prev_ema_fast = data.get(f"ema_{self.params.get('fast_ema_period', 9)}_lag_1")
        prev_ema_slow = data.get(f"ema_{self.params.get('slow_ema_period', 21)}_lag_1")

        symbol = data.get("symbol", "")
        timeframe = data.get("timeframe", "1h")

        if ema_fast is None or ema_slow is None or close is None:
            return None

        if prev_ema_fast is None or prev_ema_slow is None:
            return None

        fast_period = self.params.get("fast_ema_period", 9)
        slow_period = self.params.get("slow_ema_period", 21)

        # Пересечение: быстрая выше медленной, но в предыдущей свече — ниже
        if ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow:
            # Long crossover
            sl = close * (1 - self.params.get("sl_pct", 0.02))
            tp = close * (1 + self.params.get("tp_pct", 0.04))

            signal = StrategySignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="long",
                confidence=0.7,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                position_size_pct=self.params.get("position_size_pct", 5.0),
                timeframe=timeframe,
                rationale=f"EMA{fast_period} ({ema_fast:.2f}) пересекла EMA{slow_period} ({ema_slow:.2f}) вверх",
            )

            logger.info(
                f"[EMA] LONG {signal.symbol} | "
                f"EMA{fast_period}={ema_fast:.2f} > EMA{slow_period}={ema_slow:.2f} | "
                f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f}"
            )

            return signal

        elif ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow:
            # Short crossover
            sl = close * (1 + self.params.get("sl_pct", 0.02))
            tp = close * (1 - self.params.get("tp_pct", 0.04))

            signal = StrategySignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="short",
                confidence=0.7,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                position_size_pct=self.params.get("position_size_pct", 5.0),
                timeframe=timeframe,
                rationale=f"EMA{fast_period} ({ema_fast:.2f}) пересекла EMA{slow_period} ({ema_slow:.2f}) вниз",
            )

            logger.info(
                f"[EMA] SHORT {signal.symbol} | "
                f"EMA{fast_period}={ema_fast:.2f} < EMA{slow_period}={ema_slow:.2f} | "
                f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f}"
            )

            return signal

        return None


# === Bollinger Bands Strategy ===

class BollingerBandsStrategy(BaseStrategy):
    """
    Стратегия на основе Bollinger Bands.
    Mean-reversion: покупаем когда цена касается нижней полосы, продаем при касании верхней.
    Breakout: покупаем при пробое верхней полосы (тренд).
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(
            strategy_id="bb_strategy",
            name="Bollinger Bands",
            params=params or {},
        )

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "bb_length": {"type": "integer", "default": 20, "min": 5, "max": 50},
                "bb_std": {"type": "number", "default": 2.0, "min": 1.0, "max": 4.0},
                "mode": {"type": "string", "enum": ["mean_reversion", "breakout"], "default": "mean_reversion"},
                "position_size_pct": {"type": "number", "default": 5.0},
                "sl_pct": {"type": "number", "default": 0.02},
                "tp_pct": {"type": "number", "default": 0.04},
            },
        }

    def generate_signal(self, data: dict[str, Any]) -> Optional[StrategySignal]:
        bb_upper = data.get("bb_upper")
        bb_lower = data.get("bb_lower")
        bb_mid = data.get("bb_mid")
        close = data.get("close")
        bb_pct = data.get("bb_pct")
        volume_ratio = data.get("volume_ratio", 1.0)

        symbol = data.get("symbol", "")
        timeframe = data.get("timeframe", "1h")

        if bb_upper is None or bb_lower is None or close is None or bb_pct is None:
            return None

        sl_pct = self.params.get("sl_pct", 0.02)
        tp_pct = self.params.get("tp_pct", 0.04)
        size_pct = self.params.get("position_size_pct", 5.0)
        mode = self.params.get("mode", "mean_reversion")

        signal = None

        if mode == "mean_reversion":
            min_margin = min_profitable_margin_pct()

            # Цена у показателя нижней полосы + объём выше нормы
            if close <= bb_lower * 1.005 and volume_ratio > 1.2:
                confidence = max(0.5, 1.0 - bb_pct / 0.5)
                sl = close * (1 - sl_pct)
                # Цель — средняя полоса, но не ближе порога окупаемости
                # комиссий+слиппеджа: при узких полосах bb_mid мог быть
                # в паре десятых процента от входа — номинальный take_profit
                # после округления фактически всегда давал чистый убыток.
                tp = max(bb_mid, close * (1 + min_margin))

                signal = StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side="long",
                    confidence=confidence,
                    entry_price=close,
                    stop_loss=sl,
                    take_profit=tp,
                    position_size_pct=size_pct,
                    timeframe=timeframe,
                    rationale=f"Цена у нижней BB (bb_pct={bb_pct:.2f}) + объём в 1.2x",
                )

            elif close >= bb_upper * 0.995 and volume_ratio > 1.2:
                confidence = max(0.5, bb_pct / 0.5)
                sl = close * (1 + sl_pct)
                tp = min(bb_mid, close * (1 - min_margin))

                signal = StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side="short",
                    confidence=confidence,
                    entry_price=close,
                    stop_loss=sl,
                    take_profit=tp,
                    position_size_pct=size_pct,
                    timeframe=timeframe,
                    rationale=f"Цена у верхней BB (bb_pct={bb_pct:.2f}) + объём в 1.2x",
                )

        elif mode == "breakout":
            # Пробой верхней полосы с высоким объёмом
            if close > bb_upper and volume_ratio > 1.5:
                signal = StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side="long",
                    confidence=0.8,
                    entry_price=close,
                    stop_loss=close * (1 - sl_pct * 0.5),
                    take_profit=close * (1 + tp_pct * 1.5),
                    position_size_pct=size_pct,
                    timeframe=timeframe,
                    rationale=f"Пробой верхней BB (close={close:.2f}, bb_upper={bb_upper:.2f}) + объём {volume_ratio:.1f}x",
                )

            elif close < bb_lower and volume_ratio > 1.5:
                signal = StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side="short",
                    confidence=0.8,
                    entry_price=close,
                    stop_loss=close * (1 + sl_pct * 0.5),
                    take_profit=close * (1 - tp_pct * 1.5),
                    position_size_pct=size_pct,
                    timeframe=timeframe,
                    rationale=f"Пробой нижней BB (close={close:.2f}, bb_lower={bb_lower:.2f}) + объём {volume_ratio:.1f}x",
                )

        if signal:
            logger.info(
                f"[BB] {signal.side.upper()} {signal.symbol} | "
                f"conf={signal.confidence:.2f} | SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} | "
                f"{signal.rationale}"
            )

        return signal


# === Funding Rate Strategy ===

class FundingRateStrategy(BaseStrategy):
    """
    Стратегия на основе фандинг-рейта.
    Если фандинг высокий (> threshold) — короткие позиции переоценены → long bias
    Если фандинг отрицательный — long позиции переоценены → short bias
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(
            strategy_id="funding_rate",
            name="Funding Rate",
            params=params or {},
        )

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "high_funding_threshold": {"type": "number", "default": 0.01},
                "low_funding_threshold": {"type": "number", "default": -0.01},
                "position_size_pct": {"type": "number", "default": 5.0},
                "sl_pct": {"type": "number", "default": 0.02},
                "tp_pct": {"type": "number", "default": 0.04},
                "require_oi_confirmation": {"type": "boolean", "default": True},
            },
        }

    def generate_signal(self, data: dict[str, Any]) -> Optional[StrategySignal]:
        funding_rate = data.get("funding_rate")
        current_funding = data.get("current_funding_rate")
        oi_change = data.get("oi_change_1h")
        close = data.get("close")

        symbol = data.get("symbol", "")
        timeframe = data.get("timeframe", "1h")

        if funding_rate is None or close is None:
            return None

        high_threshold = self.params.get("high_funding_threshold", 0.01)
        low_threshold = self.params.get("low_funding_threshold", -0.01)
        sl_pct = self.params.get("sl_pct", 0.02)
        tp_pct = self.params.get("tp_pct", 0.04)
        size_pct = self.params.get("position_size_pct", 5.0)
        require_oi = self.params.get("require_oi_confirmation", True)

        signal = None

        # Высокий фандинг → short переоценены → long bias
        if current_funding is not None and current_funding > high_threshold:
            if require_oi and oi_change is not None and oi_change < 0:
                # OI падает → short liquidation, можно идти long
                confidence = min(0.9, (current_funding - high_threshold) * 10 + 0.5)
                sl = close * (1 - sl_pct)
                tp = close * (1 + tp_pct * 1.2)

                signal = StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side="long",
                    confidence=confidence,
                    entry_price=close,
                    stop_loss=sl,
                    take_profit=tp,
                    position_size_pct=size_pct,
                    timeframe=timeframe,
                    rationale=f"Funding rate={current_funding:.4f} > {high_threshold} + OI падает ({oi_change:.1f}%)",
                )

        # Отрицательный фандинг → long переоценены → short bias
        elif current_funding is not None and current_funding < low_threshold:
            if require_oi and oi_change is not None and oi_change > 0:
                confidence = min(0.9, (low_threshold - current_funding) * 10 + 0.5)
                sl = close * (1 + sl_pct)
                tp = close * (1 - tp_pct * 1.2)

                signal = StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side="short",
                    confidence=confidence,
                    entry_price=close,
                    stop_loss=sl,
                    take_profit=tp,
                    position_size_pct=size_pct,
                    timeframe=timeframe,
                    rationale=f"Funding rate={current_funding:.4f} < {low_threshold} + OI растёт ({oi_change:.1f}%)",
                )

        if signal:
            logger.info(
                f"[FUNDING] {signal.side.upper()} {signal.symbol} | "
                f"conf={signal.confidence:.2f} | funding={current_funding:.4f} | "
                f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} | {signal.rationale}"
            )

        return signal


# === Liquidation Strategy ===

class LiquidationStrategy(BaseStrategy):
    """
    Стратегия на основе уровней ликвидаций (из CoinGlass).
    Если цена близка к крупному кластеру ликвидаций и объём показывает напряжение —
    можно торговать в направлении, противоположном доминантной стороне.
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(
            strategy_id="liquidation",
            name="Liquidation Zones",
            params=params or {},
        )

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "distance_threshold_pct": {"type": "number", "default": 2.0},
                "side_bias_threshold": {"type": "number", "default": 0.6},
                "position_size_pct": {"type": "number", "default": 5.0},
                "sl_pct": {"type": "number", "default": 0.02},
                "tp_pct": {"type": "number", "default": 0.04},
            },
        }

    def generate_signal(self, data: dict[str, Any]) -> Optional[StrategySignal]:
        liquidation_distance = data.get("liquidation_distance_pct")
        long_short_ratio = data.get("long_short_ratio")
        close = data.get("close")
        liquidation_imbalance = data.get("liquidation_imbalance", 0)

        symbol = data.get("symbol", "")
        timeframe = data.get("timeframe", "1h")

        if liquidation_distance is None or close is None:
            return None

        distance_threshold = self.params.get("distance_threshold_pct", 2.0)
        side_bias_threshold = self.params.get("side_bias_threshold", 0.6)
        sl_pct = self.params.get("sl_pct", 0.02)
        tp_pct = self.params.get("tp_pct", 0.04)
        size_pct = self.params.get("position_size_pct", 5.0)

        signal = None

        # Если близко к уровню ликвидаций и соотношение Long/Short показывает доминирование
        if liquidation_distance < distance_threshold:
            if long_short_ratio is not None:
                if long_short_ratio > side_bias_threshold:
                    # Большинство long → потенциальный short squeeze при ликвидации
                    confidence = max(0.5, (side_bias_threshold - (1 - long_short_ratio)) * 2)
                    sl = close * (1 + sl_pct)
                    tp = close * (1 - tp_pct * 0.5)  # короткий таймфрейм

                    signal = StrategySignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side="short",
                        confidence=confidence,
                        entry_price=close,
                        stop_loss=sl,
                        take_profit=tp,
                        position_size_pct=size_pct,
                        timeframe=timeframe,
                        rationale=f"Близость к зоне ликвидаций ({liquidation_distance:.1f}%) + Long/Short={long_short_ratio:.2f}",
                    )

                elif long_short_ratio < (1 - side_bias_threshold):
                    # Большинство short → потенциальный long squeeze
                    confidence = max(0.5, (side_bias_threshold - long_short_ratio) * 2)
                    sl = close * (1 - sl_pct)
                    tp = close * (1 + tp_pct * 0.5)

                    signal = StrategySignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side="long",
                        confidence=confidence,
                        entry_price=close,
                        stop_loss=sl,
                        take_profit=tp,
                        position_size_pct=size_pct,
                        timeframe=timeframe,
                        rationale=f"Близость к зоне ликвидаций ({liquidation_distance:.1f}%) + Short доминирует (LSR={long_short_ratio:.2f})",
                    )

        if signal:
            logger.info(
                f"[LIQ] {signal.side.upper()} {signal.symbol} | "
                f"conf={signal.confidence:.2f} | dist={liquidation_distance:.1f}% | "
                f"LSR={long_short_ratio:.2f} | SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f}"
            )

        return signal


# === ML Direction Classifier Strategy ===

class MLDirectionClassifierStrategy(BaseStrategy):
    """
    Стратегия, использующая ML модель для предсказания направления.
    Предсказывает P(up), P(down), P(neutral) и генерирует сигнал при достаточной уверенности.
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(
            strategy_id="ml_classifier",
            name="ML Direction Classifier",
            params=params or {},
        )
        self.model_version = self.params.get("model_version", 1)
        self.confidence_threshold = self.params.get("confidence_threshold", 0.65)

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "model_version": {"type": "integer", "default": 1},
                "confidence_threshold": {"type": "number", "default": 0.65, "min": 0.5, "max": 0.95},
                "position_size_pct": {"type": "number", "default": 5.0},
                "horizon": {"type": "integer", "default": 5, "min": 1, "max": 20},
                "sl_pct": {"type": "number", "default": 0.02},
                "tp_multiplier": {"type": "number", "default": 2.0},
                "require_indicators_confirm": {"type": "boolean", "default": True},
            },
        }

    def generate_signal(self, data: dict[str, Any]) -> Optional[StrategySignal]:
        ml_proba_up = data.get("ml_proba_up")
        ml_proba_down = data.get("ml_proba_down")
        ml_proba_neutral = data.get("ml_proba_neutral")
        close = data.get("close")
        ml_features_used = data.get("ml_features_used", "")

        symbol = data.get("symbol", "")
        timeframe = data.get("timeframe", "1h")

        if ml_proba_up is None or ml_proba_down is None or close is None:
            return None

        confidence_threshold = self.params.get("confidence_threshold", 0.65)
        sl_pct = self.params.get("sl_pct", 0.02)
        tp_mult = self.params.get("tp_multiplier", 2.0)
        size_pct = self.params.get("position_size_pct", 5.0)
        require_indicators = self.params.get("require_indicators_confirm", True)

        signal = None

        # Long: высокая вероятность роста
        if ml_proba_up > confidence_threshold and ml_proba_up > ml_proba_down:
            if require_indicators:
                # Дополнительная фильтрация индикаторами
                rsi = data.get("rsi_14", 50)
                ema20 = data.get("ema_20")
                ema50 = data.get("ema_50")

                # Не входим в long если RSI > 70 (перекупленность) или цена далеко от EMA
                if rsi > 75 or (ema50 and close < ema50 * 0.99):
                    logger.debug(f"[ML] Long отклонён индикаторами: RSI={rsi:.1f}, close={close:.2f}, ema50={ema50:.2f}")
                    return None

            confidence = ml_proba_up
            sl = close * (1 - sl_pct)
            tp = close * (1 + sl_pct * tp_mult)

            signal = StrategySignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="long",
                confidence=confidence,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                position_size_pct=size_pct,
                timeframe=timeframe,
                rationale=f"ML P(up)={ml_proba_up:.2f} > {confidence_threshold} | фичи: {ml_features_used[:80]}",
            )

        # Short: высокая вероятность падения
        elif ml_proba_down > confidence_threshold and ml_proba_down > ml_proba_up:
            if require_indicators:
                rsi = data.get("rsi_14", 50)
                ema20 = data.get("ema_20")
                ema50 = data.get("ema_50")

                if rsi < 25 or (ema50 and close > ema50 * 1.01):
                    logger.debug(f"[ML] Short отклонён индикаторами: RSI={rsi:.1f}, close={close:.2f}, ema50={ema50:.2f}")
                    return None

            confidence = ml_proba_down
            sl = close * (1 + sl_pct)
            tp = close * (1 - sl_pct * tp_mult)

            signal = StrategySignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="short",
                confidence=confidence,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                position_size_pct=size_pct,
                timeframe=timeframe,
                rationale=f"ML P(down)={ml_proba_down:.2f} > {confidence_threshold} | фичи: {ml_features_used[:80]}",
            )

        if signal:
            logger.info(
                f"[ML] {signal.side.upper()} {signal.symbol} | "
                f"conf={signal.confidence:.2f} | P(up)={ml_proba_up:.2f} P(down)={ml_proba_down:.2f} | "
                f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} | {signal.rationale}"
            )

        return signal


# === Ensemble Voter Strategy ===

class EnsembleVoterStrategy(BaseStrategy):
    """
    Мета-стратегия: агрегирует сигналы от других стратегий с весами.
    Веса могут быть от ML (ensemble weight optimizer).
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(
            strategy_id="ensemble_voter",
            name="Ensemble Voter",
            params=params or {},
        )
        self.strategy_weights: dict[str, float] = {}
        self.min_confidence = self.params.get("min_confidence", 0.6)
        self.min_strategies_count = self.params.get("min_strategies_count", 2)

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "strategy_weights": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Веса стратегий {strategy_id: weight}",
                },
                "min_confidence": {"type": "number", "default": 0.6, "min": 0.5, "max": 0.9},
                "min_strategies_count": {"type": "integer", "default": 2, "min": 1, "max": 10},
                "vote_method": {"type": "string", "enum": ["weighted", "majority", "confidence_sum"], "default": "weighted"},
            },
        }

    def set_strategy_weight(self, strategy_id: str, weight: float):
        """Установить вес стратегии."""
        self.strategy_weights[strategy_id] = weight

    def generate_signal(self, data: dict[str, Any]) -> Optional[StrategySignal]:
        """
        В Ensemble VoterStrategy сигнал генерируется не напрямую,
        а через агрегацию сигналов от strategy_engine.
        Этот метод вызывается для совместимости, но реальная логика — в engine.
        """
        # В реальном коде сигнал приходит от engine, который собрал сигналы от всех стратегий
        # Здесь просто возвращаем None, так как это мета-стратегия
        return None

    def aggregate_signals(
        self,
        signals: list[StrategySignal],
    ) -> Optional[StrategySignal]:
        """
        Агрегировать сигналы от разных стратегий в один.
        Возвращает объединённый сигнал или None.
        """
        if not signals:
            return None

        # Считаем взвешенные голоса
        long_votes = 0.0
        short_votes = 0.0
        total_weight = 0.0
        best_signal = None
        best_confidence = 0.0

        for signal in signals:
            weight = self.strategy_weights.get(signal.strategy_id, 1.0)
            total_weight += weight

            if signal.side == "long":
                long_votes += weight * signal.confidence
            elif signal.side == "short":
                short_votes += weight * signal.confidence

            if signal.confidence > best_confidence:
                best_confidence = signal.confidence
                best_signal = signal

        # Если есть однозначный победитель
        if long_votes > short_votes * 1.5 and long_votes > 0.5:
            # Преобладают long
            avg_confidence = long_votes / (total_weight + 1e-10)
            if avg_confidence >= self.min_confidence:
                signal = StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=best_signal.symbol if best_signal else "",
                    side="long",
                    confidence=avg_confidence,
                    entry_price=best_signal.entry_price if best_signal else None,
                    stop_loss=best_signal.stop_loss if best_signal else None,
                    take_profit=best_signal.take_profit if best_signal else None,
                    position_size_pct=best_signal.position_size_pct if best_signal else 5.0,
                    timeframe=best_signal.timeframe if best_signal else "1h",
                    rationale=f"Ensemble: {len(signals)} стратегий, long_votes={long_votes:.2f} > short_votes={short_votes:.2f}",
                )
                logger.info(
                    f"[ENSEMBLE] LONG {signal.symbol} | "
                    f"conf={signal.confidence:.2f} | long={long_votes:.2f} short={short_votes:.2f} | "
                    f"{signal.rationale}"
                )
                return signal

        elif short_votes > long_votes * 1.5 and short_votes > 0.5:
            # Преобладают short
            avg_confidence = short_votes / (total_weight + 1e-10)
            if avg_confidence >= self.min_confidence:
                signal = StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=best_signal.symbol if best_signal else "",
                    side="short",
                    confidence=avg_confidence,
                    entry_price=best_signal.entry_price if best_signal else None,
                    stop_loss=best_signal.stop_loss if best_signal else None,
                    take_profit=best_signal.take_profit if best_signal else None,
                    position_size_pct=best_signal.position_size_pct if best_signal else 5.0,
                    timeframe=best_signal.timeframe if best_signal else "1h",
                    rationale=f"Ensemble: {len(signals)} стратегий, short_votes={short_votes:.2f} > long_votes={long_votes:.2f}",
                )
                logger.info(
                    f"[ENSEMBLE] SHORT {signal.symbol} | "
                    f"conf={signal.confidence:.2f} | short={short_votes:.2f} long={long_votes:.2f} | "
                    f"{signal.rationale}"
                )
                return signal

        return None


# === Реестр стратегий ===

class StrategyRegistry:
    """Реестр всех доступных стратегий."""

    def __init__(self):
        self.strategies: dict[str, BaseStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Зарегистрировать стандартные стратегии."""
        self.register(RSIMeanReversionStrategy())
        self.register(EMACrossoverStrategy())
        self.register(BollingerBandsStrategy())
        self.register(FundingRateStrategy())
        self.register(LiquidationStrategy())
        self.register(MLDirectionClassifierStrategy())
        self.register(EnsembleVoterStrategy())

    def register(self, strategy: BaseStrategy):
        """Зарегистрировать стратегию."""
        self.strategies[strategy.strategy_id] = strategy
        logger.info(f"Стратегия зарегистрирована: {strategy.name} ({strategy.strategy_id})")

    def get(self, strategy_id: str) -> Optional[BaseStrategy]:
        """Получить стратегию по ID."""
        return self.strategies.get(strategy_id)

    def get_all(self) -> list[BaseStrategy]:
        """Получить все зарегистрированные стратегии."""
        return list(self.strategies.values())

    def get_active(self) -> list[BaseStrategy]:
        """Получить активные стратегии."""
        return [s for s in self.strategies.values() if s.active]

    def list_strategies(self) -> list[dict]:
        """Список стратегий для API."""
        return [
            {
                "id": s.strategy_id,
                "name": s.name,
                "type": "rule" if not isinstance(s, MLDirectionClassifierStrategy) else "ml",
                "active": s.active,
                "weight": s.weight,
                "params": s.params,
            }
            for s in self.strategies.values()
        ]


# Глобальный реестр
strategy_registry = StrategyRegistry()
