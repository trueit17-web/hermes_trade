"""Тесты для крипто-трейдер бота."""
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, PropertyMock

import numpy as np
import pandas as pd

from src.config import settings
from src.event_bus import (
    event_bus,
    Event,
    MarketDataEvent,
    SignalGeneratedEvent,
    TradeEvent,
)
from src.risk.risk_manager import RiskProfile, RiskState, RiskManager
from src.execution.executor import ExecutionEngine
from src.strategy import (
    RSIMeanReversionStrategy,
    EMACrossoverStrategy,
    BollingerBandsStrategy,
    FundingRateStrategy,
    LiquidationStrategy,
    MLDirectionClassifierStrategy,
    EnsembleVoterStrategy,
)
from src.utils.logging import setup_logging


class TestConfig(unittest.TestCase):
    """Тесты для конфигурации."""

    def setUp(self):
        self.original_trading_mode = settings.trading_mode
        self.original_mode = settings.is_paper

    def tearDown(self):
        settings.trading_mode = self.original_trading_mode
        settings.is_paper = self.original_mode

    def test_paper_mode_detection(self):
        """Paper режим корректно определяется."""
        settings.trading_mode = "paper"
        self.assertTrue(settings.is_paper)
        self.assertFalse(settings.is_real)

    def test_real_mode_detection(self):
        """Real режим корректно определяется."""
        settings.trading_mode = "real"
        self.assertFalse(settings.is_paper)
        self.assertTrue(settings.is_real)

    def test_default_capital(self):
        """Стартовый капитал по умолчанию."""
        self.assertEqual(settings.startup_capitol_usdt, 10000.0)

    def test_risk_defaults(self):
        """Риск-параметры по умолчанию."""
        self.assertEqual(settings.risk_daily_loss_limit_usd, 500.0)
        self.assertEqual(settings.risk_max_open_positions, 8)
        self.assertEqual(settings.risk_max_position_size_pct, 10.0)
        self.assertEqual(settings.risk_cooldown_seconds, 300)


class TestEventBus(unittest.IsolatedAsyncioTestCase):
    """Тесты для event bus."""

    async def asyncSetUp(self):
        event_bus.clear_history()

    async def asyncTearDown(self):
        event_bus.clear_history()

    async def test_publish_and_subscribe(self):
        """Публикация и подписка на события."""
        received = []

        async def callback(event):
            received.append(event)

        event_bus.subscribe("test_event", callback)
        event = Event(type="test_event", source="test", payload={"data": "test"})
        await event_bus.publish(event)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].type, "test_event")

    async def test_wildcard_subscription(self):
        """Wildcard подписка получает все события."""
        received = []

        async def callback(event):
            received.append(event)

        event_bus.subscribe_all(callback)

        for i in range(3):
            event = Event(type=f"event_{i}", source="test")
            await event_bus.publish(event)

        self.assertEqual(len(received), 3)

    async def test_unsubscribe(self):
        """Отписка от событий."""
        received = []

        async def callback(event):
            received.append(event)

        event_bus.subscribe("test", callback)
        event_bus.unsubscribe("test", callback)

        event = Event(type="test", source="test")
        await event_bus.publish(event)

        self.assertEqual(len(received), 0)

    async def test_history(self):
        """История событий."""
        for i in range(5):
            event = Event(type=f"event_{i}", source="test")
            await event_bus.publish(event)

        history = event_bus.get_history()
        self.assertEqual(len(history), 5)

        filtered = event_bus.get_history("event_2")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].type, "event_2")


class TestRiskProfile(unittest.TestCase):
    """Тесты для RiskProfile."""

    def test_default_values(self):
        """Значения по умолчанию."""
        profile = RiskProfile()
        self.assertEqual(profile.daily_loss_limit_usd, 500.0)
        self.assertEqual(profile.max_open_positions, 8)
        self.assertEqual(profile.max_position_size_pct, 10.0)
        self.assertEqual(profile.max_drawdown_pct, 15.0)
        self.assertEqual(profile.cooldown_seconds, 300)

    def test_update_params(self):
        """Обновление параметров."""
        profile = RiskProfile()
        profile.update({
            "daily_loss_limit_usd": 1000.0,
            "max_open_positions": 10,
            "max_position_size_pct": 5.0,
        })
        self.assertEqual(profile.daily_loss_limit_usd, 1000.0)
        self.assertEqual(profile.max_open_positions, 10)
        self.assertEqual(profile.max_position_size_pct, 5.0)


class TestRiskState(unittest.TestCase):
    """Тесты для RiskState."""

    def test_initial_state(self):
        """Начальное состояние."""
        state = RiskState()
        self.assertEqual(state.current_balance, 0.0)
        self.assertEqual(state.start_balance, 0.0)
        self.assertEqual(state.open_positions_count, 0)
        self.assertEqual(state.daily_loss_limit_reached, False)
        self.assertEqual(state.paused, False)
        self.assertEqual(state.kill_switch_active, False)
        self.assertEqual(state.cooldown_active, False)

    def test_daily_loss_limit_reached(self):
        """Достижение дневного лимита убытков."""
        state = RiskState()
        state.daily_loss_limit_usd = 500.0
        state.update_daily_pnl(-500.0)
        self.assertTrue(state.daily_loss_limit_reached)
        self.assertEqual(state.daily_pnl, -500.0)

    def test_add_open_position(self):
        """Добавление открытой позиции."""
        state = RiskState()
        state.add_open_position("BTC/USDT", 10.0)
        self.assertEqual(state.open_positions_count, 1)
        self.assertEqual(state.open_positions["BTC/USDT"], 10.0)

    def test_max_positions_check(self):
        """Проверка максимального количества позиций."""
        state = RiskState()
        state.max_open_positions = 3
        state.add_open_position("BTC/USDT", 5.0)
        state.add_open_position("ETH/USDT", 5.0)
        self.assertFalse(state.check_max_positions())

        state.add_open_position("SOL/USDT", 5.0)
        self.assertTrue(state.check_max_positions())

    def test_kill_switch(self):
        """Kill switch."""
        state = RiskState()
        state.trigger_kill_switch()
        self.assertTrue(state.kill_switch_active)
        self.assertTrue(state.paused)

        state.clear_kill_switch()
        self.assertFalse(state.kill_switch_active)
        self.assertFalse(state.paused)


class TestRiskManager(unittest.IsolatedAsyncioTestCase):
    """Тесты для RiskManager."""

    async def asyncSetUp(self):
        self.risk = RiskManager()
        self.risk.state.kill_switch_active = False
        self.risk.state.paused = False
        self.risk.state.daily_loss_limit_reached = False
        self.risk.state.cooldown_active = False
        self.risk.state.open_positions.clear()
        self.risk.state.open_positions_count = 0

    async def test_can_trade_initial(self):
        """Можно торговать в начальном состоянии."""
        self.assertTrue(await self.risk.can_trade())

    async def test_cannot_trade_kill_switch(self):
        """Нельзя торговать при kill switch."""
        self.risk.state.trigger_kill_switch()
        self.assertFalse(await self.risk.can_trade())

    async def test_cannot_trade_paused(self):
        """Нельзя торговать при паузе."""
        self.risk.state.paused = True
        self.assertFalse(await self.risk.can_trade())

    async def test_cannot_trade_daily_loss_reached(self):
        """Нельзя торговать при достижении daily loss."""
        self.risk.state.daily_loss_limit_reached = True
        self.assertFalse(await self.risk.can_trade())

    async def test_check_signal_ok(self):
        """Сигнал проходит проверку."""
        signal = MagicMock()
        signal.symbol = "BTC/USDT"
        signal.position_size_pct = 5.0

        can_execute, reason = self.risk.check_signal(signal)
        self.assertTrue(can_execute)
        self.assertEqual(reason, "OK")

    async def test_check_signal_size_exceeded(self):
        """Размер позиции превышает лимит."""
        signal = MagicMock()
        signal.symbol = "BTC/USDT"
        signal.position_size_pct = 15.0

        can_execute, reason = self.risk.check_signal(signal)
        self.assertFalse(can_execute)
        self.assertIn("превышает лимит", reason)

    async def test_check_signal_duplicate_symbol(self):
        """Дубликат символа."""
        self.risk.state.open_positions["BTC/USDT"] = 5.0
        self.risk.state.open_positions_count = 1

        signal = MagicMock()
        signal.symbol = "BTC/USDT"
        signal.position_size_pct = 5.0

        can_execute, reason = self.risk.check_signal(signal)
        self.assertFalse(can_execute)
        self.assertIn("Уже есть позиция", reason)

    async def test_adjust_position_size(self):
        """Расчёт размера позиции."""
        signal = MagicMock()
        signal.position_size_pct = 10.0

        current_balance = 10000.0
        size = self.risk.adjust_position_size(signal, current_balance)
        expected = current_balance * (10.0 / 100)
        self.assertEqual(size, expected)

    async def test_adjust_position_size_with_volatility(self):
        """Расчёт размера с учётом волатильности."""
        signal = MagicMock()
        signal.position_size_pct = 10.0

        current_balance = 10000.0
        size = self.risk.adjust_position_size(signal, current_balance, volatility_adj=0.5)
        expected = current_balance * (10.0 / 100) * 0.5
        self.assertEqual(size, expected)

    async def test_on_trade_closed(self):
        """Обработка закрытия сделки."""
        self.risk.on_trade_closed(100.0)
        self.assertEqual(self.risk.state.daily_pnl, 100.0)
        self.assertFalse(self.risk.state.daily_loss_limit_reached)

        self.risk.on_trade_closed(-600.0)
        self.assertTrue(self.risk.state.daily_loss_limit_reached)

    async def test_on_position_added_and_removed(self):
        """Добавление и удаление позиции."""
        self.risk.on_position_added("BTC/USDT", 10.0)
        self.assertEqual(self.risk.state.open_positions_count, 1)

        self.risk.on_position_closed("BTC/USDT")
        self.assertEqual(self.risk.state.open_positions_count, 0)


class TestRSIMeanReversionStrategy(unittest.TestCase):
    """Тесты для RSI стратегии."""

    def test_long_signal(self):
        """Long сигнал при oversold RSI."""
        strategy = RSIMeanReversionStrategy()
        strategy.params = {
            "rsi_period": 14,
            "oversold_level": 30,
            "overbought_level": 70,
            "position_size_pct": 5.0,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }

        data = {
            "rsi_14": 25.0,
            "close": 50000.0,
            "ema_20": 49000.0,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "ml_confidence_long": 1.0,
        }

        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "long")
        self.assertEqual(signal.symbol, "BTC/USDT")
        self.assertEqual(signal.entry_price, 50000.0)
        self.assertGreater(signal.confidence, 0.5)

    def test_short_signal(self):
        """Short сигнал при overbought RSI."""
        strategy = RSIMeanReversionStrategy()
        strategy.params = {
            "rsi_period": 14,
            "oversold_level": 30,
            "overbought_level": 70,
            "position_size_pct": 5.0,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }

        data = {
            "rsi_14": 75.0,
            "close": 50000.0,
            "ema_20": 51000.0,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "ml_confidence_short": 1.0,
        }

        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "short")
        self.assertEqual(signal.symbol, "BTC/USDT")
        self.assertEqual(signal.entry_price, 50000.0)

    def test_no_signal_neutral(self):
        """Нет сигнала в нейтральном состоянии."""
        strategy = RSIMeanReversionStrategy()
        strategy.params = {"position_size_pct": 5.0}

        data = {
            "rsi_14": 50.0,
            "close": 50000.0,
            "ema_20": 50000.0,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }

        signal = strategy.generate_signal(data)
        self.assertIsNone(signal)


class TestEMACrossoverStrategy(unittest.TestCase):
    """Тесты для EMA Crossover стратегии."""

    def test_long_crossover(self):
        """Long сигнал при пересечении EMA вверх."""
        strategy = EMACrossoverStrategy()
        strategy.params = {
            "fast_ema_period": 9,
            "slow_ema_period": 21,
            "position_size_pct": 5.0,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }

        data = {
            "ema_9": 50000.0,
            "ema_21": 49000.0,
            "ema_9_lag_1": 48000.0,
            "ema_21_lag_1": 49000.0,
            "close": 50000.0,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }

        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "long")

    def test_short_crossover(self):
        """Short сигнал при пересечении EMA вниз."""
        strategy = EMACrossoverStrategy()
        strategy.params = {
            "fast_ema_period": 9,
            "slow_ema_period": 21,
            "position_size_pct": 5.0,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }

        data = {
            "ema_9": 48000.0,
            "ema_21": 49000.0,
            "ema_9_lag_1": 49500.0,
            "ema_21_lag_1": 49000.0,
            "close": 48000.0,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }

        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "short")

    def test_no_crossover(self):
        """Нет пересечения — нет сигнала."""
        strategy = EMACrossoverStrategy()
        strategy.params = {"position_size_pct": 5.0}

        data = {
            "ema_9": 50000.0,
            "ema_21": 49000.0,
            "ema_9_lag_1": 50000.0,
            "ema_21_lag_1": 49000.0,
            "close": 50000.0,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }

        signal = strategy.generate_signal(data)
        self.assertIsNone(signal)


class TestBollingerBandsStrategy(unittest.TestCase):
    """Тесты для Bollinger Bands стратегии."""

    def test_mean_reversion_long(self):
        """Long сигнал — цена у нижней полосы."""
        strategy = BollingerBandsStrategy()
        strategy.params = {
            "mode": "mean_reversion",
            "position_size_pct": 5.0,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }

        data = {
            "bb_upper": 51000.0,
            "bb_lower": 49000.0,
            "bb_mid": 50000.0,
            "bb_pct": 0.0,
            "volume_ratio": 1.3,
            "close": 49000.0,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }

        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "long")

    def test_breakout_long(self):
        """Long сигнал — пробой верхней полосы."""
        strategy = BollingerBandsStrategy()
        strategy.params = {
            "mode": "breakout",
            "position_size_pct": 5.0,
        }

        data = {
            "bb_upper": 50000.0,
            "bb_lower": 49000.0,
            "bb_mid": 49500.0,
            "bb_pct": 1.0,
            "volume_ratio": 2.0,
            "close": 50500.0,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }

        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "long")

    def test_no_signal_inside_bands(self):
        """Нет сигнала — цена внутри полос."""
        strategy = BollingerBandsStrategy()
        strategy.params = {"position_size_pct": 5.0}

        data = {
            "bb_upper": 51000.0,
            "bb_lower": 49000.0,
            "bb_mid": 50000.0,
            "bb_pct": 0.5,
            "volume_ratio": 1.0,
            "close": 50000.0,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }

        signal = strategy.generate_signal(data)
        self.assertIsNone(signal)


class TestEnsembleVoterStrategy(unittest.TestCase):
    """Тесты для Ensemble Voter стратегии."""

    def test_aggregate_long_dominant(self):
        """Агрегация: преобладают long."""
        strategy = EnsembleVoterStrategy()
        strategy.strategy_weights = {
            "rsi_mr": 1.0,
            "ema_cross": 1.0,
        }

        signals = [
            MagicMock(strategy_id="rsi_mr", side="long", confidence=0.8, symbol="BTC/USDT",
                      entry_price=50000.0, stop_loss=49000.0, take_profit=51000.0,
                      position_size_pct=5.0, timeframe="1h", rationale="test"),
            MagicMock(strategy_id="ema_cross", side="long", confidence=0.7, symbol="BTC/USDT",
                      entry_price=50000.0, stop_loss=49000.0, take_profit=51000.0,
                      position_size_pct=5.0, timeframe="1h", rationale="test"),
        ]

        aggregated = strategy.aggregate_signals(signals)
        self.assertIsNotNone(aggregated)
        self.assertEqual(aggregated.side, "long")

    def test_aggregate_short_dominant(self):
        """Агрегация: преобладают short."""
        strategy = EnsembleVoterStrategy()
        strategy.strategy_weights = {
            "rsi_mr": 1.0,
            "ema_cross": 1.0,
        }

        signals = [
            MagicMock(strategy_id="rsi_mr", side="short", confidence=0.8, symbol="BTC/USDT",
                      entry_price=50000.0, stop_loss=51000.0, take_profit=49000.0,
                      position_size_pct=5.0, timeframe="1h", rationale="test"),
            MagicMock(strategy_id="ema_cross", side="short", confidence=0.7, symbol="BTC/USDT",
                      entry_price=50000.0, stop_loss=51000.0, take_profit=49000.0,
                      position_size_pct=5.0, timeframe="1h", rationale="test"),
        ]

        aggregated = strategy.aggregate_signals(signals)
        self.assertIsNotNone(aggregated)
        self.assertEqual(aggregated.side, "short")

    def test_aggregate_conflicting_no_signal(self):
        """Конфликтующие сигналы — нет агрегированного сигнала."""
        strategy = EnsembleVoterStrategy()
        strategy.strategy_weights = {
            "rsi_mr": 1.0,
            "ema_cross": 1.0,
        }

        signals = [
            MagicMock(strategy_id="rsi_mr", side="long", confidence=0.6, symbol="BTC/USDT",
                      entry_price=50000.0, stop_loss=49000.0, take_profit=51000.0,
                      position_size_pct=5.0, timeframe="1h", rationale="test"),
            MagicMock(strategy_id="ema_cross", side="short", confidence=0.6, symbol="BTC/USDT",
                      entry_price=50000.0, stop_loss=51000.0, take_profit=49000.0,
                      position_size_pct=5.0, timeframe="1h", rationale="test"),
        ]

        aggregated = strategy.aggregate_signals(signals)
        self.assertIsNone(aggregated)


class TestFeatureEngine(unittest.TestCase):
    """Тесты для FeatureEngine."""

    def setUp(self):
        from src.data_ingest.feature_engine import FeatureEngine
        self.engine = FeatureEngine()

    def test_empty_dataframe(self):
        """Пустой DataFrame."""
        df = pd.DataFrame()
        result = self.engine.compute_all_indicators(df)
        self.assertTrue(result.empty)

    def test_compute_rsi(self):
        """Вычисление RSI."""
        df = pd.DataFrame({
            "close": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                      110, 108, 107, 106, 108, 110, 112, 111, 113, 115,
                      114, 116, 118, 117, 119, 120, 118, 117, 119, 121],
        })
        result = self.engine.compute_all_indicators(df)
        self.assertIn("rsi_14", result.columns)
        self.assertGreater(result["rsi_14"].iloc[-1], 50)  # растущий тренд

    def test_compute_macd(self):
        """Вычисление MACD."""
        df = pd.DataFrame({
            "close": np.random.uniform(100, 110, 50),
        })
        result = self.engine.compute_all_indicators(df)
        self.assertIn("macd", result.columns)
        self.assertIn("macd_hist", result.columns)

    def test_compute_bb(self):
        """Вычисление Bollinger Bands."""
        df = pd.DataFrame({
            "close": np.random.uniform(100, 110, 50),
        })
        result = self.engine.compute_all_indicators(df)
        self.assertIn("bb_upper", result.columns)
        self.assertIn("bb_lower", result.columns)
        self.assertIn("bb_mid", result.columns)

    def test_extract_ml_features(self):
        """Извлечение признаков для ML."""
        df = pd.DataFrame({
            "close": np.random.uniform(100, 110, 100),
        })
        result = self.engine.compute_all_indicators(df)
        features = self.engine.extract_features_for_ml(result, include_target=True)

        self.assertFalse(features.empty)
        self.assertIn("target_direction", features.columns)


class TestCoinGlassClient(unittest.IsolatedAsyncioTestCase):
    """Тесты для CoinGlass клиента (мок-тесты)."""

    async def test_client_initialization(self):
        """Инициализация клиента."""
        from src.data_ingest.coinglass_client import CoinGlassClient

        client = CoinGlassClient(api_key="test_key")
        self.assertEqual(client.api_key, "test_key")
        self.assertIsNotNone(client.client)

        await client.close()


class TestExecutionEngine(unittest.IsolatedAsyncioTestCase):
    """Тесты для ExecutionEngine (мок-тесты)."""

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    async def test_paper_mode_initialization(self):
        """Инициализация в paper режиме."""
        settings.trading_mode = "paper"
        settings.startup_capitol_usdt = 10000.0

        await self.engine.initialize("binance")
        self.assertTrue(self.engine.is_paper)
        self.assertEqual(self.engine.get_paper_balance(), 10000.0)

    async def test_can_execute_initial(self):
        """Можно исполнять ордер в начальном состоянии."""
        self.assertTrue(self.engine.can_execute())

    async def test_cannot_execute_kill_switch(self):
        """Нельзя исполнять при kill switch."""
        # Эмулируем kill switch через risk_manager
        from src.risk.risk_manager import risk_manager
        risk_manager.state.kill_switch_active = True
        self.assertFalse(self.engine.can_execute())
        risk_manager.state.kill_switch_active = False

    async def test_paper_create_order(self):
        """Создание paper ордера."""
        settings.trading_mode = "paper"
        settings.startup_capitol_usdt = 10000.0

        await self.engine.initialize("binance")

        order = await self.engine.create_order(
            symbol="BTC/USDT",
            side="buy",
            amount=0.01,
            price=50000.0,
            order_type="market",
            stop_loss=49000.0,
            take_profit=51000.0,
        )

        self.assertIsNotNone(order)
        self.assertEqual(order.side, "buy")
        self.assertEqual(order.status, "filled")
        self.assertEqual(self.engine.get_paper_balance(), 10000.0 - 0.01 * 50000.0 * 1.001)  # с учётом fee


class TestTelegramSignalParser(unittest.TestCase):
    """Тесты для парсера Telegram сигналов."""

    def setUp(self):
        from src.telegram.signal_parser import telegram_signal_parser
        self.parser = telegram_signal_parser

    def test_parse_btc_long(self):
        """Парсинг BTC/USDT Long сигнала."""
        text = "BTC/USDT Long 69000 SL 68000 TP 72000"
        result = self.parser.parse(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "BTC/USDT")
        self.assertEqual(result["side"], "long")
        self.assertEqual(result["entry"], 69000.0)
        self.assertEqual(result["sl"], 68000.0)
        self.assertEqual(result["tp"], 72000.0)

    def test_parse_eth_short(self):
        """Парсинг ETH/USDT Short сигнала."""
        text = "ETH/USDT Short | Entry: 3500 | Stop: 3600 | Target: 3200"
        result = self.parser.parse(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "ETH/USDT")
        self.assertEqual(result["side"], "short")
        self.assertEqual(result["entry"], 3500.0)
        self.assertEqual(result["sl"], 3600.0)
        self.assertEqual(result["tp"], 3200.0)

    def test_parse_no_signal(self):
        """Текст без сигнала."""
        text = "Привет! Как дела? https://t.me/channel"
        result = self.parser.parse(text)
        self.assertIsNone(result)

    def test_parse_invalid_pair(self):
        """Невалидная пара."""
        text = "XYZ/USD Long 100 SL 90 TP 110"
        result = self.parser.parse(text)
        self.assertIsNone(result)

    def test_validate_signal(self):
        """Валидация сигнала."""
        valid_signal = {
            "pair": "BTC/USDT",
            "side": "long",
            "entry": 50000.0,
            "sl": 49000.0,
            "tp": 51000.0,
        }
        self.assertTrue(self.parser.validate_signal(valid_signal))

        invalid_signal = {
            "pair": "BTC/USDT",
            "side": "invalid",
            "entry": 50000.0,
        }
        self.assertFalse(self.parser.validate_signal(invalid_signal))

    def test_validate_signal_missing_fields(self):
        """Недостающие поля."""
        incomplete_signal = {
            "pair": "BTC/USDT",
            "side": "long",
            # "entry" отсутствует
        }
        self.assertFalse(self.parser.validate_signal(incomplete_signal))


class TestQualificationScorer(unittest.TestCase):
    """Тесты для scorer качества сигналов."""

    def setUp(self):
        from src.telegram.quality_scorer import signal_quality_scorer
        self.scorer = signal_quality_scorer

    def test_score_signal_good_channel(self):
        """Высокий score для хорошего канала."""
        signal = {
            "pair": "BTC/USDT",
            "side": "long",
            "entry": 50000.0,
            "sl": 49000.0,
            "tp": 51000.0,
            "confidence": 0.9,
        }

        self.scorer.channel_stats["good_channel"] = {
            "signals_count": 100,
            "good_signals": 80,
            "bad_signals": 20,
            "win_rate": 0.8,
        }

        score = self.scorer.score_signal(signal, "good_channel")
        self.assertGreater(score, 0.7)

    def test_score_signal_bad_channel(self):
        """Низкий score для плохого канала."""
        signal = {
            "pair": "BTC/USDT",
            "side": "long",
            "entry": 50000.0,
            "sl": 49000.0,
            "tp": 51000.0,
            "confidence": 0.5,
        }

        self.scorer.channel_stats["bad_channel"] = {
            "signals_count": 100,
            "good_signals": 30,
            "bad_signals": 70,
            "win_rate": 0.3,
        }

        score = self.scorer.score_signal(signal, "bad_channel")
        self.assertLess(score, 0.5)

    def test_score_without_sl(self):
        """Уменьшение score без SL."""
        signal_no_sl = {
            "pair": "BTC/USDT",
            "side": "long",
            "entry": 50000.0,
            "sl": None,
            "tp": 51000.0,
            "confidence": 0.7,
        }

        score_no_sl = self.scorer.score_signal(signal_no_sl, "test_channel")
        self.scorer.channel_stats["test_channel"] = {"win_rate": 0.5}

        signal_with_sl = {
            "pair": "BTC/USDT",
            "side": "long",
            "entry": 50000.0,
            "sl": 49000.0,
            "tp": 51000.0,
            "confidence": 0.7,
        }

        score_with_sl = self.scorer.score_signal(signal_with_sl, "test_channel")
        self.assertLess(score_no_sl, score_with_sl)


class TestLogging(unittest.TestCase):
    """Тесты для логирования."""

    def test_setup_logging(self):
        """Setup logging не вызывает ошибок."""
        setup_logging("INFO")

    def test_log_trade_win(self):
        """Логирование выигрышной сделки."""
        from src.utils.logging import log_trade
        log_trade(1, "BTC/USDT", "long", 100.0, 1.0, "win", "RSI Strategy")


if __name__ == "__main__":
    unittest.main()
