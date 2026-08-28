"""Тесты для крипто-трейдер бота."""
import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, PropertyMock, patch

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

    def tearDown(self):
        settings.trading_mode = self.original_trading_mode

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
        self.assertEqual(settings.startup_capital_usdt, 10000.0)

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
        # start_balance/current_balance предзаполняются стартовым капиталом
        # аккаунта (а не 0), иначе после рестарта базой для расчёта
        # просадки становился бы текущий баланс, скрывая уже случившуюся
        # просадку от max_drawdown_pct.
        self.assertEqual(state.current_balance, settings.startup_capital_usdt)
        self.assertEqual(state.start_balance, settings.startup_capital_usdt)
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

    def test_add_open_position(self):
        """Добавление открытой позиции."""
        state = RiskState()
        state.add_open_position("BTC/USDT", 10.0)
        self.assertEqual(state.open_positions_count, 1)

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
        self.assertTrue(self.risk.can_trade())

    async def test_cannot_trade_kill_switch(self):
        """Нельзя торговать при kill switch."""
        self.risk.state.trigger_kill_switch()
        self.assertFalse(self.risk.can_trade())

    async def test_cannot_trade_paused(self):
        """Нельзя торговать при паузе."""
        self.risk.state.paused = True
        self.assertFalse(self.risk.can_trade())

    async def test_check_signal_ok(self):
        """Сигнал проходит проверку."""
        signal = MagicMock()
        signal.symbol = "BTC/USDT"
        signal.position_size_pct = 5.0
        can_execute, reason = self.risk.check_signal(signal)
        self.assertTrue(can_execute)

    async def test_check_signal_size_exceeded(self):
        """Размер позиции превышает лимит."""
        signal = MagicMock()
        signal.symbol = "BTC/USDT"
        signal.position_size_pct = 15.0
        can_execute, reason = self.risk.check_signal(signal)
        self.assertFalse(can_execute)
        self.assertIn("превышает", reason)

    async def test_adjust_position_size(self):
        """Расчёт размера позиции."""
        signal = MagicMock()
        signal.position_size_pct = 10.0
        size = self.risk.adjust_position_size(signal, 10000.0)
        self.assertEqual(size, 1000.0)

    async def test_on_trade_closed(self):
        """Обработка закрытия сделки."""
        self.risk.on_trade_closed(100.0)
        self.assertEqual(self.risk.state.daily_pnl, 100.0)
        self.risk.on_trade_closed(-600.0)
        self.assertTrue(self.risk.state.daily_loss_limit_reached)

    async def test_cannot_trade_immediately_after_trade_closed(self):
        """Сразу после закрытия сделки (в т.ч. частичного TP) кулдаун активен."""
        self.risk.on_trade_closed(10.0)
        self.assertFalse(self.risk.can_trade())

    async def test_can_trade_again_after_cooldown_elapses(self):
        """
        Кулдаун должен автоматически сниматься по истечении cooldown_seconds,
        а не оставаться активным до рестарта процесса. can_trade() раньше
        читал голый state.cooldown_active — его выключало только
        check_cooldown() по прошедшему времени, но эта проверка нигде не
        вызывалась, поэтому после первого же закрытия сделки за время
        работы процесса кулдаун блокировал вообще все новые входы навсегда,
        независимо от значения risk_cooldown_seconds.
        """
        from src.utils.timeutils import utcnow
        self.risk.profile.cooldown_seconds = 5
        self.risk.state.cooldown_seconds = 5
        self.risk.on_trade_closed(10.0)
        self.assertFalse(self.risk.can_trade())

        self.risk.state.last_trade_time = utcnow() - timedelta(seconds=6)
        self.assertTrue(self.risk.can_trade())

    async def test_changing_cooldown_seconds_takes_effect_on_active_cooldown(self):
        """Уменьшение cooldown_seconds в настройках должно сократить уже идущий кулдаун."""
        self.risk.profile.cooldown_seconds = 300
        self.risk.state.cooldown_seconds = 300
        self.risk.on_trade_closed(10.0)
        self.assertFalse(self.risk.can_trade())

        self.risk.configure({"cooldown_seconds": 1})
        self.risk.state.last_trade_time -= timedelta(seconds=2)
        self.assertTrue(self.risk.can_trade())


class TestRSIMeanReversionStrategy(unittest.TestCase):
    """Тесты для RSI стратегии."""

    def test_long_signal(self):
        """Long сигнал при oversold RSI."""
        strategy = RSIMeanReversionStrategy()
        strategy.params = {
            "rsi_period": 14, "oversold_level": 30, "overbought_level": 70,
            "position_size_pct": 5.0, "sl_pct": 0.02, "tp_pct": 0.04,
        }
        data = {"rsi_14": 25.0, "close": 50000.0, "ema_20": 49000.0,
                "symbol": "BTC/USDT", "timeframe": "1h", "ml_confidence_long": 1.0}
        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "long")
        self.assertEqual(signal.entry_price, 50000.0)

    def test_short_signal(self):
        """Short сигнал при overbought RSI."""
        strategy = RSIMeanReversionStrategy()
        strategy.params = {
            "rsi_period": 14, "oversold_level": 30, "overbought_level": 70,
            "position_size_pct": 5.0, "sl_pct": 0.02, "tp_pct": 0.04,
        }
        data = {"rsi_14": 75.0, "close": 50000.0, "ema_20": 51000.0,
                "symbol": "BTC/USDT", "timeframe": "1h", "ml_confidence_short": 1.0}
        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "short")

    def test_no_signal_neutral(self):
        """Нет сигнала в нейтральном состоянии."""
        strategy = RSIMeanReversionStrategy()
        strategy.params = {"position_size_pct": 5.0}
        data = {"rsi_14": 50.0, "close": 50000.0, "ema_20": 50000.0,
                "symbol": "BTC/USDT", "timeframe": "1h"}
        signal = strategy.generate_signal(data)
        self.assertIsNone(signal)


class TestEMACrossoverStrategy(unittest.TestCase):
    """Тесты для EMA Crossover стратегии."""

    def test_long_crossover(self):
        """Long сигнал при пересечении EMA вверх."""
        strategy = EMACrossoverStrategy()
        strategy.params = {"fast_ema_period": 9, "slow_ema_period": 21,
                          "position_size_pct": 5.0, "sl_pct": 0.02, "tp_pct": 0.04}
        data = {"ema_9": 50000.0, "ema_21": 49000.0, "ema_9_lag_1": 48000.0,
                "ema_21_lag_1": 49000.0, "close": 50000.0,
                "symbol": "BTC/USDT", "timeframe": "1h"}
        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "long")

    def test_short_crossover(self):
        """Short сигнал при пересечении EMA вниз."""
        strategy = EMACrossoverStrategy()
        strategy.params = {"fast_ema_period": 9, "slow_ema_period": 21,
                          "position_size_pct": 5.0, "sl_pct": 0.02, "tp_pct": 0.04}
        data = {"ema_9": 48000.0, "ema_21": 49000.0, "ema_9_lag_1": 49500.0,
                "ema_21_lag_1": 49000.0, "close": 48000.0,
                "symbol": "BTC/USDT", "timeframe": "1h"}
        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "short")


class TestBollingerBandsStrategy(unittest.TestCase):
    """Тесты для Bollinger Bands стратегии."""

    def test_mean_reversion_long(self):
        """Long сигнал — цена у нижней полосы."""
        strategy = BollingerBandsStrategy()
        strategy.params = {"mode": "mean_reversion", "position_size_pct": 5.0,
                          "sl_pct": 0.02, "tp_pct": 0.04}
        data = {"bb_upper": 51000.0, "bb_lower": 49000.0, "bb_mid": 50000.0,
                "bb_pct": 0.0, "volume_ratio": 1.3, "close": 49000.0,
                "symbol": "BTC/USDT", "timeframe": "1h"}
        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "long")

    def test_breakout_long(self):
        """Long сигнал — пробой верхней полосы."""
        strategy = BollingerBandsStrategy()
        strategy.params = {"mode": "breakout", "position_size_pct": 5.0}
        data = {"bb_upper": 50000.0, "bb_lower": 49000.0, "bb_mid": 49500.0,
                "bb_pct": 1.0, "volume_ratio": 2.0, "close": 50500.0,
                "symbol": "BTC/USDT", "timeframe": "1h"}
        signal = strategy.generate_signal(data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "long")


class TestEnsembleVoterStrategy(unittest.TestCase):
    """Тесты для Ensemble Voter стратегии."""

    def test_aggregate_long_dominant(self):
        """Агрегация: преобладают long."""
        strategy = EnsembleVoterStrategy()
        strategy.strategy_weights = {"rsi_mr": 1.0, "ema_cross": 1.0}
        signals = [
            MagicMock(strategy_id="rsi_mr", side="long", confidence=0.8,
                      symbol="BTC/USDT", entry_price=50000.0, stop_loss=49000.0,
                      take_profit=51000.0, position_size_pct=5.0, timeframe="1h",
                      rationale="test"),
            MagicMock(strategy_id="ema_cross", side="long", confidence=0.7,
                      symbol="BTC/USDT", entry_price=50000.0, stop_loss=49000.0,
                      take_profit=51000.0, position_size_pct=5.0, timeframe="1h",
                      rationale="test"),
        ]
        aggregated = strategy.aggregate_signals(signals)
        self.assertIsNotNone(aggregated)
        self.assertEqual(aggregated.side, "long")


def _make_ohlcv(n: int) -> pd.DataFrame:
    """Синтетические OHLCV-свечи для тестов индикаторов."""
    close = np.random.uniform(100, 110, n)
    high = close + np.random.uniform(0.5, 1.5, n)
    low = close - np.random.uniform(0.5, 1.5, n)
    open_ = close + np.random.uniform(-0.5, 0.5, n)
    volume = np.random.uniform(1000, 5000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


class TestFeatureEngine(unittest.TestCase):
    """Тесты для FeatureEngine."""

    def setUp(self):
        from src.data_ingest.feature_engine import FeatureEngine
        self.engine = FeatureEngine()

    def test_compute_rsi(self):
        """Вычисление RSI."""
        df = _make_ohlcv(50)
        result = self.engine.compute_all_indicators(df)
        self.assertIn("rsi_14", result.columns)

    def test_compute_macd(self):
        """Вычисление MACD."""
        df = _make_ohlcv(50)
        result = self.engine.compute_all_indicators(df)
        self.assertIn("macd", result.columns)

    def test_extract_ml_features(self):
        """Извлечение признаков для ML."""
        df = _make_ohlcv(100)
        result = self.engine.compute_all_indicators(df)
        features = self.engine.extract_features_for_ml(result, include_target=True)
        self.assertFalse(features.empty)
        self.assertIn("target_direction", features.columns)


class TestCoinGlassClient(unittest.IsolatedAsyncioTestCase):
    """Тесты для CoinGlass клиента."""

    async def test_client_initialization(self):
        """Инициализация клиента."""
        from src.data_ingest.coinglass_client import CoinGlassClient
        client = CoinGlassClient(api_key="test_key")
        self.assertEqual(client.api_key, "test_key")
        await client.close()


class TestExecutionEngine(unittest.IsolatedAsyncioTestCase):
    """Тесты для ExecutionEngine."""

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    async def test_fetch_confirmed_order_exits_early_on_filled_without_average_price(self):
        """
        Реальный инцидент (RLUSD/USDT, демо-счёт Bybit): ответ fetch_order
        мог уже содержать реальный filled без ещё подтянувшихся
        average/price — раньше условие выхода из цикла проверяло только
        average/price, поэтому такой снимок игнорировался, цикл впустую
        доходил до конца попыток и вызывающий код (_execute_real_order)
        считал ордер неисполненным, хотя биржа его уже реально исполнила.
        """
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_order = AsyncMock(
            return_value={"id": "early-filled-1", "filled": 1.03, "average": None, "price": None}
        )
        with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
            result = await self.engine._fetch_confirmed_order(
                {"id": "early-filled-1", "filled": None, "average": None, "price": None},
                "RLUSD/USDT",
            )
        self.assertEqual(result["filled"], 1.03)
        self.engine.exchange.fetch_order.assert_called_once()

    async def test_fetch_confirmed_order_returns_latest_snapshot_not_stale_original(self):
        """
        Если ни за одну попытку ни average/price, ни filled так и не
        появились — нужно вернуть ПОСЛЕДНИЙ полученный от биржи снимок
        (пусть и неполный), а не исходный ответ СОЗДАНИЯ ордера, который
        точно устарел (в нём filled всегда None/0 по конструкции Bybit v5).
        """
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_order = AsyncMock(
            return_value={"id": "never-confirms-1", "filled": None, "average": None, "price": None, "status": "open"}
        )
        original = {"id": "never-confirms-1", "filled": None, "average": None, "price": None}
        with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
            result = await self.engine._fetch_confirmed_order(original, "RLUSD/USDT", attempts=3, delay=0.01)
        self.assertIsNot(result, original)
        self.assertEqual(result["status"], "open")
        self.assertEqual(self.engine.exchange.fetch_order.call_count, 3)

    async def test_execute_real_order_confirms_via_balance_diff_when_status_never_confirms(self):
        """
        Реальный инцидент (демо-счёт Bybit, RLUSD/USDT, USDC/USDT,
        BTC/USDT): биржа реально исполняла ордер (видно в истории сделок
        самой биржи), но fetch_order по его ID стабильно не показывал
        filled ни разу — даже после расширенного окна поллинга. Без
        второго способа подтверждения _execute_real_order слепо считал
        такой ордер проваленным и НЕ регистрировал позицию, хотя деньги
        были реально потрачены на бирже. Сверка баланса базовой валюты
        до/после (независимая от статуса самого ордера) должна поймать
        этот случай.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(side_effect=[
            {"free": {"BALDIFF1": 0.0}, "BALDIFF1": {"free": 0.0, "used": 0, "total": 0.0}},
            {"free": {"BALDIFF1": 2011.85}, "BALDIFF1": {"free": 2011.85, "used": 0, "total": 2011.85}},
        ])
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "never-confirms-buy-1", "filled": None, "average": None, "price": None,
        }
        self.engine.exchange.fetch_order = AsyncMock(return_value={
            "id": "never-confirms-buy-1", "filled": None, "average": None, "price": None, "status": "open",
        })

        with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
            order = await self.engine.create_order(
                symbol="BALDIFF1/USDT", side="buy", amount=2011.85, price=1.0003, order_type="market",
            )

        self.assertIsNotNone(order)
        self.assertIn("BALDIFF1/USDT", self.engine.real_positions)
        # Ни fetch_order, ни история сделок не дали комиссию — она
        # рассчитывается по стандартной ставке (см. _resolve_fee) и
        # вычитается из объёма позиции (комиссия в base-валюте при покупке),
        # поэтому реально доступный остаток чуть меньше исполненного объёма.
        expected_fee = 2011.85 * 1.0003 * (settings.paper_fee_pct / 100)
        self.assertAlmostEqual(
            self.engine.real_positions["BALDIFF1/USDT"]["amount"], 2011.85 - expected_fee,
        )

    async def test_close_real_position_confirms_via_balance_diff_when_status_never_confirms(self):
        """Симметричный случай на закрытии — см. тест выше на открытии."""
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"BALDIFFCLOSE1": 0.0}, "BALDIFFCLOSE1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "baldiffclose-open-1", "filled": 100.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }
        opening_order = await self.engine.create_order(
            symbol="BALDIFFCLOSE1/USDT", side="buy", amount=100.0, price=0.5, order_type="market",
        )
        self.assertIsNotNone(opening_order)

        self.engine.exchange.fetch_balance = AsyncMock(side_effect=[
            {"free": {"BALDIFFCLOSE1": 100.0}, "BALDIFFCLOSE1": {"free": 100.0, "used": 0, "total": 100.0}},
            {"free": {"BALDIFFCLOSE1": 0.0}, "BALDIFFCLOSE1": {"free": 0.0, "used": 0, "total": 0.0}},
        ])
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "never-confirms-sell-1", "filled": None, "average": None, "price": None,
        }
        self.engine.exchange.fetch_order = AsyncMock(return_value={
            "id": "never-confirms-sell-1", "filled": None, "average": None, "price": None, "status": "open",
        })

        with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
            result = await self.engine.close_real_position(
                symbol="BALDIFFCLOSE1/USDT", side="long", entry_price=0.5, amount=100.0,
                reason="stop_loss", entry_fee=0.05, holding_seconds=60, order_open_id=opening_order.id,
            )

        self.assertIsNotNone(result)
        self.assertNotIn("BALDIFFCLOSE1/USDT", self.engine.real_positions)

    async def test_fetch_fill_details_via_trades_returns_weighted_average_and_fee(self):
        """
        Юнит-тест самого хелпера: несколько частичных сделок по одному
        ордеру должны сворачиваться в средневзвешенную цену и суммарную
        комиссию, а не просто в цену первой сделки.
        """
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=[
            {"amount": 10.0, "price": 100.0, "cost": 1000.0, "fee": {"cost": 0.1, "currency": "USDT"}},
            {"amount": 5.0, "price": 102.0, "cost": 510.0, "fee": {"cost": 0.05, "currency": "USDT"}},
        ])
        result = await self.engine._fetch_fill_details_via_trades("some-order-1", "TESTTRADES1/USDT")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["amount"], 15.0)
        self.assertAlmostEqual(result["average"], 1510.0 / 15.0)
        self.assertAlmostEqual(result["fee"]["cost"], 0.15)
        self.assertEqual(result["fee"]["currency"], "USDT")

    async def test_close_real_position_computes_correct_pnl_via_trade_history_price(self):
        """
        Реальный сценарий пользователя (LINK/USDT, демо-счёт Bybit): куплено
        34.458 LINK по 11.729, продано 34.423 LINK по 11.776 — реально
        прибыльная сделка на бирже. Без приоритета истории сделок над
        сверкой по балансу (см. предыдущий тест) exit_price откатился бы на
        entry_price — сверка по балансу подтверждает только ОБЪЁМ, не цену —
        и PnL показывал бы ровно 0.00 независимо от факта на бирже.

        Комиссия открытия здесь в BASE-валюте (0.034458 LINK) — PnL должен
        учитывать её в USDT-эквиваленте по цене входа (0.034458 * 11.729),
        а не вычитать "0.034458" напрямую как будто это уже доллары.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"LINKPNL1": 0.0}, "LINKPNL1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "linkpnl-open-1", "filled": 34.458, "average": 11.729, "price": None,
            "fee": {"cost": 0.034458, "currency": "LINKPNL1"},
        }
        opening_order = await self.engine.create_order(
            symbol="LINKPNL1/USDT", side="buy", amount=34.458, price=11.729, order_type="market",
        )
        self.assertIsNotNone(opening_order)
        entry_amount = self.engine.real_positions["LINKPNL1/USDT"]["amount"]

        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"LINKPNL1": entry_amount},
            "LINKPNL1": {"free": entry_amount, "used": 0, "total": entry_amount},
        })
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "linkpnl-close-1", "filled": None, "average": None, "price": None,
        }
        self.engine.exchange.fetch_order = AsyncMock(return_value={
            "id": "linkpnl-close-1", "filled": None, "average": None, "price": None, "status": "open",
        })
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=[
            {"amount": 34.423, "price": 11.776, "cost": 34.423 * 11.776,
             "fee": {"cost": 0.405365248, "currency": "USDT"}},
        ])

        with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
            result = await self.engine.close_real_position(
                symbol="LINKPNL1/USDT", side="long", entry_price=11.729, amount=34.423,
                reason="stop_loss", entry_fee=0.034458, holding_seconds=1900, order_open_id=opening_order.id,
            )

        self.assertIsNotNone(result)
        self.assertGreater(result["pnl"], 0.5, "реально прибыльная сделка не должна показывать PnL≈0")
        self.assertAlmostEqual(result["pnl"], 0.8083578700000205, places=4)

    async def test_execute_real_order_stores_exchange_trade_id_not_internal_order_id(self):
        """
        На бирже "ID ордера" в истории сделок — это короткий ID сделки
        (execId, ~8 символов), а не длинный внутренний order["id"], который
        нигде в интерфейсе биржи не показывается. order_id_exchange должен
        хранить ID сделки, даже когда fetch_order уже подтвердил filled сам
        по себе (без похода в _fetch_fill_details_via_trades по цене/объёму).
        """
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import Order

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"TRADEID1": 0.0}, "TRADEID1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "1234567890123456789-very-long-internal-order-id",
            "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=[
            {"id": "56173056", "amount": 10.0, "price": 2.0, "cost": 20.0,
             "fee": {"cost": 0.01, "currency": "USDT"}},
        ])

        order = await self.engine.create_order(
            symbol="TRADEID1/USDT", side="buy", amount=10.0, price=2.0, order_type="market",
        )
        self.assertIsNotNone(order)

        async with get_session() as session:
            db_order = (
                await session.execute(select(Order).where(Order.id == order.id))
            ).scalar_one()
        self.assertEqual(db_order.order_id_exchange, "56173056")

    async def test_close_real_position_stores_exchange_trade_id_not_internal_order_id(self):
        """Симметричный случай на закрытии — см. тест выше на открытии."""
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import Order, Trade

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"TRADEIDCLOSE1": 0.0}, "TRADEIDCLOSE1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "open-long-internal-id", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=None)
        opening_order = await self.engine.create_order(
            symbol="TRADEIDCLOSE1/USDT", side="buy", amount=10.0, price=2.0, order_type="market",
        )
        self.assertIsNotNone(opening_order)

        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"TRADEIDCLOSE1": 10.0},
            "TRADEIDCLOSE1": {"free": 10.0, "used": 0, "total": 10.0},
        })
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "close-long-internal-id-much-longer-than-8-chars", "filled": 10.0, "average": 2.5, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=[
            {"id": "35242741", "amount": 10.0, "price": 2.5, "cost": 25.0,
             "fee": {"cost": 0.01, "currency": "USDT"}},
        ])

        result = await self.engine.close_real_position(
            symbol="TRADEIDCLOSE1/USDT", side="long", entry_price=2.0, amount=10.0,
            reason="stop_loss", entry_fee=0.01, holding_seconds=60, order_open_id=opening_order.id,
        )
        self.assertIsNotNone(result)

        async with get_session() as session:
            trade = (
                await session.execute(select(Trade).where(Trade.id == result["trade_id"]))
            ).scalar_one()
            close_order = (
                await session.execute(select(Order).where(Order.id == trade.order_close_id))
            ).scalar_one()
        self.assertEqual(close_order.order_id_exchange, "35242741")

    async def test_execute_real_order_prefers_trade_history_fee_over_order_response(self):
        """
        Реальный инцидент (демо-счёт Bybit, TAC/USDT): комиссия покупки на
        бирже — 105.4915 TAC, но order["fee"] из create_market_buy_order
        может быть неполным/нулевым на момент снимка (Bybit v5 иногда
        подтверждает объём/статус раньше, чем комиссию) — приоритет должен
        быть у истории сделок биржи (fetch_order_trades), а не у уже как бы
        подтверждённого order["fee"].
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"TACFEE1": 0.0}, "TACFEE1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "tacfee-open-1", "filled": 10000.0, "average": 0.01, "price": None,
            "fee": {"cost": 0, "currency": None},
        }
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=[
            {"id": "56173056", "amount": 10000.0, "price": 0.01, "cost": 100.0,
             "fee": {"cost": 105.4915, "currency": "TACFEE1"}},
        ])

        order = await self.engine.create_order(
            symbol="TACFEE1/USDT", side="buy", amount=10000.0, price=0.01, order_type="market",
        )
        self.assertIsNotNone(order)
        self.assertAlmostEqual(float(order.fee), 105.4915)
        self.assertEqual(order.fee_currency, "TACFEE1")
        self.assertEqual(order.order_id_exchange, "56173056")

    async def test_close_real_position_prefers_trade_history_fee_over_order_response(self):
        """Симметричный случай на закрытии — комиссия продажи в USDT (как в примере пользователя)."""
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import Order, Trade

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"TACFEECLOSE1": 0.0}, "TACFEECLOSE1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "tacfeeclose-open-1", "filled": 10000.0, "average": 0.01, "price": None,
            "fee": {"cost": 1.0, "currency": "TACFEECLOSE1"},
        }
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=None)
        self.engine.exchange.fetch_my_trades = AsyncMock(return_value=None)
        opening_order = await self.engine.create_order(
            symbol="TACFEECLOSE1/USDT", side="buy", amount=10000.0, price=0.01, order_type="market",
        )
        self.assertIsNotNone(opening_order)
        entry_amount = self.engine.real_positions["TACFEECLOSE1/USDT"]["amount"]

        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"TACFEECLOSE1": entry_amount},
            "TACFEECLOSE1": {"free": entry_amount, "used": 0, "total": entry_amount},
        })
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "tacfeeclose-close-1", "filled": entry_amount, "average": 0.0105, "price": None,
            "fee": {"cost": 0, "currency": None},
        }
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=[
            {"id": "35242741", "amount": entry_amount, "price": 0.0105, "cost": entry_amount * 0.0105,
             "fee": {"cost": 0.2950808, "currency": "USDT"}},
        ])

        result = await self.engine.close_real_position(
            symbol="TACFEECLOSE1/USDT", side="long", entry_price=0.01, amount=entry_amount,
            reason="stop_loss", entry_fee=1.0, holding_seconds=60, order_open_id=opening_order.id,
        )
        self.assertIsNotNone(result)

        async with get_session() as session:
            trade = (
                await session.execute(select(Trade).where(Trade.id == result["trade_id"]))
            ).scalar_one()
            close_order = (
                await session.execute(select(Order).where(Order.id == trade.order_close_id))
            ).scalar_one()
        self.assertAlmostEqual(float(close_order.fee), 0.2950808)
        self.assertEqual(close_order.fee_currency, "USDT")
        self.assertEqual(close_order.order_id_exchange, "35242741")

    async def test_resolve_fee_calculates_estimate_when_exchange_data_unavailable(self):
        """
        Если ни order["fee"], ни история сделок не дали реальную комиссию —
        нужно посчитать её по стандартной ставке spot-таксы, а не оставлять
        0 (иначе PnL завышался бы на величину реальной, но неучтённой
        комиссии биржи).
        """
        fee, currency = self.engine._resolve_fee(None, 1000.0, 2.0, "buy", "RESOLVEFEE1/USDT")
        self.assertAlmostEqual(fee, 1000.0 * 2.0 * (settings.paper_fee_pct / 100))
        self.assertEqual(currency, "RESOLVEFEE1")

        fee, currency = self.engine._resolve_fee({"cost": 0, "currency": "USDT"}, 1000.0, 2.0, "sell", "RESOLVEFEE1/USDT")
        self.assertAlmostEqual(fee, 1000.0 * 2.0 * (settings.paper_fee_pct / 100))
        self.assertEqual(currency, "USDT")

        fee, currency = self.engine._resolve_fee({"cost": 3.5, "currency": "USDT"}, 1000.0, 2.0, "sell", "RESOLVEFEE1/USDT")
        self.assertEqual(fee, 3.5)
        self.assertEqual(currency, "USDT")

    async def test_close_real_position_converts_base_currency_entry_fee_to_quote(self):
        """
        Комиссия ОТКРЫТИЯ в BASE-валюте (например TAC при покупке TAC/USDT)
        должна учитываться в PnL в USDT-эквиваленте по цене входа, а не
        вычитаться как есть — иначе "105.4915 TAC" считалась бы как
        "105.4915 USDT", искажая PnL на порядки. Комиссия в QUOTE-валюте
        (уже USDT) не должна конвертироваться повторно.
        """
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import Order, Trade

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"FEECONV1": 0.0}, "FEECONV1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        # Комиссия открытия — 10 FEECONV1 (base-валюта), т.е. эквивалент
        # 10 * 2.0 = 20 USDT по цене входа.
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "feeconv-open-1", "filled": 1000.0, "average": 2.0, "price": None,
            "fee": {"cost": 10.0, "currency": "FEECONV1"},
        }
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=None)
        opening_order = await self.engine.create_order(
            symbol="FEECONV1/USDT", side="buy", amount=1000.0, price=2.0, order_type="market",
        )
        self.assertIsNotNone(opening_order)
        entry_amount = self.engine.real_positions["FEECONV1/USDT"]["amount"]

        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"FEECONV1": entry_amount},
            "FEECONV1": {"free": entry_amount, "used": 0, "total": entry_amount},
        })
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "feeconv-close-1", "filled": entry_amount, "average": 2.0, "price": None,
            "fee": {"cost": 0.5, "currency": "USDT"},
        }

        result = await self.engine.close_real_position(
            symbol="FEECONV1/USDT", side="long", entry_price=2.0, amount=entry_amount,
            reason="stop_loss", entry_fee=10.0, holding_seconds=60, order_open_id=opening_order.id,
        )
        self.assertIsNotNone(result)
        # Цена входа == цена выхода (2.0), т.е. весь PnL — это (-1) * учтённая
        # комиссия открытия в USDT-эквиваленте (-10 * 2.0 = -20, а не -10) и
        # комиссия закрытия (уже в USDT, без конвертации): -20 - 0.5 = -20.5.
        self.assertAlmostEqual(result["pnl"], -20.5, places=4)

    async def test_execute_real_order_places_exchange_stop_loss_when_set(self):
        """
        SL должен уходить на биржу отдельным условным ордером сразу после
        открытия реальной позиции (Bybit spot не поддерживает stopLoss,
        прикреплённый к самому маркет-ордеру, — только отдельный условный
        ордер с 'stopLossPrice') — иначе защита позиции существует только
        пока жив процесс бота и успевает внутренний поллинг цены.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"SLPLACE1": 0.0}, "SLPLACE1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "slplace-open-1", "filled": 100.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.02, "currency": "USDT"},
        }
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=None)
        self.engine.exchange.fetch_my_trades = AsyncMock(return_value=None)
        self.engine.exchange.create_market_sell_order.return_value = {"id": "slplace-sl-order-1"}

        order = await self.engine.create_order(
            symbol="SLPLACE1/USDT", side="buy", amount=100.0, price=2.0, order_type="market",
            stop_loss=1.8,
        )
        self.assertIsNotNone(order)
        self.engine.exchange.create_market_sell_order.assert_called_once_with(
            "SLPLACE1/USDT", 100.0, params={"stopLossPrice": 1.8},
        )
        self.assertEqual(self.engine.real_positions["SLPLACE1/USDT"]["sl_order_id"], "slplace-sl-order-1")

    async def test_place_stop_loss_order_clamps_to_available_balance(self):
        """
        Отслеживаемый объём позиции мог разойтись с реальным остатком на
        бирже (комиссии/округление лота, накопленные за частичные закрытия
        или рестарты — реальный инцидент: LINK/USDT после 3 частичных TP,
        XAUT/USDT со старой комиссией — бот пытался выставить SL на весь
        расчётный объём и биржа отклоняла его целиком с "Insufficient
        balance", оставляя позицию вовсе без биржевой защиты). SL должен
        выставляться на фактически доступный остаток, а не на устаревший.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"SLCLAMP1": 90.0}, "SLCLAMP1": {"free": 90.0, "used": 0, "total": 90.0}}
        )
        self.engine.exchange.create_market_sell_order.return_value = {"id": "slclamp-sl-order-1"}

        order_id = await self.engine._place_stop_loss_order("SLCLAMP1/USDT", 100.0, 1.8)

        self.assertEqual(order_id, "slclamp-sl-order-1")
        self.engine.exchange.create_market_sell_order.assert_called_once_with(
            "SLCLAMP1/USDT", 90.0, params={"stopLossPrice": 1.8},
        )

    async def test_place_stop_loss_order_uses_full_amount_when_balance_sufficient(self):
        """Обычный случай (без дрейфа) не должен ничего урезать."""
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"SLFULL1": 100.0}, "SLFULL1": {"free": 100.0, "used": 0, "total": 100.0}}
        )
        self.engine.exchange.create_market_sell_order.return_value = {"id": "slfull-sl-order-1"}

        order_id = await self.engine._place_stop_loss_order("SLFULL1/USDT", 100.0, 1.8)

        self.assertEqual(order_id, "slfull-sl-order-1")
        self.engine.exchange.create_market_sell_order.assert_called_once_with(
            "SLFULL1/USDT", 100.0, params={"stopLossPrice": 1.8},
        )

    async def test_execute_real_order_survives_stop_loss_placement_rejection(self):
        """
        Биржа может отклонить условный SL-ордер (например, триггер-цена
        невалидна) — это НЕ должно блокировать саму позицию, она просто
        остаётся под защитой только внутреннего поллинга цены, как раньше.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"SLREJECT1": 0.0}, "SLREJECT1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "slreject-open-1", "filled": 100.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.02, "currency": "USDT"},
        }
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=None)
        self.engine.exchange.fetch_my_trades = AsyncMock(return_value=None)
        self.engine.exchange.create_market_sell_order = AsyncMock(
            side_effect=Exception('bybit {"retCode":10001,"retMsg":"Invalid trigger price"}')
        )

        order = await self.engine.create_order(
            symbol="SLREJECT1/USDT", side="buy", amount=100.0, price=2.0, order_type="market",
            stop_loss=1.8,
        )
        self.assertIsNotNone(order)
        self.assertIn("SLREJECT1/USDT", self.engine.real_positions)
        self.assertIsNone(self.engine.real_positions["SLREJECT1/USDT"]["sl_order_id"])

    async def test_close_real_position_cancels_tracked_stop_loss_order(self):
        """
        Закрытие (по любой причине — TP, ручное закрытие) должно отменять
        ранее выставленный биржевой SL-ордер ДО собственной продажи —
        иначе он остаётся висеть параллельно и может конфликтовать за тот
        же остаток базовой валюты.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.real_positions["SLCANCEL1/USDT"] = {
            "amount": 100.0, "entry_price": 2.0, "side": "long",
            "sl_order_id": "sl-order-to-cancel-1",
        }
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"SLCANCEL1": 100.0}, "SLCANCEL1": {"free": 100.0, "used": 0, "total": 100.0}}
        )
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "slcancel-close-1", "filled": 100.0, "average": 2.1, "price": None,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }

        result = await self.engine.close_real_position(
            symbol="SLCANCEL1/USDT", side="long", entry_price=2.0, amount=100.0,
            reason="take_profit_3", entry_fee=0.02, holding_seconds=60,
        )
        self.assertIsNotNone(result)
        self.engine.exchange.cancel_order.assert_called_once_with("sl-order-to-cancel-1", "SLCANCEL1/USDT")

    async def test_reconcile_real_positions_finalizes_externally_triggered_stop_loss(self):
        """
        Если позиция пропала с баланса биржи (available ~0) и на неё был
        выставлен биржевой SL-ордер, который сам сработал (без участия
        бота — например, между итерациями цикла), периодическая сверка
        должна записать это как обычное закрытие с реальным PnL, а не
        молча списать позицию как фантомную (без PnL, как для настоящих
        фантомов/пыли).
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.real_positions["SLFIRED1/USDT"] = {
            "amount": 100.0, "entry_price": 2.0, "side": "long",
            "strategy_id": None, "entry_fee": 0.02, "order_id": None,
            "opened_at": datetime.now(), "sl_order_id": "sl-order-fired-1",
        }
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"SLFIRED1": 0.0}, "SLFIRED1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.fetch_order = AsyncMock(
            return_value={"id": "sl-order-fired-1", "status": "closed", "filled": 100.0, "average": 1.8}
        )
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=[
            {"id": "exec-1", "amount": 100.0, "price": 1.8, "cost": 180.0,
             "fee": {"cost": 0.18, "currency": "USDT"}},
        ])

        await self.engine.reconcile_real_positions()

        self.assertNotIn("SLFIRED1/USDT", self.engine.real_positions)
        from sqlalchemy import select, desc
        from src.db.session import get_session
        from src.db.models import Trade, Symbol
        async with get_session() as session:
            symbol_row = (
                await session.execute(select(Symbol).where(Symbol.symbol == "SLFIRED1/USDT"))
            ).scalar_one()
            trade = (
                await session.execute(
                    select(Trade).where(Trade.symbol_id == symbol_row.id).order_by(desc(Trade.id))
                )
            ).scalars().first()
        self.assertIsNotNone(trade)
        # (1.8 - 2.0) * 100 - entry_fee(0.02) - exit_fee(0.18) = -20.2
        self.assertAlmostEqual(float(trade.pnl), -20.2, places=4)
        self.assertFalse(trade.is_open)

    async def test_reconcile_real_positions_finds_close_via_trade_history_without_sl_order(self):
        """
        Позиция без выставленного биржевого SL-ордера (например, SL не был
        задан) пропала с баланса биржи — закрыта либо вручную на самой
        бирже, либо через биржевой TP. Без известного order id единственный
        способ восстановить реальные данные — история сделок биржи: если
        найденная недавняя продажа по объёму совпадает с отслеживаемой
        позицией, закрытие должно записаться с реальными ценой/PnL, а не
        потеряться как фантомная позиция без PnL.
        """
        from sqlalchemy import select, desc
        from src.db.session import get_session
        from src.db.models import Trade, Symbol, Order

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        opened_at = datetime.now() - timedelta(hours=1)
        self.engine.real_positions["TRHIST1/USDT"] = {
            "amount": 100.0, "entry_price": 2.0, "side": "long",
            "strategy_id": None, "entry_fee": 0.02, "order_id": None,
            "opened_at": opened_at, "sl_order_id": None,
        }
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"TRHIST1": 0.0}, "TRHIST1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        recent_ts_ms = int((opened_at + timedelta(minutes=30)).timestamp() * 1000)
        self.engine.exchange.fetch_my_trades = AsyncMock(return_value=[
            {"id": "closed-manually-1", "side": "sell", "amount": 100.0, "price": 2.1,
             "cost": 210.0, "timestamp": recent_ts_ms, "fee": {"cost": 0.05, "currency": "USDT"}},
        ])

        await self.engine.reconcile_real_positions()

        self.assertNotIn("TRHIST1/USDT", self.engine.real_positions)
        async with get_session() as session:
            symbol_row = (
                await session.execute(select(Symbol).where(Symbol.symbol == "TRHIST1/USDT"))
            ).scalar_one()
            trade = (
                await session.execute(
                    select(Trade).where(Trade.symbol_id == symbol_row.id).order_by(desc(Trade.id))
                )
            ).scalars().first()
            close_order = (
                await session.execute(select(Order).where(Order.id == trade.order_close_id))
            ).scalar_one()
        self.assertIsNotNone(trade)
        # (2.1 - 2.0) * 100 - entry_fee(0.02) - exit_fee(0.05) = 9.93
        self.assertAlmostEqual(float(trade.pnl), 9.93, places=4)
        self.assertFalse(trade.is_open)
        self.assertEqual(close_order.order_id_exchange, "closed-manually-1")

    async def test_reconcile_real_positions_does_not_misattribute_unrelated_trade(self):
        """
        Если недавние продажи по символу сильно расходятся по объёму с
        отслеживаемой позицией (например, аккаунт биржи используется не
        только этим ботом — см. docstring _finalize_via_recent_trade_history),
        подставлять чужую сделку нельзя — реальные деньги, неверно
        приписанный PnL хуже отсутствия записи. Должен сработать привычный
        фолбэк на _reconcile_phantom_position (без PnL, но и без риска).
        """
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import Trade, Symbol

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        opened_at = datetime.now() - timedelta(hours=1)
        self.engine.real_positions["TRMISMATCH1/USDT"] = {
            "amount": 100.0, "entry_price": 2.0, "side": "long",
            "strategy_id": None, "entry_fee": 0.02, "order_id": None,
            "opened_at": opened_at, "sl_order_id": None,
        }
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"TRMISMATCH1": 0.0}, "TRMISMATCH1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        recent_ts_ms = int((opened_at + timedelta(minutes=30)).timestamp() * 1000)
        # Совсем другой объём (5.0 против отслеживаемых 100.0) — похоже на
        # сделку постороннего процесса на том же аккаунте, а не на закрытие
        # именно этой позиции.
        self.engine.exchange.fetch_my_trades = AsyncMock(return_value=[
            {"id": "unrelated-1", "side": "sell", "amount": 5.0, "price": 2.1,
             "cost": 10.5, "timestamp": recent_ts_ms, "fee": {"cost": 0.01, "currency": "USDT"}},
        ])

        await self.engine.reconcile_real_positions()

        self.assertNotIn("TRMISMATCH1/USDT", self.engine.real_positions)
        # Фолбэк на _reconcile_phantom_position не создаёт Trade-запись —
        # никакого PnL для этой позиции быть не должно (в отличие от
        # предыдущего теста, где совпадение по объёму принимается).
        async with get_session() as session:
            symbol_row = (
                await session.execute(select(Symbol).where(Symbol.symbol == "TRMISMATCH1/USDT"))
            ).scalar_one_or_none()
            trade = (
                await session.execute(select(Trade).where(Trade.symbol_id == symbol_row.id))
            ).scalars().first() if symbol_row else None
        self.assertIsNone(trade)

    async def test_paper_mode_initialization(self):
        """Инициализация в paper режиме восстанавливает баланс/позиции из БД."""
        settings.trading_mode = "paper"
        settings.startup_capital_usdt = 10000.0
        await self.engine.initialize("binance")
        self.assertTrue(self.engine.is_paper)
        self.assertIsInstance(self.engine.get_paper_balance(), float)

    async def test_restore_paper_state_matches_db(self):
        """paper_balance после initialize() должен совпадать с независимо
        пересчитанным по Order/Trade значением (тестовая БД общая на сессию
        и может содержать данные от других тестов, в т.ч. real-режима —
        поэтому пересчёт здесь тоже скопирован по Exchange.is_paper, как и
        в самой реализации, а не сравнивается с константой)."""
        from sqlalchemy import select, func
        from src.db.session import get_session
        from src.db.models import Trade, Order, Symbol, Exchange

        settings.trading_mode = "paper"
        settings.startup_capital_usdt = 10000.0

        async with get_session() as session:
            realized = (
                await session.execute(
                    select(func.sum(Trade.pnl))
                    .join(Symbol, Trade.symbol_id == Symbol.id)
                    .join(Exchange, Symbol.exchange_id == Exchange.id)
                    .where(Exchange.is_paper == True)  # noqa: E712
                )
            ).scalar() or 0
            close_order_ids = set(
                (await session.execute(
                    select(Trade.order_close_id).where(Trade.order_close_id.is_not(None))
                )).scalars().all()
            )
            partial_trades = (
                await session.execute(
                    select(Trade.order_open_id, Trade.amount).where(Trade.order_open_id.is_not(None))
                )
            ).all()
            closed_by_order: dict = {}
            for order_open_id, trade_amount in partial_trades:
                closed_by_order[order_open_id] = closed_by_order.get(order_open_id, 0.0) + float(trade_amount)
            open_orders = (
                await session.execute(
                    select(Order)
                    .join(Exchange, Order.exchange_id == Exchange.id)
                    .where(Order.side == "buy", Order.status == "filled", Exchange.is_paper == True)  # noqa: E712
                )
            ).scalars().all()

        # Позиция может быть закрыта ЧАСТИЧНО (TP1/TP2) — считаем остаток
        # объёма и пропорциональную ему долю комиссии, а не только "открыт
        # целиком или закрыт целиком", как в старой версии этого пересчёта.
        cost_basis = 0.0
        for o in open_orders:
            if o.id in close_order_ids:
                continue
            filled_amount = float(o.filled_amount or o.amount)
            remaining = filled_amount - closed_by_order.get(o.id, 0.0)
            if remaining <= 1e-9:
                continue
            fee_share = float(o.fee or 0) * (remaining / filled_amount) if filled_amount else 0.0
            cost_basis += remaining * float(o.filled_price or o.price) + fee_share
        expected_balance = settings.startup_capital_usdt + float(realized) - cost_basis

        await self.engine.initialize("binance")
        self.assertAlmostEqual(self.engine.get_paper_balance(), expected_balance, places=2)

    async def test_can_execute_initial(self):
        """Можно исполнять в начальном состоянии."""
        self.assertTrue(self.engine.can_execute())

    async def test_paper_create_order(self):
        """Создание paper ордера."""
        settings.trading_mode = "paper"
        settings.startup_capital_usdt = 10000.0
        await self.engine.initialize("binance")
        order = await self.engine.create_order(
            symbol="BTC/USDT", side="buy", amount=0.01, price=50000.0,
            order_type="market", stop_loss=49000.0, take_profit=51000.0,
        )
        self.assertIsNotNone(order)
        self.assertEqual(order.side, "buy")
        self.assertEqual(order.status, "filled")

    async def test_close_paper_position_profit(self):
        """Закрытие long-позиции в плюс: PnL положительный, баланс растёт, Trade сохранён."""
        settings.trading_mode = "paper"
        settings.startup_capital_usdt = 10000.0
        await self.engine.initialize("binance")

        order = await self.engine.create_order(
            symbol="BTC/USDT", side="buy", amount=0.1, price=50000.0,
            order_type="market", stop_loss=49000.0, take_profit=52000.0,
            strategy_id="rsi_mr",
        )
        self.assertIsNotNone(order)
        balance_after_open = self.engine.get_paper_balance()
        self.assertLess(balance_after_open, 10000.0)

        result = await self.engine.close_paper_position(
            symbol="BTC/USDT", side="long", entry_price=50000.0, amount=0.1,
            exit_price=52000.0, reason="take_profit", entry_fee=order.fee,
            holding_seconds=3600, strategy_id="rsi_mr", order_open_id=order.id,
        )
        self.assertIsNotNone(result)
        self.assertGreater(result["pnl"], 0)
        self.assertEqual(result["outcome"], "win")
        self.assertGreater(self.engine.get_paper_balance(), balance_after_open)

        from src.db.session import get_session
        from src.db.models import Trade
        from sqlalchemy import select

        async with get_session() as session:
            trade = (
                await session.execute(select(Trade).where(Trade.order_open_id == order.id))
            ).scalar_one()
        self.assertFalse(trade.is_open)
        self.assertEqual(trade.order_open_id, order.id)
        self.assertIsNotNone(trade.order_close_id)
        self.assertNotEqual(trade.order_close_id, order.id)

    async def test_close_paper_position_loss(self):
        """Закрытие short-позиции в минус: PnL отрицательный."""
        settings.trading_mode = "paper"
        await self.engine.initialize("binance")

        result = await self.engine.close_paper_position(
            symbol="ETH/USDT", side="short", entry_price=3000.0, amount=1.0,
            exit_price=3100.0, reason="stop_loss", entry_fee=1.0,
            holding_seconds=600, strategy_id="ema_cross",
        )
        self.assertIsNotNone(result)
        self.assertLess(result["pnl"], 0)
        self.assertEqual(result["outcome"], "loss")

    async def test_partial_close_reduces_position_instead_of_removing_it(self):
        """
        close_paper_position(amount=часть открытой позиции) — TP1/TP2 из
        3-уровневого take-profit — должен уменьшить объём позиции, а не
        стереть её целиком (иначе main.py на следующей итерации решил бы,
        что остаток закрыт в обход основного цикла).
        """
        settings.trading_mode = "paper"
        await self.engine.initialize("binance")

        # Уникальный символ — тестовая БД общая на сессию, а другие тесты
        # (например test_paper_create_order) тоже используют BTC/USDT и
        # оставляют там открытые позиции, что ломает точные ассерты по amount.
        symbol = "PARTIALCLOSE1/USDT"
        order = await self.engine.create_order(
            symbol=symbol, side="buy", amount=10.0, price=100.0,
            order_type="market", stop_loss=90.0, take_profit=130.0,
        )
        self.assertIsNotNone(order)

        result = await self.engine.close_paper_position(
            symbol=symbol, side="long", entry_price=100.0, amount=5.0,
            exit_price=110.0, reason="take_profit_1", entry_fee=order.fee / 2,
            holding_seconds=60, order_open_id=order.id,
        )
        self.assertIsNotNone(result)
        self.assertIn(symbol, self.engine.paper_positions)
        self.assertAlmostEqual(self.engine.paper_positions[symbol]["amount"], 5.0)

        # Второе частичное закрытие остатка (TP2) — тоже уменьшает, не удаляет
        result2 = await self.engine.close_paper_position(
            symbol=symbol, side="long", entry_price=100.0, amount=2.5,
            exit_price=120.0, reason="take_profit_2", entry_fee=order.fee / 4,
            holding_seconds=90, order_open_id=order.id,
        )
        self.assertIsNotNone(result2)
        self.assertAlmostEqual(self.engine.paper_positions[symbol]["amount"], 2.5)

        # Финальное закрытие остатка — теперь позиция должна исчезнуть
        result3 = await self.engine.close_paper_position(
            symbol=symbol, side="long", entry_price=100.0, amount=2.5,
            exit_price=130.0, reason="take_profit_3", entry_fee=order.fee / 4,
            holding_seconds=120, order_open_id=order.id,
        )
        self.assertIsNotNone(result3)
        self.assertNotIn(symbol, self.engine.paper_positions)

    async def test_restore_partially_closed_position(self):
        """
        _load_open_positions_from_db должен реконструировать частично
        закрытую позицию как открытую (с уменьшенным остатком и
        tp_hit_count = число уже сработавших уровней TP), а не считать её
        закрытой только потому, что её order_open_id уже встречается в
        каких-то Trade.
        """
        settings.trading_mode = "paper"
        await self.engine.initialize("binance")

        order = await self.engine.create_order(
            symbol="RESTORECOIN/USDT", side="buy", amount=8.0, price=50.0,
            order_type="market", stop_loss=45.0, take_profit=65.0,
        )
        self.assertIsNotNone(order)

        result = await self.engine.close_paper_position(
            symbol="RESTORECOIN/USDT", side="long", entry_price=50.0, amount=4.0,
            exit_price=55.0, reason="take_profit_1", entry_fee=order.fee / 2,
            holding_seconds=60, order_open_id=order.id,
        )
        self.assertIsNotNone(result)

        positions, _, _ = await self.engine._load_open_positions_from_db(is_paper=True)
        self.assertIsNotNone(positions)
        self.assertIn("RESTORECOIN/USDT", positions)
        restored = positions["RESTORECOIN/USDT"]
        self.assertAlmostEqual(restored["amount"], 4.0)
        self.assertEqual(restored["tp_hit_count"], 1)
        # После TP1 остаток должен восстановиться с SL в безубытке
        self.assertAlmostEqual(restored["stop_loss"], restored["entry_price"])

    async def test_restore_open_short_position(self):
        """
        _load_open_positions_from_db раньше запрашивал только Order.side ==
        "buy" — открытая SHORT-позиция (side="sell", без предшествующего
        buy на этот символ) при рестарте молча пропадала из paper_positions
        без единой закрывающей Trade-записи, поэтому никогда не появлялась
        и в истории закрытых сделок. Проверяем, что short восстанавливается
        с правильной стороной/объёмом, и что его условная стоимость не
        списывается с paper_balance (маржа при открытии short не
        резервируется — см. _execute_paper_order).
        """
        settings.trading_mode = "paper"
        settings.startup_capital_usdt = 10000.0
        await self.engine.initialize("binance")
        balance_before = self.engine.get_paper_balance()

        order = await self.engine.create_order(
            symbol="RESTORESHORT1/USDT", side="sell", amount=2.0, price=100.0,
            order_type="market", stop_loss=110.0, take_profit=80.0,
        )
        self.assertIsNotNone(order)
        self.assertIn("RESTORESHORT1/USDT", self.engine.paper_positions)
        # Открытие short не резервирует маржу — баланс не должен меняться
        # (кроме, возможно, комиссии, которая тоже не списывается при открытии).
        self.assertAlmostEqual(self.engine.get_paper_balance(), balance_before, places=6)

        positions, realized_pnl, cost_basis = await self.engine._load_open_positions_from_db(is_paper=True)
        self.assertIsNotNone(positions)
        self.assertIn("RESTORESHORT1/USDT", positions)
        restored = positions["RESTORESHORT1/USDT"]
        self.assertEqual(restored["side"], "short")
        self.assertAlmostEqual(restored["amount"], 2.0)
        # ~100 минус paper-слиппедж на sell-стороне, не ровно 100.
        self.assertAlmostEqual(restored["entry_price"], 100.0, delta=1.0)
        # cost_basis здесь может быть ненулевым из-за прочих открытых long-
        # позиций в общей тестовой БД (см. docstring теста класса) — сам факт,
        # что restored_balance ниже совпадает с balance_before, доказывает,
        # что именно ЭТОТ short не внёс вклад в cost_basis.

        # Симулируем полный рестарт процесса новым экземпляром движка.
        engine2 = ExecutionEngine()
        try:
            await engine2.initialize("binance")
            self.assertIn("RESTORESHORT1/USDT", engine2.paper_positions)
            self.assertEqual(engine2.paper_positions["RESTORESHORT1/USDT"]["side"], "short")
            self.assertAlmostEqual(engine2.get_paper_balance(), balance_before, places=2)
        finally:
            await engine2.close()

    async def test_real_order_registers_position_with_correct_fee_and_price(self):
        """
        _execute_real_order раньше писал в БД сырой ccxt fee-dict вместо
        числа и брал order["price"] (у market-ордеров обычно None) вместо
        order["average"]; открытая реальная позиция нигде не регистрировалась,
        поэтому SL/TP по ней никогда не проверялись.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "binance"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "ex-1", "filled": 0.1, "price": None, "average": 50000.0,
            "fee": {"cost": 5.0, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="BTC/USDT", side="buy", amount=0.1, price=50000.0,
            order_type="market", stop_loss=48000.0, take_profit=55000.0,
        )
        self.assertIsNotNone(order)
        self.assertEqual(float(order.filled_price), 50000.0)
        self.assertEqual(float(order.fee), 5.0)
        self.assertIn("BTC/USDT", self.engine.real_positions)
        self.assertEqual(self.engine.real_positions["BTC/USDT"]["entry_price"], 50000.0)

    async def test_execute_real_order_polls_fetch_order_when_create_response_is_a_skeleton(self):
        """
        Bybit v5 на создание маркет-ордера отдаёт ТОЛЬКО orderId — цена,
        объём и комиссия исполнения туда не попадают (сопоставление
        асинхронное). average/price в таком ответе всегда None — без
        догоняющего fetch_order exit_price/fill_price падал бы на
        entry_price/запрошенную цену, а комиссия терялась бы — из-за чего
        PnL ЛЮБОЙ закрытой на Bybit сделки получался ровно 0.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "bybit-order-1", "filled": None, "price": None, "average": None, "fee": None,
        }
        self.engine.exchange.fetch_order = AsyncMock(return_value={
            "id": "bybit-order-1", "filled": 0.2, "price": None, "average": 3000.0,
            "fee": {"cost": 1.5, "currency": "USDT"},
        })

        with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
            order = await self.engine.create_order(
                symbol="ETH/USDT", side="buy", amount=0.2, price=3000.0,
                order_type="market", stop_loss=2900.0, take_profit=3200.0,
            )

        self.assertIsNotNone(order)
        self.engine.exchange.fetch_order.assert_called_once_with("bybit-order-1", "ETH/USDT")
        self.assertEqual(float(order.filled_price), 3000.0)
        self.assertEqual(float(order.fee), 1.5)
        self.assertEqual(self.engine.real_positions["ETH/USDT"]["entry_price"], 3000.0)

    async def test_fetch_confirmed_order_gives_up_after_all_attempts_fail(self):
        """Если fetch_order так и не вернул цену — используем то, что было (не зависаем/не падаем)."""
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_order = AsyncMock(return_value={
            "id": "stuck-order", "filled": None, "price": None, "average": None,
        })
        skeleton = {"id": "stuck-order", "filled": None, "price": None, "average": None}

        with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
            result = await self.engine._fetch_confirmed_order(skeleton, "BTC/USDT", attempts=2, delay=0.01)

        self.assertEqual(result, skeleton)
        self.assertEqual(self.engine.exchange.fetch_order.call_count, 2)

    async def test_execute_real_order_rejects_unconfirmed_fill(self):
        """
        Bybit может вернуть orderId даже для ордера, который на самом деле
        НЕ исполнился (отклонён движком сопоставления) — сам факт ответа
        без исключения не значит, что сделка реально произошла. Раньше
        бот всё равно регистрировал позицию — реального актива на бирже
        не было (ни самого баланса, ни истории операций), а закрыть такую
        фантомную позицию было невозможно ("Insufficient balance" на
        каждой попытке, бесконечно).
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "phantom-order-1", "filled": None, "price": None, "average": None, "fee": None,
        }
        self.engine.exchange.fetch_order = AsyncMock(return_value={
            "id": "phantom-order-1", "filled": None, "price": None, "average": None,
            "status": "rejected",
        })

        with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
            order = await self.engine.create_order(
                symbol="PHANTOM1/USDT", side="buy", amount=100.0, price=0.5,
                order_type="market", stop_loss=0.45, take_profit=0.6,
            )

        self.assertIsNone(order)
        self.assertNotIn("PHANTOM1/USDT", self.engine.real_positions)

    async def test_close_real_position_rejects_unconfirmed_fill(self):
        """Симметричный случай на закрытии: неподтверждённое исполнение не засчитывается как закрытие."""
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"PHANTOM2": 100.0}, "PHANTOM2": {"free": 100.0, "used": 0, "total": 100.0}}
        )
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "phantom-close-1", "filled": None, "price": None, "average": None, "fee": None,
        }
        self.engine.exchange.fetch_order = AsyncMock(return_value={
            "id": "phantom-close-1", "filled": None, "price": None, "average": None,
        })

        with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
            result = await self.engine.close_real_position(
                symbol="PHANTOM2/USDT", side="long", entry_price=0.5, amount=100.0,
                reason="stop_loss", entry_fee=0.1, holding_seconds=60,
            )

        self.assertIsNone(result)

    async def test_close_real_position_reconciles_phantom_when_exchange_confirms_zero_balance(self):
        """
        Когда И наша предварительная проверка баланса, И сама попытка
        продажи независимо согласны, что на бирже 0 актива — это почти
        наверняка фантомная позиция (открывающий BUY на самом деле не
        исполнился). Позиция должна сняться с учёта, а не зависать в
        вечных повторных попытках продать то, чего никогда не было —
        исходный открывающий ордер помечается rejected, чтобы реконструкция
        при рестарте не воссоздала фантом заново.
        """
        from src.db.models import Order
        from src.db.session import get_session

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "recon-open-1", "filled": 100.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }
        opening_order = await self.engine.create_order(
            symbol="RECONCILE1/USDT", side="buy", amount=100.0, price=0.5,
            order_type="market", stop_loss=0.45, take_profit=0.6,
        )
        self.assertIsNotNone(opening_order)
        self.assertIn("RECONCILE1/USDT", self.engine.real_positions)

        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"RECONCILE1": 0.0}, "RECONCILE1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_sell_order = AsyncMock(
            side_effect=Exception('bybit {"retCode":170131,"retMsg":"Insufficient balance."}')
        )

        result = await self.engine.close_real_position(
            symbol="RECONCILE1/USDT", side="long", entry_price=0.5, amount=100.0,
            reason="stop_loss", entry_fee=0.05, holding_seconds=60,
            order_open_id=opening_order.id,
        )

        self.assertIsNone(result)
        self.assertNotIn("RECONCILE1/USDT", self.engine.real_positions)

        async with get_session() as session:
            refreshed = await session.get(Order, opening_order.id)
            self.assertEqual(refreshed.status, "rejected")

    async def test_close_real_position_reconciles_phantom_when_remaining_dust_below_minimum(self):
        """
        Второй вариант того же класса бага: доступный остаток положительный
        (не 0), но настолько мал, что биржа отклоняет продажу как ниже
        минимального торгуемого объёма ("precision"/"minimum" в тексте
        ошибки, как реально было у AVAX/USDT: наш учёт 416.54, доступно
        0.00046). Продать такую пыль невозможно в принципе — повторные
        попытки не помогут, позиция должна сняться с учёта так же, как и
        при available == 0.
        """
        from src.db.models import Order
        from src.db.session import get_session

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "dust-open-1", "filled": 100.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }
        opening_order = await self.engine.create_order(
            symbol="DUST1/USDT", side="buy", amount=100.0, price=0.5,
            order_type="market", stop_loss=0.45, take_profit=0.6,
        )
        self.assertIsNotNone(opening_order)
        self.assertIn("DUST1/USDT", self.engine.real_positions)

        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"DUST1": 0.0001}, "DUST1": {"free": 0.0001, "used": 0, "total": 0.0001}}
        )
        self.engine.exchange.create_market_sell_order = AsyncMock(
            side_effect=Exception(
                "bybit amount of DUST1/USDT must be greater than minimum amount precision of 0.001"
            )
        )

        result = await self.engine.close_real_position(
            symbol="DUST1/USDT", side="long", entry_price=0.5, amount=100.0,
            reason="stop_loss", entry_fee=0.05, holding_seconds=60,
            order_open_id=opening_order.id,
        )

        self.assertIsNone(result)
        self.assertNotIn("DUST1/USDT", self.engine.real_positions)

        async with get_session() as session:
            refreshed = await session.get(Order, opening_order.id)
            self.assertEqual(refreshed.status, "rejected")

    async def test_execute_real_order_skips_below_exchange_minimum_cost(self):
        """
        Реальный сценарий: DATA/USDT, retCode 170140 "Order value exceeded
        lower limit" — рассчитанный от текущего (малого) баланса объём
        оказался ниже минимальной стоимости ордера на бирже. Такой ордер не
        должен даже отправляться на биржу (и падать оттуда ERROR'ом,
        выглядящим как поломка) — если markets[symbol].limits.cost.min
        известен, проверяем ДО отправки и просто пропускаем сигнал.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.markets = {
            "DATA1/USDT": {"limits": {"amount": {"min": 1.0}, "cost": {"min": 5.0}}}
        }

        # amount=1.5 * price=0.05 = 0.075 USDT — заметно ниже cost.min=5.0
        order = await self.engine.create_order(
            symbol="DATA1/USDT", side="buy", amount=1.5, price=0.05, order_type="market",
        )

        self.assertIsNone(order)
        self.engine.exchange.create_market_buy_order.assert_not_called()
        self.assertNotIn("DATA1/USDT", self.engine.real_positions)

    async def test_execute_real_order_ignores_unusable_markets_metadata(self):
        """
        Если exchange.markets отсутствует/не словарь (как в большинстве
        существующих тестов с AsyncMock()-биржей без явного .markets) —
        проверка минимумов не должна ломать обычную отправку ордера.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "mm-open-1", "filled": 10.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="MARKETSMOCK1/USDT", side="buy", amount=10.0, price=0.5, order_type="market",
        )

        self.assertIsNotNone(order)
        self.engine.exchange.create_market_buy_order.assert_called_once()

    async def test_reconcile_real_positions_purges_dust_and_returns_balance(self):
        """
        reconcile_real_positions() — ПРОАКТИВНАЯ сверка (раньше сверка
        происходила только реактивно, в момент попытки закрыть позицию —
        см. close_real_position — то есть только когда сработает SL/TP; если
        цена никогда до них не доходила, испорченная позиция могла висеть
        неограниченно долго, искажая equity/просадку каждую итерацию).
        Один fetch_balance() должен: (1) снять с учёта позицию, для которой
        реальный остаток ниже торгуемого минимума, (2) НЕ трогать позицию с
        достаточным остатком, (3) вернуть актуальный USDT-баланс из того же
        запроса.
        """
        from src.db.models import Order
        from src.db.session import get_session

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.markets = {
            "RECONDUST1/USDT": {"limits": {"amount": {"min": 0.001}}},
        }
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "recondust-open-1", "filled": 416.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }
        dust_order = await self.engine.create_order(
            symbol="RECONDUST1/USDT", side="buy", amount=416.0, price=0.5, order_type="market",
        )
        self.assertIsNotNone(dust_order)

        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "reconok-open-1", "filled": 10.0, "price": None, "average": 1.0,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }
        ok_order = await self.engine.create_order(
            symbol="RECONOK1/USDT", side="buy", amount=10.0, price=1.0, order_type="market",
        )
        self.assertIsNotNone(ok_order)

        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 123.45, "RECONDUST1": 0.00046, "RECONOK1": 10.0},
            "RECONDUST1": {"free": 0.00046, "used": 0, "total": 0.00046},
            "RECONOK1": {"free": 10.0, "used": 0, "total": 10.0},
        })

        balance = await self.engine.reconcile_real_positions()

        self.assertAlmostEqual(balance, 123.45)
        self.assertNotIn("RECONDUST1/USDT", self.engine.real_positions)
        self.assertIn("RECONOK1/USDT", self.engine.real_positions)

        async with get_session() as session:
            refreshed = await session.get(Order, dust_order.id)
            self.assertEqual(refreshed.status, "rejected")

    async def test_reconcile_phantom_position_also_clears_risk_manager_count(self):
        """
        _reconcile_phantom_position() снимает позицию с учёта в
        execution_engine.real_positions, но раньше не уведомляла об этом
        risk_manager — risk_manager.state.open_positions_count оставался
        завышенным навсегда (единственные места, вызывающие
        on_position_closed — POST /positions/close и обычное закрытие по
        SL/TP в _check_position_exit — про реконсиляцию не знают). Живой
        симптом в проде: символ числился в risk_state.open_positions, но
        отсутствовал среди реально отслеживаемых позиций — искусственно
        занижая доступное число новых сделок относительно
        max_open_positions.
        """
        from src.risk.risk_manager import risk_manager as global_risk_manager

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "riskcount-open-1", "filled": 100.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }
        opening_order = await self.engine.create_order(
            symbol="RISKCOUNT1/USDT", side="buy", amount=100.0, price=0.5, order_type="market",
        )
        self.assertIsNotNone(opening_order)
        # Симулируем то, что в реальном цикле делает main.py после открытия.
        global_risk_manager.on_position_added("RISKCOUNT1/USDT", 5.0)
        self.assertIn("RISKCOUNT1/USDT", global_risk_manager.state.open_positions)

        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"RISKCOUNT1": 0.0}, "RISKCOUNT1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_sell_order = AsyncMock(
            side_effect=Exception('bybit {"retCode":170131,"retMsg":"Insufficient balance."}')
        )

        result = await self.engine.close_real_position(
            symbol="RISKCOUNT1/USDT", side="long", entry_price=0.5, amount=100.0,
            reason="stop_loss", entry_fee=0.05, holding_seconds=60, order_open_id=opening_order.id,
        )

        self.assertIsNone(result)
        self.assertNotIn("RISKCOUNT1/USDT", global_risk_manager.state.open_positions)

    async def test_close_real_position_uses_actual_fill_price(self):
        """
        close_real_position должен считать PnL по фактической цене исполнения
        закрывающего ордера, а не по цене, переданной вызывающим кодом
        (последняя известная цена в торговом цикле может отличаться от
        реального fill на бирже).
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "binance"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "ex-2", "filled": 0.1, "price": None, "average": 52000.0,
            "fee": {"cost": 5.2, "currency": "USDT"},
        }

        result = await self.engine.close_real_position(
            symbol="BTC/USDT", side="long", entry_price=50000.0, amount=0.1,
            reason="take_profit", entry_fee=5.0, holding_seconds=60,
            exit_price=99999.0,  # должен игнорироваться
        )
        self.assertIsNotNone(result)
        expected_pnl = (52000.0 - 50000.0) * 0.1 - 5.0 - 5.2
        self.assertAlmostEqual(result["pnl"], expected_pnl, places=6)
        self.assertNotIn("BTC/USDT", self.engine.real_positions)

    async def test_close_real_position_polls_fetch_order_when_response_is_a_skeleton(self):
        """
        Тот же баг "PnL закрытой сделки = 0" на закрывающей стороне: Bybit
        v5 отдаёт на создание ордера только orderId, без average/price/fee —
        без догоняющего fetch_order exit_price падал бы на entry_price и
        PnL получался бы ровно 0 (минус потерянная комиссия) для КАЖДОЙ
        закрытой на Bybit сделки.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "bybit-close-1", "filled": None, "price": None, "average": None, "fee": None,
        }
        self.engine.exchange.fetch_order = AsyncMock(return_value={
            "id": "bybit-close-1", "filled": 0.1, "price": None, "average": 52000.0,
            "fee": {"cost": 5.2, "currency": "USDT"},
        })

        with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
            result = await self.engine.close_real_position(
                symbol="BTC/USDT", side="long", entry_price=50000.0, amount=0.1,
                reason="take_profit", entry_fee=5.0, holding_seconds=60,
            )

        self.assertIsNotNone(result)
        expected_pnl = (52000.0 - 50000.0) * 0.1 - 5.0 - 5.2
        self.assertAlmostEqual(result["pnl"], expected_pnl, places=6)
        self.assertNotEqual(result["pnl"], 0)

    async def test_close_real_position_clamps_to_available_balance(self):
        """
        Отслеживаемый объём позиции — оценка (комиссии/округление лота на
        бирже могут понемногу расходиться с реальным остатком). Раньше
        close_real_position всегда пытался продать полный отслеживаемый
        объём — если на бирже реально чуть меньше (частый случай после
        комиссии, удержанной из base-валюты), ордер падал с "Insufficient
        balance", и позиция навсегда зависала открытой. Теперь при
        расхождении продаём фактически доступный остаток.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"ARB": 99.9}, "ARB": {"free": 99.9, "used": 0, "total": 99.9}}
        )
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "ex-clamp-1", "filled": 99.9, "price": None, "average": 0.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }

        result = await self.engine.close_real_position(
            symbol="ARB/USDT", side="long", entry_price=0.5, amount=100.0,
            reason="stop_loss", entry_fee=0.1, holding_seconds=60,
        )
        self.assertIsNotNone(result)
        self.engine.exchange.create_market_sell_order.assert_called_once_with("ARB/USDT", 99.9)

    async def test_close_real_position_sells_full_amount_when_balance_sufficient(self):
        """Когда доступного баланса достаточно, объём ордера не подрезается."""
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"ARB": 150.0}, "ARB": {"free": 150.0, "used": 0, "total": 150.0}}
        )
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "ex-clamp-2", "filled": 100.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }

        result = await self.engine.close_real_position(
            symbol="ARB/USDT", side="long", entry_price=0.5, amount=100.0,
            reason="stop_loss", entry_fee=0.1, holding_seconds=60,
        )
        self.assertIsNotNone(result)
        self.engine.exchange.create_market_sell_order.assert_called_once_with("ARB/USDT", 100.0)

    async def test_close_real_position_logs_available_balance_on_failure(self):
        """
        Когда биржа реально отдаёт 0 доступного баланса (позиция уже продана
        не через бота, запрос ушёл не в тот account type и т.п.) — клэмп
        (0 < available < amount) не помогает, продажа падает с той же
        "Insufficient balance". Раньше в этом случае лог ошибки не показывал
        вообще ничего о том, что сам бот считает доступным по своей
        проверке — расследовать было нечем, кроме голого текста ошибки
        биржи.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"1INCH": 0.0}, "1INCH": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_sell_order = AsyncMock(
            side_effect=Exception('bybit {"retCode":170131,"retMsg":"Insufficient balance."}')
        )

        with self.assertLogs("src.execution.executor", level="ERROR") as logs:
            result = await self.engine.close_real_position(
                symbol="1INCH/USDT", side="long", entry_price=0.5, amount=100.0,
                reason="stop_loss", entry_fee=0.1, holding_seconds=60,
            )

        self.assertIsNone(result)
        self.engine.exchange.create_market_sell_order.assert_called_once_with("1INCH/USDT", 100.0)
        self.assertTrue(any("доступно на бирже: 0.0" in msg for msg in logs.output))

    async def test_close_real_position_reconciles_when_order_value_below_lower_limit(self):
        """
        Реальный инцидент (прод, реальный счёт Bybit, SUI/USDT): доступного
        объёма было БОЛЬШЕ отслеживаемого (available > amount — старая
        проверка available < amount вообще не срабатывала), но биржа всё
        равно отклоняла продажу с retCode 170140 "Order value exceeded
        lower limit" — стоимость позиции в USDT ниже минимальной для пары.
        Без отдельного распознавания этого случая бот пытался закрыть её
        снова каждую итерацию бесконечно (наблюдалось ~26 минут подряд).
        Продать такую позицию одним ордером в принципе невозможно (дробить
        её на более мелкие ордера — только уменьшает стоимость каждого), и
        сверка должна снять её с учёта так же, как классическую "пыль".
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"LOWERLIMIT1": 0.39879}, "LOWERLIMIT1": {"free": 0.39879, "used": 0, "total": 0.39879}}
        )
        self.engine.exchange.create_market_sell_order = AsyncMock(
            side_effect=Exception(
                'bybit {"retCode":170140,"retMsg":"Order value exceeded lower limit.",'
                '"result":{},"retExtInfo":{},"time":1787843738428}'
            )
        )

        result = await self.engine.close_real_position(
            symbol="LOWERLIMIT1/USDT", side="long", entry_price=0.7664, amount=0.39339472,
            reason="stop_loss", entry_fee=0.001, holding_seconds=60, order_open_id=None,
        )

        self.assertIsNone(result)
        self.assertNotIn("LOWERLIMIT1/USDT", self.engine.real_positions)

    async def test_restore_positions_separates_paper_and_real(self):
        """
        Order.exchange_id раньше был общим для paper и real (одно и то же
        имя "binance"), поэтому Exchange.is_paper на этом ряду отражал
        только то, какой режим создал его первым — восстановление open
        positions при рестарте не могло отличить paper-ордера от реальных.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "binance"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "ex-3", "filled": 0.2, "price": None, "average": 3000.0,
            "fee": {"cost": 1.5, "currency": "USDT"},
        }
        order = await self.engine.create_order(
            symbol="ETH/USDT", side="buy", amount=0.2, price=3000.0,
            order_type="market", stop_loss=2900.0, take_profit=3200.0,
        )
        self.assertIsNotNone(order)

        real_positions, _, _ = await self.engine._load_open_positions_from_db(is_paper=False)
        self.assertIsNotNone(real_positions)
        self.assertIn("ETH/USDT", real_positions)

        paper_positions, _, _ = await self.engine._load_open_positions_from_db(is_paper=True)
        self.assertIsNotNone(paper_positions)
        self.assertNotIn("ETH/USDT", paper_positions)

    async def test_execute_real_order_rejects_sell_side_short_open(self):
        """
        Реальный инцидент (прод, реальный счёт Bybit, ENA/USDT,
        bb_strategy): сигнал side="short" дошёл до _execute_real_order и
        создал market SELL, который реально ИСПОЛНИЛСЯ на бирже (аккаунт
        держал ENA не через этого бота) — распродав реальный актив и
        оставив в БД "осиротевший" sell-ордер, который на каждом рестарте
        реконструировался как незакрываемая "short-позиция" (close_real_position
        принципиально отклоняет side != long). На споте нет встроенного
        шорта — _execute_real_order должен отклонять side="sell" СРАЗУ, не
        доходя до биржи вообще.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()

        order = await self.engine.create_order(
            symbol="ENASHORT1/USDT", side="sell", amount=1000.0, price=0.15, order_type="market",
        )
        self.assertIsNone(order)
        self.engine.exchange.create_market_sell_order.assert_not_called()
        self.assertNotIn("ENASHORT1/USDT", self.engine.real_positions)

    async def test_restore_real_positions_skips_orphaned_short_order(self):
        """
        Уже существующий в БД "осиротевший" real short-ордер (от бага до
        добавления защиты выше) не должен реконструироваться как открытая
        позиция при каждом рестарте — закрыть его всё равно принципиально
        невозможно (close_real_position отклоняет side != long), только
        засоряет логи той же неустранимой ошибкой на каждой попытке.
        """
        from src.db.session import get_session
        from src.db.models import Order

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        async with get_session() as session:
            exchange_id, symbol_id = await self.engine._resolve_symbol_id(session, "ORPHANSHORT1/USDT")
            orphan = Order(
                exchange_id=exchange_id, symbol_id=symbol_id,
                side="sell", order_type="market", amount=1000.0, price=0.15,
                status="filled", filled_amount=1000.0, filled_price=0.15,
                fee=0.01, order_id_exchange="orphan-short-1", client_order_id="orphanshort1",
            )
            session.add(orphan)
            await session.commit()

        real_positions, _, _ = await self.engine._load_open_positions_from_db(is_paper=False)
        self.assertIsNotNone(real_positions)
        self.assertNotIn("ORPHANSHORT1/USDT", real_positions)

        # Paper-режим по-прежнему поддерживает шорт (только real сломан) —
        # тот же самый sell-ордер, но под paper-биржей, должен по-прежнему
        # восстанавливаться как открытая short-позиция.
        self.engine.is_paper = True
        async with get_session() as session:
            exchange_id, symbol_id = await self.engine._resolve_symbol_id(session, "ORPHANSHORT1/USDT")
            paper_short = Order(
                exchange_id=exchange_id, symbol_id=symbol_id,
                side="sell", order_type="market", amount=1000.0, price=0.15,
                status="filled", filled_amount=1000.0, filled_price=0.15,
                fee=0.01, order_id_exchange=None, client_order_id="paperorphanshort1",
            )
            session.add(paper_short)
            await session.commit()

        paper_positions, _, _ = await self.engine._load_open_positions_from_db(is_paper=True)
        self.assertIsNotNone(paper_positions)
        self.assertIn("ORPHANSHORT1/USDT", paper_positions)
        self.assertEqual(paper_positions["ORPHANSHORT1/USDT"]["side"], "short")

    async def test_real_buy_with_base_currency_fee_reduces_tracked_amount(self):
        """
        На споте комиссия обычно списывается из полученного актива: купили
        100 1INCH, но комиссия 0.1 1INCH удержана биржей — реально на счету
        осталось 99.9. Раньше real_positions запоминал полный запрошенный
        объём без вычета комиссии, и первая же попытка закрыть позицию тем
        же объёмом падала на бирже с "Insufficient balance".
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "ex-fee-1", "filled": 100.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.1, "currency": "1INCH"},
        }

        order = await self.engine.create_order(
            symbol="1INCH/USDT", side="buy", amount=100.0, price=0.5,
            order_type="market", stop_loss=0.45, take_profit=0.6,
        )
        self.assertIsNotNone(order)
        self.assertAlmostEqual(self.engine.real_positions["1INCH/USDT"]["amount"], 99.9)

    async def test_real_buy_with_quote_currency_fee_keeps_full_amount(self):
        """Комиссия в USDT (квота) не уменьшает объём удерживаемого актива."""
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "ex-fee-2", "filled": 100.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="QUOTEFEE1/USDT", side="buy", amount=100.0, price=0.5,
            order_type="market", stop_loss=0.45, take_profit=0.6,
        )
        self.assertIsNotNone(order)
        self.assertAlmostEqual(self.engine.real_positions["QUOTEFEE1/USDT"]["amount"], 100.0)

    async def test_restore_real_position_subtracts_entry_fee(self):
        """
        _load_open_positions_from_db(is_paper=False) должен вычитать
        комиссию покупки из восстановленного объёма при рестарте — та же
        логика, что и при живом исполнении (см.
        test_real_buy_with_base_currency_fee_reduces_tracked_amount), иначе
        баг воспроизводился бы заново после каждого рестарта бота.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "ex-fee-3", "filled": 100.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.1, "currency": "1INCH"},
        }
        order = await self.engine.create_order(
            symbol="RESTOREFEE1/USDT", side="buy", amount=100.0, price=0.5,
            order_type="market", stop_loss=0.45, take_profit=0.6,
        )
        self.assertIsNotNone(order)

        real_positions, _, _ = await self.engine._load_open_positions_from_db(is_paper=False)
        self.assertAlmostEqual(real_positions["RESTOREFEE1/USDT"]["amount"], 99.9)

    async def test_restore_paper_position_does_not_subtract_fee(self):
        """
        Paper-комиссия условная и списывается только с cash-баланса
        (paper_balance), а не с количества актива — реконструкция paper-
        позиций не должна вычитать её из amount, в отличие от real.
        """
        settings.trading_mode = "paper"
        await self.engine.initialize("binance")

        order = await self.engine.create_order(
            symbol="RESTOREPAPERFEE1/USDT", side="buy", amount=100.0, price=0.5,
            order_type="market", stop_loss=0.45, take_profit=0.6,
        )
        self.assertIsNotNone(order)
        self.assertGreater(float(order.fee), 0)

        paper_positions, _, _ = await self.engine._load_open_positions_from_db(is_paper=True)
        self.assertAlmostEqual(paper_positions["RESTOREPAPERFEE1/USDT"]["amount"], 100.0)


class TestTelegramSignalParser(unittest.TestCase):
    """Тесты для парсера Telegram сигналов (живой parse_with_regex из channel_monitor —
    src/telegram/signal_parser.py был неиспользуемым дублем и был удалён)."""

    def setUp(self):
        from src.telegram.channel_monitor import parse_with_regex
        self.parse = parse_with_regex

    def test_parse_btc_long(self):
        """Парсинг BTC/USDT Long сигнала."""
        result = self.parse("BTC/USDT Long 69000 SL 68000 TP 72000")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "BTC/USDT")
        self.assertEqual(result["side"], "long")
        self.assertEqual(result["entry"], 69000.0)

    def test_parse_eth_short(self):
        """Парсинг ETH/USDT Short сигнала."""
        result = self.parse("ETH/USDT Short | Entry: 3500 | Stop: 3600 | Target: 3200")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "ETH/USDT")
        self.assertEqual(result["side"], "short")
        self.assertEqual(result["entry"], 3500.0)

    def test_parse_no_signal(self):
        """Текст без сигнала."""
        result = self.parse("Привет! Как дела?")
        self.assertIsNone(result)

    def test_parse_pair_without_slash_is_normalized(self):
        """"BTCUSDT" (без слэша) должен нормализоваться в ccxt-формат "BTC/USDT"."""
        result = self.parse("BTCUSDT LONG 1.85 SL 1.70 TP 2.10")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "BTC/USDT")


class TestLlmSignalParser(unittest.IsolatedAsyncioTestCase):
    """
    LLM-фолбэк парсинга (перенос из clonerbot: parser/llm_parser.py) —
    подключается, когда регулярки не смогли разобрать сообщение. Мокаем
    Anthropic-клиент, чтобы не делать реальных сетевых запросов.
    """

    def setUp(self):
        self._saved = {
            "telegram_llm_fallback_enabled": settings.telegram_llm_fallback_enabled,
            "anthropic_api_key": settings.anthropic_api_key,
        }

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(settings, key, value)
        import src.telegram.llm_parser as llm_parser_module
        llm_parser_module._client = None

    def _mock_tool_use_response(self, data: dict):
        block = MagicMock()
        block.type = "tool_use"
        block.name = "emit_signal"
        block.input = data
        resp = MagicMock()
        resp.content = [block]
        return resp

    async def test_disabled_returns_none_without_calling_api(self):
        from src.telegram.llm_parser import parse_with_llm
        settings.telegram_llm_fallback_enabled = False
        settings.anthropic_api_key = "test-key"
        result = await parse_with_llm("BTC to the moon, going long soon maybe")
        self.assertIsNone(result)

    async def test_parses_valid_signal(self):
        import src.telegram.llm_parser as llm_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.anthropic_api_key = "test-key"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=self._mock_tool_use_response({
            "is_signal": True, "base": "BTC", "quote": "USDT", "side": "long",
            "entry": 69000.0, "take_profits": [70000.0, 72000.0], "stop_loss": 68000.0,
            "confidence": 0.9,
        }))
        llm_parser_module._client = mock_client

        result = await llm_parser_module.parse_with_llm("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "BTC/USDT")
        self.assertEqual(result["side"], "long")
        self.assertEqual(result["entry"], 69000.0)
        self.assertEqual(result["sl"], 68000.0)
        self.assertEqual(result["tp"], 72000.0)  # long -> максимальный TP

    async def test_low_confidence_is_rejected(self):
        import src.telegram.llm_parser as llm_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.anthropic_api_key = "test-key"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=self._mock_tool_use_response({
            "is_signal": True, "base": "BTC", "side": "long", "entry": 69000.0,
            "take_profits": [], "stop_loss": None, "confidence": 0.2,
        }))
        llm_parser_module._client = mock_client

        result = await llm_parser_module.parse_with_llm("может быть покупать биток?")
        self.assertIsNone(result)

    async def test_not_a_signal_returns_none(self):
        import src.telegram.llm_parser as llm_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.anthropic_api_key = "test-key"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=self._mock_tool_use_response({
            "is_signal": False, "confidence": 0.95,
        }))
        llm_parser_module._client = mock_client

        result = await llm_parser_module.parse_with_llm("сегодня биток вырос на 3%, отличный день")
        self.assertIsNone(result)

    async def test_api_error_returns_none_not_raises(self):
        import src.telegram.llm_parser as llm_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.anthropic_api_key = "test-key"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=RuntimeError("network down"))
        llm_parser_module._client = mock_client

        result = await llm_parser_module.parse_with_llm("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNone(result)

    async def test_parse_telegram_signal_falls_back_to_llm_when_regex_fails(self):
        import src.telegram.llm_parser as llm_parser_module
        from src.telegram.channel_monitor import parse_telegram_signal
        settings.telegram_llm_fallback_enabled = True
        settings.anthropic_api_key = "test-key"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=self._mock_tool_use_response({
            "is_signal": True, "base": "ETH", "quote": "USDT", "side": "short",
            "entry": 3500.0, "take_profits": [3300.0, 3200.0], "stop_loss": 3600.0,
            "confidence": 0.8,
        }))
        llm_parser_module._client = mock_client

        # Регулярки не найдут здесь пару/направление в явном виде — только LLM.
        result = await parse_telegram_signal("шортим эфир около 3500, стоп 3600, цели 3300 и 3200")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "ETH/USDT")
        self.assertEqual(result["side"], "short")
        self.assertEqual(result["tp"], 3200.0)  # short -> минимальный TP


class TestGeminiSignalParser(unittest.IsolatedAsyncioTestCase):
    """
    Gemini LLM-фолбэк парсинга — второй уровень, после Anthropic (см.
    src/telegram/gemini_parser.py и TestLlmSignalParser выше). Мокаем
    google-genai клиент, чтобы не делать реальных сетевых запросов.
    """

    def setUp(self):
        self._saved = {
            "telegram_llm_fallback_enabled": settings.telegram_llm_fallback_enabled,
            "anthropic_api_key": settings.anthropic_api_key,
            "gemini_api_key": settings.gemini_api_key,
        }

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(settings, key, value)
        import src.telegram.gemini_parser as gemini_parser_module
        gemini_parser_module._client = None
        import src.telegram.llm_parser as llm_parser_module
        llm_parser_module._client = None

    def _mock_response(self, data: dict):
        import json
        resp = MagicMock()
        resp.text = json.dumps(data)
        return resp

    async def test_disabled_returns_none_without_calling_api(self):
        from src.telegram.gemini_parser import parse_with_gemini
        settings.telegram_llm_fallback_enabled = False
        settings.gemini_api_key = "test-key"
        result = await parse_with_gemini("BTC to the moon, going long soon maybe")
        self.assertIsNone(result)

    async def test_no_api_key_returns_none_without_calling_api(self):
        from src.telegram.gemini_parser import parse_with_gemini
        settings.telegram_llm_fallback_enabled = True
        settings.gemini_api_key = None
        result = await parse_with_gemini("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNone(result)

    async def test_parses_valid_signal(self):
        import src.telegram.gemini_parser as gemini_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.gemini_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=self._mock_response({
            "is_signal": True, "base": "BTC", "quote": "USDT", "side": "long",
            "entry": 69000.0, "take_profits": [70000.0, 72000.0], "stop_loss": 68000.0,
            "confidence": 0.9,
        }))
        gemini_parser_module._client = mock_client

        result = await gemini_parser_module.parse_with_gemini("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "BTC/USDT")
        self.assertEqual(result["side"], "long")
        self.assertEqual(result["entry"], 69000.0)
        self.assertEqual(result["sl"], 68000.0)
        self.assertEqual(result["tp"], 72000.0)  # long -> максимальный TP

    async def test_low_confidence_is_rejected(self):
        import src.telegram.gemini_parser as gemini_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.gemini_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=self._mock_response({
            "is_signal": True, "base": "BTC", "side": "long", "entry": 69000.0,
            "take_profits": [], "stop_loss": None, "confidence": 0.2,
        }))
        gemini_parser_module._client = mock_client

        result = await gemini_parser_module.parse_with_gemini("может быть покупать биток?")
        self.assertIsNone(result)

    async def test_not_a_signal_returns_none(self):
        import src.telegram.gemini_parser as gemini_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.gemini_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=self._mock_response({
            "is_signal": False, "confidence": 0.95,
        }))
        gemini_parser_module._client = mock_client

        result = await gemini_parser_module.parse_with_gemini("сегодня биток вырос на 3%, отличный день")
        self.assertIsNone(result)

    async def test_api_error_returns_none_not_raises(self):
        import src.telegram.gemini_parser as gemini_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.gemini_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("network down"))
        gemini_parser_module._client = mock_client

        result = await gemini_parser_module.parse_with_gemini("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNone(result)

    async def test_invalid_json_response_returns_none_not_raises(self):
        import src.telegram.gemini_parser as gemini_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.gemini_api_key = "test-key"

        mock_client = MagicMock()
        bad_resp = MagicMock()
        bad_resp.text = "не json вообще"
        mock_client.aio.models.generate_content = AsyncMock(return_value=bad_resp)
        gemini_parser_module._client = mock_client

        result = await gemini_parser_module.parse_with_gemini("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNone(result)

    async def test_parse_telegram_signal_falls_back_to_gemini_when_anthropic_not_configured(self):
        """
        Anthropic не настроен (нет ключа) — цепочка должна дойти до
        Gemini как второго, резервного уровня фолбэка.
        """
        import src.telegram.gemini_parser as gemini_parser_module
        from src.telegram.channel_monitor import parse_telegram_signal
        settings.telegram_llm_fallback_enabled = True
        settings.anthropic_api_key = None
        settings.gemini_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=self._mock_response({
            "is_signal": True, "base": "ETH", "quote": "USDT", "side": "short",
            "entry": 3500.0, "take_profits": [3300.0, 3200.0], "stop_loss": 3600.0,
            "confidence": 0.8,
        }))
        gemini_parser_module._client = mock_client

        result = await parse_telegram_signal("шортим эфир около 3500, стоп 3600, цели 3300 и 3200")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "ETH/USDT")
        self.assertEqual(result["side"], "short")
        self.assertEqual(result["tp"], 3200.0)  # short -> минимальный TP


class TestQualificationScorer(unittest.TestCase):
    """Тесты для scorer качества сигналов."""

    def setUp(self):
        from src.telegram.quality_scorer import signal_quality_scorer
        self.scorer = signal_quality_scorer

    def test_score_signal_good_channel(self):
        """Высокий score для хорошего канала."""
        signal = {"pair": "BTC/USDT", "side": "long", "entry": 50000.0,
                  "sl": 49000.0, "tp": 51000.0, "confidence": 0.9}
        self.scorer.channel_stats["good_channel"] = {
            "signals_count": 100, "good_signals": 80, "bad_signals": 20,
            "win_rate": 0.8,
        }
        score = self.scorer.score_signal(signal, "good_channel")
        self.assertGreater(score, 0.7)

    def test_score_signal_bad_channel(self):
        """Низкий score для плохого канала."""
        signal = {"pair": "BTC/USDT", "side": "long", "entry": 50000.0,
                  "sl": 49000.0, "tp": 51000.0, "confidence": 0.5}
        self.scorer.channel_stats["bad_channel"] = {
            "signals_count": 100, "good_signals": 30, "bad_signals": 70,
            "win_rate": 0.3,
        }
        score = self.scorer.score_signal(signal, "bad_channel")
        self.assertLess(score, 0.5)


class TestQualityScorerRestoreFromDb(unittest.IsolatedAsyncioTestCase):
    """
    channel_stats существовал только в памяти — рестарт бота обнулял
    накопленную историческую точность канала обратно к нейтральным 50%,
    хотя вся история решений и исходов сделок хранится в БД.
    """

    async def test_restore_channel_stats_from_db(self):
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import TelegramChannel, TelegramSignal, Trade, Symbol, Exchange
        from src.telegram.quality_scorer import SignalQualityScorer
        from src.utils.timeutils import utcnow

        async with get_session() as session:
            exchange = (
                await session.execute(select(Exchange).where(Exchange.name == "binance_paper"))
            ).scalar_one_or_none()
            if exchange is None:
                exchange = Exchange(name="binance_paper", is_paper=True)
                session.add(exchange)
                await session.flush()

            symbol = Symbol(
                exchange_id=exchange.id, symbol="TESTCOIN/USDT",
                base_asset="TESTCOIN", quote_asset="USDT",
            )
            session.add(symbol)
            await session.flush()

            channel = TelegramChannel(channel_id="@qs_restore_test", channel_title="QS Restore Test", active=True)
            session.add(channel)
            await session.flush()

            for outcome, pnl in [("win", 10.0), ("win", 5.0), ("loss", -3.0)]:
                trade = Trade(
                    symbol_id=symbol.id, direction="long", entry_price=100.0, exit_price=101.0,
                    amount=1.0, pnl=pnl, pnl_pct=1.0, outcome=outcome, is_open=False, closed_at=utcnow(),
                )
                session.add(trade)
                await session.flush()
                session.add(TelegramSignal(
                    channel_id=channel.id, raw_message="test", message_date=utcnow(),
                    parsed_pair="TESTCOIN/USDT", parsed_side="long", parsed_entry=100.0,
                    decision="executed", executed_trade_id=trade.id,
                ))
            await session.commit()

        fresh_scorer = SignalQualityScorer()
        self.assertEqual(fresh_scorer.channel_stats, {})
        await fresh_scorer.restore_channel_stats_from_db()

        self.assertIn("@qs_restore_test", fresh_scorer.channel_stats)
        stats = fresh_scorer.channel_stats["@qs_restore_test"]
        self.assertEqual(stats["good_signals"], 2)
        self.assertEqual(stats["bad_signals"], 1)
        self.assertAlmostEqual(stats["win_rate"], 2 / 3)


class TestLogging(unittest.TestCase):
    """Тесты для логирования."""

    def test_setup_logging(self):
        """Setup logging не вызывает ошибок."""
        setup_logging("INFO")

    def test_log_trade_win(self):
        """Логирование выигрышной сделки."""
        from src.utils.logging import log_trade
        log_trade(1, "BTC/USDT", "long", 100.0, 1.0, "win", "RSI Strategy")


class TestBacktestPosition(unittest.TestCase):
    """Тесты для BacktestPosition."""

    def test_long_position_pnl(self):
        """PnL для long позиции."""
        from src.backtest.engine import BacktestPosition
        pos = BacktestPosition(
            symbol="BTC/USDT", direction="long", entry_price=50000.0,
            amount=0.1, entry_time=datetime.now(), stop_loss=49000.0,
            take_profit=51000.0,
        )
        pos._close(51000.0, datetime.now(), "take_profit")
        self.assertTrue(pos.closed)
        self.assertEqual(pos.pnl, 100.0)

    def test_short_position_pnl(self):
        """PnL для short позиции."""
        from src.backtest.engine import BacktestPosition
        pos = BacktestPosition(
            symbol="BTC/USDT", direction="short", entry_price=50000.0,
            amount=0.1, entry_time=datetime.now(), stop_loss=51000.0,
            take_profit=49000.0,
        )
        pos._close(49000.0, datetime.now(), "take_profit")
        self.assertTrue(pos.closed)
        self.assertEqual(pos.pnl, 100.0)


class TestBacktestResult(unittest.TestCase):
    """Тесты для BacktestResult."""

    def setUp(self):
        from src.backtest.engine import BacktestResult
        self.result = BacktestResult()
        self.result.initial_capital = 10000.0

    def test_single_win_trade(self):
        """Одна выигрышная сделка."""
        from src.backtest.engine import BacktestTrade
        trade = BacktestTrade(
            symbol="BTC/USDT", direction="long", entry_price=50000.0,
            exit_price=51000.0, entry_time=datetime(2024, 1, 1, 12, 0, 0),
            exit_time=datetime(2024, 1, 1, 14, 0, 0), amount=0.1,
            pnl=100.0, pnl_pct=2.0, strategy_id="rsi_mr",
        )
        self.result.trades.append(trade)
        self.result.compute_metrics()
        self.assertEqual(self.result.total_trades, 1)
        self.assertEqual(self.result.total_pnl, 100.0)
        self.assertEqual(self.result.win_rate, 100.0)
        self.assertEqual(self.result.final_capital, 10100.0)


class TestDecisionLogger(unittest.TestCase):
    """Тесты для DecisionLogger."""

    def setUp(self):
        from src.execution.decision_logger import DecisionLogger
        self.logger = DecisionLogger()

    def test_log_market_data(self):
        """Лог market data."""
        self.logger.begin()
        self.logger.log_market_data(
            symbol="BTC/USDT", timeframe="1h", price=50000.0,
            features={"rsi_14": 28.5, "ema_20": 49500.0},
        )
        self.assertEqual(len(self.logger._active_steps), 1)
        self.assertEqual(self.logger._active_steps[0]["step_type"], "market_data")

    def test_log_strategy_signal(self):
        """Лог сигнала."""
        self.logger.begin()
        self.logger.log_strategy_signal(
            strategy_id="rsi_mr", strategy_name="RSI Mean Reversion",
            signal_side="long", confidence=0.85, entry_price=50000.0,
            stop_loss=49000.0, take_profit=51000.0, rationale="RSI oversold",
        )
        self.assertEqual(len(self.logger._active_steps), 1)
        self.assertEqual(self.logger._active_steps[0]["step_type"], "strategy_signal")

    def test_log_ml_score(self):
        """Лог ML score."""
        self.logger.begin()
        self.logger.log_ml_score(
            model_type="direction_classifier", model_version=1,
            proba_up=0.75, proba_down=0.15, proba_neutral=0.10,
        )
        self.assertEqual(len(self.logger._active_steps), 1)
        self.assertEqual(self.logger._active_steps[0]["step_type"], "ml_score")

    def test_log_risk_check(self):
        """Лог risk check."""
        self.logger.begin()
        self.logger.log_risk_check("allowed", "All checks passed", {})
        self.assertEqual(len(self.logger._active_steps), 1)
        self.assertEqual(self.logger._active_steps[0]["step_type"], "risk_check")

    def test_log_execution(self):
        """Лог исполнения."""
        self.logger.begin()
        self.logger.log_execution("ORD-123", "market", 0.01, 50000.0, "filled", 5.0)
        self.assertEqual(len(self.logger._active_steps), 1)
        self.assertEqual(self.logger._active_steps[0]["step_type"], "execution")

    def test_attach_and_flush_roundtrip(self):
        """attach_to_order копит шаги до закрытия сделки, flush_for_trade пишет их в БД под trade_id."""
        self.logger.begin()
        self.logger.log_market_data(symbol="BTC/USDT", timeframe="1h", price=50000.0, features={})
        self.logger.log_execution("ORD-1", "market", 0.01, 50000.0, "filled", 5.0)
        self.logger.attach_to_order(999)

        self.assertEqual(self.logger._active_steps, [])
        self.assertIn(999, self.logger._pending_by_order)
        self.assertEqual(len(self.logger._pending_by_order[999]), 2)

        saved_ids = asyncio.run(
            self.logger.flush_for_trade(999, trade_id=1, close_description="closed")
        )
        self.assertEqual(len(saved_ids), 3)
        self.assertNotIn(999, self.logger._pending_by_order)

    def test_rejected_signal_without_order_is_discarded(self):
        """Если сигнал не привёл к ордеру, накопленные шаги должны просто пропадать при следующем begin()."""
        self.logger.begin()
        self.logger.log_market_data(symbol="ETH/USDT", timeframe="1h", price=3000.0, features={})
        self.logger.log_risk_check("rejected", "max positions", {})
        self.logger.begin()
        self.assertEqual(self.logger._active_steps, [])
        self.assertEqual(self.logger._pending_by_order, {})


class TestTelegramChannelDelete(unittest.IsolatedAsyncioTestCase):
    """
    Удаление канала с сигналами падало с IntegrityError (FK
    telegram_signals.channel_id без каскада) на Postgres, куда бот
    деплоится в проде — практически гарантированно после любого периода
    мониторинга у канала уже есть хотя бы один сигнал.
    """

    async def test_delete_channel_with_signals_cascades(self):
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import TelegramChannel, TelegramSignal
        from src.utils.timeutils import utcnow

        async with get_session() as session:
            channel = TelegramChannel(channel_id="@delete_cascade_test", channel_title="X", active=True)
            session.add(channel)
            await session.flush()
            for i in range(2):
                session.add(TelegramSignal(
                    channel_id=channel.id, raw_message=f"m{i}", message_date=utcnow(),
                    parsed_pair="BTC/USDT", parsed_side="long", parsed_entry=100.0,
                    decision="pending",
                ))
            await session.commit()
            channel_id = channel.id

        async with get_session() as session:
            ch = (
                await session.execute(select(TelegramChannel).where(TelegramChannel.id == channel_id))
            ).scalar_one()
            await session.delete(ch)
            await session.commit()  # раньше падало здесь

        async with get_session() as session:
            remaining = (
                await session.execute(select(TelegramSignal).where(TelegramSignal.channel_id == channel_id))
            ).scalars().all()
        self.assertEqual(remaining, [])


class TestPaperAccountReset(unittest.IsolatedAsyncioTestCase):
    """
    Кнопка "Сбросить paper-аккаунт" (для случаев, когда накопленная paper-
    история искажена уже исправленными багами): должна удалить только
    paper-историю, не трогая real, и корректно отвязывать/каскадировать
    связанные Telegram-сигналы и decision log.
    """

    async def test_reset_deletes_only_paper_history(self):
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import (
            Order, Trade, Symbol, Exchange, TelegramChannel, TelegramSignal, TradeDecisionLog,
        )
        from src.execution.executor import ExecutionEngine
        from src.risk.risk_manager import RiskManager
        from src.utils.timeutils import utcnow

        async with get_session() as session:
            paper_ex = (
                await session.execute(select(Exchange).where(Exchange.name == "binance_paper"))
            ).scalar_one_or_none()
            if paper_ex is None:
                paper_ex = Exchange(name="binance_paper", is_paper=True)
                session.add(paper_ex)
            real_ex = (
                await session.execute(select(Exchange).where(Exchange.name == "binance", Exchange.is_paper == False))
            ).scalar_one_or_none()
            if real_ex is None:
                real_ex = Exchange(name="binance", is_paper=False)
                session.add(real_ex)
            await session.flush()

            paper_sym = Symbol(exchange_id=paper_ex.id, symbol="RESETCOIN/USDT", base_asset="RESETCOIN", quote_asset="USDT")
            real_sym = Symbol(exchange_id=real_ex.id, symbol="RESETCOIN/USDT", base_asset="RESETCOIN", quote_asset="USDT")
            session.add_all([paper_sym, real_sym])
            await session.flush()

            paper_order = Order(exchange_id=paper_ex.id, symbol_id=paper_sym.id, side="buy", order_type="market",
                                 amount=1.0, price=100.0, status="filled", filled_amount=1.0, filled_price=100.0, fee=1.0)
            real_order = Order(exchange_id=real_ex.id, symbol_id=real_sym.id, side="buy", order_type="market",
                                amount=1.0, price=100.0, status="filled", filled_amount=1.0, filled_price=100.0, fee=1.0)
            session.add_all([paper_order, real_order])
            await session.flush()

            paper_trade = Trade(symbol_id=paper_sym.id, direction="long", entry_price=100, exit_price=90,
                                 amount=1.0, pnl=-10.0, pnl_pct=-10.0, outcome="loss", is_open=False,
                                 order_open_id=paper_order.id, closed_at=utcnow())
            real_trade = Trade(symbol_id=real_sym.id, direction="long", entry_price=100, exit_price=110,
                                amount=1.0, pnl=10.0, pnl_pct=10.0, outcome="win", is_open=False,
                                order_open_id=real_order.id, closed_at=utcnow())
            session.add_all([paper_trade, real_trade])
            await session.flush()

            session.add(TradeDecisionLog(trade_id=paper_trade.id, step_order=1, step_type="execution", description="x", details={}))
            session.add(TradeDecisionLog(trade_id=real_trade.id, step_order=1, step_type="execution", description="y", details={}))

            channel = TelegramChannel(channel_id="@reset_unittest", channel_title="X", active=True)
            session.add(channel)
            await session.flush()
            session.add(TelegramSignal(
                channel_id=channel.id, raw_message="m", message_date=utcnow(),
                parsed_pair="RESETCOIN/USDT", parsed_side="long", parsed_entry=100.0,
                decision="executed", executed_order_id=paper_order.id, executed_trade_id=paper_trade.id,
            ))
            await session.commit()
            paper_trade_id, real_trade_id = paper_trade.id, real_trade.id
            paper_order_id, real_order_id, channel_id = paper_order.id, real_order.id, channel.id

        engine = ExecutionEngine()
        engine.is_paper = True
        engine.paper_balance = 100.0
        engine.paper_positions = {"RESETCOIN/USDT": {"amount": 1.0, "entry_price": 100.0, "side": "long"}}
        rm = RiskManager()
        rm.state.total_drawdown_pct = 90.0
        rm.state.paused = True

        result = await engine.reset_paper_account()
        rm.reset_for_new_paper_account()

        self.assertEqual(engine.paper_positions, {})
        self.assertEqual(engine.paper_balance, settings.startup_capital_usdt)
        self.assertEqual(rm.state.total_drawdown_pct, 0.0)
        self.assertFalse(rm.state.paused)

        async with get_session() as session:
            self.assertIsNone(
                (await session.execute(select(Order).where(Order.id == paper_order_id))).scalar_one_or_none()
            )
            self.assertIsNone(
                (await session.execute(select(Trade).where(Trade.id == paper_trade_id))).scalar_one_or_none()
            )
            self.assertIsNotNone(
                (await session.execute(select(Order).where(Order.id == real_order_id))).scalar_one_or_none()
            )
            self.assertIsNotNone(
                (await session.execute(select(Trade).where(Trade.id == real_trade_id))).scalar_one_or_none()
            )
            remaining_logs = (await session.execute(select(TradeDecisionLog))).scalars().all()
            self.assertTrue(all(log.trade_id != paper_trade_id for log in remaining_logs))
            self.assertTrue(any(log.trade_id == real_trade_id for log in remaining_logs))

            sig = (
                await session.execute(select(TelegramSignal).where(TelegramSignal.channel_id == channel_id))
            ).scalar_one()
            self.assertIsNone(sig.executed_order_id)
            self.assertIsNone(sig.executed_trade_id)


class TestCreateOrderWithoutSlTp(unittest.IsolatedAsyncioTestCase):
    """
    create_order()'s own log line did f"{stop_loss:.2f}" unconditionally —
    a Telegram signal from a channel that posts no stop-loss (very common)
    passes stop_loss=None, which crashed with an unhandled TypeError before
    the order was ever created. Not wrapped in a try/except anywhere in the
    call chain, so it silently killed the whole trading iteration.
    """

    async def test_create_order_without_stop_loss_or_take_profit(self):
        from src.execution.executor import ExecutionEngine

        engine = ExecutionEngine()
        settings.trading_mode = "paper"
        await engine.initialize("binance")

        order = await engine.create_order(
            symbol="NOSLTP/USDT", side="buy", amount=1.0, price=10.0,
            order_type="market", stop_loss=None, take_profit=None,
        )
        self.assertIsNotNone(order)


class TestTradesGrouping(unittest.IsolatedAsyncioTestCase):
    """GET /trades должен объединять частичные закрытия одной позиции
    (TP1/TP2/TP3) в одну строку с суммарными показателями."""

    async def test_partial_closes_grouped_into_one_row(self):
        from src.execution.executor import ExecutionEngine
        from src.web.api import list_trades

        engine = ExecutionEngine()
        settings.trading_mode = "paper"
        await engine.initialize("binance")

        symbol = "TRADESGROUP1/USDT"
        order = await engine.create_order(
            symbol=symbol, side="buy", amount=10.0, price=200.0, order_type="market",
        )
        self.assertIsNotNone(order)
        await engine.close_paper_position(
            symbol=symbol, side="long", entry_price=200.0, amount=5.0,
            exit_price=220.0, reason="take_profit_1", entry_fee=1.0,
            holding_seconds=60, order_open_id=order.id,
        )
        await engine.close_paper_position(
            symbol=symbol, side="long", entry_price=200.0, amount=5.0,
            exit_price=240.0, reason="take_profit_3", entry_fee=1.0,
            holding_seconds=180, order_open_id=order.id,
        )

        result = await list_trades(limit=200, offset=0)
        rows = [t for t in result["trades"] if t["symbol"] == symbol]
        self.assertEqual(len(rows), 1, "две частичные сделки одной позиции должны стать одной строкой")
        row = rows[0]
        self.assertEqual(row["parts"], 2)
        self.assertAlmostEqual(row["amount"], 10.0)
        self.assertEqual(row["holding_seconds"], 180)

    async def test_list_trades_uses_opening_order_time_and_exchange_order_id(self):
        """
        Тот же баг, что и в GET /trades/{id}/detail (см. TestTradeDetail):
        created_at в списке сделок должен быть временем реального открытия
        позиции (created_at открывающего Order), а не моментом вставки
        строки Trade в БД (который совпадает с closed_at). Плюс проверка
        ID открывающего/закрывающего ордера с биржи.
        """
        from src.execution.executor import ExecutionEngine
        from src.web.api import list_trades
        from src.db.session import get_session

        engine = ExecutionEngine()
        settings.trading_mode = "real"
        engine.is_paper = False
        engine.exchange_id = "bybit"
        engine.exchange = AsyncMock()
        engine.exchange.create_market_buy_order.return_value = {
            "id": "list-open-ex-1", "filled": 10.0, "price": None, "average": 200.0,
            "fee": {"cost": 0.5, "currency": "USDT"},
        }

        symbol = "TRADESLISTEX1/USDT"
        order = await engine.create_order(
            symbol=symbol, side="buy", amount=10.0, price=200.0, order_type="market",
        )
        self.assertIsNotNone(order)

        async with get_session() as session:
            refreshed_order = await session.get(type(order), order.id)
            expected_opened_at = refreshed_order.created_at.isoformat() + "Z"

        engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"TRADESLISTEX1": 10.0}, "TRADESLISTEX1": {"free": 10.0, "used": 0, "total": 10.0}}
        )
        engine.exchange.create_market_sell_order.return_value = {
            "id": "list-close-ex-1", "filled": 10.0, "price": None, "average": 220.0,
            "fee": {"cost": 0.5, "currency": "USDT"},
        }
        result = await engine.close_real_position(
            symbol=symbol, side="long", entry_price=200.0, amount=10.0,
            reason="take_profit_3", entry_fee=0.5, holding_seconds=60, order_open_id=order.id,
        )
        self.assertIsNotNone(result)

        listed = await list_trades(limit=200, offset=0)
        row = next(t for t in listed["trades"] if t["symbol"] == symbol)
        self.assertEqual(row["created_at"], expected_opened_at)
        self.assertEqual(row["order_id_exchange_open"], "list-open-ex-1")
        self.assertEqual(row["order_id_exchange_close"], "list-close-ex-1")


class TestStatusExposesExchangeOrderId(unittest.IsolatedAsyncioTestCase):
    """GET /status должен показывать ID открывающего ордера с биржи для
    каждой открытой реальной позиции — раньше в ответе был только
    внутренний DB id ордера, недоступный для сверки с самой биржей."""

    async def test_open_real_position_includes_order_id_exchange(self):
        from src.execution.executor import ExecutionEngine
        from src.web.api import get_status

        engine = ExecutionEngine()
        settings.trading_mode = "real"
        engine.is_paper = False
        engine.exchange_id = "bybit"
        engine.exchange = AsyncMock()
        engine.exchange.create_market_buy_order.return_value = {
            "id": "status-open-ex-1", "filled": 10.0, "price": None, "average": 300.0,
            "fee": {"cost": 0.5, "currency": "USDT"},
        }

        symbol = "STATUSEX1/USDT"
        order = await engine.create_order(
            symbol=symbol, side="buy", amount=10.0, price=300.0, order_type="market",
        )
        self.assertIsNotNone(order)

        # api.py импортирует execution_engine на уровне модуля (`from
        # src.execution.executor import execution_engine`) — это отдельная
        # привязка имени, переприсваивание src.execution.executor.execution_engine
        # её не затронуло бы; подменяем именно ссылку внутри api.py.
        with patch("src.web.api.execution_engine", engine):
            status = await get_status()

        self.assertIn(symbol, status["paper_positions"])
        self.assertEqual(status["paper_positions"][symbol]["order_id_exchange"], "status-open-ex-1")


class TestTradeDetail(unittest.IsolatedAsyncioTestCase):
    """
    GET /trades/{trade_id}/detail — разворачиваемая строка на дашборде по
    клику на закрытую сделку. Должен собрать ВСЕ части частично закрытой
    позиции (не только последнюю) и decision log по каждой из них, т.к.
    каждое частичное закрытие пишет свой decision log под своим Trade.id.
    """

    async def test_aggregates_legs_and_decision_log_across_partial_closes(self):
        from src.execution.executor import ExecutionEngine
        from src.web.api import get_trade_detail
        from src.db.session import get_session
        from src.db.models import TradeDecisionLog

        engine = ExecutionEngine()
        settings.trading_mode = "paper"
        await engine.initialize("binance")

        symbol = "TRADEDETAIL1/USDT"
        order = await engine.create_order(
            symbol=symbol, side="buy", amount=10.0, price=100.0, order_type="market",
        )
        self.assertIsNotNone(order)

        result1 = await engine.close_paper_position(
            symbol=symbol, side="long", entry_price=100.0, amount=5.0,
            exit_price=110.0, reason="take_profit_1", entry_fee=1.0,
            holding_seconds=60, order_open_id=order.id,
        )
        result2 = await engine.close_paper_position(
            symbol=symbol, side="long", entry_price=100.0, amount=5.0,
            exit_price=130.0, reason="take_profit_3", entry_fee=1.0,
            holding_seconds=180, order_open_id=order.id,
        )

        async with get_session() as session:
            session.add(TradeDecisionLog(
                trade_id=result1["trade_id"], step_order=1, step_type="execution",
                description="TP1 leg step", details={},
            ))
            session.add(TradeDecisionLog(
                trade_id=result2["trade_id"], step_order=1, step_type="execution",
                description="TP3 leg step", details={},
            ))
            await session.commit()

        detail = await get_trade_detail(result2["trade_id"])
        self.assertEqual(detail["symbol"], symbol)
        self.assertEqual(len(detail["legs"]), 2)
        self.assertAlmostEqual(detail["amount"], 10.0)
        descriptions = {log["description"] for log in detail["decision_log"]}
        self.assertEqual(descriptions, {"TP1 leg step", "TP3 leg step"})

    async def test_unknown_trade_raises_404(self):
        from fastapi import HTTPException
        from src.web.api import get_trade_detail

        with self.assertRaises(HTTPException) as ctx:
            await get_trade_detail(999999999)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_detail_uses_opening_order_time_and_exchange_order_ids(self):
        """
        Trade.created_at — момент вставки строки Trade в БД, а Trade
        создаётся только при ЗАКРЫТИИ позиции — то есть почти совпадает
        с closed_at, и "Открыта"/"Закрыта" в дашборде показывали одно и
        то же время. Настоящее время открытия — created_at связанного
        открывающего Order. Заодно проверяем ID ордеров с биржи (открытие
        и закрытие) — должны браться из реальных данных ордеров, а не
        придумываться.
        """
        from src.execution.executor import ExecutionEngine
        from src.web.api import get_trade_detail
        from src.db.session import get_session

        engine = ExecutionEngine()
        settings.trading_mode = "real"
        engine.is_paper = False
        engine.exchange_id = "bybit"
        engine.exchange = AsyncMock()
        engine.exchange.create_market_buy_order.return_value = {
            "id": "detail-open-ex-1", "filled": 10.0, "price": None, "average": 100.0,
            "fee": {"cost": 0.5, "currency": "USDT"},
        }

        symbol = "TRADEDETAILEX1/USDT"
        order = await engine.create_order(
            symbol=symbol, side="buy", amount=10.0, price=100.0, order_type="market",
        )
        self.assertIsNotNone(order)

        async with get_session() as session:
            refreshed_order = await session.get(type(order), order.id)
            expected_opened_at = refreshed_order.created_at.isoformat() + "Z"

        engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"TRADEDETAILEX1": 10.0}, "TRADEDETAILEX1": {"free": 10.0, "used": 0, "total": 10.0}}
        )
        engine.exchange.create_market_sell_order.return_value = {
            "id": "detail-close-ex-1", "filled": 10.0, "price": None, "average": 110.0,
            "fee": {"cost": 0.5, "currency": "USDT"},
        }
        result = await engine.close_real_position(
            symbol=symbol, side="long", entry_price=100.0, amount=10.0,
            reason="take_profit_3", entry_fee=0.5, holding_seconds=60, order_open_id=order.id,
        )
        self.assertIsNotNone(result)

        detail = await get_trade_detail(result["trade_id"])
        self.assertEqual(detail["created_at"], expected_opened_at)
        self.assertEqual(detail["order_id_exchange_open"], "detail-open-ex-1")
        self.assertEqual(len(detail["legs"]), 1)
        self.assertEqual(detail["legs"][0]["order_id_exchange_close"], "detail-close-ex-1")


class TestClosePositionAtomicity(unittest.IsolatedAsyncioTestCase):
    """
    close_paper_position mutated paper_balance/paper_positions BEFORE
    writing to the DB — if that write failed for any reason (a transient
    DB hiccup), the position vanished from memory with zero Order/Trade
    record: it disappeared from "open positions" and never showed up in
    "closed trades" either. It must only mutate live state after the DB
    write actually commits.
    """

    async def test_position_survives_failed_db_write(self):
        from unittest.mock import patch
        from src.execution.executor import ExecutionEngine

        engine = ExecutionEngine()
        settings.trading_mode = "paper"
        await engine.initialize("binance")

        symbol = "ATOMICITY1/USDT"
        order = await engine.create_order(
            symbol=symbol, side="buy", amount=1.0, price=100.0, order_type="market",
        )
        self.assertIsNotNone(order)
        balance_before = engine.paper_balance
        amount_before = engine.paper_positions[symbol]["amount"]

        with patch.object(ExecutionEngine, "_resolve_symbol_id", side_effect=RuntimeError("simulated DB failure")):
            with self.assertRaises(RuntimeError):
                await engine.close_paper_position(
                    symbol=symbol, side="long", entry_price=100.0, amount=1.0,
                    exit_price=110.0, reason="take_profit_3", entry_fee=0.1,
                    holding_seconds=60, order_open_id=order.id,
                )

        self.assertIn(symbol, engine.paper_positions, "position must not vanish when the DB write fails")
        self.assertEqual(engine.paper_positions[symbol]["amount"], amount_before)
        self.assertEqual(engine.paper_balance, balance_before)

        # A subsequent successful close still works normally.
        result = await engine.close_paper_position(
            symbol=symbol, side="long", entry_price=100.0, amount=1.0,
            exit_price=110.0, reason="take_profit_3", entry_fee=0.1,
            holding_seconds=60, order_open_id=order.id,
        )
        self.assertIsNotNone(result)
        self.assertNotIn(symbol, engine.paper_positions)


class TestComputeEquitySkipsUntrackedPositions(unittest.IsolatedAsyncioTestCase):
    """
    self.open_positions (TradingBot) — вторичный кэш execution_engine
    .paper_positions/real_positions, синхронизируется лениво (см.
    _check_position_exit). Если позицию уже сняли с учёта в execution_engine
    (закрытие в обход основного цикла, реконсиляция фантомной/пыльной
    позиции), но self.open_positions ещё не подчищен, _compute_equity не
    должен продолжать считать её объём в equity — иначе именно так раздутый
    amount одной такой позиции (AVAX/USDT: учтено 416.5, на бирже 0.00046)
    превращал "Просадку" в дашборде в бессмысленные "-220110,7%".
    """

    def test_untracked_position_excluded_from_equity(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        bot = main_module.TradingBot()
        original_trading_mode = settings.trading_mode
        original_paper_positions = dict(main_module.execution_engine.paper_positions)
        try:
            settings.trading_mode = "paper"
            # execution_engine больше не знает об этой позиции (снята с
            # учёта), но open_positions ещё её содержит с раздутым amount.
            main_module.execution_engine.paper_positions = {}
            bot.open_positions = {
                "GHOST/USDT": {"side": "long", "entry_price": 0.5, "amount": 416.5},
            }
            bot.last_prices = {"GHOST/USDT": 0.5}

            equity = bot._compute_equity(100.0)
        finally:
            settings.trading_mode = original_trading_mode
            main_module.execution_engine.paper_positions = original_paper_positions

        self.assertEqual(equity, 100.0, "untracked position's amount must not inflate equity")

    def test_tracked_position_still_counted_in_equity(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        bot = main_module.TradingBot()
        original_trading_mode = settings.trading_mode
        original_paper_positions = dict(main_module.execution_engine.paper_positions)
        try:
            settings.trading_mode = "paper"
            main_module.execution_engine.paper_positions = {
                "REAL/USDT": {"amount": 2.0, "entry_price": 10.0},
            }
            bot.open_positions = {
                "REAL/USDT": {"side": "long", "entry_price": 10.0, "amount": 2.0},
            }
            bot.last_prices = {"REAL/USDT": 12.0}

            equity = bot._compute_equity(100.0)
        finally:
            settings.trading_mode = original_trading_mode
            main_module.execution_engine.paper_positions = original_paper_positions

        self.assertAlmostEqual(equity, 100.0 + 2.0 * 12.0)


class TestCleanupClosesExchangeConnection(unittest.IsolatedAsyncioTestCase):
    """
    _cleanup() закрывал ingest/cg_client/scheduler/telegram, но никогда не
    вызывал execution_engine.close() — ccxt-биржа держит собственную aiohttp
    ClientSession, которая при каждом рестарте/остановке процесса оставалась
    незакрытой (aiohttp сам логировал это как ERROR "Unclosed client
    session" / "Unclosed connector" уже после выхода из event loop).
    """

    async def test_cleanup_calls_execution_engine_close(self):
        from unittest.mock import AsyncMock, patch
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        bot = main_module.TradingBot()
        with patch.object(main_module.execution_engine, "close", new=AsyncMock()) as close_mock, \
                patch.object(main_module, "close_telegram", new=AsyncMock()):
            await bot._cleanup()

        close_mock.assert_awaited_once()


class TestTradingIterationPerSymbolIsolation(unittest.IsolatedAsyncioTestCase):
    """
    The for-symbol loop in _trading_iteration had no per-symbol exception
    handling — an unhandled error processing ONE symbol (a network hiccup
    fetching candles, anything) aborted the whole iteration, silently
    skipping every symbol still left in active_symbols that cycle. Since
    active_symbols barely changes between iterations, a symbol that
    consistently errors permanently blocked SL/TP checks (and price
    updates) for everything after it in the list, cycle after cycle,
    until a bot restart happened to reorder the universe.
    """

    async def test_one_symbol_error_does_not_block_the_rest(self):
        from unittest.mock import AsyncMock
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        bot = main_module.TradingBot()
        bot.active_symbols = ["A/USDT", "B/USDT", "C/USDT"]
        bot.daily_pnl_reset_date = main_module.utcnow().date()
        processed = []

        async def fake_process_symbol(symbol):
            if symbol == "B/USDT":
                raise RuntimeError("simulated transient failure for B/USDT")
            processed.append(symbol)

        bot._process_symbol = fake_process_symbol

        original_risk_manager = main_module.risk_manager
        original_get_paper_balance = main_module.execution_engine.get_paper_balance
        try:
            main_module.risk_manager = AsyncMock()
            main_module.risk_manager.state.kill_switch_active = False
            main_module.risk_manager.state.paused = False
            main_module.execution_engine.get_paper_balance = lambda: 10000.0

            await bot._trading_iteration()
        finally:
            main_module.risk_manager = original_risk_manager
            main_module.execution_engine.get_paper_balance = original_get_paper_balance

        self.assertEqual(processed, ["A/USDT", "C/USDT"])


class TestTradingIterationPausedStillUpdatesPrices(unittest.IsolatedAsyncioTestCase):
    """
    risk_manager.state.paused (пауза по просадке, которая сама себя не
    снимает — см. RiskManager.on_balance_update) раньше останавливала
    _trading_iteration() целиком ранним return — цены не обновлялись, SL/TP
    открытых позиций не проверялись, ордера не закрывались вплоть до
    /risk/resume или рестарта бота (реконнект к бирже сбрасывает paused
    через reset_for_real_account). Новые входы уже блокируются отдельно —
    risk_manager.check_signal() внутри _process_symbol — поэтому пауза не
    должна останавливать обработку символов целиком, только открытие новых
    позиций. kill switch — отдельный, более серьёзный стоп, останавливает
    итерацию полностью, как и раньше.
    """

    async def test_paused_still_processes_symbols(self):
        from unittest.mock import AsyncMock
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        bot = main_module.TradingBot()
        bot.active_symbols = ["A/USDT", "B/USDT"]
        bot.daily_pnl_reset_date = main_module.utcnow().date()
        processed = []

        async def fake_process_symbol(symbol):
            processed.append(symbol)

        bot._process_symbol = fake_process_symbol

        original_risk_manager = main_module.risk_manager
        original_get_paper_balance = main_module.execution_engine.get_paper_balance
        try:
            main_module.risk_manager = AsyncMock()
            main_module.risk_manager.state.kill_switch_active = False
            main_module.risk_manager.state.paused = True
            main_module.execution_engine.get_paper_balance = lambda: 10000.0

            await bot._trading_iteration()
        finally:
            main_module.risk_manager = original_risk_manager
            main_module.execution_engine.get_paper_balance = original_get_paper_balance

        self.assertEqual(processed, ["A/USDT", "B/USDT"])

    async def test_kill_switch_still_stops_iteration_entirely(self):
        from unittest.mock import AsyncMock
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        bot = main_module.TradingBot()
        bot.active_symbols = ["A/USDT", "B/USDT"]
        bot.daily_pnl_reset_date = main_module.utcnow().date()
        bot._kill_switch_notified = False
        processed = []

        async def fake_process_symbol(symbol):
            processed.append(symbol)

        bot._process_symbol = fake_process_symbol

        original_risk_manager = main_module.risk_manager
        with patch("src.main.send_notification", new=AsyncMock()):
            try:
                main_module.risk_manager = AsyncMock()
                main_module.risk_manager.state.kill_switch_active = True

                await bot._trading_iteration()
            finally:
                main_module.risk_manager = original_risk_manager

        self.assertEqual(processed, [])

    async def test_balance_update_crash_does_not_block_symbol_loop(self):
        """
        Обновление баланса/просадки (risk_manager.on_balance_update) идёт
        ДО цикла по active_symbols и раньше не было защищено — необработанное
        исключение там (например, повреждённая запись в open_positions)
        прерывало ВСЮ _trading_iteration() до того, как цикл по символам
        вообще начинался: цены не обновлялись и SL/TP не проверялись ни для
        одной позиции, каждую итерацию — тот же эффект "полной заморозки",
        что и исправленный ранее блокирующий return на паузе.
        """
        from unittest.mock import AsyncMock
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        bot = main_module.TradingBot()
        bot.active_symbols = ["A/USDT", "B/USDT"]
        bot.daily_pnl_reset_date = main_module.utcnow().date()
        processed = []

        async def fake_process_symbol(symbol):
            processed.append(symbol)

        bot._process_symbol = fake_process_symbol

        original_risk_manager = main_module.risk_manager
        original_get_paper_balance = main_module.execution_engine.get_paper_balance
        original_trading_mode = settings.trading_mode
        try:
            settings.trading_mode = "paper"
            main_module.risk_manager = AsyncMock()
            main_module.risk_manager.state.kill_switch_active = False
            main_module.risk_manager.state.paused = False
            main_module.risk_manager.on_balance_update = MagicMock(side_effect=RuntimeError("corrupted position"))
            main_module.execution_engine.get_paper_balance = lambda: 10000.0

            await bot._trading_iteration()
        finally:
            main_module.risk_manager = original_risk_manager
            main_module.execution_engine.get_paper_balance = original_get_paper_balance
            settings.trading_mode = original_trading_mode

        self.assertEqual(processed, ["A/USDT", "B/USDT"])


class TestProtections(unittest.IsolatedAsyncioTestCase):
    """
    Protections (перенесено из clonerbot, адаптировано под hermes_trade):
    cooldown источника после закрытия, StoplossGuard (кластер стопов ->
    глобальная пауза) и LosingStreak (серия убытков у одного источника ->
    блокировка только этого источника).
    """

    def setUp(self):
        self._saved = {
            "protections_enabled": settings.protections_enabled,
            "protections_channel_cooldown_minutes": settings.protections_channel_cooldown_minutes,
            "protections_stoploss_guard_window_min": settings.protections_stoploss_guard_window_min,
            "protections_stoploss_guard_count": settings.protections_stoploss_guard_count,
            "protections_stoploss_guard_lock_min": settings.protections_stoploss_guard_lock_min,
            "protections_losing_streak_count": settings.protections_losing_streak_count,
            "protections_losing_streak_lock_min": settings.protections_losing_streak_lock_min,
        }
        settings.protections_enabled = True

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(settings, key, value)

    async def test_lock_expires(self):
        from datetime import timedelta
        from src.risk.protections import LockStore
        from src.utils.timeutils import utcnow

        locks = LockStore()
        key = "test:lock-expiry"
        await locks.add(key, 10, "test lock")
        self.assertEqual(await locks.active_reason([key]), "test lock")

        # Истёкшая блокировка не должна больше действовать.
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import RiskLock
        async with get_session() as session:
            row = (await session.execute(select(RiskLock).where(RiskLock.scope_key == key))).scalars().first()
            row.until = utcnow() - timedelta(minutes=1)
            await session.commit()

        self.assertIsNone(await locks.active_reason([key]))

    async def test_cooldown_applied_after_any_close(self):
        from src.risk.protections import ProtectionManager

        settings.protections_channel_cooldown_minutes = 15
        pm = ProtectionManager()
        source_key = "telegram:@cooldown_unittest"

        self.assertIsNone(await pm.locked_reason([source_key]))
        await pm.on_close(source_key, "COOLDOWNCOIN/USDT", pnl=5.0, reason="take_profit_3")
        reason = await pm.locked_reason([source_key])
        self.assertIsNotNone(reason)
        self.assertIn("cooldown", reason)

    async def test_stoploss_guard_pauses_all_trading(self):
        from datetime import timedelta
        from sqlalchemy import select, func
        from src.risk.protections import ProtectionManager, GLOBAL_KEY
        from src.db.session import get_session
        from src.db.models import RiskCloseEvent
        from src.utils.timeutils import utcnow

        settings.protections_stoploss_guard_window_min = 60
        settings.protections_stoploss_guard_lock_min = 30
        settings.protections_losing_streak_count = 999  # не мешать этому тесту

        # БД общая на весь тестовый сеанс — другие тесты этого класса уже
        # могли записать свои события со stop_loss. Считаем порог от текущего
        # количества, а не от нуля, чтобы тест не зависел от порядка запуска.
        since = utcnow() - timedelta(minutes=60)
        async with get_session() as session:
            baseline = (
                await session.execute(
                    select(func.count(RiskCloseEvent.id)).where(
                        RiskCloseEvent.reason == "stop_loss", RiskCloseEvent.closed_at >= since,
                    )
                )
            ).scalar() or 0
        settings.protections_stoploss_guard_count = baseline + 2

        pm = ProtectionManager()

        self.assertIsNone(await pm.locked_reason([GLOBAL_KEY]))
        await pm.on_close("telegram:@sg_channel_1", "SGCOIN1/USDT", pnl=-10.0, reason="stop_loss")
        # После одного стопа глобальной паузы ещё нет.
        self.assertIsNone(await pm.locked_reason([GLOBAL_KEY]))
        await pm.on_close("telegram:@sg_channel_2", "SGCOIN2/USDT", pnl=-8.0, reason="stop_loss")
        # Второй стоп (от ДРУГОГО источника) добивает до порога — StoplossGuard
        # реагирует на кластер стопов по всем источникам сразу.
        reason = await pm.locked_reason([GLOBAL_KEY])
        self.assertIsNotNone(reason)
        self.assertIn("stoploss guard", reason)

    async def test_losing_streak_locks_only_that_channel(self):
        from src.risk.protections import ProtectionManager, GLOBAL_KEY, channel_key

        settings.protections_stoploss_guard_count = 999  # не мешать этому тесту
        settings.protections_losing_streak_count = 2
        settings.protections_losing_streak_lock_min = 60
        pm = ProtectionManager()
        bad_channel = "@losing_streak_unittest"
        other_channel = "@losing_streak_unrelated"

        await pm.on_close(channel_key(bad_channel), "LSCOIN/USDT", pnl=-5.0, reason="stop_loss")
        # После первого закрытия уже есть блокировка (cooldown), но ещё не
        # по причине losing streak — нужны два подряд убыточных закрытия.
        reason_after_first = await pm.locked_reason([channel_key(bad_channel)])
        self.assertNotIn("losing streak", reason_after_first or "")
        await pm.on_close(channel_key(bad_channel), "LSCOIN/USDT", pnl=-3.0, reason="take_profit_1")

        reason = await pm.locked_reason([channel_key(bad_channel)])
        self.assertIsNotNone(reason)
        self.assertIn("losing streak", reason)
        # Другой канал не пострадал, и это не глобальная блокировка.
        self.assertIsNone(await pm.locked_reason([channel_key(other_channel)]))
        self.assertIsNone(await pm.locked_reason([GLOBAL_KEY]))

    async def test_losing_streak_broken_by_a_win(self):
        from src.risk.protections import ProtectionManager, channel_key

        settings.protections_losing_streak_count = 2
        pm = ProtectionManager()
        channel = "@losing_streak_broken_unittest"

        await pm.on_close(channel_key(channel), "LSBCOIN/USDT", pnl=-5.0, reason="stop_loss")
        await pm.on_close(channel_key(channel), "LSBCOIN/USDT", pnl=1.0, reason="take_profit_3")
        # win rate — win-loss-... не подряд убыточная серия, блокировки по
        # losing streak быть не должно (кулдаун-то будет, это отдельная вещь).
        reason = await pm.locked_reason([channel_key(channel)])
        self.assertIsNotNone(reason)
        self.assertIn("cooldown", reason)
        self.assertNotIn("losing streak", reason)

    async def test_protections_disabled_is_a_noop(self):
        from src.risk.protections import ProtectionManager, channel_key

        settings.protections_enabled = False
        pm = ProtectionManager()
        channel = "@protections_disabled_unittest"

        await pm.on_close(channel_key(channel), "DISABLEDCOIN/USDT", pnl=-100.0, reason="stop_loss")
        self.assertIsNone(await pm.locked_reason([channel_key(channel)]))

    async def test_disabling_protections_bypasses_a_lock_created_earlier(self):
        """
        locked_reason() раньше не смотрел на protections_enabled вообще —
        выключатель в настройках останавливал только создание НОВЫХ
        блокировок, но уже существующая (поставленная ДО выключения)
        по-прежнему находилась и отклоняла сигналы: выключатель на практике
        ничего не менял, пока старая блокировка не истекала сама.
        """
        from src.risk.protections import ProtectionManager, channel_key

        settings.protections_enabled = True
        pm = ProtectionManager()
        channel = "@toggle_bypass_unittest"

        await pm.on_close(channel_key(channel), "TOGGLECOIN/USDT", pnl=-5.0, reason="stop_loss")
        self.assertIsNotNone(await pm.locked_reason([channel_key(channel)]))

        settings.protections_enabled = False
        self.assertIsNone(await pm.locked_reason([channel_key(channel)]))

        # Включили обратно — ещё не истёкшая блокировка снова действует.
        settings.protections_enabled = True
        self.assertIsNotNone(await pm.locked_reason([channel_key(channel)]))


class TestTrailingStop(unittest.IsolatedAsyncioTestCase):
    """
    Trailing stop-loss (портировано из clonerbot): SL подтягивается к
    текущей цене на trailing_stop_pct и только ужесточается — никогда не
    откатывается назад, даже если цена временно отступает.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    def setUp(self):
        self._saved_pct = settings.trailing_stop_pct

    def tearDown(self):
        settings.trailing_stop_pct = self._saved_pct

    def test_disabled_by_default_does_nothing(self):
        settings.trailing_stop_pct = 0.0
        bot = self._make_bot()
        bot.open_positions["TRAILOFF/USDT"] = {"side": "long", "entry_price": 100.0, "sl": 90.0}
        bot._apply_trailing_stop("TRAILOFF/USDT", 150.0)
        self.assertEqual(bot.open_positions["TRAILOFF/USDT"]["sl"], 90.0)

    def test_long_tightens_but_never_loosens(self):
        settings.trailing_stop_pct = 1.0  # 1%
        bot = self._make_bot()
        bot.open_positions["TRAILLONG/USDT"] = {"side": "long", "entry_price": 100.0, "sl": 90.0}

        bot._apply_trailing_stop("TRAILLONG/USDT", 110.0)
        self.assertAlmostEqual(bot.open_positions["TRAILLONG/USDT"]["sl"], 108.9, places=6)

        # Цена откатилась — SL не должен ослабнуть.
        bot._apply_trailing_stop("TRAILLONG/USDT", 105.0)
        self.assertAlmostEqual(bot.open_positions["TRAILLONG/USDT"]["sl"], 108.9, places=6)

        # Новый максимум — SL подтягивается дальше.
        bot._apply_trailing_stop("TRAILLONG/USDT", 120.0)
        self.assertAlmostEqual(bot.open_positions["TRAILLONG/USDT"]["sl"], 118.8, places=6)

    def test_short_tightens_but_never_loosens(self):
        settings.trailing_stop_pct = 1.0
        bot = self._make_bot()
        bot.open_positions["TRAILSHORT/USDT"] = {"side": "short", "entry_price": 100.0, "sl": 110.0}

        bot._apply_trailing_stop("TRAILSHORT/USDT", 90.0)
        self.assertAlmostEqual(bot.open_positions["TRAILSHORT/USDT"]["sl"], 90.9, places=6)

        # Цена откатилась вверх — SL не должен ослабнуть.
        bot._apply_trailing_stop("TRAILSHORT/USDT", 95.0)
        self.assertAlmostEqual(bot.open_positions["TRAILSHORT/USDT"]["sl"], 90.9, places=6)

        bot._apply_trailing_stop("TRAILSHORT/USDT", 80.0)
        self.assertAlmostEqual(bot.open_positions["TRAILSHORT/USDT"]["sl"], 80.8, places=6)

    def test_does_not_undo_breakeven_stop_if_price_has_not_moved_further(self):
        # После TP1 SL переставлен в безубыток (100). Пока цена не ушла
        # достаточно далеко за пределы trailing-зазора выше безубытка,
        # trailing не должен откатывать SL ниже него.
        settings.trailing_stop_pct = 5.0  # 5%
        bot = self._make_bot()
        bot.open_positions["TRAILBE/USDT"] = {"side": "long", "entry_price": 100.0, "sl": 100.0}

        bot._apply_trailing_stop("TRAILBE/USDT", 102.0)  # candidate = 96.9 < 100
        self.assertEqual(bot.open_positions["TRAILBE/USDT"]["sl"], 100.0)

    def test_no_open_position_is_a_noop(self):
        settings.trailing_stop_pct = 1.0
        bot = self._make_bot()
        bot._apply_trailing_stop("NOSUCHPOS/USDT", 100.0)  # не должно падать
        self.assertNotIn("NOSUCHPOS/USDT", bot.open_positions)


class TestExpectancySizing(unittest.IsolatedAsyncioTestCase):
    """
    Expectancy-based sizing (портировано из clonerbot: scoring/channel_scorer.py):
    множитель размера позиции по источнику (канал/стратегия), обученный на
    среднем pnl_pct его закрытых сделок из общего журнала risk_close_events.
    """

    def setUp(self):
        self._saved = {
            "expectancy_sizing_enabled": settings.expectancy_sizing_enabled,
            "expectancy_sizing_min_trades": settings.expectancy_sizing_min_trades,
            "expectancy_sizing_max_trades": settings.expectancy_sizing_max_trades,
            "expectancy_sizing_min_expectancy_pct": settings.expectancy_sizing_min_expectancy_pct,
        }
        settings.expectancy_sizing_enabled = True
        settings.expectancy_sizing_min_trades = 3
        settings.expectancy_sizing_max_trades = 50
        settings.expectancy_sizing_min_expectancy_pct = 0.0

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(settings, key, value)

    async def _seed(self, scope_key: str, pnl_pcts: list[float]):
        from src.db.session import get_session
        from src.db.models import RiskCloseEvent
        async with get_session() as session:
            for pct in pnl_pcts:
                session.add(RiskCloseEvent(
                    scope_key=scope_key, symbol="X/USDT", reason="take_profit_3",
                    pnl=pct, pnl_pct=pct,
                ))
            await session.commit()

    async def test_disabled_returns_neutral_multiplier(self):
        from src.risk import expectancy_sizing
        settings.expectancy_sizing_enabled = False
        await self._seed("telegram:@sizing_disabled_unittest", [-10, -10, -10])
        mult = await expectancy_sizing.size_multiplier("telegram:@sizing_disabled_unittest")
        self.assertEqual(mult, 1.0)

    async def test_low_sample_gets_min_multiplier(self):
        from src.risk import expectancy_sizing
        await self._seed("telegram:@sizing_new_unittest", [5.0])  # меньше min_trades=3
        mult = await expectancy_sizing.size_multiplier("telegram:@sizing_new_unittest")
        self.assertEqual(mult, expectancy_sizing.MIN_MULTIPLIER)

    async def test_positive_expectancy_scales_up_toward_max(self):
        from src.risk import expectancy_sizing
        await self._seed("telegram:@sizing_good_unittest", [3.0, 4.0, 2.5, 3.5])  # ~3.25% > цели 2%
        mult = await expectancy_sizing.size_multiplier("telegram:@sizing_good_unittest")
        self.assertEqual(mult, expectancy_sizing.MAX_MULTIPLIER)

    async def test_non_positive_expectancy_skips_channel(self):
        from src.risk import expectancy_sizing
        await self._seed("telegram:@sizing_bad_unittest", [-2.0, 1.0, -3.0, -1.0])  # средний < 0
        mult = await expectancy_sizing.size_multiplier("telegram:@sizing_bad_unittest")
        self.assertEqual(mult, 0.0)

    async def test_unknown_strategy_scope_uses_same_logic(self):
        from src.risk import expectancy_sizing
        from src.risk.protections import strategy_key
        await self._seed(strategy_key("ema_crossover"), [-1.0, -2.0, -1.5])
        mult = await expectancy_sizing.size_multiplier(strategy_key("ema_crossover"))
        self.assertEqual(mult, 0.0)


class TestMlTrainingReadiness(unittest.IsolatedAsyncioTestCase):
    """
    /ml/training-readiness (FeatureStore.get_training_readiness): отвечает
    на "сколько записей есть для обучения модели" — раньше единственным
    способом узнать это было читать таблицу ml_features напрямую в БД.
    """

    async def test_counts_labeled_rows_separately_per_label_type(self):
        from datetime import timedelta
        from src.db.session import get_session
        from src.db.models import MLFeature
        from src.ml import feature_store, MIN_TRAINING_SAMPLES
        from src.utils.timeutils import utcnow

        symbol = "MLREADINESS1/USDT"
        async with get_session() as session:
            # 2 строки с обоими лейблами, 1 только с direction, 1 вообще без лейблов.
            session.add(MLFeature(
                symbol=symbol, timeframe="1h", timestamp=utcnow(),
                features={"rsi_14": 50.0}, label_direction=1, label_volatility=0.02, source="live",
            ))
            session.add(MLFeature(
                symbol=symbol, timeframe="1h", timestamp=utcnow() + timedelta(seconds=1),
                features={"rsi_14": 55.0}, label_direction=-1, label_volatility=0.01, source="live",
            ))
            session.add(MLFeature(
                symbol=symbol, timeframe="1h", timestamp=utcnow() + timedelta(seconds=2),
                features={"rsi_14": 60.0}, label_direction=0, label_volatility=None, source="live",
            ))
            session.add(MLFeature(
                symbol=symbol, timeframe="1h", timestamp=utcnow() + timedelta(seconds=3),
                features={"rsi_14": 65.0}, label_direction=None, label_volatility=None, source="live",
            ))
            await session.commit()

        result = await feature_store.get_training_readiness(symbol=symbol)
        self.assertEqual(result["total_features"], 4)
        self.assertEqual(result["labeled_direction"], 3)
        self.assertEqual(result["labeled_volatility"], 2)
        self.assertEqual(result["min_training_samples"], MIN_TRAINING_SAMPLES)
        self.assertFalse(result["direction_ready"])  # 3 < 100
        self.assertFalse(result["volatility_ready"])  # 2 < 100
        self.assertIn({"symbol": symbol, "count": 4}, result["by_symbol"])

    async def test_endpoint_returns_the_same_shape(self):
        from src.ml import feature_store
        result = await feature_store.get_training_readiness()
        for key in (
            "total_features", "labeled_direction", "labeled_volatility",
            "min_training_samples", "direction_ready", "volatility_ready",
            "trades_count", "min_trades_for_retrain_attempt", "by_symbol",
        ):
            self.assertIn(key, result)


class TestChannelQualitySettings(unittest.IsolatedAsyncioTestCase):
    """
    TradingBot._get_channel_settings — раньше не существовал, main.py всегда
    использовал ГЛОБАЛЬНЫЕ settings.telegram_signals_quality_threshold/
    auto_execute для всех каналов сразу, поэтому индивидуальный порог/
    автоисполнение канала, выставленные при добавлении или через дашборд,
    не имели никакого эффекта на реальное исполнение сигналов.
    """

    async def test_reads_per_channel_threshold_and_auto_execute(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        from src.db.session import get_session
        from src.db.models import TelegramChannel

        async with get_session() as session:
            channel = TelegramChannel(
                channel_id="@channelsettings_unittest", channel_title="X",
                quality_threshold=0.85, auto_execute=True, active=True,
            )
            session.add(channel)
            await session.commit()
            db_id = channel.id

        bot = main_module.TradingBot()
        bot._telegram_channel_db_ids = {"@channelsettings_unittest": db_id}

        threshold, auto_execute = await bot._get_channel_settings("@channelsettings_unittest")
        self.assertAlmostEqual(threshold, 0.85)
        self.assertTrue(auto_execute)

    async def test_falls_back_to_global_settings_for_unknown_channel(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        bot = main_module.TradingBot()
        bot._telegram_channel_db_ids = {}

        threshold, auto_execute = await bot._get_channel_settings("@unknown_channel_unittest")
        self.assertEqual(threshold, settings.telegram_signals_quality_threshold)
        self.assertEqual(auto_execute, settings.telegram_signals_auto_execute)


class TestTelegramAutoExecuteIgnoresProtections(unittest.IsolatedAsyncioTestCase):
    """
    Автоисполнение сигнала включённого канала — явное доверие каналу по
    запросу пользователя, поэтому Protections-блокировки (кулдаун канала
    после закрытия сделки, StoplossGuard, LosingStreak) не должны его
    останавливать, в отличие от стратегийного пути. Kill switch/пауза
    (execution_engine.can_execute(), общий аварийный стоп) по-прежнему
    применяются — это не тестируется здесь напрямую (сигнал просто
    доходит до _execute_telegram_signal, тот сам упирается в can_execute()
    внутри create_order при необходимости).
    """

    async def test_auto_execute_runs_despite_active_protections_lock(self):
        from unittest.mock import AsyncMock, patch

        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        from src.db.session import get_session
        from src.db.models import TelegramChannel
        from src.risk.protections import GLOBAL_KEY, channel_key, protection_manager

        channel_id = "@autoexec_ignore_protections_unittest"
        async with get_session() as session:
            channel = TelegramChannel(
                channel_id=channel_id, channel_title="X",
                quality_threshold=0.0, auto_execute=True, active=True,
            )
            session.add(channel)
            await session.commit()
            db_id = channel.id

        # Активная блокировка и по каналу, и глобальная (StoplossGuard) —
        # обе должны быть проигнорированы для автоисполнения.
        await protection_manager.locks.add(channel_key(channel_id), 5, "test lock")
        await protection_manager.locks.add(GLOBAL_KEY, 5, "stoploss guard test")

        bot = main_module.TradingBot()
        bot._telegram_channel_db_ids = {channel_id: db_id}
        bot.open_positions = {}

        fake_order = MagicMock(id=1)
        with patch.object(bot, "_execute_telegram_signal", new=AsyncMock(return_value=fake_order)) as exec_mock:
            await bot._on_telegram_signal({
                "channel_id": channel_id,
                "parsed_pair": "BTC/USDT",
                "parsed_side": "long",
                "parsed_entry": 50000.0,
                "raw_message": "test",
            })

        exec_mock.assert_awaited_once()


class TestUpdateTelegramChannel(unittest.IsolatedAsyncioTestCase):
    """PATCH /telegram/channels/{id} — делает настройки существующего канала
    редактируемыми (раньше можно было только создать/удалить канал целиком)."""

    async def test_partial_update_only_changes_given_fields(self):
        from src.db.models import TelegramChannel
        from src.db.session import get_session
        from src.web.api import TelegramChannelUpdate, update_telegram_channel

        async with get_session() as session:
            channel = TelegramChannel(
                channel_id="@patchchannel_unittest", channel_title="Original",
                quality_threshold=0.5, auto_execute=False, active=True,
            )
            session.add(channel)
            await session.commit()
            db_id = channel.id

        result = await update_telegram_channel(
            db_id, TelegramChannelUpdate(quality_threshold=0.7)
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["updated"], ["quality_threshold"])

        async with get_session() as session:
            refreshed = await session.get(TelegramChannel, db_id)
            self.assertAlmostEqual(refreshed.quality_threshold, 0.7)
            self.assertFalse(refreshed.auto_execute)  # не тронуто
            self.assertEqual(refreshed.channel_title, "Original")  # не тронуто

        await update_telegram_channel(db_id, TelegramChannelUpdate(auto_execute=True))
        async with get_session() as session:
            refreshed = await session.get(TelegramChannel, db_id)
            self.assertTrue(refreshed.auto_execute)
            self.assertAlmostEqual(refreshed.quality_threshold, 0.7)  # сохранился с прошлого апдейта

    async def test_update_unknown_channel_raises_404(self):
        from fastapi import HTTPException

        from src.web.api import TelegramChannelUpdate, update_telegram_channel

        with self.assertRaises(HTTPException) as ctx:
            await update_telegram_channel(999999999, TelegramChannelUpdate(quality_threshold=0.9))
        self.assertEqual(ctx.exception.status_code, 404)


class TestProtectionsLockTimestampFormat(unittest.IsolatedAsyncioTestCase):
    """
    active_locks() отдавал naive-UTC datetime без суффикса 'Z' — браузер
    парсит такую ISO-строку как ЛОКАЛЬНОЕ время (не UTC), из-за чего ещё
    активная блокировка могла отображаться так, будто она уже давно истекла
    (расхождение на величину смещения часового пояса пользователя).
    """

    async def test_until_has_utc_suffix(self):
        from src.risk.protections import LockStore

        locks = LockStore()
        key = "test:iso-format-unittest"
        await locks.add(key, 10, "test lock")

        rows = await locks.active_locks()
        row = next(r for r in rows if r["scope"] == key)
        self.assertTrue(row["until"].endswith("Z"), row["until"])


class TestGetTradableSymbols(unittest.IsolatedAsyncioTestCase):
    """
    MarketDataIngest.get_tradable_symbols раньше звал fetch_tickers(candidates)
    со списком из сотен пар — регулярно ловило "'str' object has no attribute
    'keys'" в проде (кривой/неожиданный ответ биржи на такой запрос,
    который ccxt не всегда корректно отличает от ошибки). Один запрос "все
    тикеры" без списка символов + локальная фильтрация работает так же и
    без этого риска.
    """

    def _make_ingest(self, markets: dict):
        from src.data_ingest.market_data import MarketDataIngest
        ingest = MarketDataIngest("binance")
        ingest.exchange = AsyncMock()
        ingest.exchange.markets = markets
        return ingest

    def _spot_market(self, quote="USDT"):
        return {"spot": True, "active": True, "quote": quote}

    async def test_fetch_tickers_called_without_symbol_list(self):
        ingest = self._make_ingest({
            "BTC/USDT": self._spot_market(), "ETH/USDT": self._spot_market(),
        })
        ingest.exchange.fetch_tickers.return_value = {
            "BTC/USDT": {"quoteVolume": 100}, "ETH/USDT": {"quoteVolume": 200},
        }
        await ingest.get_tradable_symbols(quote="USDT", max_symbols=10)
        ingest.exchange.fetch_tickers.assert_awaited_once_with()

    async def test_sorted_by_volume_descending(self):
        ingest = self._make_ingest({
            "BTC/USDT": self._spot_market(), "ETH/USDT": self._spot_market(),
            "XRP/USDT": self._spot_market(),
        })
        ingest.exchange.fetch_tickers.return_value = {
            "BTC/USDT": {"quoteVolume": 100}, "ETH/USDT": {"quoteVolume": 300},
            "XRP/USDT": {"quoteVolume": 200},
        }
        result = await ingest.get_tradable_symbols(quote="USDT", max_symbols=10)
        self.assertEqual(result, ["ETH/USDT", "XRP/USDT", "BTC/USDT"])

    async def test_fetch_tickers_error_falls_back_without_crashing(self):
        ingest = self._make_ingest({
            "BTC/USDT": self._spot_market(), "ETH/USDT": self._spot_market(),
        })
        ingest.exchange.fetch_tickers.side_effect = RuntimeError("'str' object has no attribute 'keys'")
        result = await ingest.get_tradable_symbols(quote="USDT", max_symbols=10)
        self.assertEqual(set(result), {"BTC/USDT", "ETH/USDT"})

    async def test_excludes_leveraged_tokens_and_blacklist(self):
        ingest = self._make_ingest({
            "BTC/USDT": self._spot_market(), "BTCUP/USDT": self._spot_market(),
            "ETH/USDT": self._spot_market(), "BNB/BTC": self._spot_market(quote="BTC"),
        })
        ingest.exchange.fetch_tickers.return_value = {}
        result = await ingest.get_tradable_symbols(
            quote="USDT", blacklist=["ETH/USDT"], max_symbols=10,
        )
        self.assertEqual(set(result), {"BTC/USDT"})


class TestVolatilityAdjustment(unittest.TestCase):
    """
    TradingBot._volatility_multipliers — переводит предсказание
    volatility_predictor в коэффициенты для размера позиции (обратная
    связь с волатильностью) и ширины SL/TP (прямая связь), только для
    сигналов от стратегий. Раньше модель обучалась, но её предсказание
    нигде не использовалось.
    """

    def setUp(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        self.TradingBot = main_module.TradingBot
        self._saved = {
            k: getattr(settings, k) for k in (
                "volatility_adjustment_enabled", "volatility_baseline_pct",
                "volatility_size_min_mult", "volatility_size_max_mult",
                "volatility_sltp_min_mult", "volatility_sltp_max_mult",
            )
        }

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)

    def test_disabled_returns_neutral_multipliers(self):
        settings.volatility_adjustment_enabled = False
        self.assertEqual(self.TradingBot._volatility_multipliers(0.05), (1.0, 1.0))

    def test_none_prediction_returns_neutral(self):
        settings.volatility_adjustment_enabled = True
        self.assertEqual(self.TradingBot._volatility_multipliers(None), (1.0, 1.0))

    def test_high_volatility_shrinks_size_and_widens_sltp(self):
        settings.volatility_adjustment_enabled = True
        settings.volatility_baseline_pct = 2.0
        settings.volatility_size_min_mult, settings.volatility_size_max_mult = 0.5, 1.5
        settings.volatility_sltp_min_mult, settings.volatility_sltp_max_mult = 0.5, 2.0
        # Предсказано 4% против базовых 2% -> vol_ratio = 2.
        size_mult, sltp_mult = self.TradingBot._volatility_multipliers(0.04)
        self.assertAlmostEqual(size_mult, 0.5)
        self.assertAlmostEqual(sltp_mult, 2.0)

    def test_low_volatility_grows_size_and_narrows_sltp(self):
        settings.volatility_adjustment_enabled = True
        settings.volatility_baseline_pct = 2.0
        settings.volatility_size_min_mult, settings.volatility_size_max_mult = 0.5, 1.5
        settings.volatility_sltp_min_mult, settings.volatility_sltp_max_mult = 0.5, 2.0
        # Предсказано 1% против базовых 2% -> vol_ratio = 0.5.
        size_mult, sltp_mult = self.TradingBot._volatility_multipliers(0.01)
        self.assertAlmostEqual(size_mult, 1.5)  # 1/0.5=2.0, ограничено максимумом
        self.assertAlmostEqual(sltp_mult, 0.5)


class TestScaleSlTp(unittest.TestCase):
    """TradingBot._scale_sl_tp — отодвигает/приближает SL и TP от цены входа
    в заданное число раз, сохраняя сторону сделки."""

    def setUp(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        self.TradingBot = main_module.TradingBot

    def test_no_scaling_is_passthrough(self):
        self.assertEqual(self.TradingBot._scale_sl_tp("long", 100.0, 95.0, 110.0, 1.0), (95.0, 110.0))

    def test_widens_long_position(self):
        sl, tp = self.TradingBot._scale_sl_tp("long", 100.0, 95.0, 110.0, 2.0)
        self.assertAlmostEqual(sl, 90.0)
        self.assertAlmostEqual(tp, 120.0)

    def test_widens_short_position(self):
        sl, tp = self.TradingBot._scale_sl_tp("short", 100.0, 105.0, 90.0, 2.0)
        self.assertAlmostEqual(sl, 110.0)
        self.assertAlmostEqual(tp, 80.0)

    def test_narrows_position(self):
        sl, tp = self.TradingBot._scale_sl_tp("long", 100.0, 90.0, 120.0, 0.5)
        self.assertAlmostEqual(sl, 95.0)
        self.assertAlmostEqual(tp, 110.0)

    def test_none_values_pass_through(self):
        sl, tp = self.TradingBot._scale_sl_tp("long", 100.0, None, None, 2.0)
        self.assertIsNone(sl)
        self.assertIsNone(tp)

    def test_zero_entry_price_is_noop(self):
        self.assertEqual(self.TradingBot._scale_sl_tp("long", 0.0, 95.0, 110.0, 2.0), (95.0, 110.0))


class TestAtrSlTp(unittest.TestCase):
    """
    TradingBot._atr_sl_tp — ATR-адаптивный SL/TP (метод 1 из запроса
    пользователя): SL = ATR(14) × множитель, TP = SL × R:R, где R:R зависит
    от типа стратегии (трендовая/контртрендовая). Выключено по умолчанию.
    """

    def setUp(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        self.TradingBot = main_module.TradingBot
        self._saved = {
            k: getattr(settings, k) for k in (
                "atr_sltp_enabled", "atr_sl_multiplier", "atr_tp_rr_trend", "atr_tp_rr_countertrend",
            )
        }
        settings.atr_sltp_enabled = True
        settings.atr_sl_multiplier = 1.8
        settings.atr_tp_rr_trend = 3.0
        settings.atr_tp_rr_countertrend = 2.0

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)

    def test_disabled_returns_none(self):
        settings.atr_sltp_enabled = False
        sl, tp = self.TradingBot._atr_sl_tp("ema_cross", "long", 100.0, 5.0)
        self.assertIsNone(sl)
        self.assertIsNone(tp)

    def test_missing_atr_returns_none(self):
        sl, tp = self.TradingBot._atr_sl_tp("ema_cross", "long", 100.0, None)
        self.assertIsNone(sl)
        self.assertIsNone(tp)

    def test_nan_atr_returns_none(self):
        """
        pandas .get() на ещё не прогретом буфере свечей отдаёт NaN, а не
        None — NaN truthy в Python и не ловится ни "not atr", ни "atr <= 0",
        поэтому нужна отдельная проверка math.isnan (иначе NaN попал бы в
        SL/TP реального ордера).
        """
        sl, tp = self.TradingBot._atr_sl_tp("ema_cross", "long", 100.0, float("nan"))
        self.assertIsNone(sl)
        self.assertIsNone(tp)

    def test_trend_strategy_uses_trend_rr_long(self):
        """BTC/USD пример из запроса пользователя: ATR=1200, множитель 1.8 -> SL=2160, R:R=3 -> TP=6480."""
        sl, tp = self.TradingBot._atr_sl_tp("ema_cross", "long", 50000.0, 1200.0)
        self.assertAlmostEqual(sl, 50000.0 - 2160.0)
        self.assertAlmostEqual(tp, 50000.0 + 6480.0)

    def test_trend_strategy_uses_trend_rr_short(self):
        sl, tp = self.TradingBot._atr_sl_tp("ml_classifier", "short", 50000.0, 1200.0)
        self.assertAlmostEqual(sl, 50000.0 + 2160.0)
        self.assertAlmostEqual(tp, 50000.0 - 6480.0)

    def test_countertrend_strategy_uses_narrower_rr(self):
        sl, tp = self.TradingBot._atr_sl_tp("rsi_mr", "long", 100.0, 2.0)
        sl_distance = 2.0 * 1.8
        self.assertAlmostEqual(sl, 100.0 - sl_distance)
        self.assertAlmostEqual(tp, 100.0 + sl_distance * 2.0)

    def test_ensemble_voter_classified_as_trend(self):
        sl, tp = self.TradingBot._atr_sl_tp("ensemble_voter", "long", 100.0, 2.0)
        sl_distance = 2.0 * 1.8
        self.assertAlmostEqual(tp, 100.0 + sl_distance * 3.0)


class TestVolatilityPredictorRegistration(unittest.IsolatedAsyncioTestCase):
    """
    train_volatility_predictor раньше не писал ничего в MLModel — модель
    обучалась и сохранялась на диск, но GET /ml/models её не видел,
    activate_model() не находил ряд для обновления, а
    MLInference.predict_volatility() (которая ищет активную модель именно
    через ModelRegistry) никогда не смогла бы её загрузить.
    """

    async def test_registers_active_model_row(self):
        from sqlalchemy import select

        from src.db.models import MLModel
        from src.db.session import get_session
        from src.ml import ModelTrainer

        rng = np.random.default_rng(42)
        n = 150
        df = pd.DataFrame({
            "rsi_14": rng.uniform(20, 80, n),
            "natr_14": rng.uniform(0.5, 5.0, n),
            "realized_vol_20": rng.uniform(0.01, 0.05, n),
            "realized_vol_60": rng.uniform(0.01, 0.05, n),
            "volume_ratio": rng.uniform(0.5, 2.0, n),
            "atr_14": rng.uniform(10, 500, n),
            "return_1": rng.normal(0, 0.01, n),
            "return_3": rng.normal(0, 0.02, n),
            "label_volatility": rng.uniform(0.01, 0.05, n),
        })

        trainer = ModelTrainer()
        result = await trainer.train_volatility_predictor(training_data=df)
        self.assertIsNotNone(result)

        async with get_session() as session:
            row = (
                await session.execute(
                    select(MLModel).where(
                        MLModel.model_type == "volatility_predictor",
                        MLModel.version == result["version"],
                    )
                )
            ).scalar_one_or_none()
        self.assertIsNotNone(row)
        self.assertTrue(row.is_active)


class TestPredictionUsesNamedFeatures(unittest.IsolatedAsyncioTestCase):
    """
    predict_direction/predict_volatility передавали в model.predict[_proba]
    голый список без имён колонок, хотя модель обучалась на DataFrame с
    именованными колонками — sklearn/lightgbm логировал предупреждение
    "X does not have valid feature names, but ... was fitted with feature
    names" на КАЖДОЕ предсказание (то есть на каждую итерацию торгового
    цикла по каждому символу — не ошибка, но постоянный шум в логах).
    """

    async def test_predict_volatility_does_not_warn_about_feature_names(self):
        import warnings

        from src.ml import MLInference, ModelTrainer

        rng = np.random.default_rng(7)
        n = 150
        df = pd.DataFrame({
            "rsi_14": rng.uniform(20, 80, n),
            "natr_14": rng.uniform(0.5, 5.0, n),
            "realized_vol_20": rng.uniform(0.01, 0.05, n),
            "realized_vol_60": rng.uniform(0.01, 0.05, n),
            "volume_ratio": rng.uniform(0.5, 2.0, n),
            "atr_14": rng.uniform(10, 500, n),
            "return_1": rng.normal(0, 0.01, n),
            "return_3": rng.normal(0, 0.02, n),
            "label_volatility": rng.uniform(0.01, 0.05, n),
        })
        result = await ModelTrainer().train_volatility_predictor(training_data=df)
        self.assertIsNotNone(result)

        inference = MLInference()
        inference.load_model("volatility_predictor", result["model_path"])

        features = {
            "rsi_14": 55.0, "natr_14": 1.5, "realized_vol_20": 0.02,
            "realized_vol_60": 0.02, "volume_ratio": 1.1, "atr_14": 100.0,
            "return_1": 0.001, "return_3": 0.002,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pred = await inference.predict_volatility(features)

        self.assertIsInstance(pred, float)
        feature_name_warnings = [w for w in caught if "feature names" in str(w.message)]
        self.assertEqual(feature_name_warnings, [], [str(w.message) for w in feature_name_warnings])


class TestRealModeSwitchActuallyConnects(unittest.IsolatedAsyncioTestCase):
    """
    apply_settings_update() при переключении trading_mode -> "real" вызывал
    execution_engine.initialize(...), пока execution_engine.is_paper всё
    ещё оставался True (выставляется один раз при конструировании движка
    и больше нигде не сбрасывается) — а initialize() САМ ПЕРВЫМ ДЕЛОМ
    проверяет self.is_paper и, если он ещё True, молча остаётся в paper,
    даже не пытаясь подключиться к бирже. Переключение в real через
    дашборд не делало ровным счётом ничего: настройка менялась, а движок
    молча оставался в paper и логировал "Paper Trading режим".
    """

    def setUp(self):
        self._saved = {
            k: getattr(settings, k) for k in (
                "trading_mode", "active_exchange", "use_exchange_sandbox",
                "okx_api_key", "okx_api_secret", "okx_passphrase",
            )
        }
        from src.risk.risk_manager import risk_manager as global_risk_manager
        self._saved_risk_state = dict(global_risk_manager.state.__dict__)

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)
        from src.risk.risk_manager import risk_manager as global_risk_manager
        global_risk_manager.state.__dict__.update(self._saved_risk_state)

    async def test_switching_to_real_actually_calls_initialize(self):
        import src.execution.executor as executor_module
        from src.web.settings_store import apply_settings_update

        settings.trading_mode = "paper"
        settings.okx_api_key = "okx-key"
        settings.okx_api_secret = "okx-secret"
        settings.okx_passphrase = "okx-pass"

        original_engine = executor_module.execution_engine
        engine = ExecutionEngine()  # is_paper=True, как только что сконструированный
        executor_module.execution_engine = engine
        try:
            mock_exchange = AsyncMock()
            mock_exchange.fetch_balance = AsyncMock(return_value={})
            mock_exchange.set_sandbox_mode = MagicMock()
            with patch("src.execution.executor.ccxt.okx", return_value=mock_exchange):
                result = await apply_settings_update({
                    "trading_mode": "real", "active_exchange": "okx",
                })
            self.assertEqual(result["errors"], {})
            self.assertFalse(engine.is_paper)
            self.assertIsNotNone(engine.exchange)
        finally:
            executor_module.execution_engine = original_engine


class TestMultiExchangeCredentials(unittest.IsolatedAsyncioTestCase):
    """
    ExecutionEngine.initialize(exchange_id) раньше БЕЗУСЛОВНО брал
    settings.binance_api_key/secret независимо от exchange_id — реальное
    подключение к Bybit (или любой другой бирже) шло по Binance-ключам.
    Плюс новая поддержка OKX (требует passphrase) и демо-режима
    (ccxt set_sandbox_mode).
    """

    def setUp(self):
        self._saved = {
            k: getattr(settings, k) for k in (
                "trading_mode", "binance_api_key", "binance_api_secret",
                "bybit_api_key", "bybit_api_secret",
                "okx_api_key", "okx_api_secret", "okx_passphrase",
                "use_exchange_sandbox",
            )
        }
        from src.risk.risk_manager import risk_manager as global_risk_manager
        self._saved_risk_state = dict(global_risk_manager.state.__dict__)

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)
        from src.risk.risk_manager import risk_manager as global_risk_manager
        global_risk_manager.state.__dict__.update(self._saved_risk_state)

    async def test_bybit_uses_bybit_credentials_not_binance(self):
        settings.trading_mode = "real"
        settings.binance_api_key = "WRONG-should-not-be-used"
        settings.binance_api_secret = "WRONG-should-not-be-used"
        settings.bybit_api_key = "correct-bybit-key"
        settings.bybit_api_secret = "correct-bybit-secret"
        settings.use_exchange_sandbox = True

        engine = ExecutionEngine()
        engine.is_paper = False
        mock_exchange = AsyncMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={})
        mock_exchange.enable_demo_trading = MagicMock()  # ccxt: синхронный
        with patch("src.execution.executor.ccxt.bybit", return_value=mock_exchange) as mock_cls:
            await engine.initialize("bybit")

        mock_cls.assert_called_once()
        config = mock_cls.call_args.args[0]
        self.assertEqual(config["apiKey"], "correct-bybit-key")
        self.assertEqual(config["secret"], "correct-bybit-secret")
        # Bybit: demo-ключ живёт на api-demo.bybit.com (enable_demo_trading),
        # а НЕ на testnet.bybit.com (set_sandbox_mode) — это разные песочницы
        # с разными ключами, см. комментарий в executor.py.
        mock_exchange.enable_demo_trading.assert_called_once_with(True)
        mock_exchange.set_sandbox_mode.assert_not_called()
        self.assertFalse(engine.is_paper)

    async def test_okx_passes_passphrase_as_password(self):
        settings.trading_mode = "real"
        settings.okx_api_key = "okx-key"
        settings.okx_api_secret = "okx-secret"
        settings.okx_passphrase = "okx-passphrase"
        settings.use_exchange_sandbox = True

        engine = ExecutionEngine()
        engine.is_paper = False
        mock_exchange = AsyncMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={})
        mock_exchange.set_sandbox_mode = MagicMock()
        with patch("src.execution.executor.ccxt.okx", return_value=mock_exchange) as mock_cls:
            await engine.initialize("okx")

        config = mock_cls.call_args.args[0]
        self.assertEqual(config["apiKey"], "okx-key")
        self.assertEqual(config["secret"], "okx-secret")
        self.assertEqual(config["password"], "okx-passphrase")

    async def test_okx_missing_passphrase_falls_back_to_paper(self):
        settings.trading_mode = "real"
        settings.okx_api_key = "okx-key"
        settings.okx_api_secret = "okx-secret"
        settings.okx_passphrase = None

        engine = ExecutionEngine()
        engine.is_paper = False
        await engine.initialize("okx")

        self.assertTrue(engine.is_paper)

    async def test_sandbox_disabled_does_not_call_set_sandbox_mode(self):
        settings.trading_mode = "real"
        settings.binance_api_key = "key"
        settings.binance_api_secret = "secret"
        settings.use_exchange_sandbox = False

        engine = ExecutionEngine()
        engine.is_paper = False
        mock_exchange = AsyncMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={})
        with patch("src.execution.executor.ccxt.binance", return_value=mock_exchange):
            await engine.initialize("binance")

        mock_exchange.set_sandbox_mode.assert_not_called()

    async def test_bybit_sandbox_disabled_does_not_call_enable_demo_trading(self):
        settings.trading_mode = "real"
        settings.bybit_api_key = "key"
        settings.bybit_api_secret = "secret"
        settings.use_exchange_sandbox = False

        engine = ExecutionEngine()
        engine.is_paper = False
        mock_exchange = AsyncMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={})
        with patch("src.execution.executor.ccxt.bybit", return_value=mock_exchange):
            await engine.initialize("bybit")

        mock_exchange.enable_demo_trading.assert_not_called()
        mock_exchange.set_sandbox_mode.assert_not_called()


class TestExtractUsdtBalance(unittest.TestCase):
    """
    ccxt fetch_balance() кладёт баланс валюты во ВЛОЖЕННЫЙ словарь
    (balance['free']['USDT'] и/или balance['USDT'] = {'free':,'used':,'total':}),
    а не как плоское число на верхнем уровне. Старый код фильтровал
    isinstance(v, (int, float)) по balance.items() верхнего уровня и
    поэтому всегда получал 0, независимо от реального остатка на счёте.
    """

    def test_reads_from_free_dict(self):
        balance = {"free": {"USDT": 4987.65, "BTC": 0.0}, "used": {}, "total": {}}
        self.assertEqual(ExecutionEngine._extract_usdt_balance(balance), 4987.65)

    def test_falls_back_to_nested_currency_dict(self):
        balance = {"USDT": {"free": 5000.0, "used": 12.5, "total": 5012.5}}
        self.assertEqual(ExecutionEngine._extract_usdt_balance(balance), 5000.0)

    def test_missing_usdt_returns_zero(self):
        balance = {"free": {"BTC": 0.1}, "BTC": {"free": 0.1, "used": 0, "total": 0.1}}
        self.assertEqual(ExecutionEngine._extract_usdt_balance(balance), 0.0)

    def test_empty_balance_returns_zero(self):
        self.assertEqual(ExecutionEngine._extract_usdt_balance({}), 0.0)


class TestRealBalanceReseedsRiskState(unittest.IsolatedAsyncioTestCase):
    """
    RiskState.start_balance был захардкожен на settings.startup_capital_usdt
    (paper-дефолт, напр. 10000) и никогда не пересчитывался от реального
    баланса биржи при входе в real-режим — после переключения в real
    просадка считалась от чужого числа и почти сразу давала ложное
    срабатывание max_drawdown_pct с автопаузой торговли (в связке с багом
    _extract_usdt_balance выше — ровно 100% просадки, т.к. баланс к тому
    же читался как 0).
    """

    def setUp(self):
        self._saved = {
            k: getattr(settings, k) for k in (
                "trading_mode", "binance_api_key", "binance_api_secret",
                "use_exchange_sandbox", "risk_max_drawdown_pct",
            )
        }
        from src.risk.risk_manager import risk_manager as global_risk_manager
        self._saved_risk_state = dict(global_risk_manager.state.__dict__)

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)
        from src.risk.risk_manager import risk_manager as global_risk_manager
        global_risk_manager.state.__dict__.update(self._saved_risk_state)

    async def test_initialize_reseeds_start_balance_from_real_exchange(self):
        from src.risk.risk_manager import risk_manager as global_risk_manager

        settings.trading_mode = "real"
        settings.binance_api_key = "key"
        settings.binance_api_secret = "secret"
        settings.use_exchange_sandbox = True
        settings.risk_max_drawdown_pct = 15

        global_risk_manager.state.start_balance = 10000.0
        global_risk_manager.state.current_balance = 0.0
        global_risk_manager.state.paused = True  # как после ложной 100%-просадки

        engine = ExecutionEngine()
        engine.is_paper = False
        mock_exchange = AsyncMock()
        mock_exchange.fetch_balance = AsyncMock(
            return_value={"free": {"USDT": 4987.65}, "USDT": {"free": 4987.65, "used": 0, "total": 4987.65}}
        )
        mock_exchange.set_sandbox_mode = MagicMock()
        with patch("src.execution.executor.ccxt.binance", return_value=mock_exchange):
            await engine.initialize("binance")

        # Тестовая БД общая на сессию — другие тесты могут оставлять свои
        # незакрытые real-позиции для биржи "binance" (не относится к делу
        # здесь: они не куплены по этому fetch_balance и не влияют на смысл
        # проверки). start_balance честно учитывает стоимость ЛЮБЫХ
        # восстановленных позиций (см. test_initialize_baseline_includes_
        # preexisting_real_positions рядом), поэтому ожидание считаем от
        # того же restored-состояния, а не от жёстко зашитого числа —
        # без открытых позиций оно совпадёт с голым кэшем 4987.65.
        expected_positions_value = sum(
            pos["amount"] * pos["entry_price"]
            for pos in engine.real_positions.values() if pos.get("side") == "long"
        )
        expected_baseline = 4987.65 + expected_positions_value

        self.assertAlmostEqual(global_risk_manager.state.start_balance, expected_baseline)
        self.assertAlmostEqual(global_risk_manager.state.current_balance, expected_baseline)
        self.assertFalse(global_risk_manager.state.paused)
        self.assertEqual(global_risk_manager.state.total_drawdown_pct, 0.0)
        self.assertEqual(engine.paper_balance, 4987.65)

    async def test_initialize_baseline_includes_preexisting_real_positions(self):
        """
        total_usdt из fetch_balance() — это ТОЛЬКО свободный (неинвестированный)
        кэш на бирже. Если на счету уже есть открытые real-позиции (обычный
        случай при КАЖДОМ рестарте процесса, а не только при первом
        подключении), старый код всё равно брал за базу для просадки один
        голый кэш — а _compute_equity() в main.py дальше на каждой итерации
        считает cash + стоимость позиций. Несопоставимые базы для
        start_balance и current_balance превращали "просадку" в дашборде в
        гигантский фиктивный "профит" (реальный инцидент: свободный кэш
        $12.79, equity с учётом позиций — $7156, просадка показывала
        -55830%). База для просадки должна включать стоимость уже открытых
        позиций по цене входа — так же, как их посчитал бы _compute_equity.
        """
        from sqlalchemy import select, update
        from src.db.session import get_session
        from src.db.models import Order, Symbol, Exchange
        from src.risk.risk_manager import risk_manager as global_risk_manager

        async with get_session() as session:
            real_ex = (
                await session.execute(select(Exchange).where(Exchange.name == "binance", Exchange.is_paper == False))
            ).scalar_one_or_none()
            if real_ex is None:
                real_ex = Exchange(name="binance", is_paper=False)
                session.add(real_ex)
                await session.flush()
            sym = Symbol(exchange_id=real_ex.id, symbol="BASELINE1/USDT", base_asset="BASELINE1", quote_asset="USDT")
            session.add(sym)
            await session.flush()
            order = Order(
                exchange_id=real_ex.id, symbol_id=sym.id, side="buy", order_type="market",
                amount=100.0, price=39.5, status="filled", filled_amount=100.0, filled_price=39.5, fee=0.0,
            )
            session.add(order)
            await session.commit()
            order_id = order.id

        settings.trading_mode = "real"
        settings.binance_api_key = "key"
        settings.binance_api_secret = "secret"
        settings.use_exchange_sandbox = True

        try:
            engine = ExecutionEngine()
            engine.is_paper = False
            mock_exchange = AsyncMock()
            mock_exchange.fetch_balance = AsyncMock(
                return_value={"free": {"USDT": 12.79}, "USDT": {"free": 12.79, "used": 0, "total": 12.79}}
            )
            mock_exchange.set_sandbox_mode = MagicMock()
            with patch("src.execution.executor.ccxt.binance", return_value=mock_exchange):
                await engine.initialize("binance")

            self.assertIn("BASELINE1/USDT", engine.real_positions)
            our_position = engine.real_positions["BASELINE1/USDT"]
            self.assertAlmostEqual(our_position["amount"] * our_position["entry_price"], 3950.0)

            # Тестовая БД общая на сессию — у других тестов в этом файле могли
            # остаться свои незакрытые real-позиции (не относится к делу здесь),
            # поэтому ожидаемое значение считаем от того же restored-состояния,
            # а не от жёстко зашитого числа: главное утверждение теста — что
            # start_balance учитывает ВСЕ восстановленные позиции, а не только
            # свободный кэш.
            expected_positions_value = sum(
                pos["amount"] * pos["entry_price"]
                for pos in engine.real_positions.values() if pos.get("side") == "long"
            )
            expected_baseline = 12.79 + expected_positions_value
            self.assertAlmostEqual(global_risk_manager.state.start_balance, expected_baseline)
            self.assertAlmostEqual(global_risk_manager.state.current_balance, expected_baseline)
            self.assertEqual(global_risk_manager.state.total_drawdown_pct, 0.0)
            # Ключевая регрессия: раньше start_balance был бы равен голому кэшу
            # (12.79), полностью игнорируя стоимость уже открытых позиций.
            self.assertGreater(global_risk_manager.state.start_balance, 12.79 + 3950.0 - 1.0)
        finally:
            # Тестовая БД общая на сессию — не оставляем открытую позицию,
            # иначе она "просочится" в другие тесты этого класса/файла,
            # которые тоже восстанавливают real-позиции из БД (проверено:
            # без этой очистки следующий по алфавиту тест в этом же классе
            # получал заниженный ожидаемый start_balance, т.к. видел и эту
            # позицию тоже).
            async with get_session() as session:
                await session.execute(
                    update(Order).where(Order.id == order_id).values(status="rejected")
                )
                await session.commit()


class TestOkxTradePermissionCheck(unittest.IsolatedAsyncioTestCase):
    """
    OKX отдаёт реально выданные API-ключу права прямо в GET /account/config
    (поле "perm", напр. "read_only,trade") — раньше отсутствие права
    "trade" обнаруживалось только когда первый реальный ордер падал с
    50123 "API Key does not have trading permission for the Crypto",
    иногда спустя долгое время после подключения. Это read-only проверка
    при initialize(), не блокирующая подключение.
    """

    def setUp(self):
        self._saved = {
            k: getattr(settings, k) for k in (
                "trading_mode", "okx_api_key", "okx_api_secret",
                "okx_passphrase", "use_exchange_sandbox",
            )
        }
        settings.trading_mode = "real"
        settings.okx_api_key = "okx-key"
        settings.okx_api_secret = "okx-secret"
        settings.okx_passphrase = "okx-pass"
        settings.use_exchange_sandbox = True

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)

    async def test_warns_when_trade_permission_missing(self):
        engine = ExecutionEngine()
        engine.is_paper = False
        mock_exchange = AsyncMock()
        mock_exchange.set_sandbox_mode = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={})
        mock_exchange.fetch_accounts = AsyncMock(return_value=[{"info": {"perm": "read_only"}}])
        with patch("src.execution.executor.ccxt.okx", return_value=mock_exchange):
            with self.assertLogs("src.execution.executor", level="WARNING") as logs:
                await engine.initialize("okx")

        self.assertTrue(any("Trade" in msg for msg in logs.output))

    async def test_no_warning_when_trade_permission_present(self):
        engine = ExecutionEngine()
        engine.is_paper = False
        mock_exchange = AsyncMock()
        mock_exchange.set_sandbox_mode = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={})
        mock_exchange.fetch_accounts = AsyncMock(return_value=[{"info": {"perm": "read_only,trade"}}])
        with patch("src.execution.executor.ccxt.okx", return_value=mock_exchange):
            await engine.initialize("okx")

        self.assertFalse(engine.is_paper)

    async def test_fetch_accounts_failure_does_not_break_initialize(self):
        engine = ExecutionEngine()
        engine.is_paper = False
        mock_exchange = AsyncMock()
        mock_exchange.set_sandbox_mode = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={})
        mock_exchange.fetch_accounts = AsyncMock(side_effect=Exception("network error"))
        with patch("src.execution.executor.ccxt.okx", return_value=mock_exchange):
            await engine.initialize("okx")

        self.assertFalse(engine.is_paper)


class TestWebSocketBroadcast(unittest.IsolatedAsyncioTestCase):
    """
    broadcast_event() раньше отправлял клиентам только event.payload — у
    TradeEvent (и других подклассов Event) собственные данные (symbol,
    pnl, direction, outcome, is_opening...) хранятся в типизированных
    полях датакласса, а не в payload, который у них всегда None: клиент
    получал "event_type": "trade_event" без единого реального поля сделки.
    Отдельно setup_websocket_broadcast() существовал, но нигде не
    вызывался — event_bus ни разу не подписывался на трансляцию, поэтому
    вообще ничего не долетало до /ws вне зависимости от первого бага.
    """

    async def test_broadcast_event_includes_dataclass_fields(self):
        from src.web.websocket import broadcast_event

        with patch("src.web.websocket.ws_manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            event = TradeEvent(
                trade_id=42, symbol="BTC/USDT", direction="long",
                entry_price=50000.0, exit_price=52000.0, amount=0.1,
                pnl=200.0, pnl_pct=4.0, outcome="win", is_opening=False,
            )
            await broadcast_event(event)

        mock_manager.broadcast.assert_called_once()
        sent = mock_manager.broadcast.call_args.args[0]
        self.assertEqual(sent["type"], "event")
        self.assertEqual(sent["event_type"], "trade_event")
        self.assertEqual(sent["symbol"], "BTC/USDT")
        self.assertEqual(sent["pnl"], 200.0)
        self.assertEqual(sent["outcome"], "win")
        self.assertFalse(sent["is_opening"])

    async def test_setup_websocket_broadcast_subscribes_to_event_bus(self):
        from src.web.websocket import setup_websocket_broadcast

        original_subscribers = dict(event_bus._subscribers)
        try:
            event_bus._subscribers = {}
            setup_websocket_broadcast()

            with patch("src.web.websocket.ws_manager") as mock_manager:
                mock_manager.broadcast = AsyncMock()
                await event_bus.publish(TradeEvent(symbol="ETH/USDT", is_opening=True))

            mock_manager.broadcast.assert_called_once()
            sent = mock_manager.broadcast.call_args.args[0]
            self.assertEqual(sent["symbol"], "ETH/USDT")
        finally:
            event_bus._subscribers = original_subscribers

    async def test_websocket_sends_keepalive_ping_on_idle_timeout(self):
        """
        Реальный инцидент: фронтенд ничего не шлёт на /ws (только слушает
        broadcast), поэтому receive_text() без таймаута ждал неограниченно —
        соединение простаивало в обе стороны, и обратные прокси/браузер
        тихо рвали "неактивный" WebSocket через некоторое время. Сервер
        теперь должен сам отправлять ping при таймауте ожидания.
        """
        import src.web.api as api_module
        from fastapi import WebSocketDisconnect

        mock_ws = AsyncMock()
        mock_ws.cookies = {api_module.auth.SESSION_COOKIE_NAME: "valid-token"}
        mock_ws.receive_text = AsyncMock(
            side_effect=[TimeoutError(), WebSocketDisconnect()]
        )

        with patch.object(api_module.auth, "verify_session", return_value=True):
            await api_module.websocket_endpoint(mock_ws)

        mock_ws.send_json.assert_any_call({"type": "ping"})

    async def test_websocket_rejects_unauthenticated(self):
        import src.web.api as api_module

        mock_ws = AsyncMock()
        mock_ws.cookies = {}

        with patch.object(api_module.auth, "verify_session", return_value=False):
            await api_module.websocket_endpoint(mock_ws)

        mock_ws.close.assert_called_once_with(code=4401)
        mock_ws.accept.assert_not_called()


class TestManualTrading(unittest.IsolatedAsyncioTestCase):
    """
    Вкладка 'Ручная торговля': POST /manual/order открывает сделку через
    тот же execution_engine.create_order(), что и стратегии/Telegram, с
    strategy_id="manual" — и регистрирует её в основном торговом цикле
    (src.main.current_bot), иначе SL/TP такой позиции никогда бы не
    проверялись. POST /positions/{symbol}/edit меняет SL/TP на лету.
    """

    def setUp(self):
        import src.main as main_module
        import src.web.api as api_module
        from fastapi import HTTPException

        self.main_module = main_module
        self.api_module = api_module
        self.HTTPException = HTTPException
        self._saved_trading_mode = settings.trading_mode
        self._saved_current_bot = main_module.current_bot
        self._saved_engine = api_module.execution_engine

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode
        self.main_module.current_bot = self._saved_current_bot
        self.api_module.execution_engine = self._saved_engine

    async def _install_engine_and_bot(self, exchange_id="binance", is_paper=True):
        from src.execution.executor import ExecutionEngine

        engine = ExecutionEngine()
        settings.trading_mode = "paper" if is_paper else "real"
        engine.is_paper = is_paper
        engine.exchange_id = exchange_id
        self.api_module.execution_engine = engine

        bot = self.main_module.TradingBot()
        bot.ingest = AsyncMock()
        bot.ingest.fetch_ohlcv = AsyncMock(return_value=None)
        self.main_module.current_bot = bot
        return engine, bot

    async def test_create_manual_order_paper_registers_in_bot_and_engine(self):
        engine, bot = await self._install_engine_and_bot(is_paper=True)

        result = await self.api_module.create_manual_order(self.api_module.ManualOrderCreate(
            symbol="MANUALPAPER1/USDT", side="buy", order_type="market",
            amount=2.0, price=100.0, stop_loss=90.0, take_profit=120.0,
        ))
        self.assertTrue(result["success"])
        self.assertIn("MANUALPAPER1/USDT", engine.paper_positions)
        self.assertIn("MANUALPAPER1/USDT", bot.open_positions)
        self.assertEqual(bot.open_positions["MANUALPAPER1/USDT"]["strategy_id"], "manual")
        self.assertEqual(bot.open_positions["MANUALPAPER1/USDT"]["sl"], 90.0)
        self.assertEqual(bot.open_positions["MANUALPAPER1/USDT"]["tp"], 120.0)
        self.assertIn("MANUALPAPER1/USDT", bot.active_symbols)

    async def test_create_manual_order_computes_amount_from_usdt(self):
        await self._install_engine_and_bot(is_paper=True)

        result = await self.api_module.create_manual_order(self.api_module.ManualOrderCreate(
            symbol="MANUALPAPER2/USDT", side="buy", order_type="limit",
            amount_usdt=500.0, price=50.0,
        ))
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["amount"], 10.0)

    async def test_create_manual_order_rejects_short_on_real_mode(self):
        await self._install_engine_and_bot(exchange_id="bybit", is_paper=False)

        with self.assertRaises(self.HTTPException) as ctx:
            await self.api_module.create_manual_order(self.api_module.ManualOrderCreate(
                symbol="MANUALREAL1/USDT", side="sell", amount=1.0, price=100.0,
            ))
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_create_manual_order_rejects_duplicate_position(self):
        await self._install_engine_and_bot(is_paper=True)

        await self.api_module.create_manual_order(self.api_module.ManualOrderCreate(
            symbol="MANUALDUP1/USDT", side="buy", amount=1.0, price=100.0,
        ))
        with self.assertRaises(self.HTTPException) as ctx:
            await self.api_module.create_manual_order(self.api_module.ManualOrderCreate(
                symbol="MANUALDUP1/USDT", side="buy", amount=1.0, price=100.0,
            ))
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_edit_position_updates_engine_and_bot_state(self):
        engine, bot = await self._install_engine_and_bot(is_paper=True)

        await self.api_module.create_manual_order(self.api_module.ManualOrderCreate(
            symbol="MANUALEDIT1/USDT", side="buy", amount=1.0, price=100.0,
            stop_loss=90.0, take_profit=110.0,
        ))

        result = await self.api_module.edit_position(
            self.api_module.PositionEditRequest(symbol="MANUALEDIT1/USDT", stop_loss=95.0)
        )
        self.assertEqual(result["stop_loss"], 95.0)
        self.assertEqual(result["take_profit"], 110.0, "take_profit не был передан — не должен измениться")
        self.assertEqual(engine.paper_positions["MANUALEDIT1/USDT"]["stop_loss"], 95.0)
        self.assertEqual(bot.open_positions["MANUALEDIT1/USDT"]["sl"], 95.0)

        cleared = await self.api_module.edit_position(
            self.api_module.PositionEditRequest(symbol="MANUALEDIT1/USDT", clear_take_profit=True)
        )
        self.assertIsNone(cleared["take_profit"])
        self.assertIsNone(bot.open_positions["MANUALEDIT1/USDT"]["tp"])

    async def test_edit_position_resyncs_exchange_stop_loss_in_real_mode(self):
        """
        Ручное изменение SL из дашборда (POST /positions/edit) должно сразу
        переставлять биржевой SL-ордер под новую цену в real-режиме —
        иначе выставленный ранее условный ордер продолжил бы защищать
        позицию по старой, уже неактуальной цене.
        """
        engine, bot = await self._install_engine_and_bot(exchange_id="bybit", is_paper=False)
        engine.exchange = AsyncMock()
        engine.exchange.create_market_sell_order.return_value = {"id": "resync-sl-order-2"}
        engine.real_positions["MANUALEDITREAL1/USDT"] = {
            "amount": 10.0, "entry_price": 100.0, "side": "long",
            "stop_loss": 90.0, "take_profit": 110.0, "strategy_id": "manual",
            "sl_order_id": "resync-sl-order-1",
        }
        bot.open_positions["MANUALEDITREAL1/USDT"] = {"sl": 90.0, "tp": 110.0}

        result = await self.api_module.edit_position(
            self.api_module.PositionEditRequest(symbol="MANUALEDITREAL1/USDT", stop_loss=95.0)
        )
        self.assertEqual(result["stop_loss"], 95.0)
        engine.exchange.cancel_order.assert_called_once_with("resync-sl-order-1", "MANUALEDITREAL1/USDT")
        engine.exchange.create_market_sell_order.assert_called_once_with(
            "MANUALEDITREAL1/USDT", 10.0, params={"stopLossPrice": 95.0},
        )
        self.assertEqual(engine.real_positions["MANUALEDITREAL1/USDT"]["sl_order_id"], "resync-sl-order-2")

    async def test_edit_position_unknown_symbol_404(self):
        await self._install_engine_and_bot(is_paper=True)
        with self.assertRaises(self.HTTPException) as ctx:
            await self.api_module.edit_position(
                self.api_module.PositionEditRequest(symbol="NOPOSITION1/USDT", stop_loss=1.0)
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_list_trades_filters_by_strategy_id(self):
        engine, bot = await self._install_engine_and_bot(is_paper=True)

        order = await self.api_module.create_manual_order(self.api_module.ManualOrderCreate(
            symbol="MANUALLIST1/USDT", side="buy", amount=1.0, price=100.0,
        ))
        del bot.open_positions["MANUALLIST1/USDT"]
        await engine.close_paper_position(
            symbol="MANUALLIST1/USDT", side="long", entry_price=100.0, amount=1.0,
            exit_price=110.0, reason="manual", entry_fee=0.1, holding_seconds=30,
            order_open_id=order["order_id"], strategy_id="manual",
        )

        result = await self.api_module.list_trades(limit=200, offset=0, strategy_id="manual")
        symbols = {t["symbol"] for t in result["trades"]}
        self.assertIn("MANUALLIST1/USDT", symbols)

        result_other = await self.api_module.list_trades(limit=200, offset=0, strategy_id="telegram_signal")
        symbols_other = {t["symbol"] for t in result_other["trades"]}
        self.assertNotIn("MANUALLIST1/USDT", symbols_other)


if __name__ == "__main__":
    unittest.main()
