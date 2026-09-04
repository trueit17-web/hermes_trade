"""Тесты для крипто-трейдер бота."""
import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, PropertyMock, patch

import ccxt.async_support as ccxt
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
        # рассчитывается по стандартной ставке (см. _resolve_fee), но её
        # валюта в этом расчётном случае всегда quote (см.
        # test_resolve_fee_calculates_estimate_when_exchange_data_unavailable) —
        # объём позиции не уменьшается.
        self.assertAlmostEqual(self.engine.real_positions["BALDIFF1/USDT"]["amount"], 2011.85)

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
        комиссии биржи). Валюта оценки — всегда quote (даже для buy):
        estimated = filled_amount(base) × fill_price(quote/base) × pct —
        число в quote-валюте по построению формулы, независимо от side.
        Реальный инцидент: HYPE/USDT — оценка в USDT была помечена как
        "в HYPE", close_real_position домножила её на entry_price ЕЩЁ РАЗ
        при конвертации, и прибыльный Take Profit 1 показал PnL -13.47.
        """
        fee, currency = self.engine._resolve_fee(None, 1000.0, 2.0, "buy", "RESOLVEFEE1/USDT")
        self.assertAlmostEqual(fee, 1000.0 * 2.0 * (settings.paper_fee_pct / 100))
        self.assertEqual(currency, "USDT")

        fee, currency = self.engine._resolve_fee({"cost": 0, "currency": "USDT"}, 1000.0, 2.0, "sell", "RESOLVEFEE1/USDT")
        self.assertAlmostEqual(fee, 1000.0 * 2.0 * (settings.paper_fee_pct / 100))
        self.assertEqual(currency, "USDT")

        fee, currency = self.engine._resolve_fee({"cost": 3.5, "currency": "USDT"}, 1000.0, 2.0, "sell", "RESOLVEFEE1/USDT")
        self.assertEqual(fee, 3.5)
        self.assertEqual(currency, "USDT")

    async def test_resolve_fee_uses_requested_vs_filled_diff_for_buy_when_available(self):
        """
        На споте комиссия покупки часто списывается биржей из самого
        актива ДО того, как объём попадает в order["filled"] — разница
        между запрошенным и фактически исполненным объёмом (в base-валюте)
        точнее стандартной ставки, когда её можно посчитать (запрошено
        строго больше исполненного).
        """
        fee, currency = self.engine._resolve_fee(
            None, 998.5, 2.0, "buy", "RESOLVEFEE2/USDT", amount_requested=1000.0,
        )
        self.assertAlmostEqual(fee, 1.5)
        self.assertEqual(currency, "RESOLVEFEE2")

    async def test_resolve_fee_ignores_diff_for_sell_side(self):
        """
        Для sell комиссия обычно из quote (полученной от продажи), а не из
        проданного base-объёма — разница запрошенного/исполненного там не
        комиссия, а округление лота. Должен остаться процентный фолбэк.
        """
        fee, currency = self.engine._resolve_fee(
            None, 998.5, 2.0, "sell", "RESOLVEFEE3/USDT", amount_requested=1000.0,
        )
        self.assertAlmostEqual(fee, 998.5 * 2.0 * (settings.paper_fee_pct / 100))
        self.assertEqual(currency, "USDT")

    async def test_resolve_fee_falls_back_to_percentage_when_filled_exceeds_requested(self):
        """Исполнено не меньше запрошенного — разница не сигнал о комиссии, используем ставку."""
        fee, currency = self.engine._resolve_fee(
            None, 1000.0, 2.0, "buy", "RESOLVEFEE4/USDT", amount_requested=1000.0,
        )
        self.assertAlmostEqual(fee, 1000.0 * 2.0 * (settings.paper_fee_pct / 100))
        self.assertEqual(currency, "USDT")

    async def test_close_real_position_profitable_tp_stays_profitable_when_fee_estimated(self):
        """
        Реальный инцидент: HYPE/USDT, buy без реальных данных о комиссии от
        биржи (fetch_order_trades/order["fee"] недоступны) — оценочная
        комиссия _resolve_fee (всегда в quote-валюте, см. тест выше) раньше
        помечалась как base для покупки. close_real_position конвертирует
        комиссию открытия в USDT-эквивалент, ТОЛЬКО если её валюта — base
        (opening_order.fee_currency == base_currency) — при неверной метке
        уже quote-валютное число домножалось на entry_price ЕЩЁ РАЗ, и
        реально прибыльное частичное закрытие (цена выросла с 83.12 до
        84.22) показывало PnL -13.47 вместо примерно +2.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"HYPEPNL1": 0.0}, "HYPEPNL1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "hypepnl-open-1", "filled": 4.506489, "average": 83.12, "price": None,
            "fee": {"cost": 0, "currency": None},
        }
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=None)
        self.engine.exchange.fetch_my_trades = AsyncMock(return_value=None)
        opening_order = await self.engine.create_order(
            symbol="HYPEPNL1/USDT", side="buy", amount=4.506489, price=83.12, order_type="market",
        )
        self.assertIsNotNone(opening_order)
        self.assertEqual(opening_order.fee_currency, "USDT")
        full_amount = self.engine.real_positions["HYPEPNL1/USDT"]["amount"]
        entry_fee_total = float(opening_order.fee)
        close_amount = full_amount * 0.5

        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"HYPEPNL1": close_amount},
            "HYPEPNL1": {"free": close_amount, "used": 0, "total": close_amount},
        })
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "hypepnl-close-1", "filled": close_amount, "average": 84.22, "price": None,
            "fee": {"cost": close_amount * 84.22 * (settings.paper_fee_pct / 100), "currency": "USDT"},
        }

        result = await self.engine.close_real_position(
            symbol="HYPEPNL1/USDT", side="long", entry_price=83.12, amount=close_amount,
            reason="take_profit_1", entry_fee=entry_fee_total * 0.5, holding_seconds=11488,
            order_open_id=opening_order.id,
        )
        self.assertIsNotNone(result)
        self.assertGreater(result["pnl"], 0, "цена выросла с 83.12 до 84.22 — закрытие должно быть в плюсе")

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

    async def test_close_real_position_skips_sell_when_sl_cancel_unconfirmed(self):
        """
        Реальный инцидент (прод, CHIP/USDT): если отмена старого SL-ордера
        падает с неоднозначной ошибкой (не "ордера уже нет", а что-то
        другое — тот же 170131 Insufficient balance), собственная продажа
        всё равно неизбежно конфликтует за тот же объём с ещё живым
        SL-ордером и падает той же ошибкой — КАЖДЫЙ раз, без исключения.
        close_real_position должен пропустить попытку продажи в этом
        цикле, а не штамповать заведомо провальный ордер.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.cancel_order.side_effect = Exception("bybit 170131 Insufficient balance")
        self.engine.real_positions["SLCANCELFAIL1/USDT"] = {
            "amount": 100.0, "entry_price": 2.0, "side": "long",
            "sl_order_id": "sl-order-still-live-1",
        }

        result = await self.engine.close_real_position(
            symbol="SLCANCELFAIL1/USDT", side="long", entry_price=2.0, amount=100.0,
            reason="take_profit_3", entry_fee=0.02, holding_seconds=60,
        )

        self.assertIsNone(result)
        self.engine.exchange.cancel_order.assert_called_once_with("sl-order-still-live-1", "SLCANCELFAIL1/USDT")
        self.engine.exchange.create_market_sell_order.assert_not_called()
        self.assertIn("SLCANCELFAIL1/USDT", self.engine.real_positions)

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

    async def test_finalize_externally_closed_position_converts_base_currency_fees(self):
        """
        Регресс на прод-баг ("неправильно считается pnl - в формуле
        количество выражается в разных монетах, а итог в usdt не
        пересчитывается относительно usdt"): entry_fee/exit_fee, снятые
        БИРЖЕЙ в BASE-валюте (не USDT), вычитались из PnL как есть в
        закрытии "вне цикла бота" (сработал сам биржевой SL) — тот же
        класс бага, что уже исправлен в close_real_position (см. её
        комментарий про инцидент "105.4915 TAC" вместо USDT), но
        _record_external_close его не унаследовал вообще никак.
        """
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import Order, Trade

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()

        symbol = "FEECONV1/USDT"
        async with get_session() as session:
            exchange_id, symbol_id = await self.engine._resolve_symbol_id(session, symbol)
            opening_order = Order(
                exchange_id=exchange_id, symbol_id=symbol_id,
                side="buy", order_type="market", amount=1000.0, price=1.0,
                status="filled", filled_amount=1000.0, filled_price=1.0,
                fee=10.0, fee_currency="FEECONV1",
                client_order_id="feeconv1-open",
            )
            session.add(opening_order)
            await session.commit()
            opening_order_id = opening_order.id

        self.engine.real_positions[symbol] = {
            "amount": 990.0, "entry_price": 1.0, "side": "long",
            "strategy_id": None, "entry_fee": 10.0, "order_id": opening_order_id,
            "opened_at": datetime.now(), "sl_order_id": "sl-feeconv-1",
        }
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"FEECONV1": 0.0}, "FEECONV1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.fetch_order = AsyncMock(
            return_value={"id": "sl-feeconv-1", "status": "closed", "filled": 990.0, "average": 1.2}
        )
        self.engine.exchange.fetch_order_trades = AsyncMock(return_value=[
            {"id": "exec-feeconv-1", "amount": 990.0, "price": 1.2, "cost": 1188.0,
             "fee": {"cost": 5.0, "currency": "FEECONV1"}},
        ])

        await self.engine.reconcile_real_positions()

        self.assertNotIn(symbol, self.engine.real_positions)
        async with get_session() as session:
            trade = (
                await session.execute(select(Trade).where(Trade.order_open_id == opening_order_id))
            ).scalar_one()
        # entry_fee_quote = 10.0 FEECONV1 * entry_price(1.0) = 10.0 USDT
        # exit_fee_quote = 5.0 FEECONV1 * exit_price(1.2) = 6.0 USDT
        # pnl = (1.2 - 1.0) * 990 - 10.0 - 6.0 = 182.0
        self.assertAlmostEqual(float(trade.pnl), 182.0, places=4)

    async def test_finalize_via_trade_history_converts_mixed_currency_exit_fees(self):
        """Та же конвертация, но для пути без известного order id (найдено
        по истории сделок биржи) — несколько закрывающих сделок могут быть
        с РАЗНОЙ валютой комиссии (часть в USDT, часть в base)."""
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import Trade, Symbol

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        opened_at = datetime.now() - timedelta(hours=1)
        self.engine.real_positions["MIXEDFEE1/USDT"] = {
            "amount": 100.0, "entry_price": 2.0, "side": "long",
            "strategy_id": None, "entry_fee": 0.02, "order_id": None,
            "opened_at": opened_at, "sl_order_id": None,
        }
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"MIXEDFEE1": 0.0}, "MIXEDFEE1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        recent_ts_ms = int((opened_at + timedelta(minutes=30)).timestamp() * 1000)
        self.engine.exchange.fetch_my_trades = AsyncMock(return_value=[
            {"id": "mixed-1", "side": "sell", "amount": 50.0, "price": 2.1, "cost": 105.0,
             "timestamp": recent_ts_ms, "fee": {"cost": 0.5, "currency": "USDT"}},
            {"id": "mixed-2", "side": "sell", "amount": 50.0, "price": 2.1, "cost": 105.0,
             "timestamp": recent_ts_ms, "fee": {"cost": 1.0, "currency": "MIXEDFEE1"}},
        ])

        await self.engine.reconcile_real_positions()

        self.assertNotIn("MIXEDFEE1/USDT", self.engine.real_positions)
        async with get_session() as session:
            symbol_row = (
                await session.execute(select(Symbol).where(Symbol.symbol == "MIXEDFEE1/USDT"))
            ).scalar_one()
            trade = (
                await session.execute(select(Trade).where(Trade.symbol_id == symbol_row.id))
            ).scalar_one()
        # exit_fee = 0.5 USDT (уже в quote) + 1.0 MIXEDFEE1 * price(2.1) = 2.1 USDT -> 2.6 USDT total
        # pnl = (2.1 - 2.0) * 100 - entry_fee(0.02) - exit_fee(2.6) = 10.0 - 0.02 - 2.6 = 7.38
        self.assertAlmostEqual(float(trade.pnl), 7.38, places=4)

    async def test_record_external_close_uses_short_formula_for_futures_short(self):
        """_record_external_close раньше всегда считала long-формулу и
        direction="long", даже для фьючерсного short, закрытого вне цикла
        бота — знак PnL был бы перевёрнут."""
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import Trade, Symbol

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        symbol = "EXTSHORT1/USDT"
        self.engine.real_positions[symbol] = {
            "amount": 10.0, "entry_price": 100.0, "side": "short",
            "strategy_id": None, "entry_fee": 0.0, "order_id": None,
            "opened_at": datetime.now(), "sl_order_id": None, "market_type": "futures",
        }

        # Short: цена упала со 100 до 90 -> прибыль (entry - exit) * amount = 100.
        await self.engine._record_external_close(
            symbol, self.engine.real_positions[symbol], exit_price=90.0, amount=10.0, exit_fee=0.0,
            order_id_exchange="ext-short-1", log_note="test",
        )

        async with get_session() as session:
            symbol_row = (
                await session.execute(select(Symbol).where(Symbol.symbol == symbol))
            ).scalar_one()
            trade = (
                await session.execute(select(Trade).where(Trade.symbol_id == symbol_row.id))
            ).scalar_one()
        self.assertEqual(trade.direction, "short")
        self.assertAlmostEqual(float(trade.pnl), 100.0, places=4)

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

    async def test_external_close_via_trade_history_cancels_leftover_sl_order(self):
        """
        Регресс на прод-инцидент: ASTER/QTUM/TIA годами держали часть
        баланса заблокированной ("used" в /balances) — позиция закрылась
        ЧУЖИМ путём (найдено по истории сделок биржи, _record_external_close),
        а биржевой SL-ордер, выставленный под неё, никогда не отменялся
        (в отличие от _reconcile_phantom_position, где отмена уже была).
        Сам ордер продолжал резервировать актив на бирже бессрочно.
        """
        from sqlalchemy import select
        from src.db.session import get_session
        from src.db.models import Symbol

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        opened_at = datetime.now() - timedelta(hours=1)
        self.engine.real_positions["TRSLLEAK1/USDT"] = {
            "amount": 100.0, "entry_price": 2.0, "side": "long",
            "strategy_id": None, "entry_fee": 0.02, "order_id": None,
            "opened_at": opened_at, "sl_order_id": "sl-leftover-1",
        }
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"TRSLLEAK1": 0.0}, "TRSLLEAK1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        # fetch_order (использует _finalize_externally_closed_position) —
        # SL-ордер ещё НЕ исполнен (open), поэтому этот путь возвращает
        # False и сверка падает дальше на _finalize_via_recent_trade_history.
        self.engine.exchange.fetch_order = AsyncMock(
            return_value={"id": "sl-leftover-1", "status": "open", "filled": 0.0}
        )
        recent_ts_ms = int((opened_at + timedelta(minutes=30)).timestamp() * 1000)
        self.engine.exchange.fetch_my_trades = AsyncMock(return_value=[
            {"id": "closed-manually-2", "side": "sell", "amount": 100.0, "price": 2.1,
             "cost": 210.0, "timestamp": recent_ts_ms, "fee": {"cost": 0.05, "currency": "USDT"}},
        ])

        await self.engine.reconcile_real_positions()

        self.assertNotIn("TRSLLEAK1/USDT", self.engine.real_positions)
        self.engine.exchange.cancel_order.assert_called_once_with("sl-leftover-1", "TRSLLEAK1/USDT")
        async with get_session() as session:
            symbol_row = (
                await session.execute(select(Symbol).where(Symbol.symbol == "TRSLLEAK1/USDT"))
            ).scalar_one_or_none()
        self.assertIsNotNone(symbol_row)

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

    async def test_close_real_position_skips_sell_order_when_amount_already_zero(self):
        """
        Реальный инцидент: ETH/USDT — из-за бага реконструкции при рестарте
        (комиссия в quote-валюте вычиталась как если бы была в base) объём
        позиции стал ровно 0.0, хотя на бирже реально лежал весь объём.
        close_real_position(amount=0.0) НЕ должен вообще пытаться создать
        ордер на бирже (0.0 — невалидное количество, Bybit отклоняет с
        "Data sent for paramter '' is not valid" на каждой попытке) —
        нечего продавать, позиция сразу снимается с учёта.
        """
        from src.db.models import Order
        from src.db.session import get_session

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "zeroamt-open-1", "filled": 0.157, "price": None, "average": 2490.41,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }
        opening_order = await self.engine.create_order(
            symbol="ZEROAMT1/USDT", side="buy", amount=0.157, price=2490.41,
            order_type="market", stop_loss=2465.0, take_profit=2540.0,
        )
        self.assertIsNotNone(opening_order)
        # create_order уже вызвал create_market_sell_order один раз, чтобы
        # выставить биржевой SL сразу после открытия — интересует только
        # то, что происходит (не происходит) при самом закрытии.
        self.engine.exchange.create_market_sell_order.reset_mock()

        result = await self.engine.close_real_position(
            symbol="ZEROAMT1/USDT", side="long", entry_price=2490.41, amount=0.0,
            reason="stop_loss", entry_fee=0.0, holding_seconds=60,
            order_open_id=opening_order.id,
        )

        self.assertIsNone(result)
        self.engine.exchange.create_market_sell_order.assert_not_called()
        self.assertNotIn("ZEROAMT1/USDT", self.engine.real_positions)

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

    async def test_close_real_position_reconciles_dust_even_when_available_exceeds_amount(self):
        """
        Третий вариант того же класса бага (прод, реальный счёт Bybit,
        USDC/USDT): наш УЧЁТ сам по себе — пыль (0.00717323), при этом на
        бирже доступно МНОГО больше (14092.25, available > amount — не
        нехватка баланса). Биржа всё равно отклоняет продажу как ниже
        минимального торгуемого объёма ("must be greater than minimum
        amount precision of 0.01"). Старая проверка требовала available <
        amount вдобавок к ключевым словам "precision"/"minimum" — здесь она
        не срабатывала, и бот повторял один и тот же провальный ордер
        каждую итерацию бесконечно (наблюдалось на проде непрерывно, лог
        рос без остановки). Продать меньше минимального объёма нельзя
        независимо от того, сколько ещё есть на бирже — позиция должна
        сняться с учёта так же, как и при available < amount.
        """
        from src.db.models import Order
        from src.db.session import get_session

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "dust-open-2", "filled": 100.0, "price": None, "average": 0.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }
        opening_order = await self.engine.create_order(
            symbol="DUST2/USDT", side="buy", amount=100.0, price=0.5,
            order_type="market", stop_loss=0.45, take_profit=0.6,
        )
        self.assertIsNotNone(opening_order)
        self.assertIn("DUST2/USDT", self.engine.real_positions)

        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"DUST2": 14092.25}, "DUST2": {"free": 14092.25, "used": 0, "total": 14092.25}}
        )
        self.engine.exchange.create_market_sell_order = AsyncMock(
            side_effect=Exception(
                "bybit amount of DUST2/USDT must be greater than minimum amount precision of 0.01"
            )
        )

        result = await self.engine.close_real_position(
            symbol="DUST2/USDT", side="long", entry_price=0.5, amount=0.00717323,
            reason="stop_loss", entry_fee=0.05, holding_seconds=60,
            order_open_id=opening_order.id,
        )

        self.assertIsNone(result)
        self.assertNotIn("DUST2/USDT", self.engine.real_positions)

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
        запроса. Позиция искусственно "состарена" (opened_at в прошлом) —
        см. test_reconcile_real_positions_grace_period_protects_fresh_position
        ниже про grace-период для только что открытых позиций.
        """
        from src.db.models import Order
        from src.db.session import get_session
        from src.utils.timeutils import utcnow

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
        self.engine.real_positions["RECONDUST1/USDT"]["opened_at"] = utcnow() - timedelta(minutes=10)

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

    async def test_reconcile_real_positions_grace_period_protects_fresh_position(self):
        """
        Регресс на прод-инцидент: CHZ/MANA/ZRX были куплены и подтверждены
        исполненными, но ~90с спустя fetch_balance() всё ещё показывал
        available≈0 (биржа не успела отразить актив) — периодическая сверка
        ошибочно списывала свежую позицию как фантомную, а стратегия тут
        же открывала ДУБЛИРУЮЩУЮ новую на тот же символ (реальные деньги от
        первой зависали без SL/TP и без отслеживания). Позиция младше
        RECONCILE_MIN_AGE_SECONDS не должна списываться, даже если выглядит
        как несомненная пыль.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.markets = {
            "RECONFRESH1/USDT": {"limits": {"amount": {"min": 0.001}}},
        }
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "reconfresh-open-1", "filled": 39083.21766, "price": None, "average": 0.01,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }
        order = await self.engine.create_order(
            symbol="RECONFRESH1/USDT", side="buy", amount=39083.21766, price=0.01, order_type="market",
        )
        self.assertIsNotNone(order)
        # opened_at выставляется create_order() в момент вызова — позиция
        # только что открыта, никакого искусственного состаривания.

        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 50.0, "RECONFRESH1": 0.00766},
            "RECONFRESH1": {"free": 0.00766, "used": 0, "total": 0.00766},
        })

        balance = await self.engine.reconcile_real_positions()

        self.assertAlmostEqual(balance, 50.0)
        self.assertIn("RECONFRESH1/USDT", self.engine.real_positions)
        self.assertAlmostEqual(self.engine.real_positions["RECONFRESH1/USDT"]["amount"], 39083.21766, places=3)

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

    async def test_reconcile_phantom_position_cancels_orphaned_sl_order(self):
        """
        _reconcile_phantom_position() снимала позицию с учёта, не отменяя
        выставленный по ней биржевой SL-ордер (см. sync_stop_loss_order) —
        он навсегда оставался висеть на бирже (мы больше никогда не
        вернёмся к этому symbol), продолжая держать часть актива в
        "used"-балансе. Живой симптом: список балансов дашборда (GET
        /balances) показывал БОЛЬШЕ валют с ненулевым "в ордерах", чем
        реально открытых позиций — расхождение от осиротевших SL-ордеров,
        накопленных за время работы бота при каждом списании непродаваемой
        (пыль/ниже минимума биржи) позиции.
        """
        self.engine.exchange = AsyncMock()
        self.engine.real_positions["PHANTOMSL/USDT"] = {
            "amount": 10.0, "entry_price": 1.0, "side": "long", "sl_order_id": "sl-order-42",
        }

        await self.engine._reconcile_phantom_position("PHANTOMSL/USDT", None)

        self.engine.exchange.cancel_order.assert_awaited_once_with("sl-order-42", "PHANTOMSL/USDT")
        self.assertNotIn("PHANTOMSL/USDT", self.engine.real_positions)

    async def test_reconcile_phantom_position_without_sl_order_does_not_call_cancel(self):
        """Позиция без выставленного SL-ордера (sl_order_id=None) — отменять нечего."""
        self.engine.exchange = AsyncMock()
        self.engine.real_positions["NOSL/USDT"] = {
            "amount": 10.0, "entry_price": 1.0, "side": "long", "sl_order_id": None,
        }

        await self.engine._reconcile_phantom_position("NOSL/USDT", None)

        self.engine.exchange.cancel_order.assert_not_called()

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

    async def test_restore_real_positions_reconstructs_short_on_futures(self):
        """
        ЭТАП 3: на фьючерсах short — штатная позиция, а не осиротевший
        баг-артефакт (как на споте) — при рестарте бота она должна
        восстанавливаться так же, как long, а не пропускаться (см.
        test_restore_real_positions_skips_orphaned_short_order для спота —
        там пропуск остаётся верным поведением). Решение теперь принимается
        по СОБСТВЕННОМУ market_type ордера (Order.market_type), а не по
        текущему положению тумблера settings.market_type — позиция могла
        быть открыта на фьючерсах, даже если тумблер сейчас переключён на
        spot (и наоборот), поэтому market_type тумблера здесь намеренно НЕ
        трогается.
        """
        from src.db.session import get_session
        from src.db.models import Order

        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        async with get_session() as session:
            exchange_id, symbol_id = await self.engine._resolve_symbol_id(session, "FUTSHORTRESTORE1/USDT")
            fut_short = Order(
                exchange_id=exchange_id, symbol_id=symbol_id,
                side="sell", order_type="market", amount=10.0, price=2.0,
                status="filled", filled_amount=10.0, filled_price=2.0,
                fee=0.01, market_type="futures",
                order_id_exchange="fut-short-restore-1", client_order_id="futshortrestore1",
            )
            session.add(fut_short)
            await session.commit()

        real_positions, _, _ = await self.engine._load_open_positions_from_db(is_paper=False)

        self.assertIsNotNone(real_positions)
        self.assertIn("FUTSHORTRESTORE1/USDT", real_positions)
        self.assertEqual(real_positions["FUTSHORTRESTORE1/USDT"]["side"], "short")
        self.assertEqual(real_positions["FUTSHORTRESTORE1/USDT"]["market_type"], "futures")

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
            "fee": {"cost": 0.1, "currency": "RESTOREFEE1"},
        }
        order = await self.engine.create_order(
            symbol="RESTOREFEE1/USDT", side="buy", amount=100.0, price=0.5,
            order_type="market", stop_loss=0.45, take_profit=0.6,
        )
        self.assertIsNotNone(order)

        real_positions, _, _ = await self.engine._load_open_positions_from_db(is_paper=False)
        self.assertAlmostEqual(real_positions["RESTOREFEE1/USDT"]["amount"], 99.9)

    async def test_restore_real_position_does_not_subtract_quote_currency_fee(self):
        """
        Комиссия открытия не всегда в base-валюте — Bybit нередко списывает
        её в USDT (quote). Реальный инцидент: ETH/USDT, комиссия 0.39 USDT
        (в quote) БОЛЬШЕ восстановленного объёма 0.157 ETH — старый код
        вычитал её как если бы она была в ETH, объём уходил в отрицательный
        и схлопывался до 0.0 (max(0.0, ...)), хотя на бирже реально лежал
        весь объём нетронутым. Бот затем бесконечно пытался продать 0.0,
        биржа отклоняла ордер на каждой итерации цикла.
        """
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "ex-fee-quote-1", "filled": 0.157, "price": None, "average": 2490.41,
            "fee": {"cost": 0.39, "currency": "USDT"},
        }
        order = await self.engine.create_order(
            symbol="RESTOREFEEQUOTE1/USDT", side="buy", amount=0.157, price=2490.41,
            order_type="market", stop_loss=2465.0, take_profit=2540.0,
        )
        self.assertIsNotNone(order)

        real_positions, _, _ = await self.engine._load_open_positions_from_db(is_paper=False)
        self.assertAlmostEqual(real_positions["RESTOREFEEQUOTE1/USDT"]["amount"], 0.157)

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

    def test_numbered_tp_targets_do_not_swallow_the_numbering_digit(self):
        """
        Регресс: "TP1: 51000" раньше матчился как keyword="tp" + число="1"
        (первая цифра самой нумерации TP1/TP2/TP3), а не как реальная цена
        51000 — без разделителя между словом и числом ([:\\-–—]? и \\s* оба
        необязательные) регэксп охотно "хватал" саму цифру нумерации.
        Итоговый TP получался околонулевым — при 3-уровневой интерполяции
        (_tp_levels в main.py) это немедленно "срабатывало" бы как
        достигнутая цель на первой же проверке после открытия позиции,
        закрывая её почти сразу вместо реального тейк-профита. Формат
        "TP1/TP2/TP3" — стандартный для каналов с несколькими целями.
        """
        result = self.parse(
            "BTC/USDT LONG Entry: 50000 SL: 49000 TP1: 51000 TP2: 52000 TP3: 53000"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["entry"], 50000.0)
        self.assertEqual(result["sl"], 49000.0)
        # tp — итоговая (самая дальняя) цель; см. test_multiple_tp_targets_
        # captured_as_list_nearest_first ниже для полного списка TP1/TP2/TP3.
        self.assertEqual(result["tp"], 53000.0)

    def test_numbered_sl_target_does_not_swallow_the_numbering_digit(self):
        result = self.parse("ETH/USDT SHORT Entry: 3500 SL1: 3600 TP: 3200")
        self.assertIsNotNone(result)
        self.assertEqual(result["sl"], 3600.0)

    def test_multiple_tp_targets_captured_as_list_nearest_first(self):
        """
        Регресс: раньше несколько целей канала схлопывались в ОДНО число
        (parsed_tp) ещё на этапе парсинга — реальные TP1/TP2/TP3 нигде не
        сохранялись, и _tp_levels() (main.py) сам синтезировал 3 фейковых
        уровня линейной интерполяцией между входом и этим единственным
        числом вместо использования того, что канал реально указал.
        """
        result = self.parse(
            "BTC/USDT LONG Entry: 50000 SL: 49000 TP1: 51000 TP2: 52000 TP3: 53000"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["take_profits"], [51000.0, 52000.0, 53000.0])
        self.assertEqual(result["tp"], 53000.0)  # финальная (самая дальняя) цель

    def test_single_tp_still_populates_take_profits_list(self):
        result = self.parse("BTC/USDT Long 69000 SL 68000 TP 72000")
        self.assertIsNotNone(result)
        self.assertEqual(result["take_profits"], [72000.0])
        self.assertEqual(result["tp"], 72000.0)

    def test_leverage_absent_when_not_mentioned(self):
        result = self.parse("BTC/USDT Long 69000 SL 68000 TP 72000")
        self.assertIsNotNone(result)
        self.assertIsNone(result["leverage"])

    def test_leverage_parsed_cyrillic_prefix_x(self):
        """Пример из прод-запроса: "Кредитное плечо: х35" (кириллическая х)."""
        result = self.parse("BTC/USDT Long 69000 SL 68000 TP 72000 Кредитное плечо: х35")
        self.assertIsNotNone(result)
        self.assertEqual(result["leverage"], 35.0)

    def test_leverage_parsed_english_suffix_x(self):
        result = self.parse("BTCUSDT LONG 69000 SL 68000 TP 72000 Leverage: 20x")
        self.assertIsNotNone(result)
        self.assertEqual(result["leverage"], 20.0)

    def test_leverage_keyword_plecho_without_kreditnoe(self):
        result = self.parse("BTC/USDT Long 69000 SL 68000 TP 72000 плечо x10")
        self.assertIsNotNone(result)
        self.assertEqual(result["leverage"], 10.0)

    def test_real_world_wif_short_market_entry_signal(self):
        """
        Реальный формат сигнала из прод-запроса: хэштег-тикер без quote
        ("#WIF"), диапазон плеча ("25-30х"), вход "по рынку" (без числа),
        цели одной строкой через пробел под русским "Тейки:", и стоп под
        кириллическим "Стоп:" (не матчится с латинским "stop" даже при
        IGNORECASE — это разные символы Unicode).
        """
        text = (
            "#WIF SHORT \n"
            "Плечо: 25-30х  \n"
            "Диапазон входа: по рынку   \n"
            "Тейки: 0.1962 0.1933 0.1853\n"
            "Стоп: 0.2091"
        )
        result = self.parse(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "WIF/USDT")
        self.assertEqual(result["side"], "short")
        self.assertIsNone(result["entry"])
        self.assertEqual(result["sl"], 0.2091)
        self.assertEqual(result["take_profits"], [0.1962, 0.1933, 0.1853])
        self.assertEqual(result["tp"], 0.1853)
        self.assertEqual(result["leverage"], 25.0)

    def test_message_without_entry_or_market_phrase_is_not_a_signal(self):
        """Без числовой цены И без явного "по рынку"/"market" — не сигнал,
        а не молчаливый маркет-ордер по любому шуму."""
        result = self.parse("BTC/USDT Long SL 68000 TP 72000")
        self.assertIsNone(result)

    def test_cyrillic_stop_keyword_matches(self):
        result = self.parse("BTC/USDT Long 69000 Стоп: 68000 TP 72000")
        self.assertIsNotNone(result)
        self.assertEqual(result["sl"], 68000.0)

    def test_space_separated_targets_after_single_keyword(self):
        result = self.parse("BTC/USDT Long по рынку Тейки: 70000 71000 72000 Стоп: 68000")
        self.assertIsNotNone(result)
        self.assertEqual(result["take_profits"], [70000.0, 71000.0, 72000.0])

    def test_bare_ticker_before_side_without_hashtag(self):
        """"ARB SHORT x25" — просто тикер заглавными, без "#" и без quote-
        валюты, прямо перед словом стороны сделки."""
        result = self.parse("ARB SHORT x25 Entry 1.20 SL 1.30 TP 1.05")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "ARB/USDT")
        self.assertEqual(result["side"], "short")
        self.assertEqual(result["leverage"], 25.0)

    def test_bare_ticker_lowercase_word_before_side_not_matched(self):
        """Обычный текст ("watch LONG on this one") не капсом — не должен
        сходить за тикер, иначе любое слово перед LONG/SHORT стало бы
        "сигналом"."""
        result = self.parse("watch LONG on this one, no price yet")
        self.assertIsNone(result)

    def test_bare_leverage_without_keyword(self):
        """"ARB SHORT x25" — плечо без какого-либо ключевого слова
        ("плечо"/"leverage"), просто "xNN" сразу после направления сделки."""
        from src.telegram.channel_monitor import extract_leverage
        self.assertEqual(extract_leverage("ARB SHORT x25"), 25.0)

    def test_bare_leverage_full_signal(self):
        result = self.parse("ARB SHORT x25 по рынку SL 1.30 TP 1.05")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "ARB/USDT")
        self.assertIsNone(result["entry"])
        self.assertEqual(result["leverage"], 25.0)

    def test_bare_leverage_number_before_x(self):
        """"20Х" (число, потом множитель) — тот же формат плеча, что и
        "x25" (множитель, потом число), просто в обратном порядке."""
        result = self.parse("ARB SHORT 25х по рынку SL 1.30 TP 1.05")
        self.assertIsNotNone(result)
        self.assertEqual(result["leverage"], 25.0)

    def test_real_world_sand_long_market_entry_with_leverage_and_promo_link(self):
        """
        Реальный прод-инцидент: канал прислал "SAND LONG 20Х" (плечо без
        разделителя сразу после стороны) с рекламной партнёрской ссылкой
        где-то в тексте (не показана здесь дословно, но воспроизводится
        похожей структурой) — старый паттерн пары матчил "com/partner"
        РАНЬШЕ, чем очередь доходила до настоящего "SAND" (более общий
        паттерн стоял раньше в списке и матчил вообще что угодно вида
        "буквы/буквы"), а "20" из "20Х" по ошибке принимался за цену
        входа. Итог был: distorted "[TG SIGNAL] COM/PARTNER LONG |
        Entry: 20.0 | SL: 0.0375 | TP: 0.04078".
        """
        text = (
            "🚀**Заходим SAND LONG 20Х\n\n"
            "Вход: по рынку \n"
            "Тейк: 0.04078, 0.0418, 0.0435\n"
            "Стоп: 0.0375**\n\n"
            "**Подробнее на нашем сайте: crypto-signals.com/partner"
        )
        result = self.parse(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "SAND/USDT")
        self.assertEqual(result["side"], "long")
        self.assertIsNone(result["entry"])
        self.assertEqual(result["sl"], 0.0375)
        self.assertEqual(result["take_profits"], [0.04078, 0.0418, 0.0435])
        self.assertEqual(result["tp"], 0.0435)
        self.assertEqual(result["leverage"], 20.0)

    def test_comma_separated_targets_after_single_keyword(self):
        result = self.parse("BTC/USDT Long по рынку Тейк: 70000, 71000, 72000 Стоп: 68000")
        self.assertIsNotNone(result)
        self.assertEqual(result["take_profits"], [70000.0, 71000.0, 72000.0])


class TestMarketEntryDetection(unittest.TestCase):
    def setUp(self):
        from src.telegram.channel_monitor import is_market_entry
        self.is_market_entry = is_market_entry

    def test_russian_market_phrase(self):
        self.assertTrue(self.is_market_entry("Диапазон входа: по рынку"))

    def test_english_market_phrase(self):
        self.assertTrue(self.is_market_entry("Entry: at market"))

    def test_bare_market_word(self):
        self.assertTrue(self.is_market_entry("Entry: Market"))

    def test_regular_price_text_is_not_market(self):
        self.assertFalse(self.is_market_entry("BTC/USDT Long 69000 SL 68000 TP 72000"))


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
        # long: ascending цена = ascending расстояние от входа -> порядок как есть
        self.assertEqual(result["take_profits"], [70000.0, 72000.0])

    async def test_take_profits_reversed_for_short_to_stay_nearest_first(self):
        """
        Регресс: промпт отдаёт цели по возрастанию ЦЕНЫ, а _tp_levels()
        (main.py) ожидает порядок по возрастанию расстояния от входа В
        ПРИБЫЛЬНУЮ СТОРОНУ (ближайшая первая) — для short это обратный
        порядок цены (профит растёт при падении цены).
        """
        import src.telegram.llm_parser as llm_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.anthropic_api_key = "test-key"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=self._mock_tool_use_response({
            "is_signal": True, "base": "ETH", "quote": "USDT", "side": "short",
            "entry": 3500.0, "take_profits": [3200.0, 3300.0], "stop_loss": 3600.0,
            "confidence": 0.9,
        }))
        llm_parser_module._client = mock_client

        result = await llm_parser_module.parse_with_llm("шортим эфир")
        self.assertIsNotNone(result)
        # ближайшая цель для short — самая высокая цена ниже входа
        self.assertEqual(result["take_profits"], [3300.0, 3200.0])
        self.assertEqual(result["tp"], 3200.0)

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
    Gemini LLM-фолбэк парсинга — третий, последний уровень, после Anthropic
    и Groq (см. src/telegram/gemini_parser.py и TestLlmSignalParser выше).
    Мокаем google-genai клиент, чтобы не делать реальных сетевых запросов.
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
        self.assertEqual(result["take_profits"], [70000.0, 72000.0])

    async def test_take_profits_reversed_for_short_to_stay_nearest_first(self):
        import src.telegram.gemini_parser as gemini_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.gemini_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=self._mock_response({
            "is_signal": True, "base": "ETH", "quote": "USDT", "side": "short",
            "entry": 3500.0, "take_profits": [3200.0, 3300.0], "stop_loss": 3600.0,
            "confidence": 0.9,
        }))
        gemini_parser_module._client = mock_client

        result = await gemini_parser_module.parse_with_gemini("шортим эфир")
        self.assertIsNotNone(result)
        self.assertEqual(result["take_profits"], [3300.0, 3200.0])
        self.assertEqual(result["tp"], 3200.0)

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

    async def test_truncated_json_response_returns_none_not_raises(self):
        """
        Реальный инцидент (прод): ответ Gemini обрывается посередине JSON
        ("thinking"-бюджет модели съедает часть max_output_tokens ДО
        собственно ответа) — json.loads падает с "Unterminated string
        starting at:", а не с обычной "не json вообще". Тот же контракт:
        None, а не исключение наружу.
        """
        import src.telegram.gemini_parser as gemini_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.gemini_api_key = "test-key"

        mock_client = MagicMock()
        bad_resp = MagicMock()
        bad_resp.text = '{"is_signal": true, "base": "BTC", "side": "long", "entry'
        mock_client.aio.models.generate_content = AsyncMock(return_value=bad_resp)
        gemini_parser_module._client = mock_client

        result = await gemini_parser_module.parse_with_gemini("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNone(result)

    async def test_requests_larger_output_budget_to_avoid_truncation(self):
        """
        Регресс на прод-инцидент: max_output_tokens=512 было мало —
        "thinking"-бюджет модели съедал часть лимита до собственно JSON,
        ответ обрывался, сигнал тихо терялся (единственный работающий
        уровень LLM-фолбэка — Anthropic не настроен). Бюджет увеличен;
        здесь просто фиксируем, что теперь запрашивается заметно больше,
        чем было — если кто-то случайно вернёт лимит обратно к 512, тест
        упадёт.
        """
        import src.telegram.gemini_parser as gemini_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.gemini_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=self._mock_response({
            "is_signal": True, "base": "BTC", "side": "long", "entry": 69000.0,
            "take_profits": [70000.0], "stop_loss": 68000.0, "confidence": 0.9,
        }))
        gemini_parser_module._client = mock_client

        await gemini_parser_module.parse_with_gemini("покупаем биток в районе 69к, стоп 68к")

        _, kwargs = mock_client.aio.models.generate_content.call_args
        self.assertGreaterEqual(kwargs["config"]["max_output_tokens"], 2048)

    async def test_parse_telegram_signal_falls_back_to_gemini_when_anthropic_not_configured(self):
        """
        Anthropic не настроен (нет ключа) — цепочка должна дойти до
        Gemini. Groq тоже явно не настроен (None) — цепочка проходит его
        (parse_with_groq молча возвращает None без ключа) и доходит до
        Gemini как последнего, третьего уровня фолбэка.
        """
        import src.telegram.gemini_parser as gemini_parser_module
        from src.telegram.channel_monitor import parse_telegram_signal
        settings.telegram_llm_fallback_enabled = True
        settings.anthropic_api_key = None
        settings.groq_api_key = None
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


class TestGroqSignalParser(unittest.IsolatedAsyncioTestCase):
    """
    Groq LLM-фолбэк парсинга — второй уровень, между Anthropic и Gemini
    (см. src/telegram/groq_parser.py). Открытые модели на бесплатном
    тарифе, добавлен как менее склонный к rate-limit'ам вариант, чем
    Gemini. Мокаем groq-клиент (OpenAI-совместимый chat.completions с
    forced tool-call), чтобы не делать реальных сетевых запросов.
    """

    def setUp(self):
        self._saved = {
            "telegram_llm_fallback_enabled": settings.telegram_llm_fallback_enabled,
            "anthropic_api_key": settings.anthropic_api_key,
            "groq_api_key": settings.groq_api_key,
            "gemini_api_key": settings.gemini_api_key,
        }

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(settings, key, value)
        import src.telegram.groq_parser as groq_parser_module
        groq_parser_module._client = None
        import src.telegram.gemini_parser as gemini_parser_module
        gemini_parser_module._client = None

    def _mock_response(self, data: dict):
        import json
        tool_call = MagicMock()
        tool_call.function.arguments = json.dumps(data)
        message = MagicMock()
        message.tool_calls = [tool_call]
        choice = MagicMock()
        choice.message = message
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    async def test_disabled_returns_none_without_calling_api(self):
        from src.telegram.groq_parser import parse_with_groq
        settings.telegram_llm_fallback_enabled = False
        settings.groq_api_key = "test-key"
        result = await parse_with_groq("BTC to the moon, going long soon maybe")
        self.assertIsNone(result)

    async def test_no_api_key_returns_none_without_calling_api(self):
        from src.telegram.groq_parser import parse_with_groq
        settings.telegram_llm_fallback_enabled = True
        settings.groq_api_key = None
        result = await parse_with_groq("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNone(result)

    async def test_parses_valid_signal(self):
        import src.telegram.groq_parser as groq_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.groq_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self._mock_response({
            "is_signal": True, "base": "BTC", "quote": "USDT", "side": "long",
            "entry": 69000.0, "take_profits": [70000.0, 72000.0], "stop_loss": 68000.0,
            "confidence": 0.9,
        }))
        groq_parser_module._client = mock_client

        result = await groq_parser_module.parse_with_groq("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "BTC/USDT")
        self.assertEqual(result["side"], "long")
        self.assertEqual(result["entry"], 69000.0)
        self.assertEqual(result["sl"], 68000.0)
        self.assertEqual(result["tp"], 72000.0)  # long -> максимальный TP
        self.assertEqual(result["take_profits"], [70000.0, 72000.0])

    async def test_take_profits_reversed_for_short_to_stay_nearest_first(self):
        import src.telegram.groq_parser as groq_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.groq_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self._mock_response({
            "is_signal": True, "base": "ETH", "quote": "USDT", "side": "short",
            "entry": 3500.0, "take_profits": [3200.0, 3300.0], "stop_loss": 3600.0,
            "confidence": 0.9,
        }))
        groq_parser_module._client = mock_client

        result = await groq_parser_module.parse_with_groq("шортим эфир")
        self.assertIsNotNone(result)
        self.assertEqual(result["take_profits"], [3300.0, 3200.0])
        self.assertEqual(result["tp"], 3200.0)

    async def test_low_confidence_is_rejected(self):
        import src.telegram.groq_parser as groq_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.groq_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self._mock_response({
            "is_signal": True, "base": "BTC", "side": "long", "entry": 69000.0,
            "take_profits": [], "stop_loss": None, "confidence": 0.2,
        }))
        groq_parser_module._client = mock_client

        result = await groq_parser_module.parse_with_groq("может быть покупать биток?")
        self.assertIsNone(result)

    async def test_not_a_signal_returns_none(self):
        import src.telegram.groq_parser as groq_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.groq_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self._mock_response({
            "is_signal": False, "confidence": 0.95,
        }))
        groq_parser_module._client = mock_client

        result = await groq_parser_module.parse_with_groq("сегодня биток вырос на 3%, отличный день")
        self.assertIsNone(result)

    async def test_api_error_returns_none_not_raises(self):
        import src.telegram.groq_parser as groq_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.groq_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("network down"))
        groq_parser_module._client = mock_client

        result = await groq_parser_module.parse_with_groq("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNone(result)

    async def test_missing_tool_call_returns_none_not_raises(self):
        """Модель ответила текстом вместо forced tool-call — не должно падать."""
        import src.telegram.groq_parser as groq_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.groq_api_key = "test-key"

        message = MagicMock()
        message.tool_calls = None
        choice = MagicMock()
        choice.message = message
        bad_resp = MagicMock()
        bad_resp.choices = [choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=bad_resp)
        groq_parser_module._client = mock_client

        result = await groq_parser_module.parse_with_groq("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNone(result)

    async def test_invalid_json_arguments_returns_none_not_raises(self):
        import src.telegram.groq_parser as groq_parser_module
        settings.telegram_llm_fallback_enabled = True
        settings.groq_api_key = "test-key"

        tool_call = MagicMock()
        tool_call.function.arguments = "не json вообще"
        message = MagicMock()
        message.tool_calls = [tool_call]
        choice = MagicMock()
        choice.message = message
        bad_resp = MagicMock()
        bad_resp.choices = [choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=bad_resp)
        groq_parser_module._client = mock_client

        result = await groq_parser_module.parse_with_groq("покупаем биток в районе 69к, стоп 68к")
        self.assertIsNone(result)

    async def test_parse_telegram_signal_falls_back_to_groq_when_anthropic_not_configured(self):
        """
        Anthropic не настроен (нет ключа), Groq настроен — цепочка должна
        остановиться на Groq и НЕ доходить до Gemini.
        """
        import src.telegram.groq_parser as groq_parser_module
        from src.telegram.channel_monitor import parse_telegram_signal
        settings.telegram_llm_fallback_enabled = True
        settings.anthropic_api_key = None
        settings.groq_api_key = "test-key"
        settings.gemini_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self._mock_response({
            "is_signal": True, "base": "ETH", "quote": "USDT", "side": "short",
            "entry": 3500.0, "take_profits": [3300.0, 3200.0], "stop_loss": 3600.0,
            "confidence": 0.8,
        }))
        groq_parser_module._client = mock_client

        import src.telegram.gemini_parser as gemini_parser_module
        gemini_mock = MagicMock()
        gemini_mock.aio.models.generate_content = AsyncMock(
            side_effect=AssertionError("Gemini не должен вызываться, если Groq уже распознал сигнал")
        )
        gemini_parser_module._client = gemini_mock

        result = await parse_telegram_signal("шортим эфир около 3500, стоп 3600, цели 3300 и 3200")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "ETH/USDT")
        self.assertEqual(result["side"], "short")
        self.assertEqual(result["tp"], 3200.0)  # short -> минимальный TP
        gemini_mock.aio.models.generate_content.assert_not_called()

    async def test_parse_telegram_signal_falls_back_to_gemini_when_groq_also_fails(self):
        """Groq настроен, но не смог разобрать (низкая уверенность) — цепочка идёт дальше к Gemini."""
        import json

        import src.telegram.groq_parser as groq_parser_module
        import src.telegram.gemini_parser as gemini_parser_module
        from src.telegram.channel_monitor import parse_telegram_signal
        settings.telegram_llm_fallback_enabled = True
        settings.anthropic_api_key = None
        settings.groq_api_key = "test-key"
        settings.gemini_api_key = "test-key"

        groq_mock = MagicMock()
        groq_mock.chat.completions.create = AsyncMock(return_value=self._mock_response({
            "is_signal": False, "confidence": 0.9,
        }))
        groq_parser_module._client = groq_mock

        gemini_resp = MagicMock()
        gemini_resp.text = json.dumps({
            "is_signal": True, "base": "ETH", "quote": "USDT", "side": "short",
            "entry": 3500.0, "take_profits": [3300.0, 3200.0], "stop_loss": 3600.0,
            "confidence": 0.8,
        })
        gemini_mock = MagicMock()
        gemini_mock.aio.models.generate_content = AsyncMock(return_value=gemini_resp)
        gemini_parser_module._client = gemini_mock

        result = await parse_telegram_signal("шортим эфир около 3500, стоп 3600, цели 3300 и 3200")
        self.assertIsNotNone(result)
        self.assertEqual(result["pair"], "ETH/USDT")


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

    def test_score_signal_with_explicit_none_sl_tp_does_not_raise(self):
        """
        Регресс: signal.get("sl", 0)/("tp", 0) с ключом, ЯВНО присутствующим
        со значением None (а не отсутствующим), возвращают None — .get()
        подставляет default только при отсутствующем ключе — и "None > 0"
        в RR-блоке падал с TypeError. Именно такую форму (ключи sl/tp
        присутствуют и равны None, если канал не указал уровни) шлёт
        _on_telegram_signal в main.py.
        """
        signal = {"pair": "BTC/USDT", "side": "long", "entry": 50000.0,
                  "sl": None, "tp": None, "confidence": 0.9}
        score = self.scorer.score_signal(signal, "no_sl_tp_channel")
        self.assertIsInstance(score, float)


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


class TestRecalculateClosedTrade(unittest.IsolatedAsyncioTestCase):
    """
    ExecutionEngine.recalculate_closed_trade — ручной способ подтянуть
    точные цену/объём/комиссию с биржи постфактум для уже закрытой сделки,
    изначально записанной по оценке (биржа не успела вовремя отдать
    комиссию/цену), не дожидаясь следующего похожего инцидента.
    """

    async def test_returns_none_in_paper_mode(self):
        from src.execution.executor import ExecutionEngine

        engine = ExecutionEngine()
        settings.trading_mode = "paper"
        await engine.initialize("binance")

        result = await engine.recalculate_closed_trade(1)
        self.assertIsNone(result)

    async def test_returns_none_for_unknown_trade(self):
        from src.execution.executor import ExecutionEngine

        engine = ExecutionEngine()
        settings.trading_mode = "real"
        engine.is_paper = False
        engine.exchange_id = "bybit"
        engine.exchange = AsyncMock()

        result = await engine.recalculate_closed_trade(999999999)
        self.assertIsNone(result)

    async def test_refreshes_orders_and_recomputes_pnl_from_fresh_exchange_data(self):
        """
        Сделка изначально закрыта без реальных данных от биржи (оценочная
        комиссия/цена по запрошенным значениям) — recalculate находит
        настоящую историю сделок биржи и пересчитывает PnL по ней.
        """
        from src.execution.executor import ExecutionEngine
        from src.db.session import get_session
        from src.db.models import Order, Trade

        engine = ExecutionEngine()
        settings.trading_mode = "real"
        engine.is_paper = False
        engine.exchange_id = "bybit"
        engine.exchange = AsyncMock()
        engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"RECALC1": 0.0}, "RECALC1": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        engine.exchange.create_market_buy_order.return_value = {
            "id": "recalc-open-1", "filled": 10.0, "average": 100.0, "price": None,
            "fee": {"cost": 0, "currency": None},
        }
        engine.exchange.fetch_order_trades = AsyncMock(return_value=None)
        engine.exchange.fetch_my_trades = AsyncMock(return_value=None)
        order = await engine.create_order(
            symbol="RECALC1/USDT", side="buy", amount=10.0, price=100.0, order_type="market",
        )
        self.assertIsNotNone(order)
        self.assertEqual(order.order_id_exchange, "recalc-open-1")

        engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"RECALC1": 10.0}, "RECALC1": {"free": 10.0, "used": 0, "total": 10.0},
        })
        engine.exchange.create_market_sell_order.return_value = {
            "id": "recalc-close-1", "filled": 10.0, "average": 110.0, "price": None,
            "fee": {"cost": 0, "currency": None},
        }
        result = await engine.close_real_position(
            symbol="RECALC1/USDT", side="long", entry_price=100.0, amount=10.0,
            reason="take_profit_3", entry_fee=0.0, holding_seconds=60, order_open_id=order.id,
        )
        self.assertIsNotNone(result)
        trade_id = result["trade_id"]

        # Теперь "биржа отдала" настоящую историю сделок — с реальной
        # комиссией в base-валюте на открытии.
        async def fake_fetch_order_trades(order_id, symbol):
            if order_id == "recalc-open-1":
                return [{"id": "t1", "amount": 10.0, "price": 100.0, "cost": 1000.0,
                          "fee": {"cost": 0.05, "currency": "RECALC1"}}]
            if order_id == "recalc-close-1":
                return [{"id": "t2", "amount": 10.0, "price": 108.0, "cost": 1080.0,
                          "fee": {"cost": 1.08, "currency": "USDT"}}]
            return None
        engine.exchange.fetch_order_trades = AsyncMock(side_effect=fake_fetch_order_trades)

        recalced = await engine.recalculate_closed_trade(trade_id)
        self.assertIsNotNone(recalced)
        self.assertTrue(recalced["updated"])
        # (108 - 100) * 10 - (0.05 * 100) - 1.08 = 80 - 5 - 1.08 = 73.92
        self.assertAlmostEqual(recalced["pnl"], 73.92, places=4)
        self.assertEqual(recalced["outcome"], "win")

        async with get_session() as session:
            refreshed_trade = await session.get(Trade, trade_id)
            self.assertAlmostEqual(float(refreshed_trade.pnl), 73.92, places=4)
            self.assertAlmostEqual(float(refreshed_trade.exit_price), 108.0)
            refreshed_open = await session.get(Order, order.id)
            self.assertEqual(refreshed_open.fee_currency, "RECALC1")

    async def test_returns_not_updated_when_exchange_has_nothing_new(self):
        from src.execution.executor import ExecutionEngine

        engine = ExecutionEngine()
        settings.trading_mode = "real"
        engine.is_paper = False
        engine.exchange_id = "bybit"
        engine.exchange = AsyncMock()
        engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"RECALC2": 0.0}, "RECALC2": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        engine.exchange.create_market_buy_order.return_value = {
            "id": "recalc2-open-1", "filled": 5.0, "average": 50.0, "price": None,
            "fee": {"cost": 0.1, "currency": "USDT"},
        }
        engine.exchange.fetch_order_trades = AsyncMock(return_value=None)
        engine.exchange.fetch_my_trades = AsyncMock(return_value=None)
        order = await engine.create_order(
            symbol="RECALC2/USDT", side="buy", amount=5.0, price=50.0, order_type="market",
        )
        engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"RECALC2": 5.0}, "RECALC2": {"free": 5.0, "used": 0, "total": 5.0},
        })
        engine.exchange.create_market_sell_order.return_value = {
            "id": "recalc2-close-1", "filled": 5.0, "average": 55.0, "price": None,
            "fee": {"cost": 0.1, "currency": "USDT"},
        }
        result = await engine.close_real_position(
            symbol="RECALC2/USDT", side="long", entry_price=50.0, amount=5.0,
            reason="take_profit_3", entry_fee=0.1, holding_seconds=60, order_open_id=order.id,
        )
        self.assertIsNotNone(result)

        # Ни fetch_order_trades, ни fetch_my_trades по-прежнему ничего не
        # находят — как и было при закрытии.
        recalced = await engine.recalculate_closed_trade(result["trade_id"])
        self.assertEqual(recalced, {"updated": False})


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


class TestGracefulShutdownWaitsForCleanup(unittest.IsolatedAsyncioTestCase):
    """
    SIGTERM (то, чем docker/docker-compose штатно останавливает контейнер)
    раньше не имел вообще никакого обработчика — процесс убивался ОС
    мгновенно, ни разу не долетая до TradingBot._cleanup() (см.
    TestCleanupClosesExchangeConnection выше — сам _cleanup() уже был
    исправлен, но это не помогало, если его никогда не вызывали). Теперь
    main() отменяет bot_task вручную по сигналу и явно дожидается, пока
    run() дойдёт до своего _cleanup() — _run_until_shutdown() и есть эта
    логика ожидания, вынесенная отдельно, чтобы проверить её без реального
    uvicorn/сигналов ОС.
    """

    async def test_cancelling_bot_task_waits_for_its_cleanup_before_returning(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        # Тесты этого файла глобально патчат asyncio.sleep на AsyncMock (см.
        # conftest._no_real_polling_delay) — он резолвится без реальной
        # приостановки, поэтому "while: await asyncio.sleep(...)" здесь
        # превратился бы в бесконечный busy-loop, ни разу не отдающий
        # управление планировщику. asyncio.Event — обычная примитива
        # синхронизации, этим патчем не затронута.
        cleanup_marker = {"ran": False}
        started = asyncio.Event()

        async def fake_bot_run():
            started.set()
            try:
                await asyncio.Event().wait()  # никогда не установится сам — только через cancel()
            except asyncio.CancelledError:
                # Мимикрирует TradingBot.run(): ловит отмену внутри себя,
                # "делает cleanup" и возвращается нормально, не перевызывая
                # исключение — то же поведение, что и у настоящего run().
                cleanup_marker["ran"] = True
                return

        class FakeWebServer:
            def __init__(self):
                self._exit_event = asyncio.Event()

            @property
            def should_exit(self):
                return self._exit_event.is_set()

            @should_exit.setter
            def should_exit(self, value):
                if value:
                    self._exit_event.set()

            async def serve(self):
                await self._exit_event.wait()

        fake_server = FakeWebServer()
        bot_task = asyncio.create_task(fake_bot_run())
        server_task = asyncio.create_task(fake_server.serve())

        await started.wait()  # дать bot_task реально стартовать
        bot_task.cancel()  # то же самое, что делает _request_shutdown() по сигналу

        await main_module._run_until_shutdown(bot_task, server_task, fake_server)

        self.assertTrue(cleanup_marker["ran"], "bot_task должен успеть дойти до своего cleanup")
        self.assertTrue(fake_server.should_exit, "should_exit должен быть выставлен серверу")
        self.assertTrue(bot_task.done())
        self.assertTrue(server_task.done())


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


class TestTpLevels(unittest.TestCase):
    """
    _tp_levels: для сделок из Telegram-канала — 3 уровня частичной фиксации
    (TP1/TP2/TP3); для остальных источников (стратегии) — временно только
    одинарный TP (TP1=TP2=None, TP3=финальный уровень).
    """

    def _tp_levels(self, entry_price, tp, strategy_id=None):
        import src.main as main_module
        return main_module.TradingBot._tp_levels(entry_price, tp, strategy_id)

    def test_telegram_signal_splits_into_three_levels(self):
        tp1, tp2, tp3 = self._tp_levels(100.0, 130.0, "telegram_signal")
        self.assertAlmostEqual(tp1, 110.0)
        self.assertAlmostEqual(tp2, 120.0)
        self.assertAlmostEqual(tp3, 130.0)

    def test_strategy_signal_uses_single_tp(self):
        tp1, tp2, tp3 = self._tp_levels(100.0, 130.0, "ensemble_voter")
        self.assertIsNone(tp1)
        self.assertIsNone(tp2)
        self.assertEqual(tp3, 130.0)

    def test_no_strategy_id_defaults_to_single_tp(self):
        tp1, tp2, tp3 = self._tp_levels(100.0, 130.0, None)
        self.assertIsNone(tp1)
        self.assertIsNone(tp2)
        self.assertEqual(tp3, 130.0)

    def test_no_tp_returns_all_none_regardless_of_source(self):
        self.assertEqual(self._tp_levels(100.0, None, "telegram_signal"), (None, None, None))
        self.assertEqual(self._tp_levels(100.0, None, "ensemble_voter"), (None, None, None))

    def test_short_side_symmetric_for_telegram_signal(self):
        tp1, tp2, tp3 = self._tp_levels(100.0, 70.0, "telegram_signal")
        self.assertAlmostEqual(tp1, 90.0)
        self.assertAlmostEqual(tp2, 80.0)
        self.assertAlmostEqual(tp3, 70.0)


class TestTpLevelsUsesRealChannelTargets(unittest.TestCase):
    """
    Регресс: даже когда канал прислал явные TP1/TP2/TP3, _tp_levels()
    всё равно линейно интерполировал 3 фейковых уровня между entry и ОДНИМ
    числом (после того как парсер уже схлопнул несколько целей в одно) —
    реальные цены, прямо указанные каналом, отбрасывались. Теперь
    take_profits (ближайшая цель первая) используется напрямую.
    """

    def _tp_levels(self, entry_price, tp, strategy_id, take_profits):
        import src.main as main_module
        return main_module.TradingBot._tp_levels(entry_price, tp, strategy_id, take_profits)

    def test_three_real_targets_used_directly_not_interpolated(self):
        tp1, tp2, tp3 = self._tp_levels(100.0, 130.0, "telegram_signal", [111.0, 122.0, 130.0])
        self.assertEqual(tp1, 111.0)
        self.assertEqual(tp2, 122.0)
        self.assertEqual(tp3, 130.0)

    def test_two_real_targets_map_to_tp2_tp3_leaving_tp1_none(self):
        """Только 2 цели — недостающий ближний уровень (TP1) пропускается,
        а не выдумывается интерполяцией."""
        tp1, tp2, tp3 = self._tp_levels(100.0, 130.0, "telegram_signal", [120.0, 130.0])
        self.assertIsNone(tp1)
        self.assertEqual(tp2, 120.0)
        self.assertEqual(tp3, 130.0)

    def test_single_real_target_is_full_close_not_partial_split(self):
        tp1, tp2, tp3 = self._tp_levels(100.0, 130.0, "telegram_signal", [130.0])
        self.assertIsNone(tp1)
        self.assertIsNone(tp2)
        self.assertEqual(tp3, 130.0)

    def test_more_than_three_targets_uses_first_three(self):
        tp1, tp2, tp3 = self._tp_levels(
            100.0, 140.0, "telegram_signal", [110.0, 120.0, 130.0, 140.0],
        )
        self.assertEqual(tp1, 110.0)
        self.assertEqual(tp2, 120.0)
        self.assertEqual(tp3, 130.0)

    def test_empty_take_profits_falls_back_to_interpolation(self):
        tp1, tp2, tp3 = self._tp_levels(100.0, 130.0, "telegram_signal", [])
        self.assertAlmostEqual(tp1, 110.0)
        self.assertAlmostEqual(tp2, 120.0)
        self.assertAlmostEqual(tp3, 130.0)

    def test_real_targets_ignored_for_non_telegram_source(self):
        tp1, tp2, tp3 = self._tp_levels(100.0, 130.0, "ensemble_voter", [111.0, 122.0, 130.0])
        self.assertIsNone(tp1)
        self.assertIsNone(tp2)
        self.assertEqual(tp3, 130.0)


class TestSymbolBlacklistSkipsProcessing(unittest.IsolatedAsyncioTestCase):
    """
    _process_symbol должен переставать генерировать НОВЫЕ сигналы стратегий
    для блэклист-символов без открытой позиции. Раньше это фильтровалось
    только при построении active_symbols (_refresh_symbol_universe) —
    символ с уже открытой позицией на момент блокировки намеренно остаётся
    в active_symbols (чтобы SL/TP по нему продолжали проверяться), но между
    обновлениями вселенной (раз в symbol_universe_refresh_hours) он, после
    закрытия этой позиции, как ни в чём не бывало продолжал получать НОВЫЕ
    сигналы — risk_manager.check_signal() блэклист вообще не проверяет.
    Реальный инцидент (прод): RLUSD/USDT, USDE/USDT, USDC/USDT — уже
    добавленные в блэклист — продолжали открываться заново.

    ВАЖНО: первая версия этого фикса ставила проверку блэклиста ДО
    _check_position_exit() и ловила регресс именно на том сценарии, который
    тестирует test_blacklisted_symbol_with_stale_position_entry_* ниже —
    self.open_positions синхронно чистится (лениво, при закрытии в обход
    основного цикла — кнопка "Закрыть" в дашборде и т.п.) только ВНУТРИ
    _check_position_exit(); проверка до него видела устаревшую запись и
    пропускала обработку дальше как если бы позиция всё ещё была открыта —
    ровно то же самое поведение, которое чинил исходный баг.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    @staticmethod
    def _make_candles_df():
        return pd.DataFrame({
            "open": [1.0] * 60, "high": [1.0] * 60, "low": [1.0] * 60,
            "close": [1.0] * 60, "volume": [1.0] * 60,
        })

    def setUp(self):
        self._saved = {
            "symbol_blacklist": settings.symbol_blacklist,
            "trading_mode": settings.trading_mode,
        }
        settings.trading_mode = "paper"

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(settings, key, value)

    async def test_blacklisted_symbol_never_had_position_is_skipped(self):
        settings.symbol_blacklist = ["RLUSD/USDT"]
        bot = self._make_bot()
        bot.feature_engine = MagicMock()
        bot._refresh_symbol_candles = AsyncMock(return_value=self._make_candles_df())

        await bot._process_symbol("RLUSD/USDT")

        bot.feature_engine.compute_all_indicators.assert_not_called()

    async def test_blacklisted_symbol_with_stale_position_entry_is_cleaned_up_and_skipped(self):
        """
        Точный сценарий прод-инцидента: позицию закрыли в обход основного
        цикла (POST /positions/close трогает execution_engine, но НЕ
        self.open_positions напрямую) — запись в self.open_positions
        осталась устаревшей до следующего вызова _check_position_exit.
        """
        settings.symbol_blacklist = ["RLUSD/USDT"]
        bot = self._make_bot()
        bot.open_positions["RLUSD/USDT"] = {
            "side": "long", "entry_price": 1.0, "amount": 10.0, "sl": None, "tp": None,
            "tp_hit_count": 0, "strategy_id": "manual", "order_id": 1,
        }
        bot.feature_engine = MagicMock()
        bot._refresh_symbol_candles = AsyncMock(return_value=self._make_candles_df())

        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.paper_positions = {}
            mock_engine.real_positions = {}
            mock_engine.last_prices = {}
            await bot._process_symbol("RLUSD/USDT")

        self.assertNotIn("RLUSD/USDT", bot.open_positions)
        bot.feature_engine.compute_all_indicators.assert_not_called()

    async def test_blacklisted_symbol_with_genuinely_open_position_is_still_processed(self):
        settings.symbol_blacklist = ["RLUSD/USDT"]
        bot = self._make_bot()
        bot.open_positions["RLUSD/USDT"] = {
            "side": "long", "entry_price": 1.0, "amount": 10.0, "sl": None, "tp": None,
            "tp_hit_count": 0, "strategy_id": "manual", "order_id": 1,
        }
        bot.feature_engine = MagicMock()
        bot._refresh_symbol_candles = AsyncMock(return_value=self._make_candles_df())

        with patch("src.main.execution_engine") as mock_engine, \
                patch("src.main.strategy_registry.get_active", return_value=[]):
            mock_engine.paper_positions = {"RLUSD/USDT": bot.open_positions["RLUSD/USDT"]}
            mock_engine.real_positions = {}
            mock_engine.last_prices = {}
            await bot._process_symbol("RLUSD/USDT")

        self.assertIn("RLUSD/USDT", bot.open_positions)
        bot.feature_engine.compute_all_indicators.assert_called_once()

    async def test_non_blacklisted_symbol_is_processed_normally(self):
        settings.symbol_blacklist = ["RLUSD/USDT"]
        bot = self._make_bot()
        bot.feature_engine = MagicMock()
        bot._refresh_symbol_candles = AsyncMock(return_value=self._make_candles_df())

        with patch("src.main.strategy_registry.get_active", return_value=[]):
            await bot._process_symbol("BTC/USDT")

        bot.feature_engine.compute_all_indicators.assert_called_once()


class TestCheckPositionExitCleansStaleEntryOnFailedClose(unittest.IsolatedAsyncioTestCase):
    """
    Регресс на прод-инцидент: close_real_position/close_paper_position могут
    САМИ снять позицию с учёта execution_engine ВНУТРИ себя (недопродаваемая
    пыль ниже минимума биржи — см. _reconcile_phantom_position) и вернуть
    None вызывающему коду. Раньше _check_position_exit() в этом случае
    просто возвращал False, не трогая self.open_positions — запись
    оставалась устаревшей, и _process_symbol() в ТОЙ ЖЕ итерации, чуть
    ниже по коду, успевал сгенерировать и исполнить НОВЫЙ сигнал на тот же
    символ, думая, что позиция ещё открыта (реальный инцидент: неудачное
    закрытие пыли по SL на RLUSD/USDT и USDC/USDT — обоих блэклист-символов
    — тут же сменялось открытием дублирующей новой позиции в той же
    итерации, до следующего вызова _check_position_exit).
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot(), main_module.execution_engine

    def setUp(self):
        self._saved_trading_mode = settings.trading_mode
        settings.trading_mode = "paper"

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode

    async def test_stale_open_positions_entry_cleaned_up_when_close_reconciles_away(self):
        from src.utils.timeutils import utcnow

        bot, engine = self._make_bot()
        engine.paper_positions["DUSTCLOSE1/USDT"] = {
            "side": "long", "entry_price": 1.0, "amount": 0.0001,
        }
        bot.open_positions["DUSTCLOSE1/USDT"] = {
            "side": "long", "entry_price": 1.0, "amount": 0.0001, "sl": 1.0, "tp": None,
            "tp_hit_count": 0, "strategy_id": "ensemble_voter", "opened_at": utcnow(),
        }

        async def fake_close_paper_position(**kwargs):
            # Имитирует поведение _reconcile_phantom_position внутри
            # close_paper_position: позиция снята с учёта execution_engine,
            # но самому close_fn закрыть её обычным способом не удалось —
            # возвращает None, как и при реальной непродаваемой пыли.
            engine.paper_positions.pop("DUSTCLOSE1/USDT", None)
            return None

        with patch.object(engine, "close_paper_position", side_effect=fake_close_paper_position):
            closed = await bot._check_position_exit("DUSTCLOSE1/USDT", 1.0)

        self.assertFalse(closed)
        self.assertNotIn("DUSTCLOSE1/USDT", bot.open_positions)

    async def test_open_positions_entry_kept_when_close_genuinely_just_failed(self):
        """
        Отличие от предыдущего теста: close_fn вернул None, НО execution_engine
        всё ещё отслеживает позицию (просто временная ошибка — сетевой сбой
        и т.п., а не реконсиляция фантомной пыли) — запись НЕ должна
        стираться, чтобы попытка закрытия повторилась на следующей итерации.
        """
        from src.utils.timeutils import utcnow

        bot, engine = self._make_bot()
        engine.paper_positions["DUSTCLOSE2/USDT"] = {
            "side": "long", "entry_price": 1.0, "amount": 5.0,
        }
        bot.open_positions["DUSTCLOSE2/USDT"] = {
            "side": "long", "entry_price": 1.0, "amount": 5.0, "sl": 1.0, "tp": None,
            "tp_hit_count": 0, "strategy_id": "ensemble_voter", "opened_at": utcnow(),
        }

        with patch.object(engine, "close_paper_position", AsyncMock(return_value=None)):
            closed = await bot._check_position_exit("DUSTCLOSE2/USDT", 1.0)

        self.assertFalse(closed)
        self.assertIn("DUSTCLOSE2/USDT", bot.open_positions)


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
                quality_threshold=0.85, auto_execute=True, position_size_pct=7.5,
                market="futures", active=True,
            )
            session.add(channel)
            await session.commit()
            db_id = channel.id

        bot = main_module.TradingBot()
        bot._telegram_channel_db_ids = {"@channelsettings_unittest": db_id}

        threshold, auto_execute, position_size_pct, market_type = await bot._get_channel_settings(
            "@channelsettings_unittest"
        )
        self.assertAlmostEqual(threshold, 0.85)
        self.assertTrue(auto_execute)
        self.assertAlmostEqual(position_size_pct, 7.5)
        self.assertEqual(market_type, "futures")

    async def test_falls_back_to_global_settings_for_unknown_channel(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")

        bot = main_module.TradingBot()
        bot._telegram_channel_db_ids = {}

        threshold, auto_execute, position_size_pct, market_type = await bot._get_channel_settings(
            "@unknown_channel_unittest"
        )
        self.assertEqual(threshold, settings.telegram_signals_quality_threshold)
        self.assertEqual(auto_execute, settings.telegram_signals_auto_execute)
        self.assertEqual(position_size_pct, 5.0)
        self.assertEqual(market_type, settings.market_type)


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

    async def test_update_market_field(self):
        from src.db.models import TelegramChannel
        from src.db.session import get_session
        from src.web.api import TelegramChannelUpdate, update_telegram_channel

        async with get_session() as session:
            channel = TelegramChannel(
                channel_id="@marketpatch_unittest", channel_title="X",
                quality_threshold=0.5, auto_execute=False, market="spot", active=True,
            )
            session.add(channel)
            await session.commit()
            db_id = channel.id

        result = await update_telegram_channel(db_id, TelegramChannelUpdate(market="futures"))
        self.assertTrue(result["success"])
        self.assertEqual(result["updated"], ["market"])

        async with get_session() as session:
            refreshed = await session.get(TelegramChannel, db_id)
            self.assertEqual(refreshed.market, "futures")

    async def test_update_invalid_market_raises_400(self):
        from fastapi import HTTPException

        from src.web.api import TelegramChannelUpdate, update_telegram_channel

        with self.assertRaises(HTTPException) as ctx:
            await update_telegram_channel(999999999, TelegramChannelUpdate(market="margin"))
        self.assertEqual(ctx.exception.status_code, 400)


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

        # Первый вызов — клиент ТЕКУЩЕГО рынка (settings.market_type,
        # эagerly подключается всегда); в общей тестовой БД (см. conftest —
        # она не очищается между тестами) может найтись реальная позиция,
        # оставленная ДРУГИМ тестом на другом рынке — тогда initialize()
        # лениво поднимет и второй клиент для неё (см. план "два
        # параллельных подключения"). Оба вызова используют одни и те же
        # bybit-ключи — проверяем именно ПЕРВЫЙ, не количество вызовов.
        mock_cls.assert_called()
        config = mock_cls.call_args_list[0].args[0]
        self.assertEqual(config["apiKey"], "correct-bybit-key")
        self.assertEqual(config["secret"], "correct-bybit-secret")
        # Bybit: demo-ключ живёт на api-demo.bybit.com (enable_demo_trading),
        # а НЕ на testnet.bybit.com (set_sandbox_mode) — это разные песочницы
        # с разными ключами, см. комментарий в executor.py. mock_exchange —
        # один и тот же объект для ЛЮБОГО количества подключений (см.
        # комментарий выше про возможный второй клиент), поэтому проверяем
        # факт вызова, а не количество.
        mock_exchange.enable_demo_trading.assert_called_with(True)
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


class TestGetAllBalances(unittest.IsolatedAsyncioTestCase):
    """
    get_all_balances() — все ненулевые балансы аккаунта на бирже (не
    только USDT, как get_real_balance()), для отображения на дашборде
    (см. запрос "можно в дашборде отображать все актуальные балансы
    аккаунта на бирже, не только в базовой монете").
    """

    async def test_returns_only_nonzero_currencies_sorted(self):
        engine = ExecutionEngine()
        engine.exchange = MagicMock()
        engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 100.0, "BTC": 0.001, "ETH": 0.0},
            "used": {"USDT": 5.0, "BTC": 0.0},
            "total": {"USDT": 105.0, "BTC": 0.001, "ETH": 0.0},
        })
        engine.exchange.fetch_ticker = AsyncMock(return_value={"last": 60000.0})
        result = await engine.get_all_balances()
        self.assertEqual([b["currency"] for b in result], ["BTC", "USDT"])
        engine.exchange.fetch_ticker.assert_awaited_once_with("BTC/USDT")
        btc = next(b for b in result if b["currency"] == "BTC")
        self.assertEqual(
            btc, {"currency": "BTC", "free": 0.001, "used": 0.0, "total": 0.001, "usdt_value": 60.0},
        )
        usdt = next(b for b in result if b["currency"] == "USDT")
        self.assertEqual(
            usdt, {"currency": "USDT", "free": 100.0, "used": 5.0, "total": 105.0, "usdt_value": 105.0},
        )

    async def test_ticker_error_yields_none_usdt_value(self):
        engine = ExecutionEngine()
        engine.exchange = MagicMock()
        engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"SOMECOIN": 10.0}, "used": {}, "total": {"SOMECOIN": 10.0},
        })
        engine.exchange.fetch_ticker = AsyncMock(side_effect=RuntimeError("no such market"))
        result = await engine.get_all_balances()
        self.assertEqual(result, [
            {"currency": "SOMECOIN", "free": 10.0, "used": 0.0, "total": 10.0, "usdt_value": None},
        ])

    async def test_falls_back_to_nested_currency_dict(self):
        engine = ExecutionEngine()
        engine.exchange = MagicMock()
        engine.exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 50.0, "used": 0.0, "total": 50.0},
        })
        result = await engine.get_all_balances()
        self.assertEqual(
            result, [{"currency": "USDT", "free": 50.0, "used": 0.0, "total": 50.0, "usdt_value": 50.0}],
        )

    async def test_no_exchange_returns_none(self):
        engine = ExecutionEngine()
        engine.exchange = None
        self.assertIsNone(await engine.get_all_balances())

    async def test_fetch_error_returns_none(self):
        engine = ExecutionEngine()
        engine.exchange = MagicMock()
        engine.exchange.fetch_balance = AsyncMock(side_effect=RuntimeError("boom"))
        self.assertIsNone(await engine.get_all_balances())


class TestBalancesEndpoint(unittest.IsolatedAsyncioTestCase):
    """GET /balances — список всех ненулевых балансов аккаунта для дашборда."""

    def setUp(self):
        self._saved_mode = settings.trading_mode

    def tearDown(self):
        settings.trading_mode = self._saved_mode

    async def test_paper_mode_returns_empty_list(self):
        from src.web.api import get_balances

        settings.trading_mode = "paper"
        result = await get_balances()
        self.assertEqual(result, {"balances": [], "trading_mode": "paper"})

    async def test_real_mode_returns_engine_balances(self):
        import src.web.api as api_module
        from src.web.api import get_balances

        settings.trading_mode = "real"
        saved_engine = api_module.execution_engine
        mock_engine = MagicMock()
        mock_engine.get_all_balances = AsyncMock(return_value=[
            {"currency": "USDT", "free": 100.0, "used": 0.0, "total": 100.0},
        ])
        api_module.execution_engine = mock_engine
        try:
            result = await get_balances()
        finally:
            api_module.execution_engine = saved_engine

        self.assertEqual(result["trading_mode"], "real")
        self.assertEqual(result["balances"], [{"currency": "USDT", "free": 100.0, "used": 0.0, "total": 100.0}])

    async def test_real_mode_none_from_engine_becomes_empty_list(self):
        import src.web.api as api_module
        from src.web.api import get_balances

        settings.trading_mode = "real"
        saved_engine = api_module.execution_engine
        mock_engine = MagicMock()
        mock_engine.get_all_balances = AsyncMock(return_value=None)
        api_module.execution_engine = mock_engine
        try:
            result = await get_balances()
        finally:
            api_module.execution_engine = saved_engine

        self.assertEqual(result, {"balances": [], "trading_mode": "real"})


class TestRestartBotSendsSigtermNotOsExit(unittest.IsolatedAsyncioTestCase):
    """
    POST /system/restart раньше завершал процесс через os._exit(0) — это
    убивает процесс мгновенно на уровне C, минуя ЛЮБУЮ Python-очистку,
    включая уже существующий SIGTERM-обработчик в main.py, который как раз
    закрывает ccxt-биржу (execution_engine.close()) перед выходом. Реальный
    симптом: каждый рестарт через эту кнопку логировал "Unclosed client
    session"/"Unclosed connector" от aiohttp, хотя для docker stop/recreate
    (тоже SIGTERM) это уже было починено раньше. Кнопка должна слать себе
    тот же SIGTERM, каким останавливает контейнер docker — тогда сработает
    тот же graceful shutdown.
    """

    async def test_delayed_task_sends_sigterm_to_self(self):
        import src.web.api as api_module
        from src.web.api import restart_bot

        captured_tasks = []
        real_create_task = asyncio.create_task

        def _capture(coro, *args, **kwargs):
            task = real_create_task(coro, *args, **kwargs)
            captured_tasks.append(task)
            return task

        with (
            patch("src.web.api.asyncio.sleep", new=AsyncMock()),
            patch("src.web.api.os.kill") as mock_kill,
            patch("src.web.api.asyncio.create_task", side_effect=_capture),
        ):
            result = await restart_bot()
            self.assertTrue(captured_tasks)
            await captured_tasks[0]

        self.assertEqual(result, {"success": True, "message": "Бот перезапускается"})
        mock_kill.assert_called_once_with(api_module.os.getpid(), api_module.signal.SIGTERM)


class TestInitializeClosesPreviousExchangeConnection(unittest.IsolatedAsyncioTestCase):
    """
    Реальный инцидент (прод): живое переключение market_type/active_exchange/
    use_exchange_sandbox БЕЗ рестарта процесса (см. settings_store.
    apply_settings_update — вызывает execution_engine.initialize() повторно
    на уже работающем движке) оставляло старое соединение (ccxt +
    aiohttp ClientSession) незакрытым, перезаписывая его новым — та же
    "Unclosed client session"/"Unclosed connector", что и в баге #31
    (кнопка "Перезапустить бота"), но по другому пути: там процесс убивался
    целиком мимо graceful shutdown, здесь процесс живёт, а initialize()
    просто не закрывал ПРЕДЫДУЩЕЕ соединение перед созданием нового.
    """

    def setUp(self):
        self._saved = {
            k: getattr(settings, k) for k in (
                "trading_mode", "bybit_api_key", "bybit_api_secret", "use_exchange_sandbox", "market_type",
            )
        }
        settings.trading_mode = "real"
        settings.bybit_api_key = "key"
        settings.bybit_api_secret = "secret"
        settings.use_exchange_sandbox = True

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)

    async def test_second_initialize_closes_first_exchange(self):
        engine = ExecutionEngine()
        engine.is_paper = False
        old_exchange = AsyncMock()
        engine.exchange = old_exchange

        new_exchange = AsyncMock()
        new_exchange.enable_demo_trading = MagicMock()
        new_exchange.fetch_balance = AsyncMock(
            return_value={"free": {"USDT": 10.0}, "USDT": {"free": 10.0, "used": 0, "total": 10.0}}
        )
        with patch("src.execution.executor.ccxt.bybit", return_value=new_exchange):
            await engine.initialize("bybit")

        old_exchange.close.assert_awaited_once()
        self.assertIs(engine.exchange, new_exchange)

    async def test_first_initialize_with_no_prior_exchange_does_not_error(self):
        engine = ExecutionEngine()
        engine.is_paper = False
        self.assertIsNone(engine.exchange)

        new_exchange = AsyncMock()
        new_exchange.enable_demo_trading = MagicMock()
        new_exchange.fetch_balance = AsyncMock(
            return_value={"free": {"USDT": 10.0}, "USDT": {"free": 10.0, "used": 0, "total": 10.0}}
        )
        with patch("src.execution.executor.ccxt.bybit", return_value=new_exchange):
            await engine.initialize("bybit")

        self.assertIs(engine.exchange, new_exchange)


class TestDualMarketExchangeClients(unittest.IsolatedAsyncioTestCase):
    """
    ЭТАП 3 перехода на фьючерсы: settings.market_type — глобальный тумблер,
    но реальная позиция могла быть открыта на ДРУГОМ рынке и должна
    оставаться под защитой (SL, закрытие) через клиент ИМЕННО ТОГО рынка,
    даже если тумблер потом переключили. Реальный инцидент (прод): 3
    спотовые позиции (MON/RLUSD/USDC) при переключении тумблера на futures
    стали обслуживаться так, будто они фьючерсные — исполнение шло через
    единственный self.exchange, привязанный к текущему тумблеру.
    ExecutionEngine теперь держит словарь _exchanges по рынку, а каждая
    real-позиция помечена своим market_type (Order.market_type в БД).
    """

    def setUp(self):
        self._saved = {
            k: getattr(settings, k) for k in (
                "trading_mode", "bybit_api_key", "bybit_api_secret", "use_exchange_sandbox", "market_type",
            )
        }
        settings.trading_mode = "real"
        settings.bybit_api_key = "key"
        settings.bybit_api_secret = "secret"
        settings.use_exchange_sandbox = True
        settings.market_type = "spot"

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)

    async def test_initialize_lazily_connects_second_market_for_restored_position(self):
        """
        Тумблер сейчас на spot, но в БД есть открытая real-позиция,
        помеченная как futures (восстановлена раньше или осталась с другого
        рынка) — initialize() должен эagerly подключить клиент ТЕКУЩЕГО
        рынка (spot) и ДОПОЛНИТЕЛЬНО лениво поднять клиент для futures, не
        трогая текущий выбор тумблера.
        """
        from src.db.session import get_session
        from src.db.models import Order

        engine = ExecutionEngine()
        engine.is_paper = False
        engine.exchange_id = "bybit"

        symbol = "DUALMKT1/USDT"
        async with get_session() as session:
            exchange_id, symbol_id = await engine._resolve_symbol_id(session, symbol)
            order = Order(
                exchange_id=exchange_id, symbol_id=symbol_id,
                side="sell", order_type="market", amount=5.0, price=3.0,
                status="filled", filled_amount=5.0, filled_price=3.0,
                fee=0.01, market_type="futures",
                order_id_exchange="dual-mkt-1", client_order_id="dualmkt1",
            )
            session.add(order)
            await session.commit()

        spot_mock = AsyncMock()
        spot_mock.enable_demo_trading = MagicMock()
        spot_mock.fetch_balance = AsyncMock(
            return_value={"free": {"USDT": 10.0}, "USDT": {"free": 10.0, "used": 0, "total": 10.0}}
        )
        futures_mock = AsyncMock()
        futures_mock.enable_demo_trading = MagicMock()

        def make_exchange(config):
            options = config.get("options") or {}
            return futures_mock if options.get("defaultType") == "swap" else spot_mock

        with patch("src.execution.executor.ccxt.bybit", side_effect=make_exchange):
            await engine.initialize("bybit")

        self.assertIs(engine.exchange, spot_mock)  # текущий рынок (тумблер) — spot, не тронут
        self.assertIn(symbol, engine.real_positions)
        self.assertEqual(engine.real_positions[symbol]["market_type"], "futures")
        self.assertIs(engine._exchange_for(engine.real_positions[symbol]), futures_mock)

    async def test_reinitialize_closes_all_previous_market_clients(self):
        """
        Живое переключение настроек (см. settings_store.apply_settings_update)
        может вызвать initialize() повторно, когда УЖЕ подключены клиенты
        ОБОИХ рынков (см. предыдущий тест) — все они должны закрываться при
        реконнекте, а не только клиент текущего рынка (та же утечка, что и
        TestInitializeClosesPreviousExchangeConnection, но для словаря).
        """
        engine = ExecutionEngine()
        engine.is_paper = False
        old_spot = AsyncMock()
        old_futures = AsyncMock()
        engine._exchanges = {"spot": old_spot, "futures": old_futures}

        new_exchange = AsyncMock()
        new_exchange.enable_demo_trading = MagicMock()
        new_exchange.fetch_balance = AsyncMock(
            return_value={"free": {"USDT": 10.0}, "USDT": {"free": 10.0, "used": 0, "total": 10.0}}
        )
        with patch("src.execution.executor.ccxt.bybit", return_value=new_exchange):
            await engine.initialize("bybit")

        old_spot.close.assert_awaited_once()
        old_futures.close.assert_awaited_once()

    async def test_close_real_position_resolves_client_by_positions_own_market_type(self):
        """
        Тумблер сейчас на spot, но закрываемая позиция помечена как
        futures — close_real_position должен уйти именно во ФЬЮЧЕРСНЫЙ
        mock-клиент (по pos["market_type"]), а не в клиент текущего
        тумблера (который в реальности вообще может быть не подключен под
        этот символ).
        """
        from src.utils.timeutils import utcnow

        engine = ExecutionEngine()
        engine.is_paper = False
        spot_mock = AsyncMock()
        futures_mock = AsyncMock()
        engine._exchanges = {"spot": spot_mock, "futures": futures_mock}
        engine.real_positions["DUALCLOSE1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "short", "sl_order_id": None,
            "market_type": "futures", "opened_at": utcnow(),
        }
        futures_mock.create_market_buy_order.return_value = {
            "id": "dualclose-1", "filled": 10.0, "average": 2.5, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }

        result = await engine.close_real_position(
            symbol="DUALCLOSE1/USDT", side="short", entry_price=2.0, amount=10.0, reason="test",
        )

        self.assertIsNotNone(result)
        futures_mock.create_market_buy_order.assert_awaited_once()
        spot_mock.create_market_buy_order.assert_not_called()
        spot_mock.create_market_sell_order.assert_not_called()
        spot_mock.fetch_balance.assert_not_called()

    async def test_sync_stop_loss_order_resolves_client_by_positions_own_market_type(self):
        """
        Та же логика для SL: тумблер на futures (глобально биржевой SL для
        фьючерсов пока не ставится), но позиция сама помечена spot —
        sync_stop_loss_order должен по-прежнему выставить SL через
        SPOT-клиент, используя market_type самой позиции, а не текущего
        тумблера.
        """
        settings.market_type = "futures"
        engine = ExecutionEngine()
        engine.is_paper = False
        spot_mock = AsyncMock()
        futures_mock = AsyncMock()
        spot_mock.create_market_sell_order.return_value = {"id": "spot-sl-dual-1"}
        spot_mock.fetch_balance = AsyncMock(return_value={})
        engine._exchanges = {"spot": spot_mock, "futures": futures_mock}
        engine.real_positions["DUALSL1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long", "sl_order_id": None,
            "market_type": "spot",
        }

        await engine.sync_stop_loss_order("DUALSL1/USDT", 10.0, 1.8)

        spot_mock.create_market_sell_order.assert_awaited_once()
        futures_mock.create_market_sell_order.assert_not_called()
        self.assertEqual(engine.real_positions["DUALSL1/USDT"]["sl_order_id"], "spot-sl-dual-1")


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


class TestMarketTypeFuturesConnection(unittest.IsolatedAsyncioTestCase):
    """
    ЭТАП 1 перехода на фьючерсы (см. запрос "давай начнём первый этап на
    demo... переключение в шапке по аналогии сигналы/алго"): settings.market_type
    должен определять, к какому рынку ccxt подключается execution_engine —
    linear-swap (USDT-perpetual) при "futures", обычный spot при "spot"
    (значение по умолчанию, без изменений в поведении).
    """

    def setUp(self):
        self._saved = {
            k: getattr(settings, k) for k in (
                "trading_mode", "market_type", "bybit_api_key", "bybit_api_secret", "use_exchange_sandbox",
            )
        }
        settings.trading_mode = "real"
        settings.bybit_api_key = "key"
        settings.bybit_api_secret = "secret"
        settings.use_exchange_sandbox = True

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)

    async def test_futures_market_type_connects_to_linear_swap(self):
        settings.market_type = "futures"
        engine = ExecutionEngine()
        engine.is_paper = False
        mock_exchange = AsyncMock()
        mock_exchange.enable_demo_trading = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(
            return_value={"free": {"USDT": 100.0}, "USDT": {"free": 100.0, "used": 0, "total": 100.0}}
        )
        with patch("src.execution.executor.ccxt.bybit", return_value=mock_exchange) as mock_class:
            await engine.initialize("bybit")

        # Первый вызов — клиент ТЕКУЩЕГО рынка (эagerly подключается всегда);
        # последующие вызовы (если есть) — лениво поднятые клиенты ДРУГИХ
        # рынков для позиций, восстановленных из БД на них (см.
        # TestInitializeConnectsSecondMarketForRestoredPositions) — не
        # относятся к этой проверке.
        called_config = mock_class.call_args_list[0][0][0]
        self.assertEqual(called_config["options"], {"defaultType": "swap", "defaultSubType": "linear"})

    async def test_spot_market_type_still_connects_to_spot(self):
        settings.market_type = "spot"
        engine = ExecutionEngine()
        engine.is_paper = False
        mock_exchange = AsyncMock()
        mock_exchange.enable_demo_trading = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(
            return_value={"free": {"USDT": 100.0}, "USDT": {"free": 100.0, "used": 0, "total": 100.0}}
        )
        with patch("src.execution.executor.ccxt.bybit", return_value=mock_exchange) as mock_class:
            await engine.initialize("bybit")

        called_config = mock_class.call_args_list[0][0][0]
        self.assertEqual(called_config["options"], {"defaultType": "spot"})


class TestExecuteRealOrderOpensLongAndShortOnFutures(unittest.IsolatedAsyncioTestCase):
    """
    ЭТАП 2 перехода на фьючерсы: _execute_real_order теперь реально
    открывает long И short на фьючерсах (market_type=="futures"), а не
    заглушка-отказ (этап 1). На споте поведение не меняется — short
    по-прежнему отклоняется.
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type
        self._saved_trading_mode = settings.trading_mode

    def tearDown(self):
        settings.market_type = self._saved_market_type
        settings.trading_mode = self._saved_trading_mode

    async def test_opens_long_on_futures_and_sets_leverage(self):
        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "fut-long-1", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="FUTLONG1/USDT", side="buy", amount=10.0, price=2.0, order_type="market",
        )

        self.assertIsNotNone(order)
        self.engine.exchange.set_leverage.assert_awaited_once_with(1, "FUTLONG1/USDT")
        self.assertIn("FUTLONG1/USDT", self.engine.real_positions)
        self.assertEqual(self.engine.real_positions["FUTLONG1/USDT"]["side"], "long")

    async def test_opens_short_on_futures(self):
        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "fut-short-1", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="FUTSHORT1/USDT", side="sell", amount=10.0, price=2.0, order_type="market",
        )

        self.assertIsNotNone(order)
        self.engine.exchange.create_market_buy_order.assert_not_called()
        self.assertIn("FUTSHORT1/USDT", self.engine.real_positions)
        self.assertEqual(self.engine.real_positions["FUTSHORT1/USDT"]["side"], "short")

    async def test_leverage_setting_error_does_not_block_order(self):
        """best-effort: биржа может бросить "leverage not modified" — не должно ронять ордер."""
        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.set_leverage = AsyncMock(side_effect=Exception("leverage not modified"))
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "fut-long-2", "filled": 5.0, "average": 3.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="FUTLEVERR1/USDT", side="buy", amount=5.0, price=3.0, order_type="market",
        )

        self.assertIsNotNone(order)

    async def test_spot_market_type_still_rejects_short(self):
        settings.market_type = "spot"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()

        order = await self.engine.create_order(
            symbol="FUTGUARD1/USDT", side="sell", amount=10.0, price=2.0, order_type="market",
        )

        self.assertIsNone(order)
        self.engine.exchange.create_market_sell_order.assert_not_called()

    async def test_spot_market_type_still_executes_normally(self):
        settings.market_type = "spot"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"FUTGUARD2": 0.0}, "FUTGUARD2": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "futguard-order-1", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="FUTGUARD2/USDT", side="buy", amount=10.0, price=2.0, order_type="market",
        )

        self.assertIsNotNone(order)
        self.engine.exchange.set_leverage.assert_not_called()


class TestExecuteRealOrderAppliesPerSignalLeverage(unittest.IsolatedAsyncioTestCase):
    """
    Некоторые Telegram-каналы указывают плечо прямо в тексте сигнала
    ("Кредитное плечо: х35" — см. extract_leverage в channel_monitor.py).
    create_order(leverage=...) должен применить ЕГО через set_leverage
    вместо глобальной settings.futures_leverage, и сохранить фактически
    применённое значение на real_positions[symbol]["leverage"] — иначе
    бейдж плеча в дашборде показывал бы либо ничего (до первой сверки),
    либо неверное глобальное значение.
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type
        self._saved_trading_mode = settings.trading_mode
        self._saved_futures_leverage = settings.futures_leverage

    def tearDown(self):
        settings.market_type = self._saved_market_type
        settings.trading_mode = self._saved_trading_mode
        settings.futures_leverage = self._saved_futures_leverage

    async def test_per_order_leverage_overrides_global_setting(self):
        settings.market_type = "futures"
        settings.trading_mode = "real"
        settings.futures_leverage = 1.0
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "fut-lev-1", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="FUTLEV1/USDT", side="buy", amount=10.0, price=2.0, order_type="market",
            leverage=35.0,
        )

        self.assertIsNotNone(order)
        self.engine.exchange.set_leverage.assert_awaited_once_with(35, "FUTLEV1/USDT")
        self.assertEqual(self.engine.real_positions["FUTLEV1/USDT"]["leverage"], 35.0)

    async def test_no_per_order_leverage_falls_back_to_global_setting(self):
        settings.market_type = "futures"
        settings.trading_mode = "real"
        settings.futures_leverage = 5.0
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "fut-lev-2", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="FUTLEV2/USDT", side="buy", amount=10.0, price=2.0, order_type="market",
        )

        self.assertIsNotNone(order)
        self.engine.exchange.set_leverage.assert_awaited_once_with(5, "FUTLEV2/USDT")
        self.assertEqual(self.engine.real_positions["FUTLEV2/USDT"]["leverage"], 5.0)

    async def test_leverage_ignored_on_spot(self):
        settings.market_type = "spot"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(
            return_value={"free": {"FUTLEV3": 0.0}, "FUTLEV3": {"free": 0.0, "used": 0, "total": 0.0}}
        )
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "fut-lev-3", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="FUTLEV3/USDT", side="buy", amount=10.0, price=2.0, order_type="market",
            leverage=35.0,
        )

        self.assertIsNotNone(order)
        self.engine.exchange.set_leverage.assert_not_called()
        self.assertIsNone(self.engine.real_positions["FUTLEV3/USDT"]["leverage"])


class TestGetReferencePrice(unittest.IsolatedAsyncioTestCase):
    """
    execution_engine.get_reference_price() — резолвит текущую рыночную
    цену для сигналов "по рынку" (см. is_market_entry в
    channel_monitor.py и _on_telegram_signal в main.py), которым нужно
    конкретное число ДО открытия ордера (расчёт объёма позиции), а не
    только в момент самого исполнения.
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type

    def tearDown(self):
        settings.market_type = self._saved_market_type

    async def test_real_mode_fetches_ticker_from_target_market(self):
        settings.market_type = "spot"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        futures_exchange = AsyncMock()
        futures_exchange.fetch_ticker = AsyncMock(return_value={"last": 0.2034, "bid": 0.2033, "ask": 0.2035})
        self.engine._exchanges["futures"] = futures_exchange

        price = await self.engine.get_reference_price("WIF/USDT", "futures")

        self.assertEqual(price, 0.2034)
        futures_exchange.fetch_ticker.assert_awaited_once_with("WIF/USDT")
        self.engine.exchange.fetch_ticker.assert_not_called()

    async def test_real_mode_falls_back_to_bid_ask_when_last_missing(self):
        settings.market_type = "spot"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_ticker = AsyncMock(return_value={"last": None, "bid": 1.23, "ask": 1.25})

        price = await self.engine.get_reference_price("BTC/USDT")

        self.assertEqual(price, 1.23)

    async def test_returns_none_when_no_exchange_connected(self):
        settings.market_type = "spot"
        self.engine.is_paper = False
        self.engine.exchange_id = None

        price = await self.engine.get_reference_price("BTC/USDT", "futures")

        self.assertIsNone(price)

    async def test_paper_mode_uses_current_exchange(self):
        self.engine.is_paper = True
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_ticker = AsyncMock(return_value={"last": 100.0})

        price = await self.engine.get_reference_price("BTC/USDT")

        self.assertEqual(price, 100.0)

    async def test_ticker_error_returns_none(self):
        settings.market_type = "spot"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_ticker = AsyncMock(side_effect=Exception("network error"))

        price = await self.engine.get_reference_price("BTC/USDT")

        self.assertIsNone(price)


class TestSweepBalancesToUsdt(unittest.IsolatedAsyncioTestCase):
    """
    ExecutionEngine.sweep_balances_to_usdt() — ручная конвертация ВСЕХ
    ненулевых остатков в USDT по кнопке дашборда. Реальный прод-инцидент:
    десятки валют скопились на аккаунте (демо-счёт Bybit) за время работы
    бота, часть балансов заблокирована в осиротевших ордерах ("used" в
    /balances) от давно закрытых позиций — см. фикс _record_external_close
    выше (теперь отменяет sl_order_id).
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type

    def tearDown(self):
        settings.market_type = self._saved_market_type

    async def test_sells_free_balances_skips_dust_and_unmarketed_currencies(self):
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.markets = {
            "ARB/USDT": {"limits": {"amount": {"min": 1.0}, "cost": {"min": 5.0}}},
            "MON/USDT": {"limits": {"amount": {"min": 1.0}}},
        }
        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"ARB": 100.0, "MON": 0.0001, "GHOST": 5.0, "USDT": 500.0},
            "used": {"ARB": 0.0, "MON": 0.0, "GHOST": 0.0, "USDT": 0.0},
            "total": {"ARB": 100.0, "MON": 0.0001, "GHOST": 5.0, "USDT": 500.0},
        })
        self.engine.exchange.fetch_ticker = AsyncMock(side_effect=lambda symbol: {"last": 10.0})
        self.engine.exchange.create_market_sell_order = AsyncMock(return_value={"id": "sell-arb-1"})

        result = await self.engine.sweep_balances_to_usdt()

        self.assertEqual(len(result["sold"]), 1)
        self.assertEqual(result["sold"][0]["currency"], "ARB")
        self.assertEqual(result["sold"][0]["amount"], 100.0)
        skipped_currencies = {s["currency"] for s in result["skipped"]}
        self.assertEqual(skipped_currencies, {"MON", "GHOST"})
        self.assertEqual(result["errors"], [])
        self.engine.exchange.create_market_sell_order.assert_called_once_with("ARB/USDT", 100.0)

    async def test_cancels_open_orders_before_selling_locked_balance(self):
        """Тот самый прод-сценарий ASTER/QTUM/TIA: часть баланса
        заблокирована в открытых ордерах — сначала отменяем их, потом
        продаём освободившийся остаток."""
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.markets = {"LOCKED/USDT": {"limits": {}}}
        self.engine.exchange.fetch_balance = AsyncMock(side_effect=[
            {
                "free": {"LOCKED": 0.0, "USDT": 500.0},
                "used": {"LOCKED": 50.0, "USDT": 0.0},
                "total": {"LOCKED": 50.0, "USDT": 500.0},
            },
            {
                "free": {"LOCKED": 50.0, "USDT": 500.0},
                "used": {"LOCKED": 0.0, "USDT": 0.0},
                "total": {"LOCKED": 50.0, "USDT": 500.0},
            },
        ])
        self.engine.exchange.fetch_open_orders = AsyncMock(return_value=[{"id": "orphaned-order-1"}])
        self.engine.exchange.fetch_ticker = AsyncMock(return_value={"last": 2.0})
        self.engine.exchange.create_market_sell_order = AsyncMock(return_value={"id": "sell-locked-1"})

        result = await self.engine.sweep_balances_to_usdt()

        self.engine.exchange.cancel_order.assert_any_call("orphaned-order-1", "LOCKED/USDT")
        self.assertEqual(len(result["sold"]), 1)
        self.assertEqual(result["sold"][0], {"currency": "LOCKED", "amount": 50.0, "order_id": "sell-locked-1"})

    async def test_sell_order_error_recorded_without_blocking_others(self):
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.markets = {
            "FAIL/USDT": {"limits": {}}, "OK/USDT": {"limits": {}},
        }
        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"FAIL": 1.0, "OK": 1.0, "USDT": 100.0},
            "used": {"FAIL": 0.0, "OK": 0.0, "USDT": 0.0},
            "total": {"FAIL": 1.0, "OK": 1.0, "USDT": 100.0},
        })
        self.engine.exchange.fetch_ticker = AsyncMock(return_value={"last": 1.0})

        async def _sell(symbol, amount):
            if symbol == "FAIL/USDT":
                raise Exception("exchange rejected order")
            return {"id": "sell-ok-1"}

        self.engine.exchange.create_market_sell_order = AsyncMock(side_effect=_sell)

        result = await self.engine.sweep_balances_to_usdt()

        self.assertEqual(len(result["sold"]), 1)
        self.assertEqual(result["sold"][0]["currency"], "OK")
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(result["errors"][0]["currency"], "FAIL")

    async def test_no_exchange_connected_returns_error(self):
        self.engine.is_paper = False
        self.engine.exchange_id = None

        result = await self.engine.sweep_balances_to_usdt()

        self.assertEqual(result["sold"], [])
        self.assertEqual(len(result["errors"]), 1)

    async def test_rebases_risk_baseline_after_selling(self):
        """
        Регресс на прод-инцидент: после продажи ~20 монет реальный баланс
        USDT резко вырос (одноразовая консолидация ранее НЕотслеживаемых
        остатков, а не торговая прибыль) — total_drawdown_pct показал
        -1943%, потому что risk_manager.state.start_balance остался
        устаревшим (маленьким) числом с последнего рестарта. sweep должен
        пересчитать базу так же, как reset_for_real_account() при
        подключении к реальному аккаунту.
        """
        from src.risk.risk_manager import risk_manager as global_risk_manager

        saved_start = global_risk_manager.state.start_balance
        saved_current = global_risk_manager.state.current_balance
        try:
            global_risk_manager.state.start_balance = 9243.73
            global_risk_manager.state.current_balance = 9243.73

            self.engine.is_paper = False
            self.engine.exchange_id = "bybit"
            self.engine.exchange = AsyncMock()
            self.engine.exchange.markets = {"ARB/USDT": {"limits": {}}}
            self.engine.exchange.fetch_balance = AsyncMock(return_value={
                "free": {"ARB": 100.0, "USDT": 500.0},
                "used": {"ARB": 0.0, "USDT": 0.0},
                "total": {"ARB": 100.0, "USDT": 188744.43},
            })
            self.engine.exchange.fetch_ticker = AsyncMock(return_value={"last": 10.0})
            self.engine.exchange.create_market_sell_order = AsyncMock(return_value={"id": "sell-arb-1"})

            await self.engine.sweep_balances_to_usdt()

            self.assertAlmostEqual(global_risk_manager.state.start_balance, 188744.43, places=2)
        finally:
            global_risk_manager.state.start_balance = saved_start
            global_risk_manager.state.current_balance = saved_current

    async def test_does_not_rebase_when_nothing_sold(self):
        from src.risk.risk_manager import risk_manager as global_risk_manager

        saved_start = global_risk_manager.state.start_balance
        try:
            global_risk_manager.state.start_balance = 9243.73

            self.engine.is_paper = False
            self.engine.exchange_id = "bybit"
            self.engine.exchange = AsyncMock()
            self.engine.exchange.markets = {}
            self.engine.exchange.fetch_balance = AsyncMock(return_value={
                "free": {"GHOST": 5.0, "USDT": 500.0},
                "used": {"GHOST": 0.0, "USDT": 0.0},
                "total": {"GHOST": 5.0, "USDT": 500.0},
            })

            result = await self.engine.sweep_balances_to_usdt()

            self.assertEqual(result["sold"], [])
            self.assertEqual(global_risk_manager.state.start_balance, 9243.73)
        finally:
            global_risk_manager.state.start_balance = saved_start


class TestCloseRealPositionOnFutures(unittest.IsolatedAsyncioTestCase):
    """
    ЭТАП 2 перехода на фьючерсы: close_real_position закрывает и long
    (продажей), и short (покупкой) на фьючерсах, с reduceOnly — раньше
    метод работал только для long и всегда безусловно продавал.
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type
        self._saved_trading_mode = settings.trading_mode

    def tearDown(self):
        settings.market_type = self._saved_market_type
        settings.trading_mode = self._saved_trading_mode

    async def test_closes_long_via_sell_with_reduce_only(self):
        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_sell_order.return_value = {
            "id": "fut-close-long-1", "filled": 10.0, "price": None, "average": 3.0,
            "fee": {"cost": 0.1, "currency": "USDT"},
        }

        result = await self.engine.close_real_position(
            symbol="FUTCLOSE1/USDT", side="long", entry_price=2.0, amount=10.0,
            reason="take_profit", entry_fee=0.05, holding_seconds=60,
        )

        self.assertIsNotNone(result)
        self.engine.exchange.create_market_sell_order.assert_awaited_once_with(
            "FUTCLOSE1/USDT", 10.0, params={"reduceOnly": True},
        )
        expected_pnl = (3.0 - 2.0) * 10.0 - 0.05 - 0.1
        self.assertAlmostEqual(result["pnl"], expected_pnl, places=6)

    async def test_closes_short_via_buy_with_reduce_only_and_correct_pnl(self):
        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "fut-close-short-1", "filled": 10.0, "price": None, "average": 1.5,
            "fee": {"cost": 0.05, "currency": "USDT"},
        }

        result = await self.engine.close_real_position(
            symbol="FUTCLOSE2/USDT", side="short", entry_price=2.0, amount=10.0,
            reason="take_profit", entry_fee=0.05, holding_seconds=60,
        )

        self.assertIsNotNone(result)
        self.engine.exchange.create_market_buy_order.assert_awaited_once_with(
            "FUTCLOSE2/USDT", 10.0, params={"reduceOnly": True},
        )
        self.engine.exchange.create_market_sell_order.assert_not_called()
        # short: цена упала 2.0 -> 1.5, значит прибыль.
        expected_pnl = (2.0 - 1.5) * 10.0 - 0.05 - 0.05
        self.assertAlmostEqual(result["pnl"], expected_pnl, places=6)

    async def test_spot_still_rejects_short_close(self):
        settings.market_type = "spot"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()

        result = await self.engine.close_real_position(
            symbol="FUTCLOSE3/USDT", side="short", entry_price=2.0, amount=10.0,
            reason="stop_loss", entry_fee=0.0, holding_seconds=10,
        )

        self.assertIsNone(result)
        self.engine.exchange.create_market_buy_order.assert_not_called()
        self.engine.exchange.create_market_sell_order.assert_not_called()

    async def test_futures_close_failure_does_not_reconcile_phantom(self):
        """
        Неудачное закрытие на фьючерсах НЕ должно списывать позицию как
        "дуст" (это спот-концепция, читающая кошелёк) — просто ошибка,
        позиция остаётся отслеживаемой для следующей попытки закрытия.
        """
        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_sell_order = AsyncMock(
            side_effect=Exception('bybit {"retCode":170140,"retMsg":"Order value exceeded lower limit."}')
        )
        self.engine.real_positions["FUTCLOSE4/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long", "sl_order_id": None,
        }

        result = await self.engine.close_real_position(
            symbol="FUTCLOSE4/USDT", side="long", entry_price=2.0, amount=10.0,
            reason="stop_loss", entry_fee=0.0, holding_seconds=10,
        )

        self.assertIsNone(result)
        self.assertIn("FUTCLOSE4/USDT", self.engine.real_positions)


class TestReconcileFuturesPositions(unittest.IsolatedAsyncioTestCase):
    """
    ЭТАП 5 перехода на фьючерсы: reconcile_real_positions() теперь сверяет
    и фьючерсные позиции (раньше при market_type=="futures" сверка
    ВООБЩЕ не выполнялась — это уже устаревшая premise, см. историю).
    Фьючерсная сверка сравнивает отслеживаемый объём с фактическим
    размером ПОЗИЦИИ на бирже (fetch_position — контракт, а не остаток
    монеты на кошельке, как на споте).
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type
        self._saved_trading_mode = settings.trading_mode

    def tearDown(self):
        settings.market_type = self._saved_market_type
        settings.trading_mode = self._saved_trading_mode

    async def test_removes_phantom_futures_position_when_contracts_zero(self):
        from src.utils.timeutils import utcnow

        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 500.0}, "USDT": {"free": 500.0, "used": 0, "total": 500.0},
        })
        self.engine.exchange.fetch_position = AsyncMock(return_value={"contracts": 0.0})
        self.engine.real_positions["FUTRECON1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long",
            "opened_at": utcnow() - timedelta(hours=1), "sl_order_id": None, "order_id": None,
            "market_type": "futures",
        }

        balance = await self.engine.reconcile_real_positions()

        self.assertAlmostEqual(balance, 500.0)
        self.assertNotIn("FUTRECON1/USDT", self.engine.real_positions)

    async def test_keeps_futures_position_when_contracts_match(self):
        from src.utils.timeutils import utcnow

        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 500.0}, "USDT": {"free": 500.0, "used": 0, "total": 500.0},
        })
        self.engine.exchange.fetch_position = AsyncMock(return_value={"contracts": 10.0})
        self.engine.real_positions["FUTRECON2/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long",
            "opened_at": utcnow() - timedelta(hours=1), "sl_order_id": None, "order_id": None,
            "market_type": "futures",
        }

        balance = await self.engine.reconcile_real_positions()

        self.assertAlmostEqual(balance, 500.0)
        self.assertIn("FUTRECON2/USDT", self.engine.real_positions)

    async def test_caches_leverage_and_margin_from_fetch_position(self):
        """
        ЭТАП 6: leverage/margin_usdt кэшируются на pos из ТОГО ЖЕ ответа
        fetch_position (без доп. запроса к бирже) — для отображения на
        дашборде. settings.futures_leverage — глобальная настройка, могла
        измениться ПОСЛЕ открытия позиции, поэтому источник истины — сама
        биржа, не текущее значение настройки.
        """
        from src.utils.timeutils import utcnow

        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 500.0}, "USDT": {"free": 500.0, "used": 0, "total": 500.0},
        })
        self.engine.exchange.fetch_position = AsyncMock(
            return_value={"contracts": 10.0, "leverage": 5.0, "initialMargin": 40.0}
        )
        self.engine.real_positions["FUTLEV1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long",
            "opened_at": utcnow() - timedelta(hours=1), "sl_order_id": None, "order_id": None,
            "market_type": "futures",
        }

        await self.engine.reconcile_real_positions()

        self.assertEqual(self.engine.real_positions["FUTLEV1/USDT"]["leverage"], 5.0)
        self.assertEqual(self.engine.real_positions["FUTLEV1/USDT"]["margin_usdt"], 40.0)

    async def test_futures_position_within_grace_period_not_removed(self):
        from src.utils.timeutils import utcnow

        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 500.0}, "USDT": {"free": 500.0, "used": 0, "total": 500.0},
        })
        self.engine.exchange.fetch_position = AsyncMock(return_value={"contracts": 0.0})
        self.engine.real_positions["FUTRECON3/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long",
            "opened_at": utcnow(), "sl_order_id": None, "order_id": None,
            "market_type": "futures",
        }

        await self.engine.reconcile_real_positions()

        self.assertIn("FUTRECON3/USDT", self.engine.real_positions)

    async def test_mixed_markets_fetches_spot_balance_from_spot_client_not_current_toggle(self):
        """
        Реальный сценарий с прода: тумблер стоит на futures, но
        отслеживается и спотовая позиция (как MON/USDT) — сверка спота
        должна брать баланс со SPOT-клиента, а не с клиента текущего
        тумблера (futures), иначе спотовая позиция ошибочно казалась бы
        "без остатка на кошельке" и списывалась бы как фантомная.
        """
        from src.utils.timeutils import utcnow

        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        futures_mock = AsyncMock()
        futures_mock.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 500.0}, "USDT": {"free": 500.0, "used": 0, "total": 500.0},
        })
        spot_mock = AsyncMock()
        spot_mock.fetch_balance = AsyncMock(return_value={
            "free": {"MON": 5238.2565, "USDT": 10.0},
            "MON": {"free": 5238.2565, "used": 0, "total": 5238.2565},
        })
        self.engine._exchanges = {"futures": futures_mock, "spot": spot_mock}
        self.engine.real_positions["MON/USDT"] = {
            "amount": 5238.2565, "entry_price": 0.0257, "side": "long",
            "opened_at": utcnow() - timedelta(days=5), "sl_order_id": None, "order_id": None,
            "market_type": "spot",
        }

        balance = await self.engine.reconcile_real_positions()

        self.assertAlmostEqual(balance, 500.0)
        spot_mock.fetch_balance.assert_awaited_once()
        self.assertIn("MON/USDT", self.engine.real_positions)


class TestCcxtSymbolTranslationForFutures(unittest.IsolatedAsyncioTestCase):
    """
    Реальный инцидент (прод, месяцами): у ccxt спотовый и linear-swap
    (наш "futures"/USDT-perpetual) рынки одной и той же пары — это ДВА
    РАЗНЫХ unified-символа: "BCH/USDT" (спот) и "BCH/USDT:USDT" (swap,
    суффикс — расчётная валюта через двоеточие; см. parse_market в
    ccxt/bybit.py). exchange.market(symbol) резолвит ПО БУКВАЛЬНОМУ
    СОВПАДЕНИЮ СТРОКИ в self.markets и не учитывает options.defaultType —
    раз "BCH/USDT" уже есть как ключ (спотовый рынок), он и возвращается
    ВСЕГДА, что бы ни было в defaultType. Весь код в этом файле годами
    обращался к "futures"-клиенту голым "BASE/QUOTE" без суффикса — а
    значит КАЖДЫЙ вызов (create_order, fetch_position, cancel_order,
    set_leverage, fetch_open_orders...) на деле резолвился в СПОТОВЫЙ
    рынок. Отсюда и retCode 181001 "category only support linear or
    option" на fetch_position() (у спота нет позиций), и стабильный
    170131 "Insufficient balance" на закрытии/SL (не reduceOnly-закрытие
    фьючерсного контракта, а спотовая/маржинальная продажа с другой
    семантикой баланса).

    _ccxt_symbol(exchange, symbol) — единая точка перевода: добавляет
    суффикс ":QUOTE" только когда exchange.options["defaultType"] == "swap"
    (так его выставляет _connect_exchange для futures-клиента), и НЕ
    трогает наш собственный канонический "BASE/QUOTE", которым everywhere
    else (БД, real_positions, main.py, дашборд) обозначается символ —
    перевод происходит непосредственно перед вызовом ccxt.
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type
        self._saved_trading_mode = settings.trading_mode

    def tearDown(self):
        settings.market_type = self._saved_market_type
        settings.trading_mode = self._saved_trading_mode

    def _futures_exchange(self) -> AsyncMock:
        ex = AsyncMock()
        ex.options = {"defaultType": "swap", "defaultSubType": "linear"}
        return ex

    def _spot_exchange(self) -> AsyncMock:
        ex = AsyncMock()
        ex.options = {"defaultType": "spot"}
        return ex

    def test_ccxt_symbol_suffixes_quote_for_futures_exchange(self):
        futures_exchange = self._futures_exchange()
        self.assertEqual(
            self.engine._ccxt_symbol(futures_exchange, "BCH/USDT"), "BCH/USDT:USDT",
        )

    def test_ccxt_symbol_unchanged_for_spot_exchange(self):
        spot_exchange = self._spot_exchange()
        self.assertEqual(
            self.engine._ccxt_symbol(spot_exchange, "BCH/USDT"), "BCH/USDT",
        )

    def test_ccxt_symbol_idempotent_if_already_suffixed(self):
        futures_exchange = self._futures_exchange()
        self.assertEqual(
            self.engine._ccxt_symbol(futures_exchange, "BCH/USDT:USDT"), "BCH/USDT:USDT",
        )

    def test_ccxt_symbol_safe_against_mock_without_real_options_dict(self):
        """
        AsyncMock() без явного .options — сам .options становится AsyncMock,
        а не dict (см. unittest.mock: атрибуты AsyncMock рекурсивно тоже
        AsyncMock) — .get(...) на нём вернул бы корутину, а не значение.
        _ccxt_symbol должен безопасно вернуть символ как есть, а не упасть
        и не оставить незаявленную корутину.
        """
        bare_mock = AsyncMock()
        self.assertEqual(self.engine._ccxt_symbol(bare_mock, "BCH/USDT"), "BCH/USDT")

    def test_ccxt_symbol_safe_with_none_exchange(self):
        self.assertEqual(self.engine._ccxt_symbol(None, "BCH/USDT"), "BCH/USDT")

    async def test_execute_real_order_uses_suffixed_symbol_on_futures(self):
        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        futures_exchange = self._futures_exchange()
        futures_exchange.create_market_buy_order.return_value = {
            "id": "ccxtfix-open-1", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }
        self.engine.exchange = futures_exchange

        order = await self.engine.create_order(
            symbol="CCXTFIX1/USDT", side="buy", amount=10.0, price=2.0, order_type="market",
        )

        self.assertIsNotNone(order)
        futures_exchange.create_market_buy_order.assert_awaited_once_with("CCXTFIX1/USDT:USDT", 10.0)
        futures_exchange.set_leverage.assert_awaited_once()
        self.assertEqual(futures_exchange.set_leverage.await_args.args[1], "CCXTFIX1/USDT:USDT")

    async def test_execute_real_order_symbol_unchanged_on_spot(self):
        """Регресс: то же самое на споте символ НЕ должен получать суффикс."""
        settings.market_type = "spot"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        spot_exchange = self._spot_exchange()
        spot_exchange.fetch_balance = AsyncMock(return_value={
            "free": {"CCXTFIX2": 0.0}, "CCXTFIX2": {"free": 0.0, "used": 0, "total": 0.0},
        })
        spot_exchange.create_market_buy_order.return_value = {
            "id": "ccxtfix-open-2", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }
        self.engine.exchange = spot_exchange

        order = await self.engine.create_order(
            symbol="CCXTFIX2/USDT", side="buy", amount=10.0, price=2.0, order_type="market",
        )

        self.assertIsNotNone(order)
        spot_exchange.create_market_buy_order.assert_awaited_once_with("CCXTFIX2/USDT", 10.0)

    async def test_close_real_position_uses_suffixed_symbol_on_futures(self):
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        futures_exchange = self._futures_exchange()
        futures_exchange.create_market_sell_order.return_value = {
            "id": "ccxtfix-close-1", "filled": 10.0, "average": 2.1, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }
        self.engine._exchanges = {"futures": futures_exchange}
        self.engine.real_positions["CCXTFIX3/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long",
            "sl_order_id": None, "market_type": "futures",
        }

        result = await self.engine.close_real_position(
            symbol="CCXTFIX3/USDT", side="long", entry_price=2.0, amount=10.0, reason="test",
        )

        self.assertIsNotNone(result)
        futures_exchange.create_market_sell_order.assert_awaited_once_with(
            "CCXTFIX3/USDT:USDT", 10.0, params={"reduceOnly": True},
        )

    async def test_place_stop_loss_order_uses_suffixed_symbol_on_futures(self):
        futures_exchange = self._futures_exchange()
        futures_exchange.create_market_sell_order.return_value = {"id": "ccxtfix-sl-1"}

        order_id = await self.engine._place_stop_loss_order(
            "CCXTFIX4/USDT", 10.0, 1.8, futures_exchange, side="long", is_futures=True,
        )

        self.assertEqual(order_id, "ccxtfix-sl-1")
        futures_exchange.create_market_sell_order.assert_awaited_once_with(
            "CCXTFIX4/USDT:USDT", 10.0, params={"stopLossPrice": 1.8, "reduceOnly": True},
        )

    async def test_reconcile_futures_position_fetches_suffixed_symbol(self):
        from src.utils.timeutils import utcnow

        settings.market_type = "futures"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        futures_exchange = self._futures_exchange()
        futures_exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 500.0}, "USDT": {"free": 500.0, "used": 0, "total": 500.0},
        })
        futures_exchange.fetch_position.return_value = {"contracts": 10.0, "leverage": 5.0}
        self.engine.exchange = futures_exchange
        self.engine.real_positions["CCXTFIX5/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long",
            "opened_at": utcnow() - timedelta(hours=1), "sl_order_id": None, "order_id": None,
            "market_type": "futures",
        }

        await self.engine.reconcile_real_positions()

        futures_exchange.fetch_position.assert_awaited_once_with("CCXTFIX5/USDT:USDT")
        self.assertEqual(self.engine.real_positions["CCXTFIX5/USDT"]["leverage"], 5.0)


class TestSyncStopLossOrderOnFutures(unittest.IsolatedAsyncioTestCase):
    """
    ЭТАП 4 перехода на фьючерсы: биржевой SL теперь ставится и на
    фьючерсах, симметрично споту — _place_stop_loss_order сам выбирает
    направление ордера по стороне позиции (sell для long, buy для short),
    вместо прежнего безусловного пропуска. Это заодно устраняет причину
    более раннего реального инцидента (прод, demo-фьючерсы, ENA/USDT):
    восстановление short-позиции после рестарта раньше могло получить
    СПОТОВЫЙ "sell"-стоп на короткую позицию (неверно — SL шорта должен
    быть buy выше входа) — теперь направление определяется явно по side,
    а не всегда sell.
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type

    def tearDown(self):
        settings.market_type = self._saved_market_type

    async def test_places_sl_sell_order_for_long_position_on_futures(self):
        settings.market_type = "futures"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_sell_order.return_value = {"id": "fut-sl-long-1"}
        self.engine.real_positions["FUTSLLONG1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long", "sl_order_id": None,
            "market_type": "futures",
        }

        await self.engine.sync_stop_loss_order("FUTSLLONG1/USDT", 10.0, 1.8)

        self.engine.exchange.create_market_sell_order.assert_awaited_once_with(
            "FUTSLLONG1/USDT", 10.0, params={"stopLossPrice": 1.8, "reduceOnly": True},
        )
        self.engine.exchange.create_market_buy_order.assert_not_called()
        self.assertEqual(self.engine.real_positions["FUTSLLONG1/USDT"]["sl_order_id"], "fut-sl-long-1")

    async def test_places_sl_buy_order_for_short_position_on_futures(self):
        settings.market_type = "futures"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {"id": "fut-sl-short-1"}
        self.engine.real_positions["FUTSLSHORT1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "short", "sl_order_id": None,
            "market_type": "futures",
        }

        await self.engine.sync_stop_loss_order("FUTSLSHORT1/USDT", 10.0, 2.2)

        self.engine.exchange.create_market_buy_order.assert_awaited_once_with(
            "FUTSLSHORT1/USDT", 10.0, params={"stopLossPrice": 2.2, "reduceOnly": True},
        )
        self.engine.exchange.create_market_sell_order.assert_not_called()
        self.assertEqual(self.engine.real_positions["FUTSLSHORT1/USDT"]["sl_order_id"], "fut-sl-short-1")

    async def test_futures_sl_does_not_check_wallet_balance(self):
        """Спотовая сверка доступного остатка на кошельке бессмысленна для фьючерсов — не должна вызываться."""
        settings.market_type = "futures"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_sell_order.return_value = {"id": "fut-sl-nobalance-1"}
        self.engine.real_positions["FUTSLNOBAL1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long", "sl_order_id": None,
            "market_type": "futures",
        }

        await self.engine.sync_stop_loss_order("FUTSLNOBAL1/USDT", 10.0, 1.8)

        self.engine.exchange.fetch_balance.assert_not_called()

    async def test_still_places_sl_order_on_spot(self):
        settings.market_type = "spot"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_sell_order.return_value = {"id": "spot-sl-1"}
        self.engine.real_positions["SPOTSL1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long", "sl_order_id": None,
        }

        await self.engine.sync_stop_loss_order("SPOTSL1/USDT", 10.0, 1.8)

        self.engine.exchange.create_market_sell_order.assert_awaited_once_with(
            "SPOTSL1/USDT", 10.0, params={"stopLossPrice": 1.8},
        )
        self.assertEqual(self.engine.real_positions["SPOTSL1/USDT"]["sl_order_id"], "spot-sl-1")


class TestSyncStopLossOrderDoesNotOrphanUnconfirmedCancel(unittest.IsolatedAsyncioTestCase):
    """
    Реальный инцидент (прод, BCH/USDT и HYPE/USDT): sync_stop_loss_order
    раньше безусловно сбрасывал pos["sl_order_id"] в None ДО того, как
    убеждался, что отмена старого условного SL-ордера реально прошла на
    бирже (_cancel_order_safe глотает любую ошибку молча). Если cancel_order
    падал с неоднозначной ошибкой (не "ордера уже нет", а что-то другое —
    в проде это был тот же 170131 Insufficient balance), старый ордер
    оставался живым на бирже, но бот забывал его order_id — ордер
    осиротевал навсегда: не отменяем его повторно, не видим его
    исполнение через _finalize_externally_closed_position, а каждая
    следующая попытка переставить SL или закрыть остаток позиции стабильно
    падает "insufficient balance", потому что биржа резервирует объём под
    этот забытый ордер.
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type
        settings.market_type = "futures"

    def tearDown(self):
        settings.market_type = self._saved_market_type

    async def test_keeps_old_sl_order_id_when_cancel_fails_ambiguously(self):
        self.engine.exchange = AsyncMock()
        self.engine.exchange.cancel_order.side_effect = Exception("bybit 170131 Insufficient balance")
        self.engine.real_positions["ORPHANSL1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long",
            "sl_order_id": "old-sl-order-1", "market_type": "futures",
        }

        await self.engine.sync_stop_loss_order("ORPHANSL1/USDT", 5.0, 1.9)

        self.engine.exchange.cancel_order.assert_awaited_once_with("old-sl-order-1", "ORPHANSL1/USDT")
        self.engine.exchange.create_market_sell_order.assert_not_called()
        self.assertEqual(self.engine.real_positions["ORPHANSL1/USDT"]["sl_order_id"], "old-sl-order-1")

    async def test_replaces_sl_order_id_when_cancel_succeeds(self):
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_sell_order.return_value = {"id": "new-sl-order-1"}
        self.engine.real_positions["ORPHANSL2/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long",
            "sl_order_id": "old-sl-order-2", "market_type": "futures",
        }

        await self.engine.sync_stop_loss_order("ORPHANSL2/USDT", 5.0, 1.9)

        self.engine.exchange.cancel_order.assert_awaited_once_with("old-sl-order-2", "ORPHANSL2/USDT")
        self.engine.exchange.create_market_sell_order.assert_awaited_once()
        self.assertEqual(self.engine.real_positions["ORPHANSL2/USDT"]["sl_order_id"], "new-sl-order-1")

    async def test_replaces_sl_order_id_when_old_order_already_gone(self):
        """OrderNotFound (ордер уже отменён/исполнен биржей) — безопасно считать слот свободным."""
        self.engine.exchange = AsyncMock()
        self.engine.exchange.cancel_order.side_effect = ccxt.OrderNotFound("no such order")
        self.engine.exchange.create_market_sell_order.return_value = {"id": "new-sl-order-3"}
        self.engine.real_positions["ORPHANSL3/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long",
            "sl_order_id": "old-sl-order-3", "market_type": "futures",
        }

        await self.engine.sync_stop_loss_order("ORPHANSL3/USDT", 5.0, 1.9)

        self.engine.exchange.create_market_sell_order.assert_awaited_once()
        self.assertEqual(self.engine.real_positions["ORPHANSL3/USDT"]["sl_order_id"], "new-sl-order-3")


class TestRearmStopLossOrdersFilterByMarketType(unittest.IsolatedAsyncioTestCase):
    """
    _rearm_stop_loss_orders_after_restart чистит старые условные ордера
    перед переустановкой SL — orderFilter для этого запроса зависит от
    рынка позиции: 'tpslOrder' (спот) документирован в ccxt/bybit.py как
    "Valid for spot only", для фьючерсов (linear) нужен 'StopOrder'.
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type

    def tearDown(self):
        settings.market_type = self._saved_market_type

    async def test_uses_stop_order_filter_for_futures_position(self):
        settings.market_type = "futures"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_open_orders.return_value = []
        self.engine.exchange.create_market_sell_order.return_value = {"id": "rearm-fut-sl-1"}
        self.engine.real_positions["REARMFUT1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long", "sl_order_id": None,
            "stop_loss": 1.8, "market_type": "futures",
        }

        await self.engine._rearm_stop_loss_orders_after_restart()

        self.engine.exchange.fetch_open_orders.assert_awaited_once_with(
            "REARMFUT1/USDT", params={"orderFilter": "StopOrder"},
        )

    async def test_uses_tpsl_order_filter_for_spot_position(self):
        settings.market_type = "spot"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.fetch_open_orders.return_value = []
        self.engine.exchange.create_market_sell_order.return_value = {"id": "rearm-spot-sl-1"}
        self.engine.real_positions["REARMSPOT1/USDT"] = {
            "amount": 10.0, "entry_price": 2.0, "side": "long", "sl_order_id": None,
            "stop_loss": 1.8, "market_type": "spot",
        }

        await self.engine._rearm_stop_loss_orders_after_restart()

        self.engine.exchange.fetch_open_orders.assert_awaited_once_with(
            "REARMSPOT1/USDT", params={"orderFilter": "tpslOrder"},
        )


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

    async def test_create_manual_order_allows_short_on_futures_real_mode(self):
        """ЭТАП 2: ручной short на фьючерсах в real-режиме больше не блокируется — только на споте."""
        engine, bot = await self._install_engine_and_bot(exchange_id="bybit", is_paper=False)
        saved_market_type = settings.market_type
        settings.market_type = "futures"
        engine.exchange = AsyncMock()
        engine.exchange.create_market_sell_order.return_value = {
            "id": "manual-fut-short-1", "filled": 1.0, "average": 100.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }
        try:
            result = await self.api_module.create_manual_order(self.api_module.ManualOrderCreate(
                symbol="MANUALFUTSHORT1/USDT", side="sell", amount=1.0, price=100.0,
            ))
        finally:
            settings.market_type = saved_market_type

        self.assertTrue(result["success"])
        self.assertIn("MANUALFUTSHORT1/USDT", bot.open_positions)
        self.assertEqual(bot.open_positions["MANUALFUTSHORT1/USDT"]["side"], "short")

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


class TestSellBalancesToUsdtEndpoint(unittest.IsolatedAsyncioTestCase):
    """POST /balances/sell-to-usdt — кнопка дашборда "Продать все остатки
    в USDT" (см. ExecutionEngine.sweep_balances_to_usdt)."""

    def setUp(self):
        import src.web.api as api_module
        from fastapi import HTTPException

        self.api_module = api_module
        self.HTTPException = HTTPException
        self._saved_trading_mode = settings.trading_mode
        self._saved_engine = api_module.execution_engine

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode
        self.api_module.execution_engine = self._saved_engine

    async def test_rejected_in_paper_mode(self):
        settings.trading_mode = "paper"

        with self.assertRaises(self.HTTPException) as ctx:
            await self.api_module.sell_balances_to_usdt()
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_real_mode_delegates_to_execution_engine(self):
        settings.trading_mode = "real"
        mock_engine = MagicMock()
        mock_engine.sweep_balances_to_usdt = AsyncMock(return_value={
            "sold": [{"currency": "ARB", "amount": 100.0, "order_id": "1"}],
            "skipped": [], "errors": [],
        })
        self.api_module.execution_engine = mock_engine

        result = await self.api_module.sell_balances_to_usdt()

        self.assertTrue(result["success"])
        self.assertEqual(result["sold"][0]["currency"], "ARB")
        mock_engine.sweep_balances_to_usdt.assert_awaited_once()


class TestSystemRedeploy(unittest.IsolatedAsyncioTestCase):
    """
    POST /system/redeploy и GET /system/redeploy/status: бот сам НЕ трогает
    docker/git — только проксирует HTTP-запрос отдельному деплой-агенту вне
    своего контейнера (см. scripts/deploy_agent.py, docker-compose.yml) с
    общим секретом. Здесь тестируется только эта проксирующая логика;
    сам деплой-агент (чистый stdlib HTTP-сервис, никаких сторонних
    зависимостей) не покрыт этим тест-сьютом.
    """

    def setUp(self):
        from fastapi import HTTPException
        self.HTTPException = HTTPException
        self._saved = {
            "deploy_agent_url": settings.deploy_agent_url,
            "deploy_agent_token": settings.deploy_agent_token,
        }

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(settings, key, value)

    @staticmethod
    def _mock_response(status_code, json_data, raise_error=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.text = str(json_data)
        resp.raise_for_status.side_effect = raise_error
        resp.raise_for_status.return_value = None
        return resp

    @staticmethod
    def _mock_client_cm(response=None, post_side_effect=None, get_side_effect=None):
        client = MagicMock()
        client.post = AsyncMock(return_value=response, side_effect=post_side_effect)
        client.get = AsyncMock(return_value=response, side_effect=get_side_effect)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm, client

    async def test_redeploy_returns_503_when_agent_not_configured(self):
        from src.web.api import redeploy_bot
        settings.deploy_agent_url = None

        with self.assertRaises(self.HTTPException) as ctx:
            await redeploy_bot()
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_redeploy_calls_agent_with_token_and_returns_result(self):
        from src.web.api import redeploy_bot
        settings.deploy_agent_url = "http://deploy-agent:8091"
        settings.deploy_agent_token = "secret-token"

        response = self._mock_response(202, {"success": True, "message": "Деплой запущен"})
        cm, client = self._mock_client_cm(response=response)

        with patch("src.web.api.httpx.AsyncClient", return_value=cm):
            result = await redeploy_bot()

        self.assertEqual(result, {"success": True, "message": "Деплой запущен"})
        _, kwargs = client.post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-token")
        self.assertIn("http://deploy-agent:8091/deploy", client.post.call_args.args)

    async def test_redeploy_reports_already_running_without_raising(self):
        from src.web.api import redeploy_bot
        settings.deploy_agent_url = "http://deploy-agent:8091"
        settings.deploy_agent_token = "secret-token"

        response = self._mock_response(409, {"error": "деплой уже выполняется"})
        cm, _ = self._mock_client_cm(response=response)

        with patch("src.web.api.httpx.AsyncClient", return_value=cm):
            result = await redeploy_bot()

        self.assertFalse(result["success"])

    async def test_redeploy_raises_502_when_agent_unreachable(self):
        from src.web.api import redeploy_bot
        settings.deploy_agent_url = "http://deploy-agent:8091"
        settings.deploy_agent_token = "secret-token"

        cm, _ = self._mock_client_cm(post_side_effect=ConnectionError("connection refused"))

        with patch("src.web.api.httpx.AsyncClient", return_value=cm):
            with self.assertRaises(self.HTTPException) as ctx:
                await redeploy_bot()
        self.assertEqual(ctx.exception.status_code, 502)

    async def test_redeploy_status_returns_503_when_agent_not_configured(self):
        from src.web.api import redeploy_status
        settings.deploy_agent_url = None

        with self.assertRaises(self.HTTPException) as ctx:
            await redeploy_status()
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_redeploy_status_returns_agent_state(self):
        from src.web.api import redeploy_status
        settings.deploy_agent_url = "http://deploy-agent:8091"
        settings.deploy_agent_token = "secret-token"

        response = self._mock_response(200, {
            "running": False, "exit_code": 0, "started_at": 1.0, "finished_at": 2.0,
            "log_tail": ["ok"],
        })
        cm, client = self._mock_client_cm(response=response)

        with patch("src.web.api.httpx.AsyncClient", return_value=cm):
            result = await redeploy_status()

        self.assertFalse(result["running"])
        self.assertEqual(result["exit_code"], 0)
        _, kwargs = client.get.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-token")


class TestDeployAgentScript(unittest.TestCase):
    """
    scripts/deploy_agent.py — минимальный HTTP-сервис ВНЕ контейнера бота,
    единственный с доступом к docker.sock хоста (см. подробный комментарий
    в самом файле). Тестируем только чистую логику (сборка shell-команды,
    сверка токена constant-time) — сам HTTP-сервер (ThreadingHTTPServer) и
    subprocess.run здесь не поднимаются.
    """

    def _load_module(self):
        import importlib.util
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "scripts", "deploy_agent.py")
        spec = importlib.util.spec_from_file_location("deploy_agent_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_build_deploy_command_runs_steps_in_correct_order(self):
        """
        Порядок принципиален: миграции — ПОСЛЕ пересоздания контейнера (на
        новом образе/файлах миграций), иначе alembic upgrade применится ещё
        по старому коду.
        """
        module = self._load_module()
        cmd = module.build_deploy_command(repo_dir="/opt/cryptobot", branch="main", service="bot")

        pull_idx = cmd.index("git pull origin main")
        build_idx = cmd.index("docker compose build bot")
        up_idx = cmd.index("docker compose up -d bot")
        migrate_idx = cmd.index("docker compose exec -T bot alembic upgrade head")
        self.assertTrue(pull_idx < build_idx < up_idx < migrate_idx)
        self.assertTrue(cmd.startswith("cd /opt/cryptobot &&"))

    def test_build_deploy_command_uses_given_branch_and_service(self):
        module = self._load_module()
        cmd = module.build_deploy_command(repo_dir="/srv/bot", branch="release", service="worker")
        self.assertIn("git pull origin release", cmd)
        self.assertIn("docker compose build worker", cmd)
        self.assertIn("docker compose up -d worker", cmd)
        self.assertIn("docker compose exec -T worker alembic upgrade head", cmd)


class TestMarketDataMergeCandles(unittest.TestCase):
    """
    MarketDataIngest.merge_candles() — чистая функция слияния свечей,
    вынесенная из update_buffer(), чтобы TradingBot.candles_buffer в
    main.py мог применять ту же логику дедупликации/обрезки к СВОЕМУ
    собственному буферу, не завязываясь на self.ingest.candles_buffer
    (см. TestRefreshSymbolCandlesUsesOwnBuffer ниже — регресс на реальный
    прод-баг, где эти два разных dict'а перепутали).
    """

    @staticmethod
    def _df(index, value=1.0):
        return pd.DataFrame(
            {"open": value, "high": value, "low": value, "close": value, "volume": value},
            index=pd.to_datetime(index, unit="h"),
        )

    def test_merge_into_empty_buffer_returns_new_as_is(self):
        from src.data_ingest.market_data import MarketDataIngest
        new = self._df([1, 2, 3])
        merged = MarketDataIngest.merge_candles(None, new)
        self.assertEqual(len(merged), 3)

    def test_merge_deduplicates_by_index_keeping_newest(self):
        from src.data_ingest.market_data import MarketDataIngest
        existing = self._df([1, 2, 3], value=1.0)
        new = self._df([3, 4], value=2.0)
        merged = MarketDataIngest.merge_candles(existing, new)
        self.assertEqual(len(merged), 4)
        overlapping_ts = pd.to_datetime(3, unit="h")
        self.assertEqual(merged.loc[overlapping_ts, "close"], 2.0)

    def test_merge_trims_to_cache_size(self):
        from src.data_ingest.market_data import MarketDataIngest
        saved = settings.candlesticks_cache_size
        try:
            settings.candlesticks_cache_size = 5
            existing = self._df(range(10))
            new = self._df(range(10, 12))
            merged = MarketDataIngest.merge_candles(existing, new)
            self.assertEqual(len(merged), 5)
        finally:
            settings.candlesticks_cache_size = saved


class TestRefreshSymbolCandlesUsesOwnBuffer(unittest.IsolatedAsyncioTestCase):
    """
    Регресс на прод-инцидент: TIA/USDT (открыт через авто-исполнение
    Telegram-сигнала, см. _execute_telegram_signal) валился с
    "Ошибка обработки TIA/USDT: 'TIA/USDT'" на КАЖДОЙ итерации подряд —
    KeyError('TIA/USDT').

    Причина: _refresh_symbol_candles() звал self.ingest.update_buffer(...),
    который пишет в self.ingest.candles_buffer — ОТДЕЛЬНЫЙ dict от
    self.candles_buffer (буфер самого TradingBot, используемый везде
    дальше в _process_symbol) — а затем тут же читал
    self.candles_buffer[symbol], как будто он обновился. Для пары,
    впервые открытой не через _refresh_symbol_universe (которая пишет в
    self.candles_buffer напрямую), там никогда не было записи → KeyError
    на каждой итерации без единого шанса на восстановление. Для уже
    известных пар KeyError не было, но свежие свечи из инкрементального
    запроса точно так же терялись — буфер молча оставался устаревшим.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    @staticmethod
    def _df(rows, value=1.0, start_hour=0):
        return pd.DataFrame(
            {"open": value, "high": value, "low": value, "close": value, "volume": value},
            index=pd.to_datetime(range(start_hour, start_hour + rows), unit="h"),
        )

    async def test_new_symbol_not_in_own_buffer_does_not_raise_keyerror(self):
        """Символ, открытый напрямую (Telegram-сигнал/ручная сделка) и
        никогда не проходивший через _refresh_symbol_universe — буфер
        TradingBot для него пуст с самого начала."""
        from src.data_ingest.market_data import MarketDataIngest

        bot = self._make_bot()
        self.assertNotIn("TIA/USDT", bot.candles_buffer)
        bot.ingest = MarketDataIngest("bybit")
        bot.ingest.fetch_ohlcv = AsyncMock(return_value=self._df(200))

        df = await bot._refresh_symbol_candles("TIA/USDT")

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 200)
        self.assertIn("TIA/USDT", bot.candles_buffer)

    async def test_incremental_refresh_updates_own_buffer_not_stale_copy(self):
        """Пара уже в буфере TradingBot (>=50 строк) — инкрементальный
        запрос новых свечей должен реально попасть в bot.candles_buffer,
        а не потеряться в отдельном буфере ingest."""
        from src.data_ingest.market_data import MarketDataIngest

        bot = self._make_bot()
        bot.candles_buffer["BTC/USDT"] = self._df(60, value=1.0)
        fresh = self._df(3, value=999.0, start_hour=60)
        bot.ingest = MarketDataIngest("bybit")
        bot.ingest.fetch_ohlcv = AsyncMock(return_value=fresh)

        df = await bot._refresh_symbol_candles("BTC/USDT")

        self.assertIsNotNone(df)
        latest_ts = pd.to_datetime(62, unit="h")
        self.assertEqual(df.loc[latest_ts, "close"], 999.0)
        self.assertEqual(bot.candles_buffer["BTC/USDT"].loc[latest_ts, "close"], 999.0)


class TestShortSignalRejectedBeforeExecutionInRealMode(unittest.IsolatedAsyncioTestCase):
    """
    Регресс на прод-инцидент: ML/Ensemble стратегия продолжала генерировать
    SHORT-сигнал для GRAM/USDT в реальном (спот) режиме — каждый такой
    сигнал доходил до execution_engine.create_order() ->
    _execute_real_order(), которая ПРАВИЛЬНО его отклоняла (на споте
    шорта не существует — см. защиту в executor.py, добавленную после
    инцидента ENA/USDT), но КАЖДУЮ итерацию (~раз в минуту) это означало
    лишний запрос к бирже (fetch_ticker) и ERROR в логах — бесконечно,
    пока модель не изменит мнение. Итог был предрешён ещё на этапе
    сигнала, поэтому короткие сигналы в реальном режиме теперь
    отклоняются раньше, не доходя до execution_engine.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    @staticmethod
    def _make_candles_df():
        return pd.DataFrame({
            "open": [1.0] * 60, "high": [1.0] * 60, "low": [1.0] * 60,
            "close": [1.0] * 60, "volume": [1.0] * 60,
        })

    def setUp(self):
        self._saved_trading_mode = settings.trading_mode
        self._saved_active_trading_mode = settings.active_trading_mode
        settings.trading_mode = "real"
        # Тестирует путь встроенных стратегий (_process_symbol) — он
        # запускается только в режиме "algo" (см. TestTradingSourceMode*).
        settings.active_trading_mode = "algo"

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode
        settings.active_trading_mode = self._saved_active_trading_mode

    async def test_short_signal_rejected_without_hitting_execution_engine(self):
        from src.strategy import StrategySignal

        bot = self._make_bot()
        bot.feature_engine = MagicMock()
        bot.ml_inference = None
        bot._refresh_symbol_candles = AsyncMock(return_value=self._make_candles_df())

        fake_strategy = MagicMock()
        fake_strategy.strategy_id = "ml_direction"
        fake_strategy.name = "ML"
        fake_strategy.weight = 1.0
        fake_strategy.generate_signal.return_value = StrategySignal(
            strategy_id="ml_direction", symbol="GRAM/USDT", side="short", confidence=0.66,
        )

        with patch("src.main.strategy_registry.get_active", return_value=[fake_strategy]), \
                patch("src.main.strategy_registry.get", return_value=None), \
                patch("src.main.execution_engine") as mock_engine, \
                patch("src.main.risk_manager") as mock_risk:
            mock_engine.paper_positions = {}
            mock_engine.real_positions = {}
            mock_engine.last_prices = {}
            mock_engine.create_order = AsyncMock()
            mock_risk.check_signal.return_value = (True, "")

            await bot._process_symbol("GRAM/USDT")

        mock_engine.create_order.assert_not_awaited()

    async def test_long_signal_still_reaches_execution_engine(self):
        """Убедиться, что фикс не блокирует обычные long-сигналы в реальном режиме."""
        from src.strategy import StrategySignal

        bot = self._make_bot()
        bot.feature_engine = MagicMock()
        bot.ml_inference = None
        bot._refresh_symbol_candles = AsyncMock(return_value=self._make_candles_df())

        fake_strategy = MagicMock()
        fake_strategy.strategy_id = "ml_direction"
        fake_strategy.name = "ML"
        fake_strategy.weight = 1.0
        fake_strategy.generate_signal.return_value = StrategySignal(
            strategy_id="ml_direction", symbol="BTC/USDT", side="long", confidence=0.66,
            entry_price=100.0,
        )

        with patch("src.main.strategy_registry.get_active", return_value=[fake_strategy]), \
                patch("src.main.strategy_registry.get", return_value=None), \
                patch("src.main.execution_engine") as mock_engine, \
                patch("src.main.risk_manager") as mock_risk, \
                patch("src.main.protection_manager") as mock_protections, \
                patch("src.main.expectancy_sizing") as mock_sizing:
            mock_engine.paper_positions = {}
            mock_engine.real_positions = {}
            mock_engine.last_prices = {}
            mock_engine.create_order = AsyncMock(return_value=None)
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_risk.check_signal.return_value = (True, "")
            mock_protections.locked_reason = AsyncMock(return_value=None)
            mock_sizing.size_multiplier = AsyncMock(return_value=1.0)

            await bot._process_symbol("BTC/USDT")

        mock_engine.create_order.assert_awaited_once()
        self.assertEqual(mock_engine.create_order.await_args.kwargs["side"], "buy")


class TestTelegramSignalShortRejectedInRealMode(unittest.IsolatedAsyncioTestCase):
    """Тот же класс бага, что и TestShortSignalRejectedBeforeExecutionInRealMode,
    но для пути исполнения Telegram-сигналов (_execute_telegram_signal)."""

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    def setUp(self):
        self._saved_trading_mode = settings.trading_mode
        settings.trading_mode = "real"

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode

    async def test_short_telegram_signal_rejected_without_hitting_execution_engine(self):
        bot = self._make_bot()

        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.create_order = AsyncMock()
            order = await bot._execute_telegram_signal({
                "parsed_pair": "GRAM/USDT",
                "parsed_side": "short",
                "parsed_entry": 1.34,
                "parsed_sl": 1.36,
                "parsed_tp": 1.32,
                "channel_id": "@test_channel",
            })

        self.assertIsNone(order)
        mock_engine.create_order.assert_not_awaited()

    async def test_long_telegram_signal_still_reaches_execution_engine(self):
        bot = self._make_bot()

        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT",
                "parsed_side": "long",
                "parsed_entry": 50000.0,
                "parsed_sl": 49000.0,
                "parsed_tp": 52000.0,
                "channel_id": "@test_channel",
            })

        mock_engine.create_order.assert_awaited_once()
        self.assertEqual(mock_engine.create_order.await_args.kwargs["side"], "buy")


class TestShortSignalAllowedOnFutures(unittest.IsolatedAsyncioTestCase):
    """
    ЭТАП 2 перехода на фьючерсы: short-сигналы больше не отклоняются
    безусловно в реальном режиме — только на споте (см.
    TestShortSignalRejectedBeforeExecutionInRealMode/
    TestTelegramSignalShortRejectedInRealMode, где отклонение на споте
    остаётся верным поведением). При market_type=="futures" short должен
    доходить до execution_engine.create_order(side="sell").
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    @staticmethod
    def _make_candles_df():
        return pd.DataFrame({
            "open": [1.0] * 60, "high": [1.0] * 60, "low": [1.0] * 60,
            "close": [1.0] * 60, "volume": [1.0] * 60,
        })

    def setUp(self):
        self._saved_trading_mode = settings.trading_mode
        self._saved_active_trading_mode = settings.active_trading_mode
        self._saved_market_type = settings.market_type
        settings.trading_mode = "real"
        settings.market_type = "futures"
        settings.active_trading_mode = "algo"

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode
        settings.active_trading_mode = self._saved_active_trading_mode
        settings.market_type = self._saved_market_type

    async def test_short_strategy_signal_reaches_execution_engine_on_futures(self):
        from src.strategy import StrategySignal

        bot = self._make_bot()
        bot.feature_engine = MagicMock()
        bot.ml_inference = None
        bot._refresh_symbol_candles = AsyncMock(return_value=self._make_candles_df())

        fake_strategy = MagicMock()
        fake_strategy.strategy_id = "ml_direction"
        fake_strategy.name = "ML"
        fake_strategy.weight = 1.0
        fake_strategy.generate_signal.return_value = StrategySignal(
            strategy_id="ml_direction", symbol="FUTSHORTSIG1/USDT", side="short", confidence=0.66,
            entry_price=100.0,
        )

        with patch("src.main.strategy_registry.get_active", return_value=[fake_strategy]), \
                patch("src.main.strategy_registry.get", return_value=None), \
                patch("src.main.execution_engine") as mock_engine, \
                patch("src.main.risk_manager") as mock_risk, \
                patch("src.main.protection_manager") as mock_protections, \
                patch("src.main.expectancy_sizing") as mock_sizing:
            mock_engine.paper_positions = {}
            mock_engine.real_positions = {}
            mock_engine.last_prices = {}
            mock_engine.create_order = AsyncMock(return_value=None)
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_risk.check_signal.return_value = (True, "")
            mock_protections.locked_reason = AsyncMock(return_value=None)
            mock_sizing.size_multiplier = AsyncMock(return_value=1.0)

            await bot._process_symbol("FUTSHORTSIG1/USDT")

        mock_engine.create_order.assert_awaited_once()
        self.assertEqual(mock_engine.create_order.await_args.kwargs["side"], "sell")

    async def test_short_telegram_signal_reaches_execution_engine_on_futures(self):
        bot = self._make_bot()

        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            await bot._execute_telegram_signal({
                "parsed_pair": "FUTSHORTSIG2/USDT",
                "parsed_side": "short",
                "parsed_entry": 1.34,
                "parsed_sl": 1.36,
                "parsed_tp": 1.32,
                "channel_id": "@test_channel",
            })

        mock_engine.create_order.assert_awaited_once()
        self.assertEqual(mock_engine.create_order.await_args.kwargs["side"], "sell")


class TestPerChannelMarketTypeOverridesGlobalToggle(unittest.IsolatedAsyncioTestCase):
    """
    Различие типа торговли (спот/фьючерсы) по каналу, а не только по
    глобальному тумблеру settings.market_type: сигналы конкретного канала
    настроены на свой рынок (TelegramChannel.market, см. _get_channel_settings/
    _on_telegram_signal) — _execute_telegram_signal должен решать по
    signal_event["channel_market_type"], а не по текущему положению тумблера
    в шапке дашборда.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    def setUp(self):
        self._saved_trading_mode = settings.trading_mode
        self._saved_market_type = settings.market_type
        settings.trading_mode = "real"

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode
        settings.market_type = self._saved_market_type

    async def test_futures_channel_allows_short_even_when_global_toggle_is_spot(self):
        settings.market_type = "spot"
        bot = self._make_bot()

        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            await bot._execute_telegram_signal({
                "parsed_pair": "CHANFUT1/USDT",
                "parsed_side": "short",
                "parsed_entry": 1.34,
                "parsed_sl": 1.36,
                "parsed_tp": 1.32,
                "channel_id": "@futures_channel",
                "channel_market_type": "futures",
            })

        mock_engine.create_order.assert_awaited_once()
        self.assertEqual(mock_engine.create_order.await_args.kwargs["side"], "sell")
        self.assertEqual(mock_engine.create_order.await_args.kwargs["market_type"], "futures")

    async def test_spot_channel_rejects_short_even_when_global_toggle_is_futures(self):
        settings.market_type = "futures"
        bot = self._make_bot()

        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            order = await bot._execute_telegram_signal({
                "parsed_pair": "CHANSPOT1/USDT",
                "parsed_side": "short",
                "parsed_entry": 1.34,
                "parsed_sl": 1.36,
                "parsed_tp": 1.32,
                "channel_id": "@spot_channel",
                "channel_market_type": "spot",
            })

        self.assertIsNone(order)
        mock_engine.create_order.assert_not_called()

    async def test_signal_without_channel_market_type_falls_back_to_global_toggle(self):
        """Регресс: ручное подтверждение старого сигнала без известного канала
        (POST /telegram/signals/{id}/decide, канал удалён) — ведёт себя как раньше."""
        settings.market_type = "futures"
        bot = self._make_bot()

        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            await bot._execute_telegram_signal({
                "parsed_pair": "CHANFALLBACK1/USDT",
                "parsed_side": "short",
                "parsed_entry": 1.34,
                "parsed_sl": 1.36,
                "parsed_tp": 1.32,
                "channel_id": "@no_channel_row",
            })

        mock_engine.create_order.assert_awaited_once()
        self.assertEqual(mock_engine.create_order.await_args.kwargs["market_type"], "futures")


class TestCreateOrderLazilyConnectsChannelMarket(unittest.IsolatedAsyncioTestCase):
    """
    execution_engine.create_order(market_type=...) должен уметь открыть
    позицию на рынке, для которого ещё НЕТ подключённого ccxt-клиента —
    например, тумблер и все текущие real-позиции на споте, но конкретный
    Telegram-канал настроен на futures: initialize() поднимает второй
    клиент лениво только для рынков, найденных среди позиций,
    восстановленных при СТАРТЕ, а не по требованию при открытии новой.
    _ensure_exchange_connected должен подключить его на лету.
    """

    async def asyncSetUp(self):
        self.engine = ExecutionEngine()

    async def asyncTearDown(self):
        await self.engine.close()

    def setUp(self):
        self._saved_market_type = settings.market_type
        self._saved_trading_mode = settings.trading_mode

    def tearDown(self):
        settings.market_type = self._saved_market_type
        settings.trading_mode = self._saved_trading_mode

    async def test_opens_short_on_futures_via_lazily_connected_client(self):
        settings.market_type = "spot"  # текущий тумблер — spot
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        spot_mock = AsyncMock()
        futures_mock = AsyncMock()
        self.engine._exchanges = {"spot": spot_mock}  # futures ещё не подключён вовсе
        futures_mock.create_market_sell_order.return_value = {
            "id": "chan-fut-short-1", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }

        def make_exchange(config):
            options = config.get("options") or {}
            return futures_mock if options.get("defaultType") == "swap" else spot_mock

        with patch("src.execution.executor.ccxt.bybit", side_effect=make_exchange):
            order = await self.engine.create_order(
                symbol="CHANLAZY1/USDT", side="sell", amount=10.0, price=2.0,
                order_type="market", market_type="futures",
            )

        self.assertIsNotNone(order)
        futures_mock.create_market_sell_order.assert_awaited_once_with("CHANLAZY1/USDT", 10.0)
        spot_mock.create_market_sell_order.assert_not_called()
        self.assertIn("CHANLAZY1/USDT", self.engine.real_positions)
        self.assertEqual(self.engine.real_positions["CHANLAZY1/USDT"]["market_type"], "futures")
        self.assertEqual(self.engine.real_positions["CHANLAZY1/USDT"]["side"], "short")
        self.assertIs(self.engine._exchanges.get("futures"), futures_mock)

    async def test_market_type_none_still_uses_current_toggle(self):
        """Регресс: без явного market_type (сигналы стратегий/ручные ордера) поведение не меняется."""
        settings.market_type = "spot"
        settings.trading_mode = "real"
        self.engine.is_paper = False
        self.engine.exchange_id = "bybit"
        self.engine.exchange = AsyncMock()
        self.engine.exchange.create_market_buy_order.return_value = {
            "id": "chan-default-1", "filled": 10.0, "average": 2.0, "price": None,
            "fee": {"cost": 0.01, "currency": "USDT"},
        }

        order = await self.engine.create_order(
            symbol="CHANDEFAULT1/USDT", side="buy", amount=10.0, price=2.0, order_type="market",
        )

        self.assertIsNotNone(order)
        self.assertEqual(self.engine.real_positions["CHANDEFAULT1/USDT"]["market_type"], "spot")


class TestTradingSourceModeEndpoint(unittest.IsolatedAsyncioTestCase):
    """
    POST /trading-source-mode — переключатель "сигналы"/"алго" в шапке
    дашборда. Тот же apply_settings_update, что и вкладка "Настройки":
    применяется немедленно (settings.active_trading_mode) и сохраняется в
    BotConfig на будущие перезапуски.
    """

    def setUp(self):
        self._saved_mode = settings.active_trading_mode

    def tearDown(self):
        settings.active_trading_mode = self._saved_mode

    async def test_valid_mode_updates_live_setting(self):
        from src.web.api import set_trading_source_mode

        result = await set_trading_source_mode(mode="algo")

        self.assertEqual(result, {"success": True, "mode": "algo"})
        self.assertEqual(settings.active_trading_mode, "algo")

    async def test_invalid_mode_is_rejected_without_changing_setting(self):
        from fastapi import HTTPException

        from src.web.api import set_trading_source_mode

        settings.active_trading_mode = "signals"
        with self.assertRaises(HTTPException) as ctx:
            await set_trading_source_mode(mode="both")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(settings.active_trading_mode, "signals")


class TestMarketTypeEndpoint(unittest.IsolatedAsyncioTestCase):
    """
    POST /market-type — переключатель "спот"/"фьючерсы" в шапке дашборда
    (ЭТАП 1 перехода на фьючерсы). Тот же apply_settings_update, что и
    вкладка "Настройки".
    """

    def setUp(self):
        self._saved_market_type = settings.market_type
        self._saved_trading_mode = settings.trading_mode
        # paper — чтобы apply_settings_update не пытался (пере)подключить
        # execution_engine к реальной бирже внутри этого теста (market_type
        # входит в список триггеров реконнекта только при trading_mode=="real",
        # см. settings_store.apply_settings_update).
        settings.trading_mode = "paper"

    def tearDown(self):
        settings.market_type = self._saved_market_type
        settings.trading_mode = self._saved_trading_mode

    async def test_valid_type_updates_live_setting(self):
        from src.web.api import set_market_type

        result = await set_market_type(market_type="futures")

        self.assertEqual(result, {"success": True, "market_type": "futures"})
        self.assertEqual(settings.market_type, "futures")

    async def test_invalid_type_is_rejected_without_changing_setting(self):
        from fastapi import HTTPException

        from src.web.api import set_market_type

        settings.market_type = "spot"
        with self.assertRaises(HTTPException) as ctx:
            await set_market_type(market_type="margin")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(settings.market_type, "spot")


class TestTradingSourceModeGatesAlgoStrategies(unittest.IsolatedAsyncioTestCase):
    """
    Регресс/новая функциональность: переключатель "сигналы"/"алго" — в
    режиме "signals" встроенные ML/Ensemble/BB-стратегии вообще не должны
    запускаться в _process_symbol (ни инференс, ни generate_signal, ни
    исполнение) — новые позиции в этом режиме открывают только
    Telegram-каналы. Уже открытые позиции (проверка SL/TP выше по функции)
    режимом не гейтятся — это отдельная, всегда активная часть
    _process_symbol, здесь не проверяется напрямую (покрыто другими
    тестами), только то, что генерация НОВЫХ сигналов зависит от режима.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    @staticmethod
    def _make_candles_df():
        return pd.DataFrame({
            "open": [1.0] * 60, "high": [1.0] * 60, "low": [1.0] * 60,
            "close": [1.0] * 60, "volume": [1.0] * 60,
        })

    def setUp(self):
        self._saved_mode = settings.active_trading_mode

    def tearDown(self):
        settings.active_trading_mode = self._saved_mode

    async def test_signals_mode_never_calls_strategies(self):
        from src.strategy import StrategySignal

        settings.active_trading_mode = "signals"
        bot = self._make_bot()
        bot.feature_engine = MagicMock()
        bot.ml_inference = None
        bot._refresh_symbol_candles = AsyncMock(return_value=self._make_candles_df())

        fake_strategy = MagicMock()
        fake_strategy.strategy_id = "ml_direction"
        fake_strategy.generate_signal.return_value = StrategySignal(
            strategy_id="ml_direction", symbol="BTC/USDT", side="long", confidence=0.9,
            entry_price=1.0,
        )

        with patch("src.main.strategy_registry.get_active", return_value=[fake_strategy]) as get_active_mock, \
                patch("src.main.execution_engine") as mock_engine:
            mock_engine.paper_positions = {}
            mock_engine.real_positions = {}
            mock_engine.last_prices = {}
            mock_engine.create_order = AsyncMock()

            await bot._process_symbol("BTC/USDT")

        get_active_mock.assert_not_called()
        fake_strategy.generate_signal.assert_not_called()
        mock_engine.create_order.assert_not_awaited()

    async def test_algo_mode_still_calls_strategies(self):
        from src.strategy import StrategySignal

        settings.active_trading_mode = "algo"
        bot = self._make_bot()
        bot.feature_engine = MagicMock()
        bot.ml_inference = None
        bot._refresh_symbol_candles = AsyncMock(return_value=self._make_candles_df())

        fake_strategy = MagicMock()
        fake_strategy.strategy_id = "ml_direction"
        fake_strategy.name = "ML"
        fake_strategy.weight = 1.0
        fake_strategy.generate_signal.return_value = StrategySignal(
            strategy_id="ml_direction", symbol="BTC/USDT", side="long", confidence=0.9,
            entry_price=1.0,
        )

        with patch("src.main.strategy_registry.get_active", return_value=[fake_strategy]), \
                patch("src.main.strategy_registry.get", return_value=None), \
                patch("src.main.execution_engine") as mock_engine, \
                patch("src.main.risk_manager") as mock_risk, \
                patch("src.main.protection_manager") as mock_protections, \
                patch("src.main.expectancy_sizing") as mock_sizing:
            mock_engine.paper_positions = {}
            mock_engine.real_positions = {}
            mock_engine.last_prices = {}
            mock_engine.create_order = AsyncMock(return_value=None)
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.get_paper_balance = MagicMock(return_value=10000.0)
            mock_risk.check_signal.return_value = (True, "")
            mock_protections.locked_reason = AsyncMock(return_value=None)
            mock_sizing.size_multiplier = AsyncMock(return_value=1.0)

            await bot._process_symbol("BTC/USDT")

        fake_strategy.generate_signal.assert_called_once()
        mock_engine.create_order.assert_awaited_once()


class TestTradingSourceModeGatesTelegramSignals(unittest.IsolatedAsyncioTestCase):
    """Та же переключалка "сигналы"/"алго", но для пути Telegram-сигналов
    (_on_telegram_signal) — в режиме "algo" автоисполнение канала должно
    отклоняться, даже если качество прошло порог и auto_execute=True."""

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    def setUp(self):
        self._saved_mode = settings.active_trading_mode

    def tearDown(self):
        settings.active_trading_mode = self._saved_mode

    async def test_algo_mode_rejects_auto_execute_telegram_signal(self):
        settings.active_trading_mode = "algo"
        bot = self._make_bot()
        bot._telegram_channel_db_ids = {}
        bot._get_channel_settings = AsyncMock(return_value=(0.0, True, 5.0, "spot"))
        bot._save_telegram_signal = AsyncMock()

        with patch.object(bot, "_execute_telegram_signal", new=AsyncMock()) as exec_mock:
            await bot._on_telegram_signal({
                "channel_id": "@test_channel",
                "parsed_pair": "BTC/USDT",
                "parsed_side": "long",
                "parsed_entry": 50000.0,
                "raw_message": "test",
            })

        exec_mock.assert_not_awaited()
        bot._save_telegram_signal.assert_awaited_once()
        self.assertEqual(bot._save_telegram_signal.await_args.args[2], "rejected")
        self.assertIn("алго", bot._save_telegram_signal.await_args.args[0]["reject_reason"])

    async def test_signals_mode_still_auto_executes_telegram_signal(self):
        settings.active_trading_mode = "signals"
        bot = self._make_bot()
        bot._telegram_channel_db_ids = {}
        bot._get_channel_settings = AsyncMock(return_value=(0.0, True, 5.0, "spot"))
        bot._save_telegram_signal = AsyncMock()
        bot.open_positions = {}

        fake_order = MagicMock(id=1)
        with patch.object(bot, "_execute_telegram_signal", new=AsyncMock(return_value=fake_order)) as exec_mock:
            await bot._on_telegram_signal({
                "channel_id": "@test_channel",
                "parsed_pair": "BTC/USDT",
                "parsed_side": "long",
                "parsed_entry": 50000.0,
                "raw_message": "test",
            })

        exec_mock.assert_awaited_once()
        self.assertEqual(bot._save_telegram_signal.await_args.args[2], "executed")


class TestTelegramSignalMarketEntryResolution(unittest.IsolatedAsyncioTestCase):
    """
    Реальный формат сигнала без фиксированной цены входа ("Диапазон входа:
    по рынку" — см. is_market_entry в channel_monitor.py). Раньше
    _on_telegram_signal() отклонял ЛЮБОЙ сигнал без parsed_entry — до
    этого изменения parse_with_regex вообще не возвращал такой сигнал, а
    entry<=0 сравнение упало бы с TypeError на None. Канал явно попросил
    маркет-исполнение, а не забыл цену — сигнал должен резолвить текущую
    рыночную цену (execution_engine.get_reference_price) и продолжить
    как обычно.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    def setUp(self):
        self._saved_mode = settings.active_trading_mode
        settings.active_trading_mode = "signals"

    def tearDown(self):
        settings.active_trading_mode = self._saved_mode

    async def test_market_entry_resolved_via_reference_price(self):
        bot = self._make_bot()
        bot._telegram_channel_db_ids = {}
        bot._get_channel_settings = AsyncMock(return_value=(0.0, True, 5.0, "futures"))
        bot._save_telegram_signal = AsyncMock()
        bot.open_positions = {}

        fake_order = MagicMock(id=1)
        with patch("src.main.execution_engine") as mock_engine, \
                patch.object(bot, "_execute_telegram_signal", new=AsyncMock(return_value=fake_order)) as exec_mock:
            mock_engine.get_reference_price = AsyncMock(return_value=0.2034)
            await bot._on_telegram_signal({
                "channel_id": "@test_channel",
                "parsed_pair": "WIF/USDT",
                "parsed_side": "short",
                "parsed_entry": None,
                "parsed_sl": 0.2091,
                "parsed_tp": 0.1853,
                "raw_message": "test",
            })

        mock_engine.get_reference_price.assert_awaited_once_with("WIF/USDT", "futures")
        exec_mock.assert_awaited_once()
        passed_event = exec_mock.await_args.args[0]
        self.assertEqual(passed_event["parsed_entry"], 0.2034)

    async def test_market_entry_rejected_when_reference_price_unavailable(self):
        """
        Раньше этот случай отклонял сигнал через голый return — сигнал не
        попадал в БД вообще, и "Итог сделки" для него в дашборде было
        невозможно ни увидеть, ни объяснить задним числом. Теперь такой
        отказ тоже сохраняется как decision="rejected" с понятной причиной.
        """
        bot = self._make_bot()
        bot._telegram_channel_db_ids = {}
        bot._get_channel_settings = AsyncMock(return_value=(0.0, True, 5.0, "futures"))
        bot._save_telegram_signal = AsyncMock()
        bot.open_positions = {}

        with patch("src.main.execution_engine") as mock_engine, \
                patch.object(bot, "_execute_telegram_signal", new=AsyncMock()) as exec_mock:
            mock_engine.get_reference_price = AsyncMock(return_value=None)
            await bot._on_telegram_signal({
                "channel_id": "@test_channel",
                "parsed_pair": "WIF/USDT",
                "parsed_side": "short",
                "parsed_entry": None,
                "raw_message": "test",
            })

        exec_mock.assert_not_awaited()
        bot._save_telegram_signal.assert_awaited_once()
        saved_event, quality, decision, order = bot._save_telegram_signal.await_args.args
        self.assertEqual(decision, "rejected")
        self.assertIsNone(order)
        self.assertIn("цену", saved_event["reject_reason"])

    async def test_explicit_entry_does_not_trigger_reference_price_lookup(self):
        """Регресс: обычный сигнал с явной ценой не должен обращаться к
        execution_engine.get_reference_price вообще (уже есть число)."""
        bot = self._make_bot()
        bot._telegram_channel_db_ids = {}
        bot._get_channel_settings = AsyncMock(return_value=(0.0, True, 5.0, "spot"))
        bot._save_telegram_signal = AsyncMock()
        bot.open_positions = {}

        fake_order = MagicMock(id=1)
        with patch("src.main.execution_engine") as mock_engine, \
                patch.object(bot, "_execute_telegram_signal", new=AsyncMock(return_value=fake_order)) as exec_mock:
            mock_engine.get_reference_price = AsyncMock(return_value=999.0)
            await bot._on_telegram_signal({
                "channel_id": "@test_channel",
                "parsed_pair": "BTC/USDT",
                "parsed_side": "long",
                "parsed_entry": 50000.0,
                "raw_message": "test",
            })

        mock_engine.get_reference_price.assert_not_awaited()
        exec_mock.assert_awaited_once()
        self.assertEqual(exec_mock.await_args.args[0]["parsed_entry"], 50000.0)


class TestTelegramSignalQualityScoringUsesCorrectShape(unittest.IsolatedAsyncioTestCase):
    """
    Регресс: _on_telegram_signal() передавал в signal_quality_scorer.
    score_signal() сырой signal_event (ключи с префиксом parsed_* —
    parsed_side/parsed_entry/parsed_sl/parsed_tp), а score_signal() читает
    side/entry/sl/tp/confidence (без префикса — формат, который
    возвращают parse_with_regex/parse_with_llm/parse_with_gemini). Из-за
    несовпадения имён ключей signal.get("sl")/("tp") всегда возвращали
    None, signal.get("side") — всегда "", а signal.get("confidence") —
    всегда дефолтные 0.5: штраф "нет SL" (-0.15) применялся к КАЖДОМУ
    сигналу независимо от реального SL/TP в сообщении канала, бонус за
    risk/reward не срабатывал никогда, а уверенность LLM-парсера
    игнорировалась — итоговый quality был у любого сигнала практически
    одинаковым, независимо от его реального содержания.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    async def test_score_signal_receives_normalized_shape_not_raw_event(self):
        from src.telegram.quality_scorer import signal_quality_scorer

        bot = self._make_bot()
        bot._telegram_channel_db_ids = {}
        bot._get_channel_settings = AsyncMock(return_value=(1.1, False, 5.0, "spot"))  # порог недостижим — просто проверяем вызов
        bot._save_telegram_signal = AsyncMock()

        with patch.object(signal_quality_scorer, "score_signal", return_value=0.9) as score_mock:
            await bot._on_telegram_signal({
                "channel_id": "@test_channel",
                "parsed_pair": "BTC/USDT",
                "parsed_side": "long",
                "parsed_entry": 50000.0,
                "parsed_sl": 49000.0,
                "parsed_tp": 52000.0,
                "parsed_confidence": 0.87,
                "raw_message": "test",
            })

        score_mock.assert_called_once()
        passed_signal = score_mock.call_args.args[0]
        self.assertEqual(passed_signal["side"], "long")
        self.assertEqual(passed_signal["entry"], 50000.0)
        self.assertEqual(passed_signal["sl"], 49000.0)
        self.assertEqual(passed_signal["tp"], 52000.0)
        self.assertEqual(passed_signal["confidence"], 0.87)

    async def test_signal_with_sl_tp_scores_meaningfully_higher_than_without(self):
        """Интеграционная проверка через реальный (не замоканный) score_signal:
        наличие SL/TP и хороший RR должны реально поднимать quality, а не
        давать одинаковый результат независимо от содержания сигнала."""
        from src.telegram.quality_scorer import signal_quality_scorer

        bot = self._make_bot()
        bot._telegram_channel_db_ids = {}
        bot._get_channel_settings = AsyncMock(return_value=(1.1, False, 5.0, "spot"))
        bot._save_telegram_signal = AsyncMock()
        signal_quality_scorer.channel_stats.pop("@quality_shape_test", None)

        await bot._on_telegram_signal({
            "channel_id": "@quality_shape_test",
            "parsed_pair": "BTC/USDT",
            "parsed_side": "long",
            "parsed_entry": 50000.0,
            "parsed_sl": 49000.0,
            "parsed_tp": 52000.0,  # RR = 2.0 -> максимальный бонус
            "parsed_confidence": 1.0,
            "raw_message": "test",
        })
        with_sl_tp = bot._save_telegram_signal.await_args.args[1]

        bot._save_telegram_signal.reset_mock()
        await bot._on_telegram_signal({
            "channel_id": "@quality_shape_test",
            "parsed_pair": "BTC/USDT",
            "parsed_side": "long",
            "parsed_entry": 50000.0,
            "parsed_confidence": 1.0,
            "raw_message": "test",
        })
        without_sl_tp = bot._save_telegram_signal.await_args.args[1]

        self.assertGreater(with_sl_tp, without_sl_tp)


class TestTelegramSignalDefaultStopLoss(unittest.IsolatedAsyncioTestCase):
    """
    Раньше сигнал канала без явного SL открывал реальную позицию вообще
    без биржевого защитного ордера — sync_stop_loss_order() (executor.py)
    пропускает выставление SL на бирже, если stop_loss falsy, и позиция
    защищалась только опросом бота раз в торговый цикл (~60-90с): при
    падении/рестарте процесса она оставалась полностью незащищённой на
    неопределённое время. telegram_signals_default_sl_pct подставляет
    защитный SL от entry_price, когда канал сам его не указал.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    def setUp(self):
        self._saved = {
            "trading_mode": settings.trading_mode,
            "telegram_signals_default_sl_pct": settings.telegram_signals_default_sl_pct,
        }
        settings.trading_mode = "real"
        settings.telegram_signals_default_sl_pct = 3.0

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(settings, key, value)

    async def test_missing_sl_gets_default_fallback_for_long(self):
        bot = self._make_bot()
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "long",
                "parsed_entry": 50000.0, "parsed_sl": None, "parsed_tp": 52000.0,
                "channel_id": "@test_channel",
            })

        applied_sl = mock_engine.create_order.await_args.kwargs["stop_loss"]
        self.assertAlmostEqual(applied_sl, 50000.0 * 0.97)

    async def test_missing_sl_gets_default_fallback_for_short_paper(self):
        """paper-режим тоже поддерживает short — направление fallback-SL
        должно быть зеркальным (выше входа, а не ниже)."""
        bot = self._make_bot()
        settings.trading_mode = "paper"
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_paper_balance = MagicMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "short",
                "parsed_entry": 50000.0, "parsed_sl": None, "parsed_tp": 48000.0,
                "channel_id": "@test_channel",
            })

        applied_sl = mock_engine.create_order.await_args.kwargs["stop_loss"]
        self.assertAlmostEqual(applied_sl, 50000.0 * 1.03)

    async def test_explicit_sl_from_channel_is_not_overridden(self):
        bot = self._make_bot()
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "long",
                "parsed_entry": 50000.0, "parsed_sl": 49500.0, "parsed_tp": 52000.0,
                "channel_id": "@test_channel",
            })

        applied_sl = mock_engine.create_order.await_args.kwargs["stop_loss"]
        self.assertEqual(applied_sl, 49500.0)

    async def test_zero_default_pct_disables_fallback(self):
        settings.telegram_signals_default_sl_pct = 0.0
        bot = self._make_bot()
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "long",
                "parsed_entry": 50000.0, "parsed_sl": None, "parsed_tp": 52000.0,
                "channel_id": "@test_channel",
            })

        applied_sl = mock_engine.create_order.await_args.kwargs["stop_loss"]
        self.assertIsNone(applied_sl)


class TestTelegramChannelPositionSizePct(unittest.IsolatedAsyncioTestCase):
    """
    Раньше _execute_telegram_signal() всегда считал размер позиции от
    захардкоженных 5.0% для ЛЮБОГО канала (size_pct = 5.0 * mult) — доверие
    к разным каналам обычно разное, но настроить это было негде.
    TelegramChannel.position_size_pct (читается через _get_channel_settings
    в _on_telegram_signal и пробрасывается в signal_event) теперь задаёт
    базовый % персонально по каналу.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    def setUp(self):
        self._saved_trading_mode = settings.trading_mode
        settings.trading_mode = "real"

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode

    async def test_execute_uses_channel_specific_position_size_pct(self):
        bot = self._make_bot()
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "long",
                "parsed_entry": 50000.0, "parsed_sl": 49000.0, "parsed_tp": 52000.0,
                "channel_id": "@test_channel", "channel_position_size_pct": 12.0,
            })

        amount = mock_engine.create_order.await_args.kwargs["amount"]
        # balance=10000, 12% -> position_value=1200, entry=50000 -> amount=0.024
        self.assertAlmostEqual(amount, 1200.0 / 50000.0)

    async def test_execute_falls_back_to_5pct_when_not_provided(self):
        """signal_event без channel_position_size_pct (например, старый код
        пути или сбой чтения настроек канала) — прежнее поведение 5%."""
        bot = self._make_bot()
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=None)
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "long",
                "parsed_entry": 50000.0, "parsed_sl": 49000.0, "parsed_tp": 52000.0,
                "channel_id": "@test_channel",
            })

        amount = mock_engine.create_order.await_args.kwargs["amount"]
        self.assertAlmostEqual(amount, 500.0 / 50000.0)

    async def test_on_telegram_signal_passes_channel_position_size_pct_through(self):
        from src.db.models import TelegramChannel
        from src.db.session import get_session

        async with get_session() as session:
            channel = TelegramChannel(
                channel_id="@sizing_pipeline_test", channel_title="X",
                quality_threshold=0.0, auto_execute=True, position_size_pct=9.0, active=True,
            )
            session.add(channel)
            await session.commit()
            db_id = channel.id

        bot = self._make_bot()
        bot._telegram_channel_db_ids = {"@sizing_pipeline_test": db_id}
        bot._save_telegram_signal = AsyncMock()
        bot.open_positions = {}

        with patch.object(bot, "_execute_telegram_signal", new=AsyncMock(return_value=None)) as exec_mock:
            await bot._on_telegram_signal({
                "channel_id": "@sizing_pipeline_test",
                "parsed_pair": "BTC/USDT",
                "parsed_side": "long",
                "parsed_entry": 50000.0,
                "raw_message": "test",
            })

        passed_event = exec_mock.await_args.args[0]
        self.assertEqual(passed_event["channel_position_size_pct"], 9.0)


class TestExecuteTelegramSignalStoresRealTakeProfits(unittest.IsolatedAsyncioTestCase):
    """_execute_telegram_signal() должен сохранять реальные цели канала
    (parsed_take_profits) в open_positions — иначе _check_position_exit
    не может передать их в _tp_levels() и всё равно скатывается к
    интерполяции одного числа."""

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    def setUp(self):
        self._saved_trading_mode = settings.trading_mode
        settings.trading_mode = "real"

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode

    async def test_open_positions_gets_take_profits_from_signal_event(self):
        bot = self._make_bot()
        bot._refresh_symbol_candles = AsyncMock()
        fake_order = MagicMock(id=1, fee=0.0)
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=fake_order)
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "long",
                "parsed_entry": 50000.0, "parsed_sl": 49000.0, "parsed_tp": 53000.0,
                "parsed_take_profits": [51000.0, 52000.0, 53000.0],
                "channel_id": "@test_channel",
            })

        self.assertEqual(
            bot.open_positions["BTC/USDT"]["take_profits"], [51000.0, 52000.0, 53000.0],
        )

    async def test_missing_parsed_take_profits_defaults_to_empty_list(self):
        bot = self._make_bot()
        bot._refresh_symbol_candles = AsyncMock()
        fake_order = MagicMock(id=1, fee=0.0)
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=fake_order)
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "long",
                "parsed_entry": 50000.0, "parsed_sl": 49000.0, "parsed_tp": 52000.0,
                "channel_id": "@test_channel",
            })

        self.assertEqual(bot.open_positions["BTC/USDT"]["take_profits"], [])


class TestExecuteTelegramSignalPassesParsedLeverage(unittest.IsolatedAsyncioTestCase):
    """Плечо, указанное каналом в тексте сигнала (parsed_leverage — см.
    extract_leverage в channel_monitor.py), должно доходить до
    execution_engine.create_order(leverage=...), а не теряться на пути."""

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    def setUp(self):
        self._saved_trading_mode = settings.trading_mode
        settings.trading_mode = "real"

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode

    async def test_parsed_leverage_forwarded_to_create_order(self):
        bot = self._make_bot()
        bot._refresh_symbol_candles = AsyncMock()
        fake_order = MagicMock(id=1, fee=0.0)
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=fake_order)
            mock_engine.get_open_positions = MagicMock(return_value={})
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "long",
                "parsed_entry": 50000.0, "parsed_sl": 49000.0, "parsed_tp": 53000.0,
                "parsed_leverage": 35.0,
                "channel_id": "@test_channel", "channel_market_type": "futures",
            })

        mock_engine.create_order.assert_awaited_once()
        self.assertEqual(mock_engine.create_order.await_args.kwargs["leverage"], 35.0)

    async def test_missing_parsed_leverage_forwards_none(self):
        bot = self._make_bot()
        bot._refresh_symbol_candles = AsyncMock()
        fake_order = MagicMock(id=1, fee=0.0)
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=fake_order)
            mock_engine.get_open_positions = MagicMock(return_value={})
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "long",
                "parsed_entry": 50000.0, "parsed_sl": 49000.0, "parsed_tp": 53000.0,
                "channel_id": "@test_channel",
            })

        mock_engine.create_order.assert_awaited_once()
        self.assertIsNone(mock_engine.create_order.await_args.kwargs["leverage"])


class TestOpenPositionAmountUsesExecutionEngineTrackedValue(unittest.IsolatedAsyncioTestCase):
    """
    Регресс на прод-инцидент: BCH/USDT (фьючерсы, реальный режим) — после
    открытия позиции bot.open_positions[symbol]["amount"] хранил ДО-ордерную
    оценку (position_value / entry), а не реально учтённый
    execution_engine объём (net_amount из _execute_real_order: filled
    минус комиссия, если она удержана в базовой валюте). Оба счётчика
    уменьшались на один и тот же close_amount при каждом частичном
    TP-закрытии, поэтому расхождение (~размер комиссии) сохранялось
    константным — пока попытка закрыть остаток целиком не начала
    систематически (каждый цикл, без остановки) падать на бирже с
    "Insufficient balance": бот пытался продать чуть больше, чем реально
    было открыто. Фикс — брать amount из execution_engine.get_open_positions()
    сразу после успешного create_order(), а не из локальной ДО-ордерной
    переменной.
    """

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    def setUp(self):
        self._saved_trading_mode = settings.trading_mode
        self._saved_active_trading_mode = settings.active_trading_mode
        settings.trading_mode = "real"

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode
        settings.active_trading_mode = self._saved_active_trading_mode

    async def test_telegram_signal_uses_tracked_amount_not_pre_order_estimate(self):
        bot = self._make_bot()
        bot._refresh_symbol_candles = AsyncMock()
        fake_order = MagicMock(id=1, fee=0.0018392)
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=fake_order)
            # Реально учтённый execution_engine объём (net_amount) —
            # заведомо меньше, чем то, что посчитает локальная
            # ДО-ордерная оценка (position_value / entry) ниже.
            mock_engine.get_open_positions = MagicMock(
                return_value={"BCH/USDT": {"amount": 0.457943102416035, "entry_price": 244.4}}
            )
            await bot._execute_telegram_signal({
                "parsed_pair": "BCH/USDT", "parsed_side": "long",
                "parsed_entry": 244.4, "parsed_sl": 241.2108, "parsed_tp": 251.62303,
                "channel_id": "@test_channel", "channel_market_type": "futures",
            })

        self.assertEqual(bot.open_positions["BCH/USDT"]["amount"], 0.457943102416035)

    async def test_telegram_signal_falls_back_to_estimate_if_not_tracked(self):
        """Если по какой-то причине execution_engine не знает об этой
        позиции (не должно происходить в норме) — не падаем, используем
        локальную ДО-ордерную оценку, как раньше."""
        bot = self._make_bot()
        bot._refresh_symbol_candles = AsyncMock()
        fake_order = MagicMock(id=1, fee=0.0)
        with patch("src.main.execution_engine") as mock_engine:
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            mock_engine.create_order = AsyncMock(return_value=fake_order)
            mock_engine.get_open_positions = MagicMock(return_value={})
            await bot._execute_telegram_signal({
                "parsed_pair": "BTC/USDT", "parsed_side": "long",
                "parsed_entry": 50000.0, "parsed_sl": 49000.0, "parsed_tp": 53000.0,
                "channel_id": "@test_channel",
            })

        expected_amount = (10000.0 * (5.0 / 100)) / 50000.0
        self.assertEqual(bot.open_positions["BTC/USDT"]["amount"], expected_amount)

    async def test_strategy_signal_uses_tracked_amount_not_pre_order_estimate(self):
        from src.strategy import StrategySignal

        settings.active_trading_mode = "algo"
        bot = self._make_bot()
        bot.feature_engine = MagicMock()
        bot.ml_inference = None
        candles_df = pd.DataFrame({
            "open": [1.0] * 60, "high": [1.0] * 60, "low": [1.0] * 60,
            "close": [1.0] * 60, "volume": [1.0] * 60,
        })
        bot._refresh_symbol_candles = AsyncMock(return_value=candles_df)

        fake_strategy = MagicMock()
        fake_strategy.strategy_id = "ml_direction"
        fake_strategy.name = "ML"
        fake_strategy.weight = 1.0
        fake_strategy.generate_signal.return_value = StrategySignal(
            strategy_id="ml_direction", symbol="BTC/USDT", side="long", confidence=0.66,
            entry_price=100.0,
        )
        fake_order = MagicMock(id=1, fee=0.0, client_order_id="abc")

        with patch("src.main.strategy_registry.get_active", return_value=[fake_strategy]), \
                patch("src.main.strategy_registry.get", return_value=None), \
                patch("src.main.execution_engine") as mock_engine, \
                patch("src.main.risk_manager") as mock_risk, \
                patch("src.main.protection_manager") as mock_protections, \
                patch("src.main.expectancy_sizing") as mock_sizing:
            mock_engine.paper_positions = {}
            mock_engine.real_positions = {}
            mock_engine.last_prices = {}
            mock_engine.create_order = AsyncMock(return_value=fake_order)
            mock_engine.get_real_balance = AsyncMock(return_value=10000.0)
            # Реально учтённый объём меньше локальной ДО-ордерной оценки —
            # именно он должен попасть в bot.open_positions.
            mock_engine.get_open_positions = MagicMock(
                return_value={"BTC/USDT": {"amount": 0.4, "entry_price": 100.0}}
            )
            mock_risk.check_signal.return_value = (True, "")
            mock_protections.locked_reason = AsyncMock(return_value=None)
            mock_sizing.size_multiplier = AsyncMock(return_value=1.0)

            await bot._process_symbol("BTC/USDT")

        self.assertEqual(bot.open_positions["BTC/USDT"]["amount"], 0.4)


class TestDecideTelegramSignal(unittest.IsolatedAsyncioTestCase):
    """
    POST /telegram/signals/{id}/decide — раньше "⏳ Ожидает подтверждения"
    (decision="pending": канал не в автоисполнении, но сигнал прошёл порог
    качества) было тупиковым статусом — TelegramSignalConfirm существовал
    как Pydantic-модель, но ни один endpoint её не использовал, и исполнить
    или отклонить такой сигнал вручную было негде.
    """

    def setUp(self):
        import src.main as main_module
        import src.web.api as api_module
        self.main_module = main_module
        self.api_module = api_module
        self._saved_trading_mode = settings.trading_mode
        self._saved_current_bot = main_module.current_bot
        self._saved_engine = api_module.execution_engine
        self._saved_main_engine = main_module.execution_engine

    def tearDown(self):
        settings.trading_mode = self._saved_trading_mode
        self.main_module.current_bot = self._saved_current_bot
        self.api_module.execution_engine = self._saved_engine
        self.main_module.execution_engine = self._saved_main_engine

    async def _install_engine_and_bot(self):
        from src.execution.executor import ExecutionEngine

        engine = ExecutionEngine()
        settings.trading_mode = "paper"
        engine.is_paper = True
        engine.exchange_id = "binance"
        self.api_module.execution_engine = engine
        # _execute_telegram_signal (main.py) обращается к своему СОБСТВЕННОМУ
        # импортированному имени execution_engine, а не к api_module'ному —
        # без этого ордер ушёл бы в исходный (не-paper, не наш) движок.
        self.main_module.execution_engine = engine

        bot = self.main_module.TradingBot()
        bot.ingest = AsyncMock()
        bot.ingest.fetch_ohlcv = AsyncMock(return_value=None)
        self.main_module.current_bot = bot
        return engine, bot

    async def _make_pending_signal(self, **overrides) -> int:
        import uuid

        from src.db.models import TelegramChannel, TelegramSignal
        from src.db.session import get_session
        from src.utils.timeutils import utcnow

        async with get_session() as session:
            channel = TelegramChannel(
                channel_id=f"@decide_test_channel_{uuid.uuid4().hex[:8]}", channel_title="X",
                quality_threshold=0.0, auto_execute=False, position_size_pct=6.0, active=True,
            )
            session.add(channel)
            await session.flush()
            signal = TelegramSignal(
                channel_id=channel.id,
                raw_message="test message", message_date=utcnow(),
                parsed_pair=overrides.get("parsed_pair", "BTC/USDT"),
                parsed_side=overrides.get("parsed_side", "long"),
                parsed_entry=overrides.get("parsed_entry", 50000.0),
                parsed_sl=overrides.get("parsed_sl", 49000.0),
                parsed_tp=overrides.get("parsed_tp", 52000.0),
                parsed_take_profits=overrides.get("parsed_take_profits"),
                quality_score=0.9,
                decision=overrides.get("decision", "pending"),
            )
            session.add(signal)
            await session.commit()
            return signal.id

    async def test_reject_marks_signal_rejected_without_executing(self):
        from src.web.api import TelegramSignalDecision, decide_telegram_signal
        from src.db.models import TelegramSignal
        from src.db.session import get_session

        await self._install_engine_and_bot()
        signal_id = await self._make_pending_signal()

        result = await decide_telegram_signal(signal_id, TelegramSignalDecision(action="reject"))

        self.assertEqual(result, {"success": True, "decision": "rejected"})
        async with get_session() as session:
            signal = await session.get(TelegramSignal, signal_id)
            self.assertEqual(signal.decision, "rejected")
            self.assertEqual(signal.reject_reason, "отклонён вручную")
        self.assertNotIn("BTC/USDT", self.main_module.current_bot.open_positions)

    async def test_execute_failure_persists_reject_reason(self):
        """
        Ручное исполнение через дашборд, которое не проходит на бирже
        (короткая позиция на споте — единственный детерминированный способ
        гарантированно провалить исполнение без реального обращения к
        бирже), должно записать конкретную причину в reject_reason, а не
        оставить её пустой — иначе "Итог сделки" для такого сигнала в
        дашборде снова был бы неотличим от простого "—".
        """
        from src.web.api import TelegramSignalDecision, decide_telegram_signal
        from src.db.models import TelegramSignal
        from src.db.session import get_session

        engine, bot = await self._install_engine_and_bot()
        engine.is_paper = False
        settings.trading_mode = "real"
        signal_id = await self._make_pending_signal(parsed_pair="SPOTSHORT1/USDT", parsed_side="short")

        with self.assertRaises(Exception):
            await decide_telegram_signal(signal_id, TelegramSignalDecision(action="execute"))

        async with get_session() as session:
            signal = await session.get(TelegramSignal, signal_id)
            self.assertEqual(signal.decision, "rejected")
            self.assertIn("шорт", signal.reject_reason)

    async def test_execute_opens_position_with_real_take_profits(self):
        from src.web.api import TelegramSignalDecision, decide_telegram_signal
        from src.db.models import TelegramSignal
        from src.db.session import get_session

        engine, bot = await self._install_engine_and_bot()
        signal_id = await self._make_pending_signal(
            parsed_pair="EXEC1/USDT", parsed_tp=53000.0,
            parsed_take_profits=[51000.0, 52000.0, 53000.0],
        )

        result = await decide_telegram_signal(signal_id, TelegramSignalDecision(action="execute"))

        self.assertTrue(result["success"])
        self.assertEqual(result["decision"], "executed")
        self.assertIn("EXEC1/USDT", bot.open_positions)
        self.assertEqual(
            bot.open_positions["EXEC1/USDT"]["take_profits"], [51000.0, 52000.0, 53000.0],
        )
        # Размер позиции взят из TelegramChannel.position_size_pct (6.0),
        # а не захардкоженных 5.0.
        self.assertIn("EXEC1/USDT", engine.paper_positions)
        async with get_session() as session:
            signal = await session.get(TelegramSignal, signal_id)
            self.assertEqual(signal.decision, "executed")
            self.assertIsNotNone(signal.executed_order_id)

    async def test_execute_rejects_when_position_already_open(self):
        from fastapi import HTTPException

        from src.web.api import TelegramSignalDecision, decide_telegram_signal

        _, bot = await self._install_engine_and_bot()
        bot.open_positions["BTC/USDT"] = {"side": "long", "amount": 1.0}
        signal_id = await self._make_pending_signal(parsed_pair="BTC/USDT")

        with self.assertRaises(HTTPException) as ctx:
            await decide_telegram_signal(signal_id, TelegramSignalDecision(action="execute"))
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_execute_fails_when_bot_not_ready(self):
        from fastapi import HTTPException

        from src.web.api import TelegramSignalDecision, decide_telegram_signal

        await self._install_engine_and_bot()
        self.main_module.current_bot = None
        signal_id = await self._make_pending_signal()

        with self.assertRaises(HTTPException) as ctx:
            await decide_telegram_signal(signal_id, TelegramSignalDecision(action="execute"))
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_already_decided_signal_cannot_be_decided_again(self):
        from fastapi import HTTPException

        from src.web.api import TelegramSignalDecision, decide_telegram_signal

        await self._install_engine_and_bot()
        signal_id = await self._make_pending_signal(decision="executed")

        with self.assertRaises(HTTPException) as ctx:
            await decide_telegram_signal(signal_id, TelegramSignalDecision(action="reject"))
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_unknown_signal_id_returns_404(self):
        from fastapi import HTTPException

        from src.web.api import TelegramSignalDecision, decide_telegram_signal

        await self._install_engine_and_bot()
        with self.assertRaises(HTTPException) as ctx:
            await decide_telegram_signal(999999999, TelegramSignalDecision(action="reject"))
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_invalid_action_returns_400(self):
        from fastapi import HTTPException

        from src.web.api import TelegramSignalDecision, decide_telegram_signal

        await self._install_engine_and_bot()
        signal_id = await self._make_pending_signal()
        with self.assertRaises(HTTPException) as ctx:
            await decide_telegram_signal(signal_id, TelegramSignalDecision(action="bogus"))
        self.assertEqual(ctx.exception.status_code, 400)


class TestDynamicChannelMonitoring(unittest.IsolatedAsyncioTestCase):
    """
    Раньше список отслеживаемых Telegram-каналов был жёстко зашит в
    chats=entities на момент регистрации @client.on(events.NewMessage(...))
    в monitor_channels() — добавление/удаление канала через дашборд
    применялось только после рестарта бота. Обработчик теперь
    регистрируется один раз без фильтра chats= и читает live-словарь
    channel_monitor._monitored — add_channel_to_monitoring()/
    remove_channel_from_monitoring() просто мутируют его.
    """

    def setUp(self):
        import src.telegram.channel_monitor as cm
        self.cm = cm
        self._saved_client = cm._telegram_client
        self._saved_monitored = dict(cm._monitored)
        self._saved_registered = cm._handler_registered
        cm._monitored.clear()
        cm._handler_registered = False

    def tearDown(self):
        self.cm._telegram_client = self._saved_client
        self.cm._monitored.clear()
        self.cm._monitored.update(self._saved_monitored)
        self.cm._handler_registered = self._saved_registered

    def _install_fake_client(self, entity_chat_id=-100123, raise_on_resolve=False):
        client = MagicMock()
        client.add_event_handler = MagicMock()
        if raise_on_resolve:
            client.get_entity = AsyncMock(side_effect=RuntimeError("no such channel"))
        else:
            fake_entity = MagicMock()
            client.get_entity = AsyncMock(return_value=fake_entity)
            self._patcher = patch("src.telegram.channel_monitor.get_peer_id", return_value=entity_chat_id)
            self._patcher.start()
            self.addCleanup(self._patcher.stop)
        self.cm._telegram_client = client
        return client

    async def test_add_channel_to_monitoring_resolves_and_registers(self):
        client = self._install_fake_client(entity_chat_id=-100999)

        added = await self.cm.add_channel_to_monitoring(
            {"channel_id": "@newchan", "channel_title": "New", "parser_config": {}}
        )

        self.assertTrue(added)
        self.assertIn(-100999, self.cm._monitored)
        self.assertEqual(self.cm._monitored[-100999]["channel_id"], "@newchan")
        client.add_event_handler.assert_called_once()
        self.assertTrue(self.cm._handler_registered)

    async def test_add_channel_by_numeric_id_resolves_via_int(self):
        """
        Реальный инцидент: поле в дашборде подсказывает вводить и числовой
        ID канала ("-100123456789"), но get_entity() у Telethon трактует
        str-аргумент как username и никогда не резолвит числовую строку —
        добавление канала по ID выглядело рабочим, но реально никогда не
        подключало канал. get_entity должен получить именно int.
        """
        client = self._install_fake_client(entity_chat_id=-100777)

        added = await self.cm.add_channel_to_monitoring(
            {"channel_id": "-100123456789", "channel_title": "ById", "parser_config": {}}
        )

        self.assertTrue(added)
        client.get_entity.assert_awaited_once_with(-100123456789)
        self.assertIn(-100777, self.cm._monitored)

    async def test_add_channel_returns_false_without_client(self):
        self.cm._telegram_client = None
        added = await self.cm.add_channel_to_monitoring(
            {"channel_id": "@newchan", "channel_title": "New", "parser_config": {}}
        )
        self.assertFalse(added)
        self.assertEqual(self.cm._monitored, {})

    async def test_add_channel_returns_false_when_resolve_fails(self):
        self._install_fake_client(raise_on_resolve=True)
        added = await self.cm.add_channel_to_monitoring(
            {"channel_id": "@badchan", "channel_title": "Bad", "parser_config": {}}
        )
        self.assertFalse(added)
        self.assertEqual(self.cm._monitored, {})

    async def test_handler_registered_exactly_once_across_multiple_adds(self):
        client = self._install_fake_client(entity_chat_id=-1)
        await self.cm.add_channel_to_monitoring({"channel_id": "@a", "channel_title": "A", "parser_config": {}})
        await self.cm.add_channel_to_monitoring({"channel_id": "@b", "channel_title": "B", "parser_config": {}})
        client.add_event_handler.assert_called_once()

    def test_remove_channel_from_monitoring_deletes_matching_entry(self):
        self.cm._monitored[-100111] = {"channel_id": "@toremove", "channel_title": "X"}
        self.cm._monitored[-100222] = {"channel_id": "@keepme", "channel_title": "Y"}

        removed_count = self.cm.remove_channel_from_monitoring("@toremove")

        self.assertEqual(removed_count, 1)
        self.assertNotIn(-100111, self.cm._monitored)
        self.assertIn(-100222, self.cm._monitored)

    def test_remove_unknown_channel_is_a_noop(self):
        removed_count = self.cm.remove_channel_from_monitoring("@nonexistent")
        self.assertEqual(removed_count, 0)

    async def test_handler_ignores_events_from_unmonitored_chats(self):
        event = MagicMock()
        event.chat_id = -100555
        event.message.text = "BTC/USDT LONG 50000 SL 49000 TP 52000"
        # _monitored пуст -> обработчик должен тихо выйти, не пытаясь
        # парсить/уведомлять подписчиков.
        with patch.object(self.cm, "parse_telegram_signal", new=AsyncMock()) as parse_mock:
            await self.cm._handler(event)
        parse_mock.assert_not_called()

    async def test_monitor_channels_registers_handler_even_with_zero_channels(self):
        """Без этого канал, добавленный ПОСЛЕ старта (без изначально
        настроенных каналов), не имел бы работающего обработчика вообще —
        monitor_channels([]) раньше выходил раньше регистрации."""
        client = self._install_fake_client()
        await self.cm.monitor_channels([])
        client.add_event_handler.assert_called_once()
        self.assertTrue(self.cm._handler_registered)


class TestTradingBotLiveTelegramChannelWiring(unittest.IsolatedAsyncioTestCase):
    """TradingBot.add_telegram_channel_to_live_monitoring/remove_telegram_
    channel_from_live_monitoring — держат main.py:_telegram_channel_db_ids
    (используется _get_channel_settings) в синхроне с channel_monitor._monitored."""

    def _make_bot(self):
        try:
            import src.main as main_module
        except ImportError as e:
            self.skipTest(f"src.main not importable in this environment: {e}")
        return main_module.TradingBot()

    async def test_successful_add_updates_db_id_mapping(self):
        bot = self._make_bot()
        with patch("src.main.add_channel_to_monitoring", new=AsyncMock(return_value=True)) as add_mock:
            result = await bot.add_telegram_channel_to_live_monitoring(
                channel_id="@livewire", channel_title="Live", parser_config={}, db_id=42,
            )
        self.assertTrue(result)
        self.assertEqual(bot._telegram_channel_db_ids["@livewire"], 42)
        add_mock.assert_awaited_once()

    async def test_failed_add_does_not_update_db_id_mapping(self):
        bot = self._make_bot()
        with patch("src.main.add_channel_to_monitoring", new=AsyncMock(return_value=False)):
            result = await bot.add_telegram_channel_to_live_monitoring(
                channel_id="@unresolvable", channel_title="X", parser_config={}, db_id=7,
            )
        self.assertFalse(result)
        self.assertNotIn("@unresolvable", bot._telegram_channel_db_ids)

    def test_remove_clears_db_id_mapping(self):
        bot = self._make_bot()
        bot._telegram_channel_db_ids["@livewire"] = 42
        with patch("src.main.remove_channel_from_monitoring") as remove_mock:
            bot.remove_telegram_channel_from_live_monitoring("@livewire")
        remove_mock.assert_called_once_with("@livewire")
        self.assertNotIn("@livewire", bot._telegram_channel_db_ids)


class TestTelegramChannelEndpointsWireLiveMonitoring(unittest.IsolatedAsyncioTestCase):
    """POST/DELETE /telegram/channels вызывают add_telegram_channel_to_live_
    monitoring/remove_telegram_channel_from_live_monitoring на current_bot,
    если он готов — раньше добавление/удаление канала не имело вообще
    никакого эффекта до рестарта бота."""

    def setUp(self):
        import src.main as main_module
        self.main_module = main_module
        self._saved_current_bot = main_module.current_bot

    def tearDown(self):
        self.main_module.current_bot = self._saved_current_bot

    async def test_create_channel_calls_live_wiring_and_reports_result(self):
        from src.web.api import TelegramChannelCreate, create_telegram_channel

        bot = MagicMock()
        bot.add_telegram_channel_to_live_monitoring = AsyncMock(return_value=True)
        self.main_module.current_bot = bot

        result = await create_telegram_channel(TelegramChannelCreate(
            channel_id="@wiring_test_channel",
        ))

        self.assertTrue(result["success"])
        self.assertTrue(result["live_monitoring"])
        bot.add_telegram_channel_to_live_monitoring.assert_awaited_once()
        call_kwargs = bot.add_telegram_channel_to_live_monitoring.await_args.kwargs
        self.assertEqual(call_kwargs["channel_id"], "@wiring_test_channel")
        self.assertEqual(call_kwargs["db_id"], result["channel"]["id"])

    async def test_create_channel_reports_failed_live_wiring(self):
        from src.web.api import TelegramChannelCreate, create_telegram_channel

        bot = MagicMock()
        bot.add_telegram_channel_to_live_monitoring = AsyncMock(return_value=False)
        self.main_module.current_bot = bot

        result = await create_telegram_channel(TelegramChannelCreate(
            channel_id="@wiring_test_channel_2",
        ))

        self.assertFalse(result["live_monitoring"])

    async def test_create_channel_without_ready_bot_reports_not_live(self):
        from src.web.api import TelegramChannelCreate, create_telegram_channel

        self.main_module.current_bot = None
        result = await create_telegram_channel(TelegramChannelCreate(
            channel_id="@wiring_test_channel_3",
        ))
        self.assertFalse(result["live_monitoring"])

    async def test_delete_channel_calls_live_wiring(self):
        from src.db.models import TelegramChannel
        from src.db.session import get_session
        from src.web.api import delete_telegram_channel

        async with get_session() as session:
            channel = TelegramChannel(channel_id="@to_be_deleted_wiring_test", active=True)
            session.add(channel)
            await session.commit()
            channel_db_id = channel.id

        bot = MagicMock()
        bot.remove_telegram_channel_from_live_monitoring = MagicMock()
        self.main_module.current_bot = bot

        result = await delete_telegram_channel(channel_db_id)

        self.assertTrue(result["success"])
        bot.remove_telegram_channel_from_live_monitoring.assert_called_once_with("@to_be_deleted_wiring_test")


if __name__ == "__main__":
    unittest.main()
