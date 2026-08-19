"""Тесты для крипто-трейдер бота."""
import asyncio
import unittest
from datetime import datetime
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


if __name__ == "__main__":
    unittest.main()
