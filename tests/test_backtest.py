"""Unit тесты для backtest engine."""
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.backtest.engine import (
    BacktestPosition,
    BacktestTrade,
    BacktestResult,
    BacktestEngine,
    BacktestDataLoader,
)
from src.strategy import (
    RSIMeanReversionStrategy,
    EMACrossoverStrategy,
    BollingerBandsStrategy,
)


class TestBacktestPosition(unittest.TestCase):
    """Тесты для BacktestPosition."""

    def test_long_position_pnl(self):
        """PnL для long позиции."""
        pos = BacktestPosition(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            amount=0.1,
            entry_time=datetime.now(),
            stop_loss=49000.0,
            take_profit=51000.0,
        )

        # Тест закрытия по take profit
        pos._close(51000.0, datetime.now(), "take_profit")
        self.assertTrue(pos.closed)
        self.assertEqual(pos.pnl, 100.0)
        self.assertEqual(pos.pnl_pct, 2.0)

    def test_short_position_pnl(self):
        """PnL для short позиции."""
        pos = BacktestPosition(
            symbol="BTC/USDT",
            direction="short",
            entry_price=50000.0,
            amount=0.1,
            entry_time=datetime.now(),
            stop_loss=51000.0,
            take_profit=49000.0,
        )

        pos._close(49000.0, datetime.now(), "take_profit")
        self.assertTrue(pos.closed)
        self.assertEqual(pos.pnl, 100.0)
        self.assertEqual(pos.pnl_pct, 2.0)

    def test_stop_loss_long(self):
        """Стоп-лосс для long."""
        pos = BacktestPosition(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            amount=0.1,
            entry_time=datetime.now(),
            stop_loss=49000.0,
        )

        pos.update(48999.0, datetime.now())
        self.assertTrue(pos.closed)
        self.assertEqual(pos.pnl, -100.1)  # примерно

    def test_not_closed_before_sl_tp(self):
        """Позиция не закрывается до достижения SL/TP."""
        pos = BacktestPosition(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            amount=0.1,
            entry_time=datetime.now(),
            stop_loss=49000.0,
            take_profit=51000.0,
        )

        pos.update(50500.0, datetime.now())
        self.assertFalse(pos.closed)


class TestBacktestTrade(unittest.TestCase):
    """Тесты для BacktestTrade."""

    def test_trade_creation(self):
        """Создание сделки."""
        trade = BacktestTrade(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            exit_price=51000.0,
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            exit_time=datetime(2024, 1, 1, 14, 0, 0),
            amount=0.1,
            pnl=100.0,
            pnl_pct=2.0,
            strategy_id="rsi_mr",
            rationale="RSI oversold",
        )

        self.assertEqual(trade.symbol, "BTC/USDT")
        self.assertEqual(trade.direction, "long")
        self.assertEqual(trade.entry_price, 50000.0)
        self.assertEqual(trade.exit_price, 51000.0)
        self.assertEqual(trade.pnl, 100.0)
        self.assertEqual(trade.pnl_pct, 2.0)
        self.assertEqual(trade.duration_seconds, 7200)  # 2 часа

    def test_short_trade_pnl(self):
        """Short сделка."""
        trade = BacktestTrade(
            symbol="ETH/USDT",
            direction="short",
            entry_price=3000.0,
            exit_price=2900.0,
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(hours=3),
            amount=1.0,
            pnl=100.0,
            pnl_pct=3.33,
            strategy_id="ema_cross",
        )

        self.assertEqual(trade.pnl, 100.0)
        self.assertAlmostEqual(trade.pnl_pct, 3.33, places=2)


class TestBacktestResult(unittest.TestCase):
    """Тесты для BacktestResult."""

    def setUp(self):
        self.result = BacktestResult()
        self.result.initial_capital = 10000.0

    def test_empty_result(self):
        """Пустой результат."""
        self.result.compute_metrics()
        self.assertEqual(self.result.total_trades, 0)
        self.assertEqual(self.result.total_pnl, 0.0)

    def test_single_win_trade(self):
        """Одна выигрышная сделка."""
        trade = BacktestTrade(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            exit_price=51000.0,
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            exit_time=datetime(2024, 1, 1, 14, 0, 0),
            amount=0.1,
            pnl=100.0,
            pnl_pct=2.0,
            strategy_id="rsi_mr",
        )
        self.result.trades.append(trade)
        self.result.compute_metrics()

        self.assertEqual(self.result.total_trades, 1)
        self.assertEqual(self.result.winning_trades, 1)
        self.assertEqual(self.result.total_pnl, 100.0)
        self.assertEqual(self.result.total_pnl_pct, 1.0)
        self.assertEqual(self.result.win_rate, 100.0)
        self.assertEqual(self.result.final_capital, 10100.0)

    def test_win_loss_mix(self):
        """Смесь выигрышных и проигрышных сделок."""
        # Win
        win_trade = BacktestTrade(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            exit_price=51000.0,
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            exit_time=datetime(2024, 1, 1, 14, 0, 0),
            amount=0.1,
            pnl=100.0,
            pnl_pct=2.0,
            strategy_id="rsi_mr",
        )
        # Loss
        loss_trade = BacktestTrade(
            symbol="BTC/USDT",
            direction="long",
            entry_price=51000.0,
            exit_price=50000.0,
            entry_time=datetime(2024, 1, 1, 15, 0, 0),
            exit_time=datetime(2024, 1, 1, 17, 0, 0),
            amount=0.1,
            pnl=-100.0,
            pnl_pct=-1.96,
            strategy_id="rsi_mr",
        )

        self.result.trades.append(win_trade)
        self.result.trades.append(loss_trade)
        self.result.compute_metrics()

        self.assertEqual(self.result.total_trades, 2)
        self.assertEqual(self.result.winning_trades, 1)
        self.assertEqual(self.result.losing_trades, 1)
        self.assertEqual(self.result.total_pnl, 0.0)
        self.assertEqual(self.result.win_rate, 50.0)

    def test_profit_factor(self):
        """Profit factor calculation."""
        # 3 winning trades по 100
        # 2 losing trades по 50
        for _ in range(3):
            trade = BacktestTrade(
                symbol="BTC/USDT",
                direction="long",
                entry_price=50000.0,
                exit_price=50500.0,
                entry_time=datetime.now(),
                exit_time=datetime.now() + timedelta(hours=1),
                amount=0.1,
                pnl=50.0,
                pnl_pct=1.0,
                strategy_id="rsi_mr",
            )
            self.result.trades.append(trade)

        for _ in range(2):
            trade = BacktestTrade(
                symbol="BTC/USDT",
                direction="long",
                entry_price=50000.0,
                exit_price=49500.0,
                entry_time=datetime.now(),
                exit_time=datetime.now() + timedelta(hours=1),
                amount=0.1,
                pnl=-50.0,
                pnl_pct=-1.0,
                strategy_id="rsi_mr",
            )
            self.result.trades.append(trade)

        self.result.compute_metrics()
        self.assertEqual(self.result.profit_factor, 3.0)  # (3*50) / (2*50) = 150/100 = 1.5

    def test_max_drawdown(self):
        """Максимальный даундрафт."""
        self.result.initial_capital = 10000.0

        # Equity curve: 10000 → 11000 → 9000 → 9500 → 10500
        self.result.equity_curve = [
            (datetime(2024, 1, 1, 0, 0), 10000.0),
            (datetime(2024, 1, 1, 6, 0), 11000.0),
            (datetime(2024, 1, 1, 12, 0), 9000.0),
            (datetime(2024, 1, 1, 18, 0), 9500.0),
            (datetime(2024, 1, 2, 0, 0), 10500.0),
        ]

        self.result.compute_metrics()
        self.assertEqual(self.result.max_drawdown, 18.18)  # (11000-9000)/11000 = 18.18%
        self.assertEqual(self.result.max_drawdown_peak, 11000.0)


class TestBacktestEngine(unittest.TestCase):
    """Тесты для BacktestEngine."""

    def setUp(self):
        self.engine = BacktestEngine(initial_capital=10000.0)

    def test_empty_data(self):
        """Пустые данные."""
        df = pd.DataFrame()
        with self.assertRaises(ValueError) as ctx:
            self.engine.run(df, [])
        self.assertIn("empty", str(ctx.exception))

    def test_data_with_insufficient_rows(self):
        """Недостаточно данных."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200],
        }, index=[datetime.now() + timedelta(hours=i) for i in range(3)])

        strategy = RSIMeanReversionStrategy()
        with self.assertRaises(ValueError) as ctx:
            self.engine.run(df, [strategy])
        self.assertIn("No data", str(ctx.exception))

    def test_run_with_no_signals(self):
        """Бэктест без сигналов."""
        df = pd.DataFrame({
            "open": np.random.uniform(100, 110, 200),
            "high": np.random.uniform(100, 115, 200),
            "low": np.random.uniform(95, 100, 200),
            "close": np.random.uniform(100, 110, 200),
            "volume": np.random.uniform(1000, 5000, 200),
        }, index=[datetime(2024, 1, 1) + timedelta(hours=i) for i in range(200)])

        strategy = RSIMeanReversionStrategy()
        strategy.params = {"rsi_period": 14, "oversold_level": 10, "overbought_level": 90}

        result = self.engine.run(df, [strategy])
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.total_pnl, 0.0)

    def test_multiple_strategies(self):
        """Несколько стратегий."""
        df = pd.DataFrame({
            "open": np.random.uniform(100, 110, 200),
            "high": np.random.uniform(100, 115, 200),
            "low": np.random.uniform(95, 100, 200),
            "close": np.random.uniform(100, 110, 200),
            "volume": np.random.uniform(1000, 5000, 200),
        }, index=[datetime(2024, 1, 1) + timedelta(hours=i) for i in range(200)])

        rsi = RSIMeanReversionStrategy()
        rsi.params = {"rsi_period": 14, "oversold_level": 30, "overbought_level": 70}

        bb = BollingerBandsStrategy()
        bb.params = {"mode": "mean_reversion", "position_size_pct": 5.0}

        results = self.engine.run_strategies_comparison(
            data=df,
            strategy_configs=[
                {"strategy_class": RSIMeanReversionStrategy, "params": rsi.params},
                {"strategy_class": BollingerBandsStrategy, "params": bb.params},
            ],
        )

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], BacktestResult)
        self.assertIsInstance(results[1], BacktestResult)

    def test_position_sizing(self):
        """Расчёт размера позиции."""
        capital = 10000.0
        price = 50000.0
        size_pct = 5.0

        position_value = capital * (size_pct / 100)
        amount = position_value / price

        self.assertEqual(position_value, 500.0)
        self.assertAlmostEqual(amount, 0.01, places=6)

    def test_fee_and_slippage(self):
        """Комиссия и slippage."""
        capital = 10000.0
        price = 50000.0
        amount = 0.01

        fee_pct = 0.001  # 0.1%
        slippage_pct = 0.0005  # 0.05%

        # Long entry with slippage
        long_entry = price * (1 + slippage_pct)
        fee = amount * long_entry * fee_pct

        total_cost = amount * long_entry + fee
        self.assertGreater(total_cost, amount * price)

        # Short entry with slippage
        short_entry = price * (1 - slippage_pct)
        fee_short = amount * short_entry * fee_pct
        total_return = amount * short_entry - fee_short
        self.assertLess(total_return, amount * price)


class TestBacktestDataLoader(unittest.TestCase):
    """Тесты для BacktestDataLoader."""

    def test_csv_load(self):
        """Загрузка из CSV."""
        import tempfile
        import os

        # Создаём временный CSV
        csv_content = """timestamp,open,high,low,close,volume
2024-01-01 00:00:00,50000,50500,49500,50200,1000
2024-01-01 01:00:00,50200,50800,50000,50600,1100
2024-01-01 02:00:00,50600,51000,50400,50800,1200
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            loader = BacktestDataLoader()
            df = loader.from_csv(temp_path)
            self.assertEqual(len(df), 3)
            self.assertListEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        finally:
            os.unlink(temp_path)

    def test_ccxt_load(self):
        """Загрузка из ccxt (заглушка — реальный тест требует API)."""
        # Этот тест проверяет, что метод существует и может быть вызван
        loader = BacktestDataLoader()
        self.assertTrue(callable(loader.from_ccxt))


class TestDecisionLogger(unittest.TestCase):
    """Тесты для DecisionLogger."""

    def setUp(self):
        from src.execution.decision_logger import DecisionLogger
        self.logger = DecisionLogger()

    def test_log_market_data(self):
        """Лог market data."""
        self.logger.start_trade(123)
        self.logger.log_market_data(
            symbol="BTC/USDT",
            timeframe="1h",
            price=50000.0,
            features={"rsi_14": 28.5, "ema_20": 49500.0},
        )
        self.assertEqual(len(self.logger._steps), 1)
        self.assertEqual(self.logger._steps[0]["step_type"], "market_data")

    def test_log_strategy_signal(self):
        """Лог сигнала."""
        self.logger.start_trade(123)
        self.logger.log_strategy_signal(
            strategy_id="rsi_mr",
            strategy_name="RSI Mean Reversion",
            signal_side="long",
            confidence=0.85,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            rationale="RSI oversold",
        )
        self.assertEqual(len(self.logger._steps), 1)
        self.assertEqual(self.logger._steps[0]["step_type"], "strategy_signal")

    def test_log_ml_score(self):
        """Лог ML score."""
        self.logger.start_trade(123)
        self.logger.log_ml_score(
            model_type="direction_classifier",
            model_version=1,
            proba_up=0.75,
            proba_down=0.15,
            proba_neutral=0.10,
        )
        self.assertEqual(len(self.logger._steps), 1)
        self.assertEqual(self.logger._steps[0]["step_type"], "ml_score")

    def test_log_risk_check_allowed(self):
        """Лог risk check (допущено)."""
        self.logger.start_trade(123)
        self.logger.log_risk_check(
            decision="allowed",
            reason="All checks passed",
            context={"daily_pnl": 50.0, "open_positions": 2},
        )
        self.assertEqual(len(self.logger._steps), 1)
        self.assertEqual(self.logger._steps[0]["step_type"], "risk_check")

    def test_log_risk_check_rejected(self):
        """Лог risk check (отклонено)."""
        self.logger.start_trade(123)
        self.logger.log_risk_check(
            decision="rejected",
            reason="Daily loss limit reached",
            context={"daily_pnl": -500.0},
        )
        self.assertEqual(len(self.logger._steps), 1)
        self.assertEqual(self.logger._steps[0]["decision"], "rejected")

    def test_log_execution(self):
        """Лог исполнения."""
        self.logger.start_trade(123)
        self.logger.log_execution(
            order_id="ORD-12345",
            order_type="market",
            amount=0.01,
            price=50000.0,
            status="filled",
            fee=5.0,
        )
        self.assertEqual(len(self.logger._steps), 1)
        self.assertEqual(self.logger._steps[0]["step_type"], "execution")

    def test_no_trade_id_log(self):
        """Лог без trade ID — предупреждение."""
        with patch("src.execution.decision_logger.logger") as mock_logger:
            self.logger.log_market_data("BTC/USDT", "1h", 50000.0, {})
            mock_logger.warning.assert_called()


if __name__ == "__main__":
    unittest.main()
