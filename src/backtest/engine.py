"""Backtest Engine — тестирование стратегий на исторических данных."""
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Optional

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.config import settings
from src.strategy import StrategySignal, BaseStrategy, _fmt_price
from src.risk.risk_manager import RiskManager
from src.utils.logging import logger

console = Console()
logger = logging.getLogger(__name__)


class BacktestPosition:
    """Позиция в бэктесте."""

    def __init__(
        self,
        symbol: str,
        direction: str,  # long, short
        entry_price: float,
        amount: float,
        entry_time: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_id: str = "",
        rationale: str = "",
    ):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.amount = amount  # количество монет
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy_id = strategy_id
        self.rationale = rationale
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.pnl: Optional[float] = None
        self.pnl_pct: Optional[float] = None
        self.closed = False
        self.trades: list[dict] = []  # сделки внутри позиции (для частичного закрытия)

    def update(self, current_price: float, current_time: datetime):
        """Обновить позицию по текущей цене."""
        if self.closed:
            return

        # Проверка Stop Loss
        if self.stop_loss is not None:
            if self.direction == "long" and current_price <= self.stop_loss:
                self._close(current_price, current_time, "stop_loss")
                return
            elif self.direction == "short" and current_price >= self.stop_loss:
                self._close(current_price, current_time, "stop_loss")
                return

        # Проверка Take Profit
        if self.take_profit is not None:
            if self.direction == "long" and current_price >= self.take_profit:
                self._close(current_price, current_time, "take_profit")
                return
            elif self.direction == "short" and current_price <= self.take_profit:
                self._close(current_price, current_time, "take_profit")
                return

    def _close(self, exit_price: float, exit_time: datetime, reason: str):
        """Закрыть позицию."""
        self.exit_price = exit_price
        self.exit_time = exit_time

        if self.direction == "long":
            self.pnl = (exit_price - self.entry_price) * self.amount
        else:
            self.pnl = (self.entry_price - exit_price) * self.amount

        self.pnl_pct = (self.pnl / (self.entry_price * self.amount)) * 100
        self.closed = True
        self.trades.append({
            "type": "close",
            "price": exit_price,
            "time": exit_time,
            "reason": reason,
            "pnl": self.pnl,
        })

    @property
    def unrealized_pnl(self) -> float:
        """Незакрытый PnL (для открытых позиций)."""
        if self.closed:
            return self.pnl
        # Текущая цена — последняя известная цена (вычисляется externally)
        return 0.0


class BacktestTrade:
    """Результат одной сделки в бэктесте."""

    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        amount: float,
        pnl: float,
        pnl_pct: float,
        strategy_id: str,
        rationale: str = "",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.amount = amount
        self.pnl = pnl
        self.pnl_pct = pnl_pct
        self.strategy_id = strategy_id
        self.rationale = rationale
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    @property
    def duration_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()


class BacktestResult:
    """Результат бэктеста."""

    def __init__(self):
        self.trades: list[BacktestTrade] = []
        self.positions: list[BacktestPosition] = []
        self.initial_capital: float = 0.0
        self.final_capital: float = 0.0
        self.total_pnl: float = 0.0
        self.total_pnl_pct: float = 0.0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.break_even_trades: int = 0
        self.max_drawdown: float = 0.0
        self.max_drawdown_start: Optional[datetime] = None
        self.max_drawdown_end: Optional[datetime] = None
        self.max_drawdown_peak: float = 0.0
        self.equity_curve: list[tuple[datetime, float]] = []
        self.sharpe_ratio: float = 0.0
        self.win_rate: float = 0.0
        self.profit_factor: float = 0.0
        self.average_win: float = 0.0
        self.average_loss: float = 0.0
        self.average_trade_pnl: float = 0.0
        self.expectancy: float = 0.0
        self.total_trades: int = 0
        self.duration: timedelta = timedelta(0)
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.running_capital: float = 0.0

    def compute_metrics(self):
        """Вычислить все метрики на основе trades и equity curve."""
        # Даундрафт считается по equity curve и не зависит от списка сделок
        # (может быть заполнена даже без единой закрытой сделки)
        self._compute_max_drawdown()

        if not self.trades:
            return

        self.total_trades = len(self.trades)
        self.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        self.losing_trades = sum(1 for t in self.trades if t.pnl < 0)
        self.break_even_trades = sum(1 for t in self.trades if t.pnl == 0)

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        self.winning_trades_pnl = sum(t.pnl for t in self.trades if t.pnl > 0)
        self.losing_trades_pnl = sum(t.pnl for t in self.trades if t.pnl < 0)
        self.average_win = self.winning_trades_pnl / self.winning_trades if self.winning_trades > 0 else 0
        self.average_loss = abs(self.losing_trades_pnl) / self.losing_trades if self.losing_trades > 0 else 0
        self.average_trade_pnl = sum(t.pnl for t in self.trades) / self.total_trades

        self.total_pnl = sum(t.pnl for t in self.trades)
        self.total_pnl_pct = (self.total_pnl / self.initial_capital) * 100
        self.final_capital = self.initial_capital + self.total_pnl

        self.win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # Expectancy = (WinRate * AvgWin) - (LossRate * AvgLoss)
        win_rate_decimal = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        loss_rate_decimal = self.losing_trades / self.total_trades if self.total_trades > 0 else 0
        self.expectancy = (win_rate_decimal * self.average_win) - (loss_rate_decimal * self.average_loss)

        # Sharpe ratio (упрощённый — на основе retired trades)
        if self.total_trades > 1:
            returns = [t.pnl_pct for t in self.trades]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                # Annualize: предполагаем, что trades распределены равномерно за период
                years = self.duration.total_seconds() / (365.25 * 24 * 3600)
                if years > 0:
                    self.sharpe_ratio = (mean_return / std_return) * np.sqrt(252 / max(1, years * 252))
                else:
                    self.sharpe_ratio = 0.0
            else:
                self.sharpe_ratio = 0.0

    def _compute_max_drawdown(self):
        """Вычислить максимальный даундрафт на equity curve."""
        if not self.equity_curve:
            return

        peak = self.equity_curve[0][1]
        peak_time = self.equity_curve[0][0]
        max_dd = 0.0
        max_dd_start = self.equity_curve[0][0]
        max_dd_end = self.equity_curve[0][0]
        max_dd_peak = peak

        for time, equity in self.equity_curve:
            if equity > peak:
                peak = equity
                peak_time = time

            dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                max_dd_start = peak_time
                max_dd_end = time
                max_dd_peak = peak

        self.max_drawdown = round(max_dd, 2)
        self.max_drawdown_start = max_dd_start
        self.max_drawdown_end = max_dd_end
        self.max_drawdown_peak = max_dd_peak


class BacktestEngine:
    """
    Движок для бэктеста стратегий на исторических данных.

    Поддерживает:
    - Много стратегий одновременно
    - Риск-менеджмент (аналогично live)
    - Разные таймфреймы
    - Исторические данные из CSV или DataFrame
    - Подробные отчёты с метриками
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_manager: Optional[RiskManager] = None,
        fee_pct: float = 0.1,  # 0.1% комиссия
        slippage_pct: float = 0.05,  # 0.05% slippage
    ):
        self.initial_capital = initial_capital
        self.risk_manager = risk_manager or RiskManager()
        self.fee_pct = fee_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.results: list[BacktestResult] = []
        self._running_capital = initial_capital

    def run(
        self,
        data: pd.DataFrame,
        strategies: list[BaseStrategy],
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        position_size_pct: float = 5.0,
    ) -> BacktestResult:
        """
        Запустить бэктест.

        data: DataFrame с колонками [open, high, low, close, volume], индекс — datetime
        strategies: список стратегий для тестирования
        symbol: торговая пара
        timeframe: таймфрейм
        start_date, end_date: фильтр по дате
        position_size_pct: размер позиции (% от капитала)
        """
        if data.empty:
            raise ValueError("Data is empty")

        # Фильтрация по дате
        if start_date or end_date:
            data = data.copy()
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]

        if data.empty:
            raise ValueError("No data in selected date range")

        min_rows = 50  # минимум свечей для расчёта индикаторов (см. _extract_features)
        if len(data) < min_rows:
            raise ValueError(
                f"No data: недостаточно свечей для бэктеста ({len(data)} < {min_rows})"
            )

        logger.info(
            f"Запуск бэктеста: {symbol} {timeframe} | "
            f"{len(data)} свечей | {start_date or data.index[0]} — {end_date or data.index[-1]} | "
            f"стратегий: {len(strategies)} | капитал: ${self.initial_capital:,.0f}"
        )

        result = BacktestResult()
        result.initial_capital = self.initial_capital
        result.start_time = data.index[0]
        result.end_time = data.index[-1]
        result.duration = result.end_time - result.start_time

        self._running_capital = self.initial_capital
        open_positions: dict[str, BacktestPosition] = {}

        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row["close"]
            current_time = timestamp

            # Обновляем equity curve
            unrealized_pnl = sum(
                (current_price - pos.entry_price) * pos.amount if pos.direction == "long"
                else (pos.entry_price - current_price) * pos.amount
                for pos in open_positions.values()
            )
            equity = self._running_capital + unrealized_pnl
            result.equity_curve.append((current_time, equity))

            # Проверяем открытые позиции (стопы, тейк-профиты)
            for symbol_pos in list(open_positions.values()):
                symbol_pos.update(current_price, current_time)
                if symbol_pos.closed:
                    trade = BacktestTrade(
                        symbol=symbol_pos.symbol,
                        direction=symbol_pos.direction,
                        entry_price=symbol_pos.entry_price,
                        exit_price=symbol_pos.exit_price,
                        entry_time=symbol_pos.entry_time,
                        exit_time=symbol_pos.exit_time,
                        amount=symbol_pos.amount,
                        pnl=symbol_pos.pnl,
                        pnl_pct=symbol_pos.pnl_pct,
                        strategy_id=symbol_pos.strategy_id,
                        rationale=symbol_pos.rationale,
                        stop_loss=symbol_pos.stop_loss,
                        take_profit=symbol_pos.take_profit,
                    )
                    result.trades.append(trade)
                    # Обновляем капитал
                    self._running_capital += trade.pnl
                    # Удаляем позицию
                    del open_positions[symbol_pos.symbol]
                    logger.debug(
                        f"  Закрытие позиции: {symbol_pos.symbol} {symbol_pos.direction} | "
                        f"вход {symbol_pos.entry_price:.2f} → выход {symbol_pos.exit_price:.2f} | "
                        f"PnL: {trade.pnl:+.2f} ({trade.pnl_pct:+.2f}%) | {trade.exit_time}"
                    )

            # Генерируем сигналы от стратегий
            for strategy in strategies:
                if not strategy.active:
                    continue

                # Prepare data for strategy
                features = self._extract_features(data, i, strategy.params)
                features["symbol"] = symbol
                features["timeframe"] = timeframe
                features["close"] = current_price

                # Добавляем ML features если есть
                # (в реальном бэктесте — загрузить исторические ML predictions)

                signal = strategy.generate_signal(features)
                if signal is None:
                    continue

                # Риск-менеджмент
                if len(open_positions) >= self.risk_manager.profile.max_open_positions:
                    logger.debug(f"  Пропуск сигнала {signal.symbol}: достигнут лимит позиций")
                    continue

                if signal.symbol in open_positions:
                    logger.debug(f"  Пропуск сигнала {signal.symbol}: позиция уже открыта")
                    continue

                # Размер позиции
                position_value = self._running_capital * (position_size_pct / 100)
                amount = position_value / current_price

                # С учётом комиссии и slippage
                entry_price = current_price
                if signal.side == "long":
                    entry_price = current_price * (1 + self.slippage_pct)
                else:
                    entry_price = current_price * (1 - self.slippage_pct)

                # Размер позиции с учётом slippage
                amount = (self._running_capital * (position_size_pct / 100)) / entry_price

                # Создаём позицию
                position = BacktestPosition(
                    symbol=signal.symbol,
                    direction=signal.side,
                    entry_price=entry_price,
                    amount=amount,
                    entry_time=current_time,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy_id=strategy.strategy_id,
                    rationale=signal.rationale,
                )
                open_positions[signal.symbol] = position
                logger.debug(
                    f"  Открытие позиции: {signal.side.upper()} {signal.symbol} | "
                    f"цена {entry_price:.2f} | количество {amount:.6f} | "
                    f"SL {_fmt_price(signal.stop_loss)} TP {_fmt_price(signal.take_profit)} | "
                    f"{signal.rationale[:80]}"
                )

        # Закрываем все открытые позиции в конце периода (по последней цене)
        last_price = data.iloc[-1]["close"]
        last_time = data.index[-1]
        for position in list(open_positions.values()):
            position._close(last_price, last_time, "end_of_test")
            trade = BacktestTrade(
                symbol=position.symbol,
                direction=position.direction,
                entry_price=position.entry_price,
                exit_price=position.exit_price,
                entry_time=position.entry_time,
                exit_time=position.exit_time,
                amount=position.amount,
                pnl=position.pnl,
                pnl_pct=position.pnl_pct,
                strategy_id=position.strategy_id,
                rationale=position.rationale,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
            )
            result.trades.append(trade)
            self._running_capital += trade.pnl

        result.final_capital = self._running_capital
        result.compute_metrics()

        self.results.append(result)
        logger.info(
            f"Бэктест завершён: {len(result.trades)} сделок | "
            f" PnL: {result.total_pnl:+.2f} ({result.total_pnl_pct:+.2f}%) | "
            f" Win rate: {result.win_rate:.1f}% | "
            f" Profit factor: {result.profit_factor:.2f} | "
            f" Sharpe: {result.sharpe_ratio:.2f} | "
            f" Max DD: {result.max_drawdown:.1f}%"
        )

        return result

    def _extract_features(
        self,
        data: pd.DataFrame,
        current_index: int,
        params: dict,
    ) -> dict:
        """Извлечь фичи для стратегии из history data."""
        from src.data_ingest.feature_engine import get_feature_engine

        # Берём историю до current_index (не включая текущую свечу, чтобы избежать lookahead)
        history = data.iloc[:current_index].copy()
        if len(history) < 50:
            return {}

        engine = get_feature_engine()
        features_df = engine.compute_all_indicators(history)
        latest = features_df.iloc[-1]

        features = {}
        for col in features_df.columns:
            val = latest[col]
            if isinstance(val, (int, float, np.floating)):
                if not np.isnan(val):
                    features[col] = float(val)

        return features

    def run_strategies_comparison(
        self,
        data: pd.DataFrame,
        strategy_configs: list[dict],
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        position_size_pct: float = 5.0,
    ) -> list[BacktestResult]:
        """
        Сравнить несколько стратегий на одних и тех же данных.

        strategy_configs: список dict с {strategy_class, params, active}
        """
        results = []
        for config in strategy_configs:
            strategy = config["strategy_class"](config.get("params"))
            strategy.active = config.get("active", True)
            result = self.run(
                data=data,
                strategies=[strategy],
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                position_size_pct=position_size_pct,
            )
            results.append(result)
        return results

    def print_comparison(self, results: list[BacktestResult], strategies_names: list[str]):
        """Вывести сравнение результатов нескольких стратегий."""
        console.print("\n[bold]📊 Сравнение стратегий[/bold]\n")

        table = Table(
            title="Backtest Results Comparison",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Стратегия", style="bold white")
        table.add_column("Сделок", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("PnL $", justify="right")
        table.add_column("PnL %", justify="right")
        table.add_column("Profit Factor", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Max DD %", justify="right")
        table.add_column("Expectancy", justify="right")

        for name, result in zip(strategies_names, results):
            color = "green" if result.total_pnl > 0 else "red"
            table.add_row(
                name,
                str(result.total_trades),
                f"{result.win_rate:.1f}%",
                f"${result.total_pnl:,.2f}",
                f"{result.total_pnl_pct:+.2f}%",
                f"{result.profit_factor:.2f}",
                f"{result.sharpe_ratio:.2f}",
                f"{result.max_drawdown:.1f}%",
                f"${result.expectancy:,.2f}",
            )

        console.print(table)

        # Equity curve comparison
        console.print("\n[bold]📈 Equity Curves[/bold]\n")
        for name, result in zip(strategies_names, results):
            if result.equity_curve:
                equity_df = pd.DataFrame(result.equity_curve, columns=["time", "equity"])
                # Вычисляем returns
                equity_df["return"] = equity_df["equity"].pct_change()
                console.print(
                    f"[cyan]{name}:[/cyan] "
                    f"Начальный капитал: ${result.initial_capital:,.2f} → "
                    f"Конечный капитал: ${result.final_capital:,.2f} | "
                    f"Макс DD: {result.max_drawdown:.1f}%"
                )


class BacktestDataLoader:
    """Загрузчик исторических данных для бэктеста."""

    @staticmethod
    def from_csv(filepath: str, symbol: str = "BTC/USDT") -> pd.DataFrame:
        """Загрузить данные из CSV."""
        df = pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")
        df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }, inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    @staticmethod
    def from_ccxt(
        exchange_id: str = "binance",
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Загрузить данные из ccxt (исторические свечи)."""
        import ccxt.async_support as ccxt

        async def _load():
            exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            await exchange.close()

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            return df[["open", "high", "low", "close", "volume"]]

        return asyncio.run(_load())


# === Пример использования ===

def run_example_backtest():
    """Пример бэктеста RSI стратегии на BTC данных."""
    import ccxt.async_support as ccxt

    async def _example():
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = await exchange.fetch_ohlcv("BTC/USDT", "1h", limit=1000)
        await exchange.close()

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        # Стратегии
        rsi_strategy = RSIMeanReversionStrategy()
        rsi_strategy.params = {
            "rsi_period": 14,
            "oversold_level": 30,
            "overbought_level": 70,
            "position_size_pct": 5.0,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }

        bb_strategy = BollingerBandsStrategy()
        bb_strategy.params = {
            "mode": "mean_reversion",
            "position_size_pct": 5.0,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }

        ema_strategy = EMACrossoverStrategy()
        ema_strategy.params = {
            "fast_ema_period": 9,
            "slow_ema_period": 21,
            "position_size_pct": 5.0,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }

        engine = BacktestEngine(initial_capital=10000.0)

        # Запускаем сравнение
        results = engine.run_strategies_comparison(
            data=df,
            strategy_configs=[
                {"strategy_class": RSIMeanReversionStrategy, "params": rsi_strategy.params},
                {"strategy_class": BollingerBandsStrategy, "params": bb_strategy.params},
                {"strategy_class": EMACrossoverStrategy, "params": ema_strategy.params},
            ],
            symbol="BTC/USDT",
            timeframe="1h",
            position_size_pct=5.0,
        )

        engine.print_comparison(results, ["RSI Mean Reversion", "Bollinger Bands", "EMA Crossover"])

        # Печатаем детальный отчёт для лучшей стратегии
        best_result = max(results, key=lambda r: r.total_pnl)
        console.print(f"\n[bold green]🥇 Лучшая стратегия: {best_result.total_pnl:+.2f} USD[/bold green]\n")

        # Печатаем все сделки
        table = Table(title="Все сделки", box=box.ROUNDED)
        table.add_column("Time", style="bold white")
        table.add_column("Symbol")
        table.add_column("Dir")
        table.add_column("Entry")
        table.add_column("Exit")
        table.add_column("PnL $")
        table.add_column("PnL %")
        table.add_column("Strategy")
        table.add_column("Rationale")

        for trade in best_result.trades:
            table.add_row(
                str(trade.entry_time),
                trade.symbol,
                trade.direction.upper(),
                f"${trade.entry_price:.2f}",
                f"${trade.exit_price:.2f}",
                f"${trade.pnl:+,.2f}",
                f"{trade.pnl_pct:+.2f}%",
                trade.strategy_id,
                trade.rationale[:60],
            )

        console.print(table)

    asyncio.run(_example())
