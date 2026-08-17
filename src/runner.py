"""Runner — утилита для запуска бота и бэктестов."""
import argparse
import asyncio
import logging
import sys

from src.config import settings
from src.utils.logging import setup_logging
from src.backtest.engine import BacktestEngine, BacktestDataLoader
from src.strategy import (
    RSIMeanReversionStrategy,
    EMACrossoverStrategy,
    BollingerBandsStrategy,
)
from src.execution.executor import execution_engine
from src.db.session import init_db


def run_bot():
    """Запуск бота."""
    setup_logging()
    print("🚀 Запуск CryptoBot Pro...")

    try:
        from src.main import main
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n📥 Прервано пользователем")
        sys.exit(0)


def run_backtest():
    """Запуск бэктеста."""
    parser = argparse.ArgumentParser(description="Запуск бэктеста")
    parser.add_argument("--symbol", default="BTC/USDT", help="Торговая пара")
    parser.add_argument("--timeframe", default="1h", help="Таймфрейм")
    parser.add_argument("--limit", type=int, default=1000, help="Количество свечей")
    parser.add_argument("--capital", type=float, default=10000.0, help="Начальный капитал")
    parser.add_argument("--position-size", type=float, default=5.0, help="Размер позиции (%)")
    parser.add_argument("--start-date", type=str, default=None, help="Дата начала (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Дата конца (YYYY-MM-DD)")
    parser.add_argument("--strategies", nargs="+", default=["rsi", "bb", "ema"], help="Стратегии для сравнения")
    parser.add_argument("--output", type=str, default=None, help=" Сохранить отчёт в файл")

    args = parser.parse_args()

    setup_logging()

    print(f"📊 Запуск бэктеста: {args.symbol} {args.timeframe}")
    print(f"   Капитал: ${args.capital:,.2f}")
    print(f"   Размер позиции: {args.position_size}%")
    print(f"   Стратегии: {', '.join(args.strategies)}")
    print()

    # Инициализация
    asyncio.run(init_db())

    # Загрузка данных
    print("📥 Загрузка исторических данных...")
    try:
        loader = BacktestDataLoader()
        df = loader.from_ccxt(
            exchange_id="binance",
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.limit,
        )
        print(f"   Загружено: {len(df)} свечей")

        if args.start_date or args.end_date:
            from datetime import datetime
            df = df.copy()
            if args.start_date:
                df = df[df.index >= datetime.fromisoformat(args.start_date)]
            if args.end_date:
                df = df[df.index <= datetime.fromisoformat(args.end_date)]
            print(f"   После фильтрации: {len(df)} свечей")

        if df.empty:
            print("❌ Нет данных в указанном диапазоне")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        sys.exit(1)

    # Настройка стратегий
    strategies = []
    strategy_configs = []

    if "rsi" in args.strategies:
        s = RSIMeanReversionStrategy()
        s.params = {
            "rsi_period": 14,
            "oversold_level": 30,
            "overbought_level": 70,
            "position_size_pct": args.position_size,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }
        strategies.append(s)
        strategy_configs.append({"strategy_class": RSIMeanReversionStrategy, "params": s.params})

    if "bb" in args.strategies:
        s = BollingerBandsStrategy()
        s.params = {
            "mode": "mean_reversion",
            "position_size_pct": args.position_size,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }
        strategies.append(s)
        strategy_configs.append({"strategy_class": BollingerBandsStrategy, "params": s.params})

    if "ema" in args.strategies:
        s = EMACrossoverStrategy()
        s.params = {
            "fast_ema_period": 9,
            "slow_ema_period": 21,
            "position_size_pct": args.position_size,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
        }
        strategies.append(s)
        strategy_configs.append({"strategy_class": EMACrossoverStrategy, "params": s.params})

    # Запуск бэктеста
    print("🧮 Запуск бэктеста...")
    print()

    engine = BacktestEngine(
        initial_capital=args.capital,
        fee_pct=0.1,
        slippage_pct=0.05,
    )

    results = engine.run_strategies_comparison(
        data=df,
        strategy_configs=strategy_configs,
        symbol=args.symbol,
        timeframe=args.timeframe,
        position_size_pct=args.position_size,
    )

    engine.print_comparison(results, args.strategies)

    # Найти лучшую стратегию
    best_idx = max(range(len(results)), key=lambda i: results[i].total_pnl)
    best_result = results[best_idx]
    best_name = args.strategies[best_idx]

    print(f"\n🏆 Лучшая стратегия: {best_name} | PnL: {best_result.total_pnl:+.2f} USD ({best_result.total_pnl_pct:+.2f}%)")

    print()
    print("📊 Метрики лучшей стратегии:")
    print(f"   Сделок: {best_result.total_trades}")
    print(f"   Win Rate: {best_result.win_rate:.1f}%")
    print(f"   Profit Factor: {best_result.profit_factor:.2f}")
    print(f"   Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
    print(f"   Макс. даундрафт: {best_result.max_drawdown:.1f}%")
    print(f"   Ожидание (Expectancy): ${best_result.expectancy:+.2f} на сделку")
    print(f"   Средний выигрыш: ${best_result.average_win:+.2f}")
    print(f"   Средний проигрыш: ${best_result.average_loss:+.2f}")

    if args.output:
        print(f"\n📄 Отчёт сохранён в {args.output}")
        # TODO: сохранить отчёт


def run_health_check():
    """Проверка состояния бота."""
    import requests

    print("🔍 Проверка состояния бота...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health: {response.json()}")
    except Exception as e:
        print(f"❌ Бот недоступен: {e}")

    try:
        response = requests.get("http://localhost:8000/status", timeout=5)
        status = response.json()
        print(f"\n📊 Статус:")
        print(f"   Режим: {status.get('trading_mode', 'unknown')}")
        print(f"   Капитал: ${status.get('paper_balance', 0):,.2f}")
        print(f"   Позиции: {len(status.get('paper_positions', {}))}")
        print(f"   Стратегии: {len(status.get('active_strategies', []))}")
    except Exception as e:
        print(f"❌ Не удалось получить статус: {e}")


def main():
    """Основная точка входа."""
    parser = argparse.ArgumentParser(description="CryptoBot Pro — запуск и управление")
    subparsers = parser.add_subparsers(dest="command", help="Команды")

    # run
    run_parser = subparsers.add_parser("run", help="Запустить бота")
    run_parser.set_defaults(func=run_bot)

    # backtest
    bt_parser = subparsers.add_parser("backtest", help="Запустить бэктест")
    bt_parser.add_argument("--symbol", default="BTC/USDT")
    bt_parser.add_argument("--timeframe", default="1h")
    bt_parser.add_argument("--limit", type=int, default=1000)
    bt_parser.add_argument("--capital", type=float, default=10000.0)
    bt_parser.add_argument("--position-size", type=float, default=5.0)
    bt_parser.add_argument("--start-date", type=str, default=None)
    bt_parser.add_argument("--end-date", type=str, default=None)
    bt_parser.add_argument("--strategies", nargs="+", default=["rsi", "bb", "ema"])
    bt_parser.add_argument("--output", type=str, default=None)
    bt_parser.set_defaults(func=run_backtest)

    # health
    health_parser = subparsers.add_parser("health", help="Проверка состояния")
    health_parser.set_defaults(func=run_health_check)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func()


if __name__ == "__main__":
    main()
