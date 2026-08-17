"""Основной модуль — инициализация и запуск бота."""
import asyncio
import logging
import sys
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.config import settings
from src.utils.logging import setup_logging, logger
from src.db.session import init_db
from src.event_bus import event_bus
from src.data_ingest.market_data import MarketDataIngest
from src.data_ingest.coinglass_client import get_coinglass_client
from src.data_ingest.feature_engine import get_feature_engine
from src.strategy import strategy_registry
from src.risk.risk_manager import risk_manager
from src.execution.executor import execution_engine
from src.ml import model_trainer, feature_store, model_registry, ml_inference
from src.telegram.channel_monitor import init_telegram, close_telegram, subscribe_telegram_signal
from src.web.api import app
from src.web.websocket import setup_websocket_broadcast
from src.utils.crypto import generate_encryption_key

logger = logging.getLogger(__name__)


async def initialize_bot():
    """Инициализация всех компонентов бота."""
    logger.info("🚀 Инициализация CryptoBot Pro...")

    # 1. Логирование
    setup_logging()

    # 2. База данных
    await init_db()
    logger.info("✅ База данных инициализирована")

    # 3. Инженер признаков
    feature_engine = get_feature_engine()
    logger.info("✅ Feature Engine готов")

    # 4. CoinGlass клиент
    cg_client = get_coinglass_client()
    logger.info("✅ CoinGlass клиент готов")

    # 5. Execution Engine
    await execution_engine.initialize("binance")
    logger.info(f"✅ Execution Engine: {'paper' if settings.is_paper else 'real'} режим")

    # 6. Telegram клиент (опционально)
    if settings.telegram_api_id and settings.telegram_api_hash:
        await init_telegram()

    # 7. WebSocket broadcast
    setup_websocket_broadcast()

    logger.info("✅ Инициализация завершена")


async def start_market_data_feed():
    """Запуск потока рыночных данных."""
    logger.info("📡 Запуск потока рыночных данных...")

    ingest = MarketDataIngest("binance")
    await ingest.initialize()

    # Загрузить исторические свечи
    symbols = settings.default_symbols
    for symbol in symbols:
        df = await ingest.fetch_ohlcv(symbol, "1h", limit=200)
        if df is not None:
            ingest.update_buffer(symbol, df)
            logger.info(f"📊 Загружено 200 свечей для {symbol}")

    # Запустить периодическое обновление
    feed_task = asyncio.create_task(
        ingest.start_periodic_update(symbols, "1h", interval_seconds=60)
    )

    return ingest, feed_task


async def start_strategies(ingest: MarketDataIngest):
    """Запуск стратегий на обработку данных."""
    logger.info("🧠 Запуск стратегий...")

    strategy_task_running = True

    async def strategy_loop():
        nonlocal strategy_task_running
        while strategy_task_running:
            try:
                # Обрабатываем каждую пару
                for symbol in settings.default_symbols:
                    df = ingest.get_candles_for_symbol(symbol, limit=50)
                    if df is None or df.empty:
                        continue

                    latest = df.iloc[-1]

                    # Получаем фичи
                    features = {
                        "symbol": symbol,
                        "timeframe": "1h",
                        "close": latest.get("close"),
                        "rsi_14": latest.get("rsi_14"),
                        "rsi_7": latest.get("rsi_7"),
                        "rsi_21": latest.get("rsi_21"),
                        "macd": latest.get("macd"),
                        "macd_signal": latest.get("macd_signal"),
                        "macd_hist": latest.get("macd_hist"),
                        "bb_upper": latest.get("bb_upper"),
                        "bb_lower": latest.get("bb_lower"),
                        "bb_mid": latest.get("bb_mid"),
                        "bb_pct": latest.get("bb_pct"),
                        "bb_width": latest.get("bb_width"),
                        "ema_20": latest.get("ema_20"),
                        "ema_50": latest.get("ema_50"),
                        "ema_200": latest.get("ema_200"),
                        "ema_20_slope": latest.get("ema_20_slope"),
                        "ema_50_slope": latest.get("ema_50_slope"),
                        "price_above_ema20": latest.get("price_above_ema20"),
                        "price_above_ema50": latest.get("price_above_ema50"),
                        "atr_14": latest.get("atr_14"),
                        "natr_14": latest.get("natr_14"),
                        "realized_vol_20": latest.get("realized_vol_20"),
                        "realized_vol_60": latest.get("realized_vol_60"),
                        "volume_ratio": latest.get("volume_ratio"),
                        "obv": latest.get("obv"),
                        "return_1": latest.get("return_1"),
                        "return_3": latest.get("return_3"),
                        "return_5": latest.get("return_5"),
                        "return_10": latest.get("return_10"),
                        "return_20": latest.get("return_20"),
                        "log_return": latest.get("log_return"),
                        "momentum_10": latest.get("momentum_10"),
                        "momentum_20": latest.get("momentum_20"),
                        "dist_from_ema20": latest.get("dist_from_ema20"),
                        "dist_from_ema50": latest.get("dist_from_ema50"),
                        "high_low_range": latest.get("high_low_range"),
                        "range_ratio": latest.get("range_ratio"),
                        "close_position": latest.get("close_position"),
                        "stoch_k": latest.get("stoch_k"),
                        "stoch_d": latest.get("stoch_d"),
                        "wr_14": latest.get("wr_14"),
                        "mfi_14": latest.get("mfi_14"),
                        "hour": latest.get("hour", 0),
                        "day_of_week": latest.get("day_of_week", 0),
                        "is_weekend": latest.get("is_weekend", 0),
                        "roc_5": latest.get("roc_5"),
                        "roc_10": latest.get("roc_10"),
                        "roc_20": latest.get("roc_20"),
                        "ema_9_lag_1": latest.get("ema_9_lag_1"),
                        "ema_21_lag_1": latest.get("ema_21_lag_1"),
                        "close_lag_1": latest.get("close_lag_1"),
                        "close_lag_2": latest.get("close_lag_2"),
                        "volume_lag_1": latest.get("volume_lag_1"),
                    }

                    # Проверка: все ли признаки валидны
                    if not all(v is not None and not (isinstance(v, float) and v != v)
                               for v in [features.get("close"), features.get("rsi_14")]):
                        continue

                    # Запускаем ML inference для получения вероятностей
                    ml_result = ml_inference.predict_direction(features)
                    if ml_result:
                        features["ml_proba_up"] = ml_result.get("proba_up")
                        features["ml_proba_down"] = ml_result.get("proba_down")
                        features["ml_proba_neutral"] = ml_result.get("proba_neutral")
                        features["ml_features_used"] = str(ml_result.get("feature_importance", ""))[:200]

                    # Получаем сигналы от каждой активной стратегии
                    signals = []
                    for strategy in strategy_registry.get_active():
                        signal = strategy.generate_signal(features)
                        if signal:
                            signals.append(signal)

                    # Агрегация через Ensemble Voter
                    ensemble = strategy_registry.get("ensemble_voter")
                    if ensemble and signals:
                        # Установка весов по умолчанию
                        for s in signals:
                            ensemble.set_strategy_weight(s.strategy_id, s.weight)

                        aggregated = ensemble.aggregate_signals(signals)
                        if aggregated:
                            # Проверка риска
                            can_trade, reason = risk_manager.check_signal(aggregated)
                            if can_trade:
                                # Расчёт размера позиции
                                balance = execution_engine.get_paper_balance() if settings.is_paper else 10000
                                size = risk_manager.adjust_position_size(aggregated, balance)
                                logger.info(
                                    f"📈 СИГНАЛ: {aggregated.side.upper()} {aggregated.symbol} | "
                                    f"size={size:.2f} USDT | conf={aggregated.confidence:.2f} | "
                                    f"SL={aggregated.stop_loss:.2f} TP={aggregated.take_profit:.2f}"
                                )

                                # Создание ордера
                                order = await execution_engine.create_order(
                                    symbol=aggregated.symbol,
                                    side=aggregated.side,
                                    amount=size / aggregated.entry_price if aggregated.entry_price > 0 else 0,
                                    price=aggregated.entry_price,
                                    order_type="market",
                                    stop_loss=aggregated.stop_loss,
                                    take_profit=aggregated.take_profit,
                                    strategy_id=1,  # TODO: real strategy_id
                                    signal_data={
                                        "strategy_id": aggregated.strategy_id,
                                        "confidence": aggregated.confidence,
                                        "rationale": aggregated.rationale,
                                    },
                                )
                                if order:
                                    logger.info(f"✅ Ордер создан: {order.client_order_id}")

                    # Логирование
                    logger.debug(
                        f"[{symbol}] close={features.get('close'):.2f} rsi={features.get('rsi_14'):.1f} "
                        f"ema20_slope={features.get('ema_20_slope'):.4f} vol_ratio={features.get('volume_ratio'):.1f}"
                    )

            except Exception as e:
                logger.error(f"Ошибка в strategy loop: {e}")

            await asyncio.sleep(60)  # Каждую минуту

    strategy_task = asyncio.create_task(strategy_loop())
    return strategy_task


async def start_coinglass_updater():
    """Периодическое обновление данных из CoinGlass."""
    logger.info("📈 Запуск обновления CoinGlass данных...")

    async def update():
        try:
            # Получаем данные для BTC (как основного актива)
            btc_data = await cg_client.get_coins_markets(symbol="BTC")
            if btc_data:
                logger.debug(f"CoinGlass: BTC markets обновлены")

            # Funding rate для BTC
            funding = await cg_client.get_funding_rate_history(symbol="BTC/USDT", limit=50)
            if funding:
                logger.debug(f"CoinGlass: BTC funding rate обновлён")

            # OI для BTC
            oi = await cg_client.get_open_interest_history(symbol="BTC/USDT", limit=50)
            if oi:
                logger.debug(f"CoinGlass: BTC OI обновлён")

            # Fear & Greed
            fear = await cg_client.get_fear_greed_history(limit=10)
            if fear:
                logger.debug(f"CoinGlass: Fear & Greed обновлён")

        except Exception as e:
            logger.debug(f" CoinGlass обновление: {e}")

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        update,
        IntervalTrigger(hours=settings.coinglass_update_interval_hours),
        id="coinglass_updater",
        name="CoinGlass data updater",
    )
    scheduler.start()
    logger.info(f"✅ CoinGlass updater запущен (каждые {settings.coinglass_update_interval_hours} часов)")

    return scheduler


async def start_ml_retrainer():
    """Периодическое переобучение ML моделей."""
    logger.info("🧠 Запуск планировщика ML переобучения...")

    async def retrain():
        try:
            # Проверяем, есть ли достаточно данных
            trades_count = await _get_trades_count()
            if trades_count < 50:
                logger.debug(f"Слишком мало сделок для обучения: {trades_count}")
                return

            result = model_trainer.train_direction_classifier()
            if result:
                logger.info(f"ML модель переобучена: v{result['version']}")
                model_registry.activate_model("direction_classifier", result["version"])

        except Exception as e:
            logger.error(f"Ошибка ML переобучения: {e}")

    async def _get_trades_count() -> int:
        async with get_session() as session:
            from src.db.models import Trade
            result = await session.query(Trade).count()
            return result

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        retrain,
        IntervalTrigger(hours=settings.ml_retraining_interval_hours),
        id="ml_retrainer",
        name="ML model retrainer",
    )
    scheduler.start()
    logger.info(f"✅ ML retrainer запущен (каждые {settings.ml_retraining_interval_hours} часов)")

    return scheduler


async def start_telegram_signal_processor():
    """Обработка Telegram сигналов."""
    logger.info("📱 Запуск Telegram signal processor...")

    async def signal_handler(signal_event: dict):
        logger.info(f"📲 Получен Telegram сигнал: {signal_event.get('parsed_pair')} {signal_event.get('parsed_side')}")

        # Оценка качества
        quality = signal_quality_scorer.score_signal(
            signal_event,
            signal_event.get("channel_id", ""),
        )
        logger.info(f"Quality score: {quality}")

        # Проверка порога
        if quality >= settings.telegram_signals_quality_threshold:
            if settings.telegram_signals_auto_execute:
                logger.info(f"🤖 Автоматическое исполнение сигнала: {signal_event.get('parsed_pair')}")
                # TODO: исполнение сигнала
            else:
                logger.info(f"⏳ Сигнал ожидает подтверждения: {signal_event.get('parsed_pair')}")
        else:
            logger.info(f"🚫 Сигнал отклонён (quality={quality:.2f} < threshold={settings.telegram_signals_quality_threshold})")

    subscribe_telegram_signal(signal_handler)


async def run_bot():
    """Основной цикл работы бота."""
    await initialize_bot()

    # Запуск компонентов
    ingest, feed_task = await start_market_data_feed()
    strategy_task = await start_strategies(ingest)
    coinglass_scheduler = await start_coinglass_updater()
    ml_scheduler = await start_ml_retrainer()

    # Telegram
    await start_telegram_signal_processor()

    logger.info("✅ Бот полностью запущен")

    # Основной цикл — просто ждём, пока все задачи работают
    try:
        while True:
            await asyncio.sleep(60)
            # Периодический чек состояния
            logger.debug(
                f"🌐 Статус: mode={settings.trading_mode}, "
                f"balance={execution_engine.get_paper_balance():.2f}, "
                f"positions={len(execution_engine.get_paper_positions())}, "
                f"risk_paused={risk_manager.state.paused}"
            )
    except asyncio.CancelledError:
        logger.info("Бот остановлен")
    finally:
        # Очистка
        feed_task.cancel()
        strategy_task.cancel()
        coinglass_scheduler.shutdown()
        ml_scheduler.shutdown()
        await close_telegram()
        await execution_engine.close()


async def main():
    """Точка входа."""
    try:
        await run_bot()
    except KeyboardInterrupt:
        logger.info("Получен SIGINT — graceful shutdown")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
