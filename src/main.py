"""CryptoBot Pro — автономный самообучающийся крипто-трейдер бот."""
import asyncio
import logging
import sys
from typing import Optional

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.config import settings
from src.utils.logging import setup_logging, logger
from src.event_bus import event_bus
from src.data_ingest.market_data import MarketDataIngest
from src.data_ingest.coinglass_client import get_coinglass_client
from src.data_ingest.feature_engine import get_feature_engine
from src.strategy import strategy_registry
from src.risk.risk_manager import risk_manager
from src.execution.executor import execution_engine
from src.execution.decision_logger import decision_logger
from src.ml import model_trainer, model_registry, ml_inference
from src.telegram.channel_monitor import init_telegram, close_telegram, subscribe_telegram_signal
from src.telegram.notifier import send_notification
from src.utils.crypto import generate_encryption_key
from src.utils.timeutils import utcnow
from src.web.api import app as web_app


class TradingBot:
    """Основной класс торгового бота."""

    def __init__(self):
        self.running = False
        self.ingest = None
        self.cg_client = None
        self.feature_engine = None
        self.ml_inference = None
        self.ml_trainer = None
        self.scheduler = None
        self.candles_buffer = {}
        self.last_prices = {}
        self.open_positions = {}
        self.closed_trades = []
        self.daily_pnl = 0.0
        self.daily_pnl_reset_date = None
        self._kill_switch_notified = False

    async def initialize(self):
        """Инициализация всех компонентов."""
        setup_logging()
        logger.info("🚀 Инициализация CryptoBot Pro...")

        # Схема БД управляется через Alembic (см. README) — её нужно
        # применить заранее командой `alembic upgrade head`, а не при
        # каждом старте бота (иначе она конфликтует с миграциями:
        # create_all() создаёт таблицы в обход alembic_version).

        # Feature engine
        self.feature_engine = get_feature_engine()

        # CoinGlass клиент
        self.cg_client = get_coinglass_client()

        # Market data ingest
        self.ingest = MarketDataIngest("binance")
        await self.ingest.initialize()

        # Загрузка исторических данных
        for symbol in settings.default_symbols:
            df = await self.ingest.fetch_ohlcv(symbol, "1h", limit=200)
            if df is not None:
                self.ingest.update_buffer(symbol, df)
                self.candles_buffer[symbol] = df
                logger.info(f"📊 Загружено 200 свечей для {symbol}")

        # ML inference
        self.ml_inference = ml_inference
        active_model = await model_registry.get_active_model("direction_classifier")
        if active_model and active_model.get("model_path"):
            self.ml_inference.load_model("direction_classifier", active_model["model_path"])
            logger.info(f"✅ ML модель загружена: {active_model['model_type']} v{active_model['version']}")
        else:
            logger.warning("⚠️ ML модель не найдена — используются только rule-based стратегии")

        # Execution engine
        await execution_engine.initialize("binance")
        logger.info(f"✅ Execution Engine: {'paper' if settings.is_paper else 'real'} режим")

        # Telegram (опционально)
        if settings.telegram_api_id and settings.telegram_api_hash:
            telegram_client = await init_telegram()
            if telegram_client:
                subscribe_telegram_signal(self._on_telegram_signal)
                logger.info("✅ Telegram клиент инициализирован")
            else:
                logger.warning(
                    "⚠️ Telegram клиент не инициализирован (см. ошибку выше) — "
                    "мониторинг каналов отключён на этот запуск"
                )

        # Планировщик
        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(
            self._update_coinglass,
            IntervalTrigger(hours=settings.coinglass_update_interval_hours),
            id="coinglass_updater",
        )
        self.scheduler.add_job(
            self._retrain_ml,
            IntervalTrigger(hours=settings.ml_retraining_interval_hours),
            id="ml_retrainer",
        )
        self.scheduler.start()
        logger.info(f"✅ Планировщик запущен")

        # Telegram-уведомления (алерты о сделках/ошибках, без приёма команд)
        event_bus.subscribe("trade_event", self._on_trade_event)

        self.running = True
        logger.info("✅ Инициализация завершена")
        await send_notification(
            f"🚀 CryptoBot Pro запущен | режим: {settings.trading_mode} | "
            f"символы: {', '.join(settings.default_symbols)}"
        )

    async def _on_trade_event(self, event):
        """Отправить Telegram-уведомление об открытой сделке."""
        side = "LONG 📈" if event.direction == "long" else "SHORT 📉"
        await send_notification(
            f"{side} {event.symbol}\n"
            f"Цена входа: {event.entry_price:.4f}\n"
            f"Объём: {event.amount:.6f}"
        )

    async def _update_coinglass(self):
        """Обновление данных из CoinGlass."""
        try:
            btc_market = await self.cg_client.get_coins_markets(symbol="BTC")
            funding = await self.cg_client.get_funding_rate_history(symbol="BTC/USDT", limit=50)
            oi = await self.cg_client.get_open_interest_history(symbol="BTC/USDT", limit=50)
            fear = await self.cg_client.get_fear_greed_history(limit=10)
            logger.debug("CoinGlass данные обновлены")
        except Exception as e:
            logger.debug(f"CoinGlass: {e}")

    async def _retrain_ml(self):
        """Переобучение ML моделей."""
        try:
            from sqlalchemy import func, select

            from src.db.models import Trade
            from src.db.session import get_session

            async with get_session() as session:
                trades_count = (
                    await session.execute(select(func.count()).select_from(Trade))
                ).scalar_one()

            if trades_count < 50:
                logger.debug(f"ML retraining пропущен: {trades_count} сделок")
                return

            logger.info(f"ML retraining: {trades_count} сделок")
            result = await model_trainer.train_direction_classifier()
            if result:
                logger.info(f"✅ Direction classifier: v{result['version']}")
                await model_registry.activate_model("direction_classifier", result["version"])
                if self.ml_inference:
                    self.ml_inference.load_model("direction_classifier", result["model_path"])
        except Exception as e:
            logger.error(f"ML retraining: {e}")

    async def _on_telegram_signal(self, signal_event: dict):
        """Обработка Telegram сигнала."""
        channel_id = signal_event.get("channel_id", "")
        pair = signal_event.get("parsed_pair", "")
        side = signal_event.get("parsed_side", "")
        entry = signal_event.get("parsed_entry", 0)
        sl = signal_event.get("parsed_sl")
        tp = signal_event.get("parsed_tp")

        if not pair or not side or entry <= 0:
            return

        from src.telegram.quality_scorer import signal_quality_scorer
        quality = signal_quality_scorer.score_signal(signal_event, channel_id)
        logger.info(f"📲 Telegram сигнал: {pair} {side.upper()} | quality={quality:.2f}")

        if quality < settings.telegram_signals_quality_threshold:
            logger.info(f"🚫 Сигнал отклонён (quality={quality:.2f})")
            return

        if settings.telegram_signals_auto_execute:
            logger.info(f"🤖 Автоматическое исполнение")
            await self._execute_telegram_signal(signal_event)
        else:
            logger.info(f"⏳ Сигнал ожидает подтверждения")

    async def _execute_telegram_signal(self, signal_event: dict):
        """Исполнение Telegram сигнала."""
        pair = signal_event.get("parsed_pair", "")
        side = signal_event.get("parsed_side", "long")
        entry = signal_event.get("parsed_entry", 0)
        sl = signal_event.get("parsed_sl")
        tp = signal_event.get("parsed_tp")

        symbol = pair
        order_side = "buy" if side == "long" else "sell"

        balance = execution_engine.get_paper_balance() if settings.is_paper else 10000
        size_pct = 5.0
        position_value = balance * (size_pct / 100)
        amount = position_value / entry

        logger.info(f"📝 Ордер: {order_side.upper()} {amount:.6f} {symbol} @ {entry:.2f}")
        order = await execution_engine.create_order(
            symbol=symbol,
            side=order_side,
            amount=amount,
            price=entry,
            order_type="market",
            stop_loss=sl,
            take_profit=tp,
        )
        if order:
            logger.info(f"✅ Ордер исполнен: {order.client_order_id}")

    async def run(self):
        """Основной цикл торговли."""
        if not self.running:
            await self.initialize()

        logger.info("🔄 Запуск основного цикла...")

        while self.running:
            try:
                await self._trading_iteration()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Основной цикл: {e}")
                await send_notification(f"🔴 Ошибка основного цикла: {e}")
                await asyncio.sleep(60)

        await self._cleanup()

    async def _trading_iteration(self):
        """Одна итерация торговли."""
        if risk_manager.state.kill_switch_active:
            if not self._kill_switch_notified:
                self._kill_switch_notified = True
                await send_notification("🔴 KILL SWITCH активирован — вся торговля остановлена")
            return
        self._kill_switch_notified = False

        if risk_manager.state.paused:
            return

        # Daily PnL reset
        today = utcnow().date()
        if self.daily_pnl_reset_date != today:
            self.daily_pnl = 0.0
            self.daily_pnl_reset_date = today

        for symbol in settings.default_symbols:
            await self._process_symbol(symbol)

    async def _process_symbol(self, symbol: str):
        """Обработка одной пары."""
        df = self.candles_buffer.get(symbol)
        if df is None or df.empty or len(df) < 50:
            df = await self.ingest.fetch_ohlcv(symbol, "1h", limit=50)
            if df is not None:
                self.ingest.update_buffer(symbol, df)
                self.candles_buffer[symbol] = df

        if df is None or df.empty or len(df) < 50:
            return

        latest = df.iloc[-1]
        close = float(latest["close"])
        self.last_prices[symbol] = close

        # Фичи
        features = self.feature_engine.compute_all_indicators(df)
        latest_features = features.iloc[-1]

        # Сбор данных для стратегий
        strategy_data = {
            "symbol": symbol,
            "timeframe": "1h",
            "close": close,
            "rsi_14": latest_features.get("rsi_14"),
            "rsi_7": latest_features.get("rsi_7"),
            "rsi_21": latest_features.get("rsi_21"),
            "macd": latest_features.get("macd"),
            "macd_signal": latest_features.get("macd_signal"),
            "macd_hist": latest_features.get("macd_hist"),
            "bb_upper": latest_features.get("bb_upper"),
            "bb_lower": latest_features.get("bb_lower"),
            "bb_mid": latest_features.get("bb_mid"),
            "bb_pct": latest_features.get("bb_pct"),
            "bb_width": latest_features.get("bb_width"),
            "ema_20": latest_features.get("ema_20"),
            "ema_50": latest_features.get("ema_50"),
            "ema_20_slope": latest_features.get("ema_20_slope"),
            "ema_50_slope": latest_features.get("ema_50_slope"),
            "price_above_ema20": latest_features.get("price_above_ema20"),
            "price_above_ema50": latest_features.get("price_above_ema50"),
            "atr_14": latest_features.get("atr_14"),
            "natr_14": latest_features.get("natr_14"),
            "realized_vol_20": latest_features.get("realized_vol_20"),
            "volume_ratio": latest_features.get("volume_ratio"),
            "obv": latest_features.get("obv"),
            "return_1": latest_features.get("return_1"),
            "return_3": latest_features.get("return_3"),
            "return_5": latest_features.get("return_5"),
            "log_return": latest_features.get("log_return"),
            "momentum_10": latest_features.get("momentum_10"),
            "dist_from_ema20": latest_features.get("dist_from_ema20"),
            "dist_from_ema50": latest_features.get("dist_from_ema50"),
            "high_low_range": latest_features.get("high_low_range"),
            "stoch_k": latest_features.get("stoch_k"),
            "stoch_d": latest_features.get("stoch_d"),
            "wr_14": latest_features.get("wr_14"),
            "mfi_14": latest_features.get("mfi_14"),
            "hour": latest_features.get("hour", 0),
            "day_of_week": latest_features.get("day_of_week", 0),
        }

        # ML inference
        if self.ml_inference:
            ml_result = await self.ml_inference.predict_direction(strategy_data)
            if ml_result:
                strategy_data["ml_proba_up"] = ml_result.get("proba_up")
                strategy_data["ml_proba_down"] = ml_result.get("proba_down")
                strategy_data["ml_proba_neutral"] = ml_result.get("proba_neutral")

        # Decision logger: market data
        decision_logger.log_market_data(
            symbol=symbol, timeframe="1h", price=close,
            features={k: round(v, 4) if isinstance(v, float) else v
                     for k, v in list(strategy_data.items())[:15]},
        )

        # Сигналы от стратегий
        signals = []
        for strategy in strategy_registry.get_active():
            signal = strategy.generate_signal(strategy_data)
            if signal:
                signals.append(signal)
                decision_logger.log_strategy_signal(
                    strategy_id=strategy.strategy_id,
                    strategy_name=strategy.name,
                    signal_side=signal.side,
                    confidence=signal.confidence,
                    entry_price=signal.entry_price or close,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    rationale=signal.rationale,
                )

        # ML score log
        if "ml_proba_up" in strategy_data:
            active_model = await model_registry.get_active_model("direction_classifier")
            decision_logger.log_ml_score(
                model_type="direction_classifier",
                model_version=(active_model or {}).get("version", 1),
                proba_up=strategy_data["ml_proba_up"],
                proba_down=strategy_data["ml_proba_down"],
                proba_neutral=strategy_data["ml_proba_neutral"],
            )

        # Ensemble
        ensemble = strategy_registry.get("ensemble_voter")
        if ensemble and signals:
            for s in signals:
                source_strategy = strategy_registry.get(s.strategy_id)
                if source_strategy:
                    ensemble.set_strategy_weight(s.strategy_id, source_strategy.weight)
            aggregated = ensemble.aggregate_signals(signals)
            if aggregated:
                signals = [aggregated]

        if not signals:
            return

        # Риск + исполнение
        for signal in signals:
            can_execute, reason = risk_manager.check_signal(signal)
            decision_logger.log_risk_check(
                decision="allowed" if can_execute else "rejected",
                reason=reason,
                context={"symbol": symbol, "side": signal.side, "daily_pnl": self.daily_pnl},
            )

            if not can_execute:
                logger.info(f"🚫 Сигнал отклонён (risk): {symbol} {signal.side} — {reason}")
                continue

            balance = execution_engine.get_paper_balance() if settings.is_paper else 10000
            size_pct = signal.position_size_pct
            position_value = balance * (size_pct / 100)
            entry_price = signal.entry_price if signal.entry_price > 0 else close
            amount = position_value / entry_price if entry_price > 0 else 0

            if amount <= 0:
                continue

            order_side = "buy" if signal.side == "long" else "sell"
            logger.info(
                f"📝 Ордер: {order_side.upper()} {amount:.6f} {symbol} @ {entry_price:.2f} | "
                f"Conf: {signal.confidence:.2f} | SL: {signal.stop_loss} TP: {signal.take_profit}"
            )

            decision_logger.log_execution(
                order_id="pending", order_type="market", amount=amount,
                price=entry_price, status="pending", fee=0,
            )

            order = await execution_engine.create_order(
                symbol=symbol,
                side=order_side,
                amount=amount,
                price=entry_price,
                order_type="market",
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_id=1,
                signal_data={"strategy_id": signal.strategy_id, "confidence": signal.confidence},
            )

            if order:
                decision_logger.log_execution(
                    order_id=order.client_order_id, order_type="market",
                    amount=amount, price=entry_price, status="filled", fee=order.fee,
                )
                self.open_positions[symbol] = {
                    "side": signal.side, "entry_price": entry_price,
                    "amount": amount, "strategy_id": signal.strategy_id,
                    "rationale": signal.rationale, "sl": signal.stop_loss,
                    "tp": signal.take_profit, "opened_at": utcnow(),
                }
                risk_manager.on_position_added(symbol, size_pct)
                self.daily_pnl = getattr(risk_manager.state, "daily_pnl", 0.0)
                logger.info(f"✅ Ордер: {order.client_order_id}")

    async def _cleanup(self):
        """Очистка."""
        logger.info("🧹 Очистка...")

        if self.ingest:
            await self.ingest.close()
        if self.cg_client:
            await self.cg_client.close()
        if self.scheduler:
            self.scheduler.shutdown()
        await close_telegram()
        logger.info("✅ Очистка завершена")


async def main():
    """Точка входа."""
    bot = TradingBot()

    # Веб-панель (FastAPI) запускается в этом же процессе, а не отдельным
    # сервисом — она читает состояние напрямую из in-memory синглтонов
    # (execution_engine, risk_manager, strategy_registry, ...), которые
    # существуют только внутри процесса самого бота.
    web_config = uvicorn.Config(
        web_app,
        host=settings.web_host,
        port=settings.web_port,
        log_level=settings.log_level.lower(),
    )
    web_server = uvicorn.Server(web_config)
    web_server.install_signal_handlers = lambda: None  # управление сигналами — у asyncio.run()

    try:
        await asyncio.gather(bot.run(), web_server.serve())
    except KeyboardInterrupt:
        logger.info("📥 SIGINT — shutdown")
        await bot._cleanup()
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
