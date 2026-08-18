"""CryptoBot Pro — автономный самообучающийся крипто-трейдер бот."""
import asyncio
import logging
import sys
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

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
from src.ml import model_trainer, model_registry, ml_inference, feature_store
from src.telegram.channel_monitor import (
    init_telegram, close_telegram, subscribe_telegram_signal, monitor_channels,
)
from src.telegram.notifier import send_notification
from src.utils.crypto import generate_encryption_key
from src.utils.timeutils import utcnow
from src.db.session import get_session
from src.db.models import TelegramChannel, TelegramSignal
from src.web.api import app as web_app
from src.web.settings_store import load_settings_overrides


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
        self._last_ml_feature_ts: dict[str, Any] = {}
        self.active_symbols: list[str] = []
        self._telegram_channel_db_ids: dict[str, int] = {}

    async def initialize(self):
        """Инициализация всех компонентов."""
        setup_logging()
        logger.info("🚀 Инициализация CryptoBot Pro...")

        # Настройки, изменённые через веб-панель на предыдущем запуске
        # (bot_config в БД) — применяются до всего, что от них зависит
        # (ключи бирж, Telegram, CoinGlass).
        await load_settings_overrides()

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

        # Торговая вселенная: все активные spot-пары к symbol_quote_currency
        # (минус symbol_blacklist и плечевые токены), топ symbol_universe_max
        # по 24ч объёму — вместо фиксированного списка пар.
        await self._refresh_symbol_universe(initial=True)

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
        self._sync_open_positions_from_execution_engine()
        await risk_manager.restore_daily_pnl_from_db()

        # Telegram (опционально)
        if settings.telegram_api_id and settings.telegram_api_hash:
            telegram_client = await init_telegram()
            if telegram_client:
                subscribe_telegram_signal(self._on_telegram_signal)
                await self._start_telegram_monitoring()
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
        self.scheduler.add_job(
            self._refresh_symbol_universe,
            IntervalTrigger(hours=settings.symbol_universe_refresh_hours),
            id="symbol_universe_refresh",
        )
        self.scheduler.start()
        logger.info(f"✅ Планировщик запущен")

        # Telegram-уведомления (алерты о сделках/ошибках, без приёма команд)
        event_bus.subscribe("trade_event", self._on_trade_event)

        self.running = True
        logger.info("✅ Инициализация завершена")
        await send_notification(
            f"🚀 CryptoBot Pro запущен | режим: {settings.trading_mode} | "
            f"пар в торговой вселенной: {len(self.active_symbols)}"
        )

    async def _refresh_symbol_universe(self, initial: bool = False):
        """Пересчитать торговую вселенную (все активные пары symbol_quote_currency
        минус symbol_blacklist, топ symbol_universe_max по объёму) и подгрузить
        историю для новых пар. Открытые позиции никогда не выпадают из мониторинга,
        даже если пара выпала из топа по объёму или попала в блэклист."""
        try:
            new_symbols = await self.ingest.get_tradable_symbols(
                quote=settings.symbol_quote_currency,
                blacklist=settings.symbol_blacklist,
                max_symbols=settings.symbol_universe_max,
            )
        except Exception as e:
            logger.warning(f"Не удалось получить список торговых пар: {e}")
            new_symbols = []

        if not new_symbols:
            if not self.active_symbols:
                logger.warning("Торговая вселенная пуста — аварийный фоллбэк: BTC/USDT")
                new_symbols = ["BTC/USDT"]
            else:
                logger.warning("Не удалось обновить торговую вселенную — оставляем прежний список")
                return

        kept_for_open_positions = [s for s in self.open_positions if s not in new_symbols]
        combined = list(dict.fromkeys(new_symbols + kept_for_open_positions))

        added = [s for s in combined if s not in self.candles_buffer]
        removed = [s for s in self.active_symbols if s not in combined]

        for symbol in added:
            df = await self.ingest.fetch_ohlcv(symbol, "1h", limit=200)
            if df is not None:
                self.ingest.update_buffer(symbol, df)
                self.candles_buffer[symbol] = df
                logger.info(f"📊 Загружено 200 свечей для {symbol}")

        self.active_symbols = combined

        if initial:
            logger.info(f"📈 Торговая вселенная: {len(combined)} пар — {', '.join(combined)}")
        elif added or removed:
            logger.info(f"🔄 Торговая вселенная обновлена: +{len(added)} -{len(removed)} (всего {len(combined)})")

    def _sync_open_positions_from_execution_engine(self):
        """
        Пересобрать self.open_positions и счётчик открытых позиций в
        risk_manager из execution_engine.paper_positions (который на этот
        момент уже восстановлен из БД в execution_engine.initialize()).

        Без этого восстановленные из БД позиции были бы видны в дашборде
        (через /status), но: (1) _check_position_exit никогда не проверял
        бы их SL/TP, потому что self.open_positions пуст после рестарта;
        (2) risk_manager занижал бы open_positions_count и разрешил бы
        открывать больше позиций, чем реально разрешено max_open_positions.
        """
        self.open_positions = {}
        for symbol, pos in execution_engine.paper_positions.items():
            self.open_positions[symbol] = {
                "side": pos.get("side", "long"),
                "entry_price": pos.get("entry_price"),
                "amount": pos.get("amount"),
                "strategy_id": pos.get("strategy_id"),
                "rationale": "Восстановлено при старте бота",
                "sl": pos.get("stop_loss"),
                "tp": pos.get("take_profit"),
                "opened_at": pos.get("opened_at") or utcnow(),
                "order_id": pos.get("order_id"),
                "entry_fee": pos.get("entry_fee", 0.0),
            }
            position_value = (pos.get("amount") or 0) * (pos.get("entry_price") or 0)
            size_pct = (position_value / execution_engine.paper_balance * 100) if execution_engine.paper_balance else 0.0
            risk_manager.on_position_added(symbol, size_pct)

        if self.open_positions:
            logger.info(f"🔗 Синхронизировано {len(self.open_positions)} открытых позиций с риск-менеджером")

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

    async def _start_telegram_monitoring(self):
        """Загрузить активные каналы из БД и запустить их мониторинг.

        Список фиксируется на момент старта бота: добавление/удаление
        канала через дашборд начинает применяться только после рестарта
        (см. пометку "требуется рестарт" в веб-панели).
        """
        async with get_session() as session:
            channels = (
                await session.execute(select(TelegramChannel).where(TelegramChannel.active == True))
            ).scalars().all()
            channel_dicts = [
                {
                    "channel_id": c.channel_id,
                    "channel_title": c.channel_title or "",
                    "parser_config": c.parser_config or {},
                }
                for c in channels
            ]
            self._telegram_channel_db_ids = {c.channel_id: c.id for c in channels}

        if not channel_dicts:
            logger.info("Telegram: нет активных каналов для мониторинга")
            return

        await monitor_channels(channel_dicts)
        logger.info(f"👂 Мониторинг Telegram-каналов запущен: {len(channel_dicts)}")

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

        decision = "pending"
        order = None
        if quality < settings.telegram_signals_quality_threshold:
            decision = "rejected"
            logger.info(f"🚫 Сигнал отклонён (quality={quality:.2f})")
        elif settings.telegram_signals_auto_execute:
            logger.info(f"🤖 Автоматическое исполнение")
            order = await self._execute_telegram_signal(signal_event)
            decision = "executed" if order else "rejected"
        else:
            logger.info(f"⏳ Сигнал ожидает подтверждения")

        await self._save_telegram_signal(signal_event, quality, decision, order)

    async def _save_telegram_signal(
        self, signal_event: dict, quality: float, decision: str, order,
    ):
        """Сохранить сигнал в БД (для статистики по каналам)."""
        db_channel_id = self._telegram_channel_db_ids.get(signal_event.get("channel_id", ""))
        if db_channel_id is None:
            return
        try:
            async with get_session() as session:
                session.add(TelegramSignal(
                    channel_id=db_channel_id,
                    raw_message=signal_event.get("raw_message", ""),
                    message_date=signal_event.get("message_date") or utcnow(),
                    parsed_pair=signal_event.get("parsed_pair"),
                    parsed_side=signal_event.get("parsed_side"),
                    parsed_entry=signal_event.get("parsed_entry"),
                    parsed_sl=signal_event.get("parsed_sl"),
                    parsed_tp=signal_event.get("parsed_tp"),
                    quality_score=quality,
                    decision=decision,
                    executed_order_id=order.id if order else None,
                ))
                await session.commit()
        except Exception as e:
            logger.warning(f"Не удалось сохранить Telegram-сигнал в БД: {e}")

    async def _execute_telegram_signal(self, signal_event: dict):
        """Исполнение Telegram сигнала. Возвращает созданный Order или None."""
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
            strategy_id="telegram_signal",
        )
        if order:
            logger.info(f"✅ Ордер исполнен: {order.client_order_id}")
            # Регистрируем позицию так же, как для стратегийных сигналов —
            # иначе _check_position_exit никогда её не увидит, SL/TP не
            # сработают и позиция не закроется (а значит, executed_trade_id
            # у сигнала никогда не проставится).
            self.open_positions[symbol] = {
                "side": side, "entry_price": entry,
                "amount": amount, "strategy_id": "telegram_signal",
                "rationale": "Telegram сигнал", "sl": sl, "tp": tp,
                "opened_at": utcnow(),
                "order_id": order.id, "entry_fee": order.fee,
            }
            risk_manager.on_position_added(symbol, 5.0)
            self.daily_pnl = getattr(risk_manager.state, "daily_pnl", 0.0)
        return order

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

        # Баланс для контроля max_drawdown_pct — без этого вызова
        # risk_manager.state.start_balance/current_balance никогда не
        # обновлялись, и защита по просадке была мертва.
        if settings.is_paper:
            risk_manager.on_balance_update(execution_engine.get_paper_balance())
        else:
            real_balance = await execution_engine.get_real_balance()
            if real_balance is not None:
                risk_manager.on_balance_update(real_balance)

        for symbol in self.active_symbols:
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
        execution_engine.last_prices[symbol] = close

        if settings.is_paper and await self._check_position_exit(symbol, close):
            return  # позиция закрыта в этой итерации — новый сигнал сгенерируем в следующем цикле

        # Фичи
        features = self.feature_engine.compute_all_indicators(df)
        latest_features = features.iloc[-1]

        await self._record_ml_training_sample(symbol, df)

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
                strategy_id=signal.strategy_id,
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
                    "order_id": order.id, "entry_fee": order.fee,
                }
                risk_manager.on_position_added(symbol, size_pct)
                self.daily_pnl = getattr(risk_manager.state, "daily_pnl", 0.0)
                logger.info(f"✅ Ордер: {order.client_order_id}")

    async def _record_ml_training_sample(self, symbol: str, df: pd.DataFrame):
        """
        Сохранить в ml_features размеченный обучающий пример (признаки +
        целевая переменная), как только его исход становится известен из
        буфера свечей. extract_features_for_ml() смотрит на horizon свечей
        вперёд, поэтому размеченной может быть только строка не из самого
        конца буфера — это "прошлая" свеча, чьё будущее уже наступило.
        """
        try:
            labeled = self.feature_engine.extract_features_for_ml(df, include_target=True)
            if labeled.empty:
                return

            latest_ts = labeled.index[-1]
            if self._last_ml_feature_ts.get(symbol) == latest_ts:
                return  # уже сохраняли этот пример (буфер обновляется раз в час)

            latest_row = labeled.iloc[-1]
            feature_values = {
                k: float(v) for k, v in latest_row.items()
                if k not in ("target_direction", "target_volatility") and not pd.isna(v)
            }
            volatility_label = latest_row.get("target_volatility")

            timestamp = latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else latest_ts
            if not isinstance(timestamp, datetime):
                timestamp = utcnow()

            await feature_store.add_features(
                symbol=symbol,
                timeframe="1h",
                timestamp=timestamp,
                features=feature_values,
                labels={
                    "direction": float(latest_row["target_direction"]),
                    "volatility": float(volatility_label) if not pd.isna(volatility_label) else None,
                },
            )
            self._last_ml_feature_ts[symbol] = latest_ts
        except Exception as e:
            logger.debug(f"Не удалось сохранить ML-обучающий пример для {symbol}: {e}")

    async def _check_position_exit(self, symbol: str, current_price: float) -> bool:
        """
        Проверить открытую позицию на достижение SL/TP и закрыть при необходимости.
        Возвращает True, если позиция была закрыта в этом вызове.
        """
        position = self.open_positions.get(symbol)
        if not position:
            return False

        # Позиция могла быть закрыта в обход основного цикла (кнопка
        # "Закрыть" в дашборде, POST /positions/close) — execution_engine
        # уже не знает о ней, но self.open_positions ещё не подчищен.
        # Без этой проверки мы бы попытались закрыть её второй раз здесь.
        if symbol not in execution_engine.paper_positions:
            del self.open_positions[symbol]
            return False

        side = position["side"]
        sl = position.get("sl")
        tp = position.get("tp")

        reason = None
        if side == "long":
            if sl and current_price <= sl:
                reason = "stop_loss"
            elif tp and current_price >= tp:
                reason = "take_profit"
        else:  # short
            if sl and current_price >= sl:
                reason = "stop_loss"
            elif tp and current_price <= tp:
                reason = "take_profit"

        if reason is None:
            return False

        opened_at = position.get("opened_at")
        holding_seconds = int((utcnow() - opened_at).total_seconds()) if opened_at else 0

        result = await execution_engine.close_paper_position(
            symbol=symbol,
            side=side,
            entry_price=position["entry_price"],
            amount=position["amount"],
            exit_price=current_price,
            reason=reason,
            entry_fee=position.get("entry_fee", 0.0),
            holding_seconds=holding_seconds,
            strategy_id=position.get("strategy_id"),
            order_open_id=position.get("order_id"),
        )

        if result is None:
            return False

        del self.open_positions[symbol]
        risk_manager.on_position_closed(symbol)
        risk_manager.on_trade_closed(result["pnl"])
        self.daily_pnl = getattr(risk_manager.state, "daily_pnl", 0.0)

        if position.get("strategy_id") == "telegram_signal" and result.get("trade_id"):
            await self._link_telegram_signal_trade(position.get("order_id"), result["trade_id"])

        emoji = "✅" if result["pnl"] > 0 else "❌"
        reason_ru = "Stop Loss" if reason == "stop_loss" else "Take Profit"
        logger.info(
            f"{emoji} Позиция закрыта: {symbol} {side.upper()} | {reason_ru} @ {current_price:.4f} | "
            f"PnL: {result['pnl']:+.2f} ({result['pnl_pct']:+.2f}%)"
        )
        await send_notification(
            f"{emoji} Закрыта {side.upper()} {symbol}\n"
            f"Причина: {reason_ru}\n"
            f"PnL: {result['pnl']:+.2f} USDT ({result['pnl_pct']:+.2f}%)"
        )
        return True

    async def _link_telegram_signal_trade(self, order_id: Optional[int], trade_id: int):
        """Проставить executed_trade_id у Telegram-сигнала, чей ордер только что закрылся."""
        if order_id is None:
            return
        try:
            async with get_session() as session:
                result = await session.execute(
                    select(TelegramSignal).where(TelegramSignal.executed_order_id == order_id)
                )
                signal = result.scalar_one_or_none()
                if signal:
                    signal.executed_trade_id = trade_id
                    await session.commit()
        except Exception as e:
            logger.warning(f"Не удалось связать Telegram-сигнал со сделкой #{trade_id}: {e}")

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
