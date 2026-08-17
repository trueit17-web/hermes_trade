# CryptoBot Pro — MEMORY.md

## Репозиторий
- GitHub: https://github.com/trueit17-web/hermes_trade
- Ветка: `main`
- Последний коммит: `e691441` (feat: backtest engine, WebSocket feed, decision logger, runner CLI)

## Что сделано (факты)

### 1. Архитектура
- **Event-driven**: `src/event_bus.py` — асинхронная шина событий с pub/sub, history, wildcard.
- **Конфигурация**: `src/config.py` — Pydantic Settings, `.env` → настройки. `src/config_telegram.py` — отдельный Pydantic-модели для Telegram.
- **БД**: SQLite / PostgreSQL, alembic миграции, 29 таблиц, `src/db/models.py` — все модели.
- **Логирование**: `src/utils/logging.py` — JSON-логи, ротация, colorize, event_logger.

### 2. Данные
- **CoinGlass API**: `src/data_ingest/coinglass_client.py` — OI, funding rate, liquidations, long/short ratio, fear/greed, ETF flows, CVD, каналы.
- **Feature Engine**: `src/data_ingest/feature_engine.py` — RSI, MACD, BB, ATR, OBV, Momentum, Stoch, Williams %R, MFI, + ML фичи, market regime detection, label generation (direction/volatility).
- **Market Data**: `src/data_ingest/market_data.py` — ccxt OHLCV fetch, WebSocket planner, candle buffer.
- **WebSocket Feed**: `src/data_ingest/websocket_feed.py` — реальные потоковые данные от Binance/Bybit, heartbeat, reconnection, message processing.

### 3. Стратегии (7 total)
В `src/strategy/__init__.py`:
1. `RSIStrategy` — mean-reversion, long при RSI < oversold, short при RSI > overbought
2. `EMAStrategy` — trend-following, long при EMA fast > slow (rising), short при EMA fast < slow (falling)
3. `BollingerBandsStrategy` — mean-reversion / breakout, long при price < lower BB, short при price > upper BB
4. `FundingRateStrategy` — long при funding < low_threshold, short при funding > high_threshold
5. `LiquidationStrategy` — long при price near liquidation support, short при price near liquidation resistance
6. `MLDirectionStrategy` — использует LightGBM predictions, long при proba_up > threshold
7. `EnsembleVoterStrategy` — weighted voting, weight = performance score

### 4. Риск-менеджмент
`src/risk/risk_manager.py`:
- `RiskProfile` — параметры (daily_loss_limit, max_open_positions, max_position_size_pct, max_drawdown_pct, cooldown_seconds, use_kelly, kelly_fraction, max_consective_losses, min_confidence_threshold, required_profit_factor)
- `RiskState` — current_balance, start_balance, open_positions, daily_pnl, daily_loss_limit_reached, max_drawdown_reached, kill_switch_active, paused, cooldown_until, consecutive_losses, total_trades, winning_trades, losing_trades, best_trade, worst_trade, trade_history
- `RiskManager` — can_trade(), check_signal(), adjust_position_size(), on_trade_closed(), on_position_added(), update_balance(), trigger_kill_switch(), clear_kill_switch(), pause(), resume(), record_trade_result(), compute_drawdown(), compute_sharpe_ratio(), compute_win_rate()

### 5. Исполнение
`src/execution/executor.py`:
- `ExecutionEngine` — initialize(), can_execute(), get_paper_balance(), set_paper_balance(), get_exchange_balance(), create_order(), cancel_order(), get_order_status(), modify_order(), close_position(), close_all_positions(), get_open_positions(), get_trade_history(), get_daily_pnl(), get_total_pnl(), get_bot_health()

### 6. ML
`src/ml/__init__.py`:
- `FeatureStore` — offline/online storage, query_features(), get_training_data()
- `DirectionClassifier` — LightGBM trainer (train()), predict(), predict_proba(), evaluate(), cross_validate(), save_model(), load_model(), get_feature_importance()
- `VolatilityPredictor` — LightGBM Regressor, predict_volatility()
- `SignalQualityScorer` — score_telegram_signal() для Telegram сигналов
- `EnsembleWeightOptimizer` — compute_strategy_weights() на основе performance metrics
- `ModelRegistry` — save_model(), load_model(), list_models(), get_model_info(), activate_model(), deactivate_model(), get_active_model(), get_model_metrics()

### 7. Telegram
`src/telegram/`:
- `channel_monitor.py` — `ChannelMonitor`, `init_telegram()`, `close_telegram()`, `subscribe_telegram_signal()`, `start_monitoring()`, `stop_monitoring()`, `get_monitored_channels()`, `add_channel()`, `remove_channel()`, `update_channel()`, `get_channel_stats()`.
- `signal_parser.py` — `telegram_signal_parser` (parse(), parse_with_llm()), regex patterns for standard signal format.
- `quality_scorer.py` — `signal_quality_scorer` (score_signal(), update_channel_stats(), get_channel_quality(), reset_stats()).

### 8. Web API (FastAPI)
`src/web/api.py` — endpoints:
- GET `/health`, GET `/status`, GET `/config`, GET `/performance`, GET `/positions`, GET `/trades`, GET `/orders`, GET `/strategies`, GET `/ml/models`, GET `/telegram/channels`, GET `/telegram/signals`, POST `/config/update`, POST `/risk/configure`, POST `/risk/pause`, POST `/risk/resume`, POST `/risk/kill-switch`, POST `/risk/clear-kill-switch`, POST `/strategies/{id}/toggle`, POST `/strategies/{id}/update`, POST `/ml/retrain`, POST `/ml/models/{type}/{version}/activate`, POST `/telegram/channels`, DELETE `/telegram/channels/{id}`, POST `/telegram/signals/{id}/confirm`, GET `/metrics`

`src/web/schemas.py` — Pydantic модели для всех request/response.
`src/web/websocket.py` — WebSocket connect/disconnect/send_json/broadcast_event, connect() в api.py.

### 9. Backtest
`src/backtest/engine.py` — `BacktestEngine`, `BacktestPosition`, `BacktestTrade`, `BacktestResult`, `BacktestDataLoader`, `run_example_backtest()`.

### 10. Runner CLI
`src/runner.py` — argparse CLI: `python -m src.runner run` / `backtest` / `health`.

## Файловая структура (40+ файлов)

```
crypto-bot/
├── alembic.ini
├── alembic/
│   ├── env.py
│   ├── versions/
│   │   ├── 001_initial.py        # 29 таблиц, upgrade/downgrade
│   │   └── template.py           # шаблон новых миграций
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI: lint + test + docker build
├── .pre-commit-config.yaml        # ruff, black, trailing-whitespace, EOF
├── .gitignore
├── docker-compose.yml             # postgres + redis + nginx + bot + ml-trainer
├── Dockerfile                     # python:3.11-slim → pip install → uvicorn
├── deploy.sh                      # ./deploy.sh [--update] [--data-dir]
├── run.sh                         # локальный запуск, check deps, .env, init db, run
├── requirements.txt               # все зависимости (ccxt, pandas, lightgbm, fastapi, etc.)
├── scripts/
│   ├── pre-commit-hooks.sh       # линтеры и тесты перед коммитом
│   ├── git-credential-token.py   # извлечение GitHub токена
│   └── gh-env.sh                 # настройка PATH для gh CLI
├── tests/
│   ├── test_all.py               # 105 тестов: config, event_bus, risk, стратегии, ML, execution, telegram, logging, backtest
│   └── test_backtest.py         # тесты для backtest engine
├── src/
│   ├── __init__.py               # пустой пакет
│   ├── main.py                   # TradingBot: инициализация, _trading_iteration, _process_symbol, _on_telegram_signal, _execute_telegram_signal, _cleanup, run(), main()
│   ├── config.py                 # Settings: все env vars, is_paper, is_real, VALIDATION, telegram_settings
│   ├── config_telegram.py        # TelegramConfig, TelegramChannelConfig
│   ├── event_bus.py              # EventBus, Event, MarketDataEvent, SignalGeneratedEvent, TradeEvent
│   ├── runner.py                 # CLI argparse: run / backtest / health
│   ├── db/
│   │   ├── base.py               # DeclarativeBase
│   │   ├── models.py             # 29 моделей SQLAlchemy
│   │   └── session.py            # init_db(), get_session(), AsyncSessionContext
│   ├── data_ingest/
│   │   ├── coinglass_client.py   # CoinGlassClient, все endpoint'ы
│   │   ├── feature_engine.py     # FeatureEngine (indictors + ML features + labels)
│   │   ├── market_data.py        # MarketDataIngest (ccxt + WebSocket planner)
│   │   └── websocket_feed.py     # WebSocketFeed (async reconnect, subscribe, heartbeat)
│   ├── strategy/
│   │   └── __init__.py           # 7 стратегий + StrategyRegistry
│   ├── risk/
│   │   └── risk_manager.py       # RiskProfile, RiskState, RiskManager
│   ├── execution/
│   │   ├── executor.py           # ExecutionEngine (paper + real, ccxt)
│   │   └── decision_logger.py    # DecisionLogger (market_data, strategy_signal, ml_score, risk_check, execution, finalize)
│   ├── ml/
│   │   └── __init__.py           # FeatureStore, DirectionClassifier, VolatilityPredictor, SignalQualityScorer, EnsembleWeightOptimizer, ModelRegistry
│   ├── telegram/
│   │   ├── __init__.py           # re-exports всех модулей
│   │   ├── channel_monitor.py    # ChannelMonitor, init_telegram, close_telegram, subscribe_telegram_signal
│   │   ├── signal_parser.py      # telegram_signal_parser (regex + LLM)
│   │   └── quality_scorer.py     # signal_quality_scorer (score_signal, update stats)
│   ├── web/
│   │   ├── api.py                # FastAPI: все GET/POST endpoints, WebSocket mount
│   │   ├── schemas.py            # Pydantic модели для request/response
│   │   └── websocket.py          # WebSocketManager
│   └── utils/
│       ├── crypto.py             # generate_encryption_key(), encrypt_api_key(), decrypt_api_key()
│       └── logging.py            # setup_logging(), JsonFormatter, get_logger()
├── .env.example                   # все переменные с заглушками
├── README.md                      # полная документация (614 строк)
├── README.sh                      # команды запуска
└── MEMORY.md                      # этот файл
```

## Логика модулей (кратко)

### main.py — TradingBot
1. `initialize()`:
   - setup_logging()
   - init_db()
   - get_feature_engine() → self.feature_engine
   - get_coinglass_client() → self.cg_client
   - get_ws_feed("binance") → self.ws_feed (try/except, fallback)
   - MarketDataIngest("binance") → self.ingest
   - fetch OHLCV 200 свечей для default_symbols, update_buffer
   - ml_inference = ml_inference; load_model если есть
   - execution_engine.initialize("binance")
   - init_telegram() если telegram_api_id/api_hash
   - subscribe_telegram_signal(self._on_telegram_signal)
   - AsyncIOScheduler: coinglass_updater (hours), ml_retrainer (hours)
   - self.running = True

2. `_update_coinglass()`:
   - get_coins_markets("BTC"), get_funding_rate_history, get_open_interest_history, get_fear_greed_history
   - logger.debug

3. `_retrain_ml()`:
   - query trades count из БД
   - если >= 50 → model_trainer.train_direction_classifier()
   - если result → model_registry.activate_model(); ml_inference.load_model()

4. `_on_telegram_signal(signal_event)`:
   - parse pair, side, entry, sl, tp
   - если нет пары/сайда/entry > 0 → return
   - signal_quality_scorer.score_signal(signal_event, channel_id)
   - если quality < settings.telegram_signals_quality_threshold → log reject
   - если settings.telegram_signals_auto_execute → _execute_telegram_signal
   - иначе log "ожидает подтверждения"

5. `_execute_telegram_signal(signal_event)`:
   - parse pair, side, entry, sl, tp
   - symbol = pair
   - order_side = "buy" if side == "long" else "sell"
   - balance = execution_engine.get_paper_balance() (paper) или 10000 (real fallback)
   - size_pct = 5.0 (hardcoded)
   - position_value = balance * (size_pct / 100)
   - amount = position_value / entry
   - execution_engine.create_order(symbol, order_side, amount, "market", entry, sl, tp)
   - log "Ордер исполнен: {order.client_order_id}"

6. `run()`:
   - если не running → initialize()
   - logger.info "Запуск основного цикла"
   - ws_task = asyncio.create_task(self.ws_feed.start()) если ws_feed
   - while self.running:
     - try: await self._trading_iteration(); await asyncio.sleep(60)
     - except asyncio.CancelledError: break
     - except Exception: logger.error; await asyncio.sleep(60)
   - если ws_task: ws_task.cancel()
   - await _cleanup()

7. `_trading_iteration()`:
   - если kill_switch_active или paused → return
   - daily PnL reset (если дата изменилась)
   - для symbol in settings.default_symbols: await self._process_symbol(symbol)

8. `_process_symbol(symbol)`:
   - df = self.candles_buffer.get(symbol)
   - если None/empty/<50 → fetch OHLCV 50; update_buffer; self.candles_buffer[symbol] = df
   - если после fetch всё ещё None/empty/<50 → return
   - latest = df.iloc[-1]; close = float(latest["close"]); self.last_prices[symbol] = close
   - features = self.feature_engine.compute_all_indicators(df); latest_features = features.iloc[-1]
   - strategy_data = {symbol, timeframe, close, rsi_14, rsi_7, rsi_21, macd, macd_signal, macd_hist, bb_upper, bb_lower, bb_mid, bb_pct, bb_width, ema_20, ema_50, ema_20_slope, ema_50_slope, price_above_ema20, price_above_ema50, atr_14, natr_14, realized_vol_20, volume_ratio, obv, return_1, return_3, return_5, log_return, momentum_10, dist_from_ema20, dist_from_ema50, high_low_range, stoch_k, stoch_d, wr_14, mfi_14, hour, day_of_week}
   - если self.ml_inference: ml_result = self.ml_inference.predict_direction(strategy_data); если ml_result → strategy_data["ml_proba_up"] = ml_result.get("proba_up"); strategy_data["ml_proba_down"] = ml_result.get("proba_down"); strategy_data["ml_proba_neutral"] = ml_result.get("proba_neutral")
   - decision_logger.log_market_data(symbol=symbol, timeframe="1h", price=close, features={...[:15]})
   - signals = []
   - для strategy in strategy_registry.get_active(): signal = strategy.generate_signal(strategy_data); если signal → signals.append(signal); decision_logger.log_strategy_signal(...)
   - если "ml_proba_up" в strategy_data: decision_logger.log_ml_score(...)
   - ensemble = strategy_registry.get("ensemble_voter"); если ensemble и signals: для s in signals: ensemble.set_strategy_weight(s.strategy_id, s.weight); aggregated = ensemble.aggregate_signals(signals); если aggregated → signals = [aggregated]
   - если not signals → return
   - для signal in signals: can_execute, reason = risk_manager.check_signal(signal); decision_logger.log_risk_check(...)
   - если not can_execute → log reject; continue
   - balance = execution_engine.get_paper_balance() (paper) или 10000 (real fallback)
   - size_pct = signal.position_size_pct
   - position_value = balance * (size_pct / 100)
   - entry_price = signal.entry_price if signal.entry_price > 0 else close
   - amount = position_value / entry_price if entry_price > 0 else 0
   - если amount <= 0 → continue
   - order_side = "buy" if signal.side == "long" else "sell"
   - logger.info "Ордер: {order_side} {amount} {symbol} @ {entry_price} | Conf: {confidence} | SL: {stop_loss} TP: {take_profit}"
   - decision_logger.log_execution(order_id="pending", ...)
   - order = await execution_engine.create_order(symbol=symbol, side=order_side, amount=amount, price=entry_price, order_type="market", stop_loss=signal.stop_loss, take_profit=signal.take_profit, strategy_id=1, signal_data={...})
   - если order: decision_logger.log_execution(order_id=order.client_order_id, ...); self.open_positions[symbol] = {...}; risk_manager.on_position_added(symbol, size_pct); self.daily_pnl = getattr(risk_manager.state, "daily_pnl", 0.0); logger.info "Ордер: {order.client_order_id}"

9. `_cleanup()`:
   - ws_feed.close() если есть
   - ingest.close() если есть
   - cg_client.close() если есть
   - scheduler.shutdown() если есть
   - close_telegram() если есть

10. `main()`:
    - bot = TradingBot(); try: await bot.run(); except KeyboardInterrupt: await bot._cleanup(); sys.exit(0); except Exception: logger.critical; traceback.print_exc(); sys.exit(1)

### config.py — Settings
- trading_mode, startup_capitol_usdt, log_level, database_url
- binance_api_key, binance_api_secret, bybit_api_key, bybit_api_secret
- telegram_api_id, telegram_api_hash, telegram_user_id
- telegram_bot_token, telegram_chat_id, telegram_channels
- telegram_signals_auto_execute, telegram_signals_quality_threshold
- coinglass_api_key, coinglass_update_interval_hours, coinglass_retraining_after_ticks
- ml_retraining_interval_hours, ml_min_trades_for_training
- default_symbols, default_timeframes
- risk_* (все параметры)
- concurrency_* (max_workers, enable_multiprocessing)
- is_paper, is_real (property)
- VALIDATION (classmethod): проверяет coinGlass API key если есть, проверяет telegram credentials если есть, проверяет банк

### event_bus.py — EventBus
- _subscribers: Dict[str, List[Callable]]
- _history: List[Event]
- subscribe(event_type, callback)
- unsubscribe(event_type, callback)
- async publish(event)
- subscribe_all(callback)
- async publish_all(event)
- get_history(event_type=None, limit=None)
- clear_history()
- Event dataclass: type, source, payload, timestamp
- MarketDataEvent(Event): symbol, timeframe, candle, ticker
- SignalGeneratedEvent(Event): strategy_id, strategy_name, signal, symbol
- TradeEvent(Event): trade_id, order_open_id, order_close_id, symbol, direction, entry_price, exit_price, pnl, pnl_pct, status

### db/models.py — 29 моделей
- Exchange, ApiKey, Trade, Position, Order, Strategy, Signal, TelegramChannel, TelegramSignal, TelegramMessage, CoinGlassData, MLModel, MLFeature, BacktestResult, Config, DecisionLog, HealthCheck, LogEvent, WebSocketConnection, StrategyPerformance, RiskParameter, UserAction, Notification, Alert, PortfolioSnapshot, OrderBook, Market snapshot

### strategy/__init__.py — 7 стратегий
- SignalData (dataclass): symbol, timeframe, close, rsi_14, macd_histogram, bb_upper, bb_lower, bb_pct, ema_fast, ema_slow, volume_ratio, atr, funding_rate, liquidation_map, ml_proba_up, ml_proba_down, ml_proba_neutral, strategy_id=0, weight=1.0
- Signal (dataclass): side, entry_price, stop_loss, take_profit, confidence, strategy_id, strategy_name, position_size_pct, rationale, extras, timestamp
- BaseStrategy (ABC): strategy_id, name, description, params, weight, generate_signal(data) -> Optional[Signal], should_execute(data) -> bool, get_metrics() -> dict, reset_metrics()
- RSIStrategy, EMACrossoverStrategy, BollingerBandsStrategy, FundingRateStrategy, LiquidationStrategy, MLDirectionStrategy, EnsembleVoterStrategy
- StrategyRegistry: singleton, register(), get(), get_all(), get_active(), update_weight(), set_enabled(), get_strategy_metrics()

### ml/__init__.py — ML pipeline
- FeatureStore: _offline_db, _online_db, _preprocessor, add(), query_features(), get_training_data(), get_feature importance
- DirectionClassifier: LightGBMClassifier, train(), predict(), predict_proba(), evaluate(), cross_validate(), save_model(), load_model(), get_feature_importance()
- VolatilityPredictor: LightGBMRegressor, train(), predict(), predict_volatility(), evaluate()
- SignalQualityScorer: score_signal(), score_llm_signal(), _extract_features(), _compute_confidence(), _check_format_consistency()
- EnsembleWeightOptimizer: compute_strategy_weights(), update_weights(), get_weights(), _compute_sharpe(), _compute_win_rate(), _compute_drawdown(), _compute_profit_factor()
- ModelRegistry: _active_models, _model_store, _performance_history, _degradation_threshold, register(), save_model(), load_model(), list_models(), get_model_info(), activate_model(), deactivate_model(), get_active_model(), get_model_metrics(), _detect_degradation(), _compute_ensemble_weights()

### telegram/channel_monitor.py — ChannelMonitor
- _client (telethon client), _channels (Dict[str, TelegramChannelConfig]), _signal_queue, _running, _event_handlers, _message_history (Dict[str, List[Message]])
- init_telegrass() → Client() с api_id/hash; connect(); is_user_authorized(); send_code_request(); sign_in(); run_until_disconnected()
- close_telegram() → if _client: _client.disconnect()
- subscribe_telegram_signal(callback) → event_bus.subscribe("telegram_signal", callback)
- start_monitoring() → for channel in self._channels.values(): self._client.on_message = partial(self._on_message, channel_id=channel.id); self._client.add_event_handler(Callback, self._on_message)
- _on_message(event) → self._message_history.setdefault(channel_id, []).append(event.message); signal_data = parse_signal_text(event.message.text, channel_id); если signal_data: signal_data.update(channel_id=channel_id, message_id=event.message.id, message_date=event.message.date, raw_text=event.message.text); event_bus.publish(Event(type="telegram_signal", source="telegram", payload=signal_data))
- get_monitored_channels() → list(self._channels.values())
- add_channel(channel) → self._channels[channel.channel_id] = channel
- remove_channel(channel_id) → self._channels.pop(channel_id, None)
- update_channel(channel) → self._channels[channel.channel_id] = channel
- get_channel_stats(channel_id=None) → если channel_id: return self._channel_stats.get(channel_id, {}); иначе return self._channel_stats

### telegram/signal_parser.py — telegram_signal_parser
- parse(text, channel_id=None) → Regex patterns: pair_pattern, action_pattern, entry_pattern, sl_pattern, tp_pattern, size_pattern, price_pattern; если match: return dict(pair, side, entry, sl, tp, size, price, timeframe, confidence, raw_text, channel_id)
- parse_with_llm(text, llm_client) → llm_client.parse_telegram_signal(text)

### telegram/quality_scorer.py — signal_quality_scorer
- _channel_stats (Dict[str, ChannelStats])
- score_signal(signal_data, channel_id) → historical = self._channel_stats.get(channel_id); score = (historical["win_rate"] * 0.3 + successful_signals_ratio * 0.3 + accuracy_bonus * 0.2 + channel_consistency * 0.2) если historical else _score_no_history(signal_data)
- _score_no_history(signal_data) → confidence_score → 0.2; signal_quality_score ≥ 0.6 → +0.2; risk_reward_ratio ≥ 1.5 → +0.15; signal_size <= 10 → +0.15; volume > 1.5 * avg_volume → +0.15; timeframe in ["1m","5m","15m"] → +0.1; итого score = min(1.0, score)
- update_channel_stats(channel_id, signal_data, was_successful) → stats = self._channel_stats.get(channel_id, {"signals_count":0, "good_signals":0, "bad_signals":0, "win_rate":0.5}); stats["signals_count"] += 1; если was_successful: stats["good_signals"] += 1; stats["win_rate"] = stats["good_signals"] / stats["signals_count"]; иначе: stats["bad_signals"] += 1
- get_channel_quality(channel_id) → stats["win_rate"]
- reset_stats() → self._channel_stats = {}

### execution/executor.py — ExecutionEngine
- _exchange, _config, _is_paper, _paper_balance, _paper_positions, _direction, _parser, _open_orders, _order_history, _trade_history, _strategy_inst, _kill_switch_active, _orders_lock, _positions_lock
- initialize(exchange_id) → self._parser = ExchangeParser(exchange_id); self._exchange = await self._parser.get_exchange() (если real) или MockExchange (если paper); self._strategy_inst = StrategyRegistry.get("simple_momentum")
- can_execute() → return not self._kill_switch_active
- get_paper_balance() → return self._paper_balance
- set_paper_balance(balance) → self._paper_balance = balance
- get_exchange_balance() → async: if real: return await self._exchange.fetch_balance(); иначе return {"total": self._paper_balance, "free": self._paper_balance, "used": 0}
- create_order(...) → если paper: если side == sell и symbol в paper_positions → amount = min(amount, available); self._paper_balance += amount * current_price; удалить позицию; _paper_positions.remove(pos); paper_trade = PaperTrade(...); self._trade_history.append(paper_trade); PaperOrder(...); self._open_orders.append(paper_order); self._order_history.append(paper_order); return paper_order; иначе: ребаланс; return None; иначе (real): exchange_order = await self._exchange.create_order(...); self._open_orders[order_id] = exchange_order; self._order_history.append(exchange_order); return exchange_order
- cancel_order(order_id) → если real: await self._exchange.cancel_order(order_id); self._open_orders.pop(order_id, None); иначе: self._open_orders.pop(order_id, None)
- get_order_status(order_id) → return self._open_orders.get(order_id) or self._order_history return None
- modify_order(order_id, **params) → real: await self._exchange.cancel_order(order_id); new_order = await self._exchange.create_order(...); self._open_orders[order_id] = new_order; иначе: self._open_orders.pop(order_id, None); return None
- close_position(symbol, side, amount) → close_order = await self.create_order(symbol=symbol, side=opposite_side, amount=amount, order_type="market", price=None, stop_loss=None, take_profit=None, params={"reduce_only": True}); если close_order: open_order.close_order_id = close_order.id; self._trade_history.append(Trade(...)) или self._open_orders.remove(open_order)
- close_all_positions() → для pos в list(self._paper_positions): await self.close_position(pos.symbol, pos.side, pos.amount); log "Все позиции закрыты"
- get_open_positions() → return list(self._paper_positions)
- get_trade_history() → return list(self._trade_history)
- get_daily_pnl() → today = datetime.utcnow().date(); return sum(t.pnl for t in self._trade_history if t.opened_at.date() == today)
- get_total_pnl() → return sum(t.pnl for t in self._trade_history)
- get_bot_health() → return {"running": True, "mode": "paper"/"real", "paper_balance": self._paper_balance, "open_positions": len(self._paper_positions), "open_orders": len(self._open_orders), "kill_switch_active": self._kill_switch_active}

### decision_logger.py — DecisionLogger
- active_trade (Optional[Dict]), _steps (List[Dict]), engineer (DecisionEngineEngineer), _trade_id_counter
- start_trade() → self.active_trade = {"trade_id": self._trade_id_counter, "open_time": datetime.utcnow().isoformat(), "steps": []}; self._steps = []; self._trade_id_counter += 1; return self.active_trade
- finalize_trade(success=True, pnl=None, pnl_pct=None, exit_reason="closed") → trade_id = self.active_trade["trade_id"]; log_event_db(...); self.active_trade = None; self._steps = []; return trade_id
- log_market_data(symbol, timeframe, price, features) → step_type="market_data"; description=f"{symbol} @ {timeframe}: {price}"
- log_strategy_signal(strategy_id, strategy_name, signal_side, confidence, entry_price, stop_loss, take_profit, rationale) → step_type="strategy_signal"
- log_ml_score(model_type, model_version, proba_up, proba_down, proba_neutral) → step_type="ml_score"
- log_risk_check(decision, reason, context) → step_type="risk_check"; Если decision=="rejected": self.active_trade["steps"].append({"step_type":"risk_check", "result":"rejected", "reason": reason, ...})
- log_execution(order_id, order_type, amount, price, status, fee) → step_type="execution"
- _build_step(step_type, **kwargs) → {"step_order": len(self._steps)+1, "step_type": step_type, ...}

## Изменения после последнего коммита (e691441)
- Добавлены: `src/__init__.py`, `src/backtest/__init__.py`, `src/backtest/engine.py`, `src/data_ingest/websocket_feed.py`, `src/execution/decision_logger.py`, `src/runner.py`, `tests/test_backtest.py`
- Изменены: `src/main.py` (обновлён для интеграции websocket_feed, decision_logger, backtest, runner), `tests/test_all.py` (дополнен тестами)
- Telegram `__init__.py`: добавлен (ре-экспорт модулей)
- `config_telegram.py`: добавлен (Pydantic модели)

## Что есть в тестах (test_all.py + test_backtest.py)
- TestConfig (4 теста)
- TestEventBus (5 асинхронных тестов)
- TestRiskProfile (2 теста)
- TestRiskState (6 тестов)
- TestRiskManager (12 асинхронных тестов)
- TestRSIMeanReversionStrategy (3 теста)
- TestEMACrossoverStrategy (3 теста)
- TestBollingerBandsStrategy (3 теста)
- TestEnsembleVoterStrategy (3 теста)
- TestFeatureEngine (5 тестов)
- TestCoinGlassClient (1 тест)
- TestExecutionEngine (4 асинхронных теста)
- TestTelegramSignalParser (6 тестов)
- TestQualificationScorer (3 теста)
- TestLogging (2 теста)
- TestBacktestPosition (2 теста)
- TestBacktestResult (3 теста)
- TestDecisionLogger (5 тестов)
- TestBacktestEngine (из test_backtest.py) — и другие

## О-known issues / TODO
- `.hermes-tmp.QOysV3` — временный файл, удалён из индекса
- `ccxt.pro` нет в PyPI — используем `ccxt` + WebSocket напрямую или планируем заменить на `websockets` + кастомные парсеры
- WebSocket feed пока не интегрирован в main.py (получен, но не вызван в initialize/run)
- Backtest engine написан, но не вызывается из main.py или runner.py (есть интерфейс, но не задействован)
- Telegram сигналы работают только через event_bus.subscribe_telegram_signal — в main.py есть обработчик, но telethon клиент ещё не полностью настроен (api_id/api_hash из .env)
