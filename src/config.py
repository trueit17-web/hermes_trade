"""Конфигурация бота — Pydantic Settings."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки бота из .env файла."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Режим торговли: paper или real
    trading_mode: str = "paper"

    # Источник новых сигналов: "signals" — только Telegram-каналы,
    # "algo" — только встроенные ML/Ensemble/BB-стратегии. Переключается
    # кнопкой в шапке дашборда (POST /trading-source-mode). Уже открытые
    # позиции (любого источника) не затрагивает — SL/TP проверяются
    # всегда, режим гейтит только открытие НОВЫХ позиций (см.
    # TradingBot._process_symbol/_on_telegram_signal в main.py). По
    # умолчанию "signals" — совмещение обоих источников в одном
    # торговом контуре искажало статистику и риск друг друга.
    active_trading_mode: str = "signals"

    # Стартовый капитал (USDT) — виртуальный для paper, реальный для real
    startup_capital_usdt: float = 10000.0

    # Уровень логирования
    log_level: str = "INFO"

    # === Биржи ===
    # Активная биржа для real-режима (paper всегда работает поверх данных
    # Binance независимо от этого поля — см. main.py: MarketDataIngest).
    active_exchange: str = "binance"
    # Демо/testnet-счёт через тот же API вместо реальных денег (ccxt
    # set_sandbox_mode) — включено по умолчанию, чтобы переключение в
    # real-режим само по себе не начинало торговать настоящими деньгами.
    use_exchange_sandbox: bool = True
    # Тип рынка для real-режима: "spot" (обычный спот — шорт не
    # поддерживается, см. _execute_real_order) или "futures" (USDT-perpetual,
    # linear swap). Открытие/закрытие long и short на фьючерсах реализовано
    # (_execute_real_order/close_real_position), клиенты spot/futures
    # держатся раздельно и одновременно, если реальные позиции есть на
    # обоих рынках (ExecutionEngine._exchanges, см. _exchange_for). Биржевой
    # SL теперь ставится и на фьючерсах, симметрично споту
    # (_place_stop_loss_order сам выбирает sell/buy по стороне позиции) —
    # позиция защищена и биржевым стоп-ордером, и внутренним поллингом цены
    # в _check_position_exit (main.py). Тейк-профиты (TP1/TP2/TP3) остаются
    # ТОЛЬКО внутренней логикой на обоих рынках — осознанно, у Bybit нет
    # OCO-механизма частичного выхода по нескольким уровням. Дуст-сверка
    # позиций для фьючерсов (reconcile_real_positions читает через
    # fetch_balance — для фьючерсов это не применимо, нужен
    # fetch_positions) — ещё не реализована, отдельный последующий этап.
    # Переключается кнопкой в шапке дашборда (POST /market-type).
    market_type: str = "spot"
    # Плечо для фьючерсных ордеров (market_type=="futures") — глобальный
    # ДЕФОЛТ, когда сигнал/канал не задаёт своё. Telegram-канал может явно
    # указать плечо в тексте сигнала ("Кредитное плечо: х35" — см.
    # extract_leverage в channel_monitor.py) — тогда ИМЕННО оно
    # применяется через set_leverage для этого ордера (см.
    # _execute_real_order в executor.py), а не эта настройка. Полноценный
    # риск-guard по ликвидации/плечу — отдельный, более поздний этап.
    # Дефолт 1.0 — без усиления, безопасная отправная точка.
    futures_leverage: float = 1.0
    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    bybit_api_key: str | None = None
    bybit_api_secret: str | None = None
    okx_api_key: str | None = None
    okx_api_secret: str | None = None
    # OKX, в отличие от Binance/Bybit, требует третий секрет (passphrase,
    # задаётся при создании API-ключа на бирже) в каждом запросе.
    okx_passphrase: str | None = None

    # === CoinGlass API ===
    coinglass_api_key: str | None = None

    # === Telegram мониторинг ===
    telegram_api_id: int | None = None
    telegram_api_hash: str | None = None
    telegram_user_id: int | None = None

    # === Telegram уведомления ===
    telegram_bot_token: str | None = None
    telegram_chat_id: int | None = None

    # === База данных ===
    database_url: str = "sqlite+aiosqlite:///./data/cryptobot.db"

    # === Redis (опционально, для production event bus) ===
    redis_url: str | None = None

    # === Шифрование ===
    encryption_key: str | None = None

    # === ML ===
    ml_retraining_interval_hours: int = 6
    ml_max_trades_for_retrain: int = 200
    ml_active_model_version: int = 1
    ml_optuna_trials: int = 30
    ml_optuna_min_samples: int = 150

    # === Риск ===
    risk_daily_loss_limit_usd: float = 500.0
    risk_max_open_positions: int = 8
    risk_max_position_size_pct: float = 10.0
    risk_max_drawdown_pct: float = 15.0
    risk_cooldown_seconds: int = 300
    # Trailing stop-loss: 0 = выключен. SL подтягивается к текущей цене на
    # trailing_stop_pct и только ужесточается (никогда не откатывается назад).
    trailing_stop_pct: float = 0.0

    # === ATR-адаптивный SL/TP (только для сигналов от стратегий —
    # Telegram-сигналы несут собственные уровни от канала и не трогаются).
    # SL = ATR(14) * atr_sl_multiplier; TP = SL * R:R (R:R зависит от типа
    # стратегии — трендовые шире, контртрендовые уже, см. main.py
    # ATR_TREND_STRATEGY_IDS). Выключено по умолчанию — включается осознанно
    # из дашборда, как и volatility_adjustment_enabled ниже; при включении
    # ATR-уровни заменяют фиксированный %-ный SL/TP стратегии ДО того, как
    # к ним применится масштабирование по предсказанной волатильности
    # (volatility_adjustment_enabled) — эти два механизма совместимы и не
    # дублируют друг друга: ATR даёт базовую ширину, volatility её донастраивает. ===
    atr_sltp_enabled: bool = False
    atr_sl_multiplier: float = 1.8
    atr_tp_rr_trend: float = 3.0
    atr_tp_rr_countertrend: float = 2.0

    # === Volatility adjustment (predicted volatility -> размер позиции и
    # ширина SL/TP) — только для сигналов от стратегий (strategy_registry);
    # Telegram-сигналы несут собственные уровни от канала и не трогаются. ===
    volatility_adjustment_enabled: bool = False
    # "Типичная" волатильность (%), с которой сравнивается предсказание
    # volatility_predictor, чтобы получить коэффициент масштабирования.
    volatility_baseline_pct: float = 2.0
    # Предсказанная волатильность ВЫШЕ базовой -> позиция МЕНЬШЕ (защита от
    # шума на резких движениях), диапазон коэффициента ограничен снизу/сверху.
    volatility_size_min_mult: float = 0.5
    volatility_size_max_mult: float = 1.5
    # Предсказанная волатильность ВЫШЕ базовой -> SL/TP ШИРЕ (чтобы не
    # выбивало шумом раньше времени), диапазон ограничен снизу/сверху.
    volatility_sltp_min_mult: float = 0.5
    volatility_sltp_max_mult: float = 2.0

    # === Protections (freqtrade-style автопаузы после плохой серии сделок) ===
    protections_enabled: bool = True
    # После любого полного закрытия сделки источник (канал/стратегия) не
    # торгует это время — не путать с risk_cooldown_seconds (тот блокирует
    # ВСЮ торговлю, а не только источник, из которого пришла сделка).
    protections_channel_cooldown_minutes: int = 15
    # StoplossGuard: N стопов за окно minutes -> пауза ВСЕЙ торговли на lock_min.
    protections_stoploss_guard_window_min: int = 60
    protections_stoploss_guard_count: int = 4
    protections_stoploss_guard_lock_min: int = 120
    # LosingStreak: N убыточных закрытий подряд у одного источника -> блокировка
    # только этого источника (канала/стратегии) на lock_min.
    protections_losing_streak_count: int = 3
    protections_losing_streak_lock_min: int = 180

    # === Expectancy-based sizing (по источнику сигнала: канал/стратегия) ===
    expectancy_sizing_enabled: bool = False
    expectancy_sizing_min_trades: int = 5
    expectancy_sizing_max_trades: int = 50
    # Средний % доходности на сделку, ниже/при котором источник пропускается
    # целиком (множитель 0). 0.0 = пропускать только источники в минусе.
    expectancy_sizing_min_expectancy_pct: float = 0.0

    # === CoinGlass ===
    coinglass_update_interval_hours: int = 1

    # === Telegram сигналы ===
    telegram_signals_auto_execute: bool = False
    telegram_signals_quality_threshold: float = 0.5
    # Дефолтный защитный SL (% от entry), если канал сам не указал уровень —
    # без него позиция открывалась бы вообще без биржевого стоп-ордера (см.
    # TradingBot._execute_telegram_signal в main.py). 0 — не подставлять
    # (открывать как раньше, полностью без SL).
    telegram_signals_default_sl_pct: float = 3.0
    # LLM-фолбэк парсинга: когда регулярки не смогли распознать сообщение,
    # пробуем через Anthropic API (перенос из clonerbot: parser/llm_parser.py),
    # а если он не настроен/не смог — через Gemini API (второй уровень,
    # бесплатный по тарифу вариант, см. src/telegram/gemini_parser.py).
    telegram_llm_fallback_enabled: bool = False
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5-20251001"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"

    # === Веб сервер ===
    web_host: str = "0.0.0.0"
    web_port: int = 8000
    web_admin_username: str = "admin"
    web_admin_password: str = "changeme"
    web_cookie_secure: bool = False

    # === Деплой-агент (кнопка "Редеплой" в дашборде) ===
    # Отдельный сервис ВНЕ контейнера бота (см. scripts/deploy_agent.py,
    # docker-compose.yml) — сам бот НЕ получает доступ к docker.sock хоста,
    # только шлёт этому сервису HTTP-запрос с общим секретом. deploy_agent_url —
    # адрес агента (в docker-compose сети — http://deploy-agent:8091).
    deploy_agent_url: str | None = None
    deploy_agent_token: str | None = None

    # === Торговая вселенная ===
    # Вместо фиксированного списка пар — торгуются все активные spot-пары
    # с котировкой symbol_quote_currency, кроме symbol_blacklist (топ
    # symbol_universe_max по 24ч объёму; см. MarketDataIngest.get_tradable_symbols).
    symbol_quote_currency: str = "USDT"
    symbol_blacklist: list[str] = []
    symbol_universe_max: int = 30
    symbol_universe_refresh_hours: int = 12
    default_timeframes: list[str] = ["1h", "4h"]
    candlesticks_cache_size: int = 500

    # === Slippage симуляция (paper) ===
    paper_slippage_pct: float = 0.05
    paper_fee_pct: float = 0.1

    # === Аналитика ===
    performance_snapshot_interval_hours: int = 1

    @property
    def data_dir(self) -> Path:
        """Директория для данных (логи, модели, БД)."""
        return Path(__file__).parent.parent / "data"

    @property
    def is_paper(self) -> bool:
        """Бот в paper режиме?"""
        return self.trading_mode.lower() == "paper"

    @property
    def is_real(self) -> bool:
        """Бот в real режиме?"""
        return self.trading_mode.lower() == "real"


# Глобальный экземпляр настроек
settings = Settings()
