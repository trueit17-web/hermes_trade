"""Конфигурация бота — Pydantic Settings."""
from pathlib import Path
from typing import Optional

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

    # Стартовый капитал (USDT) — виртуальный для paper, реальный для real
    startup_capital_usdt: float = 10000.0

    # Уровень логирования
    log_level: str = "INFO"

    # === Биржи ===
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    bybit_api_key: Optional[str] = None
    bybit_api_secret: Optional[str] = None

    # === CoinGlass API ===
    coinglass_api_key: Optional[str] = None

    # === Telegram мониторинг ===
    telegram_api_id: Optional[int] = None
    telegram_api_hash: Optional[str] = None
    telegram_user_id: Optional[int] = None

    # === Telegram уведомления ===
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[int] = None

    # === База данных ===
    database_url: str = "sqlite+aiosqlite:///./data/cryptobot.db"

    # === Redis (опционально, для production event bus) ===
    redis_url: Optional[str] = None

    # === Шифрование ===
    encryption_key: Optional[str] = None

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
    # LLM-фолбэк парсинга: когда регулярки не смогли распознать сообщение,
    # пробуем через Anthropic API (перенос из clonerbot: parser/llm_parser.py).
    telegram_llm_fallback_enabled: bool = False
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # === Веб сервер ===
    web_host: str = "0.0.0.0"
    web_port: int = 8000
    web_admin_username: str = "admin"
    web_admin_password: str = "changeme"
    web_cookie_secure: bool = False

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
