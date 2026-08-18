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

    # === CoinGlass ===
    coinglass_update_interval_hours: int = 1

    # === Telegram сигналы ===
    telegram_signals_auto_execute: bool = False
    telegram_signals_quality_threshold: float = 0.5

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
