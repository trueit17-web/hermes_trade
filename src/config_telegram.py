"""
Telegram configuration schema.
"""

from pydantic import BaseModel, Field


class TelegramConfig(BaseModel):
    enabled: bool = False
    api_id: int = 0
    api_hash: str = ""
    phone: str = ""
    channels: list[str] = Field(default_factory=list)
    signal_quality_threshold: float = 0.5
    auto_execute: bool = False
    require_confirmation: bool = True
    max_signals_per_hour: int = 10
    allowed_symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    dry_run: bool = True  # Если True — только логируем, не исполняем


class TelegramChannelConfig(BaseModel):
    channel_name: str  # Например, "@crypto_signals_vip"
    channel_id: int = 0  # Числовой ID канала
    role: str = "signal"  # signal, news, analysis
    priority: int = 1
    keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
