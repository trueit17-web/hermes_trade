"""Реестр редактируемых настроек бота: метаданные для UI + применение/сохранение через BotConfig.

Позволяет менять большинство полей src.config.settings из веб-панели без
захода на сервер: изменения применяются немедленно к живому объекту
settings и сохраняются в таблицу bot_config, откуда подхватываются при
следующем старте бота (см. load_settings_overrides(), вызывается из
TradingBot.initialize()).
"""
from typing import Any, Optional

from sqlalchemy import select

from src.config import settings
from src.db.session import get_session
from src.db.models import BotConfig
from src.utils.logging import logger

# Поля БД/инфраструктуры сюда намеренно не включены (database_url, web_host,
# web_port, redis_url, encryption_key, web_admin_*) — их смена на лету либо
# бессмысленна (порт/хост уже забинжены), либо небезопасна через тот же UI,
# который они защищают.
SETTINGS_SCHEMA: list[dict] = [
    {"key": "trading_mode", "label": "Режим торговли", "group": "Общие", "type": "select", "options": ["paper", "real"]},
    {"key": "startup_capital_usdt", "label": "Стартовый капитал (USDT)", "group": "Общие", "type": "float"},
    {"key": "log_level", "label": "Уровень логирования", "group": "Общие", "type": "select", "options": ["DEBUG", "INFO", "WARNING", "ERROR"]},
    {"key": "default_timeframes", "label": "Таймфреймы (через запятую)", "group": "Общие", "type": "list"},
    {"key": "candlesticks_cache_size", "label": "Размер кэша свечей", "group": "Общие", "type": "int"},

    {"key": "symbol_quote_currency", "label": "Quote-валюта (напр. USDT)", "group": "Торговая вселенная", "type": "str"},
    {"key": "symbol_blacklist", "label": "Блэклист пар (через запятую)", "group": "Торговая вселенная", "type": "list"},
    {"key": "symbol_universe_max", "label": "Макс. пар в работе (топ по объёму)", "group": "Торговая вселенная", "type": "int"},
    {"key": "symbol_universe_refresh_hours", "label": "Обновление вселенной (ч)", "group": "Торговая вселенная", "type": "int"},

    {"key": "risk_daily_loss_limit_usd", "label": "Дневной лимит убытка (USD)", "group": "Риск", "type": "float"},
    {"key": "risk_max_open_positions", "label": "Макс. открытых позиций", "group": "Риск", "type": "int"},
    {"key": "risk_max_position_size_pct", "label": "Макс. размер позиции (%)", "group": "Риск", "type": "float"},
    {"key": "risk_max_drawdown_pct", "label": "Макс. просадка (%)", "group": "Риск", "type": "float"},
    {"key": "risk_cooldown_seconds", "label": "Cooldown после стоп-лосса (сек)", "group": "Риск", "type": "int"},
    {"key": "trailing_stop_pct", "label": "Trailing stop (%, 0 = выключен)", "group": "Риск", "type": "float"},

    {"key": "protections_enabled", "label": "Protections включены", "group": "Protections", "type": "bool"},
    {"key": "protections_channel_cooldown_minutes", "label": "Cooldown источника после закрытия (мин)", "group": "Protections", "type": "int"},
    {"key": "protections_stoploss_guard_window_min", "label": "StoplossGuard: окно (мин)", "group": "Protections", "type": "int"},
    {"key": "protections_stoploss_guard_count", "label": "StoplossGuard: стопов в окне", "group": "Protections", "type": "int"},
    {"key": "protections_stoploss_guard_lock_min", "label": "StoplossGuard: блокировка всей торговли (мин)", "group": "Protections", "type": "int"},
    {"key": "protections_losing_streak_count", "label": "LosingStreak: убытков подряд", "group": "Protections", "type": "int"},
    {"key": "protections_losing_streak_lock_min", "label": "LosingStreak: блокировка источника (мин)", "group": "Protections", "type": "int"},

    {"key": "expectancy_sizing_enabled", "label": "Sizing по мат. ожиданию источника включён", "group": "Protections", "type": "bool"},
    {"key": "expectancy_sizing_min_trades", "label": "Sizing: мин. сделок для оценки", "group": "Protections", "type": "int"},
    {"key": "expectancy_sizing_max_trades", "label": "Sizing: окно последних сделок", "group": "Protections", "type": "int"},
    {"key": "expectancy_sizing_min_expectancy_pct", "label": "Sizing: мин. средний PnL% (иначе источник пропускается)", "group": "Protections", "type": "float"},

    {"key": "ml_retraining_interval_hours", "label": "Интервал переобучения (ч)", "group": "ML", "type": "int"},
    {"key": "ml_max_trades_for_retrain", "label": "Сделок для переобучения", "group": "ML", "type": "int"},
    {"key": "ml_optuna_trials", "label": "Число Optuna trials", "group": "ML", "type": "int"},
    {"key": "ml_optuna_min_samples", "label": "Мин. сэмплов для тюнинга гиперпараметров", "group": "ML", "type": "int"},

    {"key": "paper_slippage_pct", "label": "Paper: слиппедж (%)", "group": "Paper trading", "type": "float"},
    {"key": "paper_fee_pct", "label": "Paper: комиссия (%)", "group": "Paper trading", "type": "float"},

    {"key": "performance_snapshot_interval_hours", "label": "Интервал снимка производительности (ч)", "group": "Аналитика", "type": "int"},

    {"key": "coinglass_update_interval_hours", "label": "Интервал обновления CoinGlass (ч)", "group": "CoinGlass", "type": "int"},
    {"key": "coinglass_api_key", "label": "CoinGlass API key", "group": "CoinGlass", "type": "secret"},

    {"key": "telegram_signals_auto_execute", "label": "Автоисполнение Telegram-сигналов", "group": "Telegram сигналы", "type": "bool"},
    {"key": "telegram_signals_quality_threshold", "label": "Порог качества сигнала", "group": "Telegram сигналы", "type": "float"},
    {"key": "telegram_llm_fallback_enabled", "label": "LLM-фолбэк парсинга (если регулярки не распознали)", "group": "Telegram сигналы", "type": "bool"},
    {"key": "anthropic_api_key", "label": "Anthropic API key", "group": "Telegram сигналы", "type": "secret"},
    {"key": "anthropic_model", "label": "Anthropic модель", "group": "Telegram сигналы", "type": "str"},
    {"key": "telegram_api_id", "label": "Telegram API ID", "group": "Telegram сигналы", "type": "secret"},
    {"key": "telegram_api_hash", "label": "Telegram API hash", "group": "Telegram сигналы", "type": "secret"},
    {"key": "telegram_user_id", "label": "Telegram user id", "group": "Telegram сигналы", "type": "secret"},

    {"key": "telegram_bot_token", "label": "Telegram bot token (уведомления)", "group": "Telegram уведомления", "type": "secret"},
    {"key": "telegram_chat_id", "label": "Telegram chat id (уведомления)", "group": "Telegram уведомления", "type": "secret"},

    {"key": "binance_api_key", "label": "Binance API key", "group": "Биржи", "type": "secret"},
    {"key": "binance_api_secret", "label": "Binance API secret", "group": "Биржи", "type": "secret"},
    {"key": "bybit_api_key", "label": "Bybit API key", "group": "Биржи", "type": "secret"},
    {"key": "bybit_api_secret", "label": "Bybit API secret", "group": "Биржи", "type": "secret"},
]

_SCHEMA_BY_KEY = {f["key"]: f for f in SETTINGS_SCHEMA}

# settings.risk_* -> имя параметра, ожидаемое RiskManager.configure()
_RISK_PARAM_MAP = {
    "risk_daily_loss_limit_usd": "daily_loss_limit_usd",
    "risk_max_open_positions": "max_open_positions",
    "risk_max_position_size_pct": "max_position_size_pct",
    "risk_max_drawdown_pct": "max_drawdown_pct",
    "risk_cooldown_seconds": "cooldown_seconds",
}


def _mask(value: Any) -> str:
    if value is None or value == "":
        return ""
    s = str(value)
    if len(s) <= 4:
        return "••••"
    return "••••" + s[-4:]


def get_settings_snapshot() -> list[dict]:
    """Текущие значения всех редактируемых настроек для UI (секреты — маскированы)."""
    result = []
    for field in SETTINGS_SCHEMA:
        raw = getattr(settings, field["key"], None)
        if field["type"] == "secret":
            value = _mask(raw)
            configured = bool(raw)
        elif field["type"] == "list":
            value = ", ".join(raw) if raw else ""
            configured = True
        else:
            value = raw
            configured = raw is not None
        result.append({**field, "value": value, "configured": configured})
    return result


def _cast(field: dict, raw_value: Any) -> Any:
    t = field["type"]
    if t == "int":
        return int(raw_value)
    if t == "float":
        return float(raw_value)
    if t == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() in ("1", "true", "yes", "on")
    if t == "list":
        if isinstance(raw_value, list):
            return [str(v).strip() for v in raw_value if str(v).strip()]
        return [v.strip() for v in str(raw_value).split(",") if v.strip()]
    if t == "select":
        raw_value = str(raw_value)
        if raw_value not in field.get("options", []):
            raise ValueError(f"допустимые значения: {field.get('options')}")
        return raw_value
    return str(raw_value)


async def apply_settings_update(updates: dict[str, Any]) -> dict:
    """Применить и сохранить изменения настроек.

    Возвращает {"updated": [...], "errors": {key: reason}}.
    Пустая строка/None для секретного поля означает "не менять" (чтобы можно
    было переотправить форму, не зная текущего значения ключа).
    """
    updated: list[str] = []
    errors: dict[str, str] = {}
    risk_changes: dict[str, Any] = {}

    async with get_session() as session:
        for key, raw_value in updates.items():
            field = _SCHEMA_BY_KEY.get(key)
            if field is None:
                errors[key] = "неизвестный параметр"
                continue
            if field["type"] == "secret" and (raw_value is None or str(raw_value).strip() == ""):
                continue

            try:
                casted = _cast(field, raw_value)
            except (ValueError, TypeError) as e:
                errors[key] = str(e)
                continue

            setattr(settings, key, casted)
            if key in _RISK_PARAM_MAP:
                risk_changes[_RISK_PARAM_MAP[key]] = casted

            existing = (
                await session.execute(select(BotConfig).where(BotConfig.config_key == key))
            ).scalar_one_or_none()
            if existing:
                existing.config_value = {"value": casted}
                existing.source = "settings_ui"
                existing.updated_by = "web"
            else:
                session.add(BotConfig(
                    config_key=key,
                    config_value={"value": casted},
                    source="settings_ui",
                    updated_by="web",
                ))
            updated.append(key)

        await session.commit()

    if risk_changes:
        from src.risk.risk_manager import risk_manager
        risk_manager.configure(risk_changes)

    if "trading_mode" in updated:
        from src.execution.executor import execution_engine
        if settings.trading_mode == "real" and execution_engine.is_paper:
            await execution_engine.initialize("binance")
        elif settings.trading_mode == "paper":
            execution_engine.is_paper = True

    if updated:
        logger.info(f"Настройки обновлены через веб-панель: {updated}")

    return {"updated": updated, "errors": errors}


async def load_settings_overrides():
    """Применить сохранённые в bot_config переопределения настроек при старте бота."""
    try:
        async with get_session() as session:
            rows = (await session.execute(select(BotConfig))).scalars().all()
    except Exception as e:
        logger.warning(f"Не удалось загрузить сохранённые настройки из БД: {e}")
        return

    applied = []
    for row in rows:
        field = _SCHEMA_BY_KEY.get(row.config_key)
        if field is None:
            continue
        value = (row.config_value or {}).get("value")
        if value is None:
            continue
        try:
            setattr(settings, row.config_key, value)
            applied.append(row.config_key)
        except Exception as e:
            logger.warning(f"Не удалось применить сохранённую настройку {row.config_key}: {e}")

    if applied:
        logger.info(f"Загружены сохранённые настройки из БД: {applied}")
