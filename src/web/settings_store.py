"""Реестр редактируемых настроек бота: метаданные для UI + применение/сохранение через BotConfig.

Позволяет менять большинство полей src.config.settings из веб-панели без
захода на сервер: изменения применяются немедленно к живому объекту
settings и сохраняются в таблицу bot_config, откуда подхватываются при
следующем старте бота (см. load_settings_overrides(), вызывается из
TradingBot.initialize()).
"""
from typing import Any

from sqlalchemy import select

from src.config import settings
from src.db.models import BotConfig
from src.db.session import get_session
from src.utils.logging import logger

# Поля БД/инфраструктуры сюда намеренно не включены (database_url, web_host,
# web_port, redis_url, encryption_key, web_admin_*) — их смена на лету либо
# бессмысленна (порт/хост уже забинжены), либо небезопасна через тот же UI,
# который они защищают.
SETTINGS_SCHEMA: list[dict] = [
    {"key": "trading_mode", "label": "Режим торговли", "group": "Общие", "type": "select", "options": ["paper", "real"],
     "description": "paper — виртуальная торговля без реальных денег (для проверки стратегии). real — реальные ордера на бирже."},
    {"key": "startup_capital_usdt", "label": "Стартовый капитал (USDT)", "group": "Общие", "type": "float",
     "description": "Виртуальный баланс paper-счёта при первом запуске или сбросе. На real-режим не влияет — там баланс берётся с биржи."},
    {"key": "log_level", "label": "Уровень логирования", "group": "Общие", "type": "select", "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
     "description": "Детальность логов. DEBUG — всё подряд (для отладки), ERROR — только ошибки. По умолчанию INFO."},
    {"key": "default_timeframes", "label": "Таймфреймы (через запятую)", "group": "Общие", "type": "list",
     "description": "Свечные интервалы, на которых стратегии считают индикаторы и принимают решения, например 1h, 4h."},
    {"key": "candlesticks_cache_size", "label": "Размер кэша свечей", "group": "Общие", "type": "int",
     "description": "Сколько последних свечей на пару хранить в памяти для расчёта индикаторов. Больше — точнее долгосрочные индикаторы, но больше памяти."},

    {"key": "symbol_quote_currency", "label": "Quote-валюта (напр. USDT)", "group": "Торговая вселенная", "type": "str",
     "description": "Котируемая валюта, по которой отбираются торговые пары (например USDT — торгуются только пары вида BASE/USDT)."},
    {"key": "symbol_blacklist", "label": "Блэклист пар (через запятую)", "group": "Торговая вселенная", "type": "list",
     "description": "Пары, которые бот никогда не будет торговать, даже если они проходят по объёму (например стейблкоины к стейблкоинам)."},
    {"key": "symbol_universe_max", "label": "Макс. пар в работе (топ по объёму)", "group": "Торговая вселенная", "type": "int",
     "description": "Сколько пар с наибольшим 24ч объёмом торгов бот отслеживает одновременно."},
    {"key": "symbol_universe_refresh_hours", "label": "Обновление вселенной (ч)", "group": "Торговая вселенная", "type": "int",
     "description": "Как часто пересчитывать список торгуемых пар заново (по свежему объёму торгов)."},

    {"key": "risk_daily_loss_limit_usd", "label": "Дневной лимит убытка (USD)", "group": "Риск", "type": "float",
     "description": "Если убыток за текущие сутки достигает этой суммы — торговля автоматически приостанавливается до следующего дня."},
    {"key": "risk_max_open_positions", "label": "Макс. открытых позиций", "group": "Риск", "type": "int",
     "description": "Сколько сделок может быть открыто одновременно. Новые сигналы сверх лимита отклоняются."},
    {"key": "risk_max_position_size_pct", "label": "Макс. размер позиции (%)", "group": "Риск", "type": "float",
     "description": "Максимальная доля баланса, которую можно вложить в одну сделку."},
    {"key": "risk_max_drawdown_pct", "label": "Макс. просадка (%)", "group": "Риск", "type": "float",
     "description": "Если баланс проседает от своего пика на этот процент — торговля ставится на паузу (защита от затяжной серии убытков)."},
    {"key": "risk_cooldown_seconds", "label": "Cooldown после стоп-лосса (сек)", "group": "Риск", "type": "int",
     "description": "Пауза перед открытием следующей сделки после любого закрытия по стоп-лоссу — не даёт бота сразу открыть новую позицию на эмоциях рынка."},
    {"key": "trailing_stop_pct", "label": "Trailing stop (%, 0 = выключен)", "group": "Риск", "type": "float",
     "description": "Стоп-лосс подтягивается к цене на этот процент и только ужесточается — фиксирует часть прибыли, если цена развернётся. 0 — выключен."},
    {"key": "atr_sltp_enabled", "label": "ATR-адаптивный SL/TP включён", "group": "Риск", "type": "bool",
     "description": "SL/TP считаются от ATR(14) вместо фиксированного %, заменяя уровни стратегии, ДО масштабирования по предсказанной волатильности. Только для сигналов от стратегий — Telegram-сигналы не затрагиваются."},
    {"key": "atr_sl_multiplier", "label": "ATR: множитель SL", "group": "Риск", "type": "float",
     "description": "SL = ATR(14) × этот множитель. Консервативно — 1.5–2.0, агрессивно (внутридневные сделки) — 0.8–1.2."},
    {"key": "atr_tp_rr_trend", "label": "ATR: R:R для трендовых стратегий", "group": "Риск", "type": "float",
     "description": "TP = SL × это значение для трендовых стратегий (EMA Crossover, ML Direction Classifier). Рекомендуемый диапазон 2.0–4.0."},
    {"key": "atr_tp_rr_countertrend", "label": "ATR: R:R для контртрендовых стратегий", "group": "Риск", "type": "float",
     "description": "TP = SL × это значение для контртрендовых/mean-reversion стратегий (RSI, Bollinger Bands, Funding Rate, Liquidation Zones). Рекомендуемый диапазон 1.5–2.5 — вероятность отката ниже, чем у тренда."},

    {"key": "protections_enabled", "label": "Protections включены", "group": "Protections", "type": "bool",
     "description": "Общий выключатель для всех автопауз ниже (Cooldown/StoplossGuard/LosingStreak). Выключено — сигналы не блокируются вообще."},
    {"key": "protections_channel_cooldown_minutes", "label": "Cooldown источника после закрытия (мин)", "group": "Protections", "type": "int",
     "description": "После любого закрытия сделки конкретный канал/стратегия на это время перестаёт открывать новые позиции (остальные источники не затрагиваются)."},
    {"key": "protections_stoploss_guard_window_min", "label": "StoplossGuard: окно (мин)", "group": "Protections", "type": "int",
     "description": "Временное окно, в котором считаются стоп-лоссы для StoplossGuard (см. ниже)."},
    {"key": "protections_stoploss_guard_count", "label": "StoplossGuard: стопов в окне", "group": "Protections", "type": "int",
     "description": "Если за окно выше накопилось столько закрытий по стопу (по всем источникам сразу) — вся торговля ставится на паузу."},
    {"key": "protections_stoploss_guard_lock_min", "label": "StoplossGuard: блокировка всей торговли (мин)", "group": "Protections", "type": "int",
     "description": "На сколько минут останавливается ВСЯ торговля, когда сработал StoplossGuard."},
    {"key": "protections_losing_streak_count", "label": "LosingStreak: убытков подряд", "group": "Protections", "type": "int",
     "description": "Сколько убыточных закрытий ПОДРЯД у одного источника (канала/стратегии) приводит к его блокировке."},
    {"key": "protections_losing_streak_lock_min", "label": "LosingStreak: блокировка источника (мин)", "group": "Protections", "type": "int",
     "description": "На сколько минут блокируется конкретный источник после серии убытков подряд (остальные источники продолжают торговать)."},

    {"key": "expectancy_sizing_enabled", "label": "Sizing по мат. ожиданию источника включён", "group": "Protections", "type": "bool",
     "description": "Размер новой позиции масштабируется по фактической прибыльности канала/стратегии в прошлом. Меняет реальный размер сделок — включайте осознанно."},
    {"key": "expectancy_sizing_min_trades", "label": "Sizing: мин. сделок для оценки", "group": "Protections", "type": "int",
     "description": "Сколько закрытых сделок должно накопиться у источника, прежде чем его мат. ожидание считается доказанным. Меньше — уменьшенный размер позиции (0.5x)."},
    {"key": "expectancy_sizing_max_trades", "label": "Sizing: окно последних сделок", "group": "Protections", "type": "int",
     "description": "Сколько последних закрытых сделок источника учитывается при расчёте среднего результата (старые сделки за пределами окна не влияют)."},
    {"key": "expectancy_sizing_min_expectancy_pct", "label": "Sizing: мин. средний PnL% (иначе источник пропускается)", "group": "Protections", "type": "float",
     "description": "Если средний % прибыли на сделку у источника ниже этого значения — источник пропускается целиком (размер позиции 0)."},

    {"key": "ml_retraining_interval_hours", "label": "Интервал переобучения (ч)", "group": "ML", "type": "int",
     "description": "Как часто ML-модели переобучаются на новых закрытых сделках."},
    {"key": "ml_max_trades_for_retrain", "label": "Сделок для переобучения", "group": "ML", "type": "int",
     "description": "Максимум последних сделок, используемых как обучающая выборка при переобучении моделей."},
    {"key": "ml_optuna_trials", "label": "Число Optuna trials", "group": "ML", "type": "int",
     "description": "Сколько комбинаций гиперпараметров перебирает Optuna при тюнинге модели. Больше — точнее, но дольше обучение."},
    {"key": "ml_optuna_min_samples", "label": "Мин. сэмплов для тюнинга гиперпараметров", "group": "ML", "type": "int",
     "description": "Минимум обучающих примеров, при котором вообще запускается подбор гиперпараметров через Optuna (иначе используются значения по умолчанию)."},
    {"key": "volatility_adjustment_enabled", "label": "Учитывать предсказанную волатильность", "group": "ML", "type": "bool",
     "description": "Модель volatility_predictor масштабирует размер позиции (меньше при высокой ожидаемой волатильности) и ширину SL/TP (шире при высокой) для сигналов от стратегий. На Telegram-сигналы не влияет — там уровни задаёт канал."},
    {"key": "volatility_baseline_pct", "label": "Базовая волатильность (%)", "group": "ML", "type": "float",
     "description": "Ориентир \"обычной\" волатильности, с которым сравнивается предсказание модели, чтобы понять, насколько текущий момент volatильнее/спокойнее нормы."},
    {"key": "volatility_size_min_mult", "label": "Sizing: мин. коэффициент", "group": "ML", "type": "float",
     "description": "Нижняя граница масштабирования размера позиции при аномально высокой предсказанной волатильности."},
    {"key": "volatility_size_max_mult", "label": "Sizing: макс. коэффициент", "group": "ML", "type": "float",
     "description": "Верхняя граница масштабирования размера позиции при аномально низкой предсказанной волатильности."},
    {"key": "volatility_sltp_min_mult", "label": "SL/TP: мин. коэффициент ширины", "group": "ML", "type": "float",
     "description": "Нижняя граница сужения SL/TP при аномально низкой предсказанной волатильности."},
    {"key": "volatility_sltp_max_mult", "label": "SL/TP: макс. коэффициент ширины", "group": "ML", "type": "float",
     "description": "Верхняя граница расширения SL/TP при аномально высокой предсказанной волатильности."},

    {"key": "paper_slippage_pct", "label": "Paper: слиппедж (%)", "group": "Paper trading", "type": "float",
     "description": "Имитация проскальзывания цены при исполнении paper-ордеров — насколько цена исполнения хуже цены сигнала."},
    {"key": "paper_fee_pct", "label": "Paper: комиссия (%)", "group": "Paper trading", "type": "float",
     "description": "Комиссия биржи, имитируемая в paper-режиме, списывается с каждой сделки при открытии и закрытии."},

    {"key": "performance_snapshot_interval_hours", "label": "Интервал снимка производительности (ч)", "group": "Аналитика", "type": "int",
     "description": "Как часто сохраняется снимок баланса/PnL для построения графиков динамики производительности."},

    {"key": "coinglass_update_interval_hours", "label": "Интервал обновления CoinGlass (ч)", "group": "CoinGlass", "type": "int",
     "description": "Как часто обновляются данные CoinGlass (открытый интерес, funding rate и т.п.), используемые как доп. фичи для ML."},
    {"key": "coinglass_api_key", "label": "CoinGlass API key", "group": "CoinGlass", "type": "secret",
     "description": "Ключ API CoinGlass. Без него данные по деривативам (OI, funding) не собираются."},

    {"key": "telegram_signals_auto_execute", "label": "Автоисполнение Telegram-сигналов", "group": "Telegram сигналы", "type": "bool",
     "description": "Открывать сделки по сигналам из Telegram-каналов автоматически. Выключено — сигналы только сохраняются, без исполнения."},
    {"key": "telegram_signals_quality_threshold", "label": "Порог качества сигнала", "group": "Telegram сигналы", "type": "float",
     "description": "Минимальная оценка качества сигнала (0–1), при которой он допускается к исполнению. Сигналы ниже порога отклоняются."},
    {"key": "telegram_llm_fallback_enabled", "label": "LLM-фолбэк парсинга (если регулярки не распознали)", "group": "Telegram сигналы", "type": "bool",
     "description": "Если регулярки не смогли разобрать сообщение канала — пробовать распознать его через Anthropic API, а если он не настроен/не смог — через Gemini API. Требует хотя бы один из ключей ниже."},
    {"key": "anthropic_api_key", "label": "Anthropic API key", "group": "Telegram сигналы", "type": "secret",
     "description": "Ключ Anthropic API для LLM-фолбэка парсинга сигналов (первый, приоритетный вариант)."},
    {"key": "anthropic_model", "label": "Anthropic модель", "group": "Telegram сигналы", "type": "str",
     "description": "Идентификатор модели Claude, используемой для разбора сигналов (например claude-haiku-4-5-20251001)."},
    {"key": "gemini_api_key", "label": "Gemini API key", "group": "Telegram сигналы", "type": "secret",
     "description": "Ключ Google Gemini API — второй, резервный вариант LLM-фолбэка (пробуется, если Anthropic не настроен или не смог разобрать сообщение). Бесплатный тариф Gemini обычно достаточен для этой задачи."},
    {"key": "gemini_model", "label": "Gemini модель", "group": "Telegram сигналы", "type": "str",
     "description": "Идентификатор модели Gemini, используемой для разбора сигналов (например gemini-2.0-flash)."},
    {"key": "telegram_api_id", "label": "Telegram API ID", "group": "Telegram сигналы", "type": "secret",
     "description": "API ID приложения Telegram (my.telegram.org) — нужен для чтения сообщений из каналов-сигналов."},
    {"key": "telegram_api_hash", "label": "Telegram API hash", "group": "Telegram сигналы", "type": "secret",
     "description": "API hash приложения Telegram (my.telegram.org), в паре с API ID выше."},
    {"key": "telegram_user_id", "label": "Telegram user id", "group": "Telegram сигналы", "type": "secret",
     "description": "ID пользователя Telegram, от имени которого бот читает каналы (используется при первом логине)."},

    {"key": "telegram_bot_token", "label": "Telegram bot token (уведомления)", "group": "Telegram уведомления", "type": "secret",
     "description": "Токен Telegram-бота (от @BotFather), которым бот отправляет уведомления об открытии/закрытии сделок."},
    {"key": "telegram_chat_id", "label": "Telegram chat id (уведомления)", "group": "Telegram уведомления", "type": "secret",
     "description": "ID чата/пользователя, куда бот отправляет уведомления. Обычно ваш личный chat id."},

    {"key": "active_exchange", "label": "Активная биржа (real-режим)", "group": "Биржи", "type": "select", "options": ["binance", "bybit", "okx"],
     "description": "На какой бирже исполняются реальные ордера в real-режиме. Требует заполненных ключей этой биржи ниже. Смена применяется сразу, без перезапуска бота."},
    {"key": "use_exchange_sandbox", "label": "Демо-счёт (sandbox/testnet)", "group": "Биржи", "type": "bool",
     "description": "Торговать на демо/testnet-счету биржи вместо реальных денег, тем же API-ключом. Рекомендуется держать включённым, пока не проверили бота вживую."},
    {"key": "binance_api_key", "label": "Binance API key", "group": "Биржи", "type": "secret",
     "description": "Ключ API Binance для торговли в real-режиме. Выдавайте права только на торговлю, без вывода средств."},
    {"key": "binance_api_secret", "label": "Binance API secret", "group": "Биржи", "type": "secret",
     "description": "Секрет API Binance, в паре с ключом выше."},
    {"key": "bybit_api_key", "label": "Bybit API key", "group": "Биржи", "type": "secret",
     "description": "Ключ API Bybit для торговли в real-режиме. Выдавайте права только на торговлю, без вывода средств."},
    {"key": "bybit_api_secret", "label": "Bybit API secret", "group": "Биржи", "type": "secret",
     "description": "Секрет API Bybit, в паре с ключом выше."},
    {"key": "okx_api_key", "label": "OKX API key", "group": "Биржи", "type": "secret",
     "description": "Ключ API OKX для торговли в real-режиме. Выдавайте права только на торговлю, без вывода средств."},
    {"key": "okx_api_secret", "label": "OKX API secret", "group": "Биржи", "type": "secret",
     "description": "Секрет API OKX, в паре с ключом выше."},
    {"key": "okx_passphrase", "label": "OKX passphrase", "group": "Биржи", "type": "secret",
     "description": "Passphrase, заданный при создании API-ключа OKX — обязателен для подписи запросов, отдельно от ключа и секрета."},
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

    if "trading_mode" in updated or "active_exchange" in updated or "use_exchange_sandbox" in updated:
        from src.execution.executor import execution_engine
        if settings.trading_mode == "real":
            # Три случая требуют (пере)подключения с нуля: первый переход в
            # real, смена активной биржи, смена sandbox/live при уже
            # включённом real — во всех трёх старое соединение ccxt уже не
            # соответствует нужному режиму.
            if (
                execution_engine.is_paper
                or execution_engine.exchange_id != settings.active_exchange
                or "active_exchange" in updated
                or "use_exchange_sandbox" in updated
            ):
                # initialize() САМ ПЕРВЫМ ДЕЛОМ проверяет self.is_paper и,
                # если он всё ещё True, молча остаётся в paper-режиме, даже
                # не пытаясь подключиться к бирже — is_paper выставляется
                # только внутри initialize() при неудаче (нет ключей и
                # т.п.), но никогда не сбрасывается перед вызовом. Без этой
                # строки переключение в real через дашборд вообще ничего не
                # делало: настройка менялась, а движок молча оставался в
                # paper и логировал "Paper Trading режим".
                execution_engine.is_paper = False
                await execution_engine.initialize(settings.active_exchange)
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
