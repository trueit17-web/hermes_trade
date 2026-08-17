# CryptoBot Pro — автономный самообучающийся крипто-трейдер бот

Полная архитектура: event-driven, modular, asyncio + FastAPI + LightGBM + Telegram + CoinGlass API.

## Быстрый старт (5 минут)

### 0. Предварительные требования

- Python 3.12+
- Git
- (опционально) Docker + Docker Compose
- (для real режима) API ключи Binance или Bybit

### 1. Клонирование

```bash
git clone https://github.com/trueit17-web/hermes_trade.git
cd hermes_trade
```

### 2. Виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# или
venv\Scripts\activate      # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка .env

```bash
cp .env.example .env
```

Отредактируйте `.env`:
- `TRADING_MODE=paper` — для тестирования (по умолчанию)
- `TRADING_MODE=real` — для реальной торговли
- `DATABASE_URL` — SQLite (по умолчанию) или PostgreSQL
- `BINANCE_API_KEY` / `BINANCE_API_SECRET` — если real режим
- `COINGLASS_API_KEY` — опционально, для аналитики

### 5. Генерация ключа шифрования

```bash
python -c "from src.utils.crypto import generate_encryption_key; print(generate_encryption_key())"
```

Скопируйте вывод в `ENCRYPTION_KEY` в `.env`.

### 6. Инициализация базы данных

```bash
# Для SQLite (по умолчанию):
python -m src.db.session init

# Или вручную:
python -c "from src.db.session import init_db; import asyncio; asyncio.run(init_db())"

# Для применения Alembic миграций:
alembic upgrade head
```

### 7. Запуск бота

```bash
python -m src.main
```

Бот запустится в режиме, указанном в `.env`.

### 8. Веб-интерфейс

Откройте в браузере:
- `http://localhost:8000/docs` — Swagger API документация
- `http://localhost:8000/health` — health check
- `http://localhost:8000/status` — полный статус

Для запуска только веб-интерфейса (если бот уже работает):
```bash
uvicorn src.web.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Развёртывание на VPS (Production)

### Вариант 1: Docker Compose (рекомендуемый)

#### 1. Подготовка VPS

- Ubuntu 22.04+ (или аналог)
- 2+ ядра CPU, 4GB+ RAM (для ML)
- Открытые порты: 22 (SSH), 80/443 (nginx), 5432 (PostgreSQL, только локально)

#### 2. Установка Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# перелогиньтесь или выполните:
newgrp docker
```

#### 3. Клонирование и настройка

```bash
sudo apt update && sudo apt install -y git

cd /opt
sudo git clone https://github.com/trueit17-web/hermes_trade.git cryptobot
cd cryptobot

cp .env.example .env
# Настройте .env для production:
# - TRADING_MODE=real (или paper)
# - DATABASE_URL=postgresql://bot:password@db:5432/cryptobot
# - ENCRYPTION_KEY=<ключ из шага 5>
# - BINANCE_API_KEY, BINANCE_API_SECRET (для real)
```

#### 4. Генерация ключа шифрования на VPS

```bash
cd /opt/cryptobot
python -c "from src.utils.crypto import generate_encryption_key; print(generate_encryption_key())"
```

Вставьте в `ENCRYPTION_KEY` в `.env`.

#### 5. Запуск Docker Compose

```bash
# Для production (PostgreSQL, Redis)
docker-compose up -d

# Мониторинг логов:
docker-compose logs -f bot
```

#### 6. Инициализация БД (первый запуск)

```bash
# Применение миграций
docker-compose exec bot alembic upgrade head

# Или для quick start (без Alembic):
docker-compose exec bot python -c "from src.db.session import init_db; import asyncio; asyncio.run(init_db())"
```

#### 7. Проверка

```bash
# Health check
curl http://localhost:8000/health

# Статус
curl http://localhost:8000/status

# Логи
docker-compose logs bot --tail=50
```

#### 8. Nginx reverse proxy (опционально)

```bash
sudo apt install nginx certbot python3-certbot-nginx

# Конфигурация nginx (см. docker-compose.yml — секция nginx)
sudo cp nginx.conf /etc/nginx/sites-available/cryptobot
sudo ln -s /etc/nginx/sites-available/cryptobot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# SSL сертификат (Let's Encrypt)
sudo certbot --nginx -d your-domain.com
```

### Вариант 2: Нативный запуск (без Docker)

```bash
# 1. Установка Python зависимостей
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Настройка .env
cp .env.example .env
# ... редактирование ...

# 3. Инициализация БД
python -c "from src.db.session import init_db; import asyncio; asyncio.run(init_db())"

# 4. Запуск бота (systemd service)
sudo tee /etc/systemd/system/cryptobot.service << 'EOF'
[Unit]
Description=CryptoBot Pro Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/cryptobot
Environment="PATH=/opt/cryptobot/venv/bin"
ExecStart=/opt/cryptobot/venv/bin/python -m src.main
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable cryptobot
sudo systemctl start cryptobot
sudo systemctl status cryptobot
```

---

## Настройка бирж (Real режим)

### Binance

1. Зайдите в Binance → API Management
2. Создайте API ключ с правами: `Enable Spot & Margin Trading`
3. Скопируйте API Key и Secret в `.env`

**Безопасность:** Не включайте `Enable Withdrawals`.

### Bybit

1. Зайдите в Bybit → API Management
2. Создайте API ключ с правами: `Trade`
3. Скопируйте в `.env`

---

## Настройка Telegram

### Для мониторинга каналов

1. Зайдите на https://my.telegram.org
2. Введите свой телефон, получите код
3. Выберите "API development tools"
4. Создайте приложение, получите `API_ID` и `API_HASH`
5. Вставьте в `.env`:
   - `TELEGRAM_API_ID=...`
   - `TELEGRAM_API_HASH=...`
   - `TELEGRAM_USER_ID=...` (ваш user_id)

### Для уведомлений (опционально)

1. Создайте Telegram бота через @BotFather
2. Получите `BOT_TOKEN`
3. Добавьте `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID` в `.env`
4. Бот будет отправлять уведомления о сделках и событиях

---

## Настройка стратегий

### Редактирование через API

```bash
# Список стратегий
curl http://localhost:8000/strategies | jq

# Включить/выключить стратегию
curl -X POST http://localhost:8000/strategies/rsi_mr/toggle \
  -H "Content-Type: application/json" \
  -d '{"active": true}'

# Обновить параметры стратегии
curl -X POST http://localhost:8000/strategies/rsi_mr/update \
  -H "Content-Type: application/json" \
  -d '{
    "params": {
      "rsi_period": 14,
      "oversold_level": 25,
      "overbought_level": 75,
      "position_size_pct": 3.0
    }
  }'
```

### Параметры стратегий

| Стратегия | Описание | Ключевые параметры |
|-----------|----------|-------------------|
| RSI Mean Reversion | Покупка при oversold, продажа при overbought | rsi_period, oversold_level, overbought_level, position_size_pct |
| EMA Crossover | Трендовый сигнал при пересечении EMA | fast_ema_period, slow_ema_period, position_size_pct |
| Bollinger Bands | Mean-reversion или breakout | mode, bb_length, bb_std, position_size_pct |
| Funding Rate | Торговля на основе фандинг-рейта | high_funding_threshold, position_size_pct |
| Liquidation Zones | Торговля вблизи зон ликвидаций | distance_threshold_pct, position_size_pct |
| ML Classifier | ML предсказание направления | model_version, confidence_threshold, position_size_pct |

---

## CoinGlass аналитика

Бот автоматически обновляет данные из CoinGlass (если `COINGLASS_API_KEY` установлен).

Доступные данные:
- Open Interest (OI)
- Funding Rate
- Long/Short Ratio
- Liquidations (включая heatmap)
- Large limit orders (whale orders)
- Taker Buy/Sell volume
- Fear & Greed Index
- ETF flows (BTC, ETH)
- Индикаторы (RSI, MACD, BB, MA, EMA)

Данные используются:
1. Как дополнительные фичи для ML-моделей
2. Как сигналы для стратегий (например, funding rate, OI)
3. Для риск-менеджмента (liquidity assessment)

---

## Telegram сигналы

Бот может мониторить Telegram каналы и автоматически исполнять сигналы.

### Добавление канала

```bash
curl -X POST http://localhost:8000/telegram/channels \
  -H "Content-Type: application/json" \
  -d '{
    "channel_id": "@crypto_signals_channel",
    "channel_title": "Crypto Signals",
    "parser_type": "regex",
    "auto_execute": false,
    "quality_threshold": 0.6
  }'
```

### Типы парсеров

- `regex` — парсинг стандартных форматов сигналов (BTC/USDT Long 69000 SL 68000 TP 72000)
- `llm` — парсинг с использованием LLM (в разработке, требует LLM API)

### Управление сигналами

```bash
# Список сигналов
curl http://localhost:8000/telegram/signals

# Подтвердить сигнал (если auto_execute=false)
curl -X POST http://localhost:8000/telegram/signals/1/confirm \
  -H "Content-Type: application/json" \
  -d '{"signal_id": 1, "action": "execute"}'
```

---

## Риск-менеджмент

### Настройка через API

```bash
# Обновить риск-параметры
curl -X POST http://localhost:8000/risk/configure \
  -H "Content-Type: application/json" \
  -d '{
    "daily_loss_limit_usd": 1000.0,
    "max_open_positions": 10,
    "max_position_size_pct": 5.0,
    "max_drawdown_pct": 20.0,
    "cooldown_seconds": 600
  }'

# Приостановить торговлю
curl -X POST http://localhost:8000/risk/pause

# Возобновить торговлю
curl -X POST http://localhost:8000/risk/resume

# Kill switch (немедленная остановка)
curl -X POST http://localhost:8000/risk/kill-switch

# Сбросить kill switch
curl -X POST http://localhost:8000/risk/clear-kill-switch
```

### Ключевые риск-параметры

| Параметр | Описание | Значение по умолчанию |
|----------|----------|---------------------|
| daily_loss_limit_usd | Максимальный убыток за день | 500 USD |
| max_open_positions | Максимум открытых позиций | 8 |
| max_position_size_pct | Максимальный размер позиции (% от капитала) | 10% |
| max_drawdown_pct | Максимальный даун-драфт перед остановкой | 15% |
| cooldown_seconds | Минимальное время между сделками | 300 сек |

---

## ML (Машинное обучение)

### Автоматическое переобучение

Бот автоматически переобучает ML-модели на основе:
- Расписание: каждые N часов (по умолчанию 6 часов)
- Событие: после M сделок (по умолчанию 200 сделок)

### Управление моделями

```bash
# Список моделей
curl http://localhost:8000/ml/models

# Активная модель
curl http://localhost:8000/ml/models/direction_classifier/active

# Переобучить вручную
curl -X POST http://localhost:8000/ml/retrain

# Активировать конкретную версию модели
curl -X POST http://localhost:8000/ml/models/direction_classifier/3/activate
```

### ML моделей

| Модель | Тип | Описание |
|--------|-----|----------|
| direction_classifier | LightGBM Classifier | Предсказывает направление цены (up/down) на 다음 N свечей |
| volatility_predictor | LightGBM Regressor | Предсказывает будущую волатильность (ATR) |
| ensemble_weights | Linear/MLP | Оптимизирует веса стратегий в ensemble |

### Фичей для ML

- Технические индикаторы (RSI, MACD, BB, EMA, и др.)
- Price-based фичи (returns, momentum, distance from MA)
- Volume-based фичи (volume ratio, OBV)
- Volatility фичи (ATR, realised volatility)
- Time-based фичи (hour, day of week)
- CoinGlass фичи (OI, funding rate, long/short ratio)

---

## Мониторинг и алёрты

### Health check

```bash
curl http://localhost:8000/health
```

### WebSocket real-time обновления

Подключиться к `ws://localhost:8000/ws` для получения событий в реальном времени.

### Telegram уведомления

Если настроен `TELEGRAM_BOT_TOKEN`, бот отправляет уведомления:
- Открытие/закрытие сделки
- Уведомления о риске (daily loss limit, max drawdown)
- ML retraining complete
- Ошибки и критические события

---

## CI/CD (GitHub Actions)

При пуше в `main` автоматически запускается:
1. **Linting** — Ruff, Flake8, Black
2. **Тесты** — pytest с asyncio-тестами
3. **Docker build** — сборка образа, проверка запуска

Файл: `.github/workflows/ci.yml`

---

## Структура проекта

```
hermes_trade/
├── src/
│   ├── __init__.py
│   ├── main.py               # Запуск бота
│   ├── config.py             # Конфигурация (Pydantic Settings)
│   ├── event_bus.py          # Event-driven шина событий
│   ├── db/
│   │   ├── base.py           # SQLAlchemy Base
│   │   ├── models.py         # Все модели БД (29 таблиц)
│   │   └── session.py        # Асинхронные сессии
│   ├── data_ingest/
│   │   ├── market_data.py    # ccxt — рыночные данные
│   │   ├── coinglass_client.py  # CoinGlass API клиент
│   │   └── feature_engine.py    # Индикаторы и фичи
│   ├── strategy/
│   │   ├── __init__.py       # Все стратегии + реестр
│   │   └── base.py           # ABC стратегии (опционально)
│   ├── risk/
│   │   └── risk_manager.py   # Риск-менеджмент
│   ├── execution/
│   │   └── executor.py       # Ордера + paper trading
│   ├── ml/
│   │   └── __init__.py       # ML pipeline (LightGBM)
│   ├── telegram/
│   │   ├── channel_monitor.py  # Мониторинг каналов
│   │   ├── signal_parser.py    # Парсинг сигналов
│   │   └── quality_scorer.py   # Оценка качества
│   ├── web/
│   │   ├── api.py            # FastAPI endpoints
│   │   ├── schemas.py        # Pydantic схемы
│   │   └── websocket.py      # WebSocket broadcast
│   └── utils/
│       ├── crypto.py         # Шифрование API ключей
│       └── logging.py        # Логирование
├── alembic/
│   ├── env.py
│   ├── versions/
│   │   └── 001_initial.py   # Начальная миграция
│   └── script.py.mako       # Шаблон миграций
├── tests/
│   └── test_all.py           # Unit + integration тесты
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI
├── scripts/
│   ├── pre-commit-hooks.sh
│   ├── git-credential-token.py
│   └── gh-env.sh
├── .env.example
├── .env                      # (не коммитится)
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
└── alembic.ini
```

---

## Безопасность

### API ключи

- Все API ключи шифруются AES-256 (Fernet) в БД
- Ключи не хранятся в plaintext в коде
- Для real режима используйте API keys с минимальными правами

### Kill switch

- Мгновенная остановка всей торговли
- Закрывает все позиции (через market ордера)
- Сбрасывается только вручную

### Дневные лимиты

- Daily loss limit — автоматическая остановка при превышении
- Maximum drawdown — остановка при значительном убытке
- Cooldown — предотвращает переторговку

---

## Развитие проекта

### Добавление новой стратегии

1. Создайте файл в `src/strategy/`
2. Наследуйтесь от `BaseStrategy` (или аналогичного класса)
3. Зарегистрируйте в `strategy/__init__.py` (StrategyRegistry)
4. Протестируйте через backtest
5. Добавьте в конфигурацию

### Добавление нового источника данных

1. Создайте клиент в `src/data_ingest/`
2. Интегрируйте с event_bus (публикация MarketDataEvent)
3. Добавьте фичи в feature_engine.py

### Добавление нового ML-модели

1. Создайте trainer в `src/ml/`
2. Определите feature set и label
3. Добавьте в ModelRegistry
4. Интегрируйте в стратегию или separate inference service

---

## Ресурсы

- **CCXT**: https://github.com/ccxt/ccxt
- **CoinGlass API**: https://docs.coinglass.com/
- **Telethon**: https://docs.telethon.dev/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **FastAPI**: https://fastapi.tiangolo.com/
- **SQLAlchemy**: https://www.sqlalchemy.org/
- **Alembic**: https://alembic.sqlalchemy.org/

---

## Лицензия

Проект распространяется как open-source.

---

## Поддержка

- **Issues**: https://github.com/trueit17-web/hermes_trade/issues
- **Telegram**: (скоро)
- **Email**: (скоро)
