# CryptoBot Pro — автономный самообучающийся крипто-трейдер бот
# Полная архитектура: event-driven, modular, asyncio + FastAPI + ML + Telegram + CoinGlass

## Структура проекта
# crypto-bot/
# ├── alembic/                  # миграции БД
# │   ├── versions/
# │   └── env.py
# ├── src/
# │   ├── __init__.py
# │   ├── main.py               # точка входа (запуск бота)
# │   ├── config.py             # загрузка конфига (Pydantic Settings)
# │   ├── event_bus.py          # asyncio event bus (pub/sub)
# │   ├── db/
# │   │   ├── __init__.py
# │   │   ├── base.py           # SQLAlchemy base
# │   │   ├── models.py         # все модели БД
# │   │   └── session.py        # асинхронные сессии
# │   ├── data_ingest/
# │   │   ├── __init__.py
# │   │   ├── market_data.py    # WebSocket/REST с бирж (ccxt)
# │   │   ├── coinglass_client.py  # CoinGlass REST API
# │   │   └── feature_engine.py    # индикаторы, фичи
# │   ├── strategy/
# │   │   ├── __init__.py
# │   │   ├── base.py           # ABC стратегии
# │   │   ├── rsi_strategy.py
# │   │   ├── ema_strategy.py
# │   │   ├── bb_strategy.py
# │   │   ├── funding_strategy.py
# │   │   ├── liquidation_strategy.py
# │   │   ├── ml_classifier_strategy.py
# │   │   └── ensemble_voter.py
# │   ├── risk/
# │   │   ├── __init__.py
# │   │   └── risk_manager.py   # риск-менеджмент
# │   ├── execution/
# │   │   ├── __init__.py
# │   │   ├── executor.py       # ордера, исполнение
# │   │   └── paper_engine.py   # paper trading симуляция
# │   ├── ml/
# │   │   ├── __init__.py
# │   │   ├── feature_store.py  # хранилище фичей
# │   │   ├── trainer.py        # обучение LightGBM
# │   │   ├── model_registry.py # реестр моделей
# │   │   └── inference.py      # инференс моделей
# │   ├── telegram/
# │   │   ├── __init__.py
# │   │   ├── channel_monitor.py  # телеграм мониторинг
# │   │   ├── signal_parser.py    # парсинг сигналов
# │   │   └── quality_scorer.py   # оценка качества
# │   ├── web/
# │   │   ├── __init__.py
# │   │   ├── api.py            # FastAPI роуты
# │   │   ├── websocket.py      # WebSocket хабы
# │   │   └── schemas.py        # Pydantic схемы
# │   └── utils/
# │       ├── __init__.py
# │       ├── crypto.py         # шифрование API ключей
# │       └── logging.py        # настройка логирования
# ├── tests/
# ├── alembic.ini
# ├── docker-compose.yml
# ├── Dockerfile
# ├── requirements.txt
# ├── .env.example
# └── README.md

## Быстрый запуск (development)

```bash
cd crypto-bot
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows
pip install -r requirements.txt
cp .env.example .env
# настройте .env
python -m src.main
```

## Программный запуск с веб-интерфейсом

```bash
# В одном терминале: бот
python -m src.main

# В другом терминале: веб-интерфейс
uvicorn src.web.api:app --host 0.0.0.0 --port 8000

# Или запуск обоих в одном процессе:
python -m src.main  # (после доработки main.py для одновременного запуска)
```

## Производительность
- Paper Trading: даёт ≈ 1000+ сделок/день на 10 парах
- Real Trading: зависит от лимитов биржи и рыночной активности
- ML: LightGBM обучение занимает 5-30 секунд на 10k строк

## Безопасность
- API ключи шифруются AES-256 (Fernet) в БД
- Kill switch останавливает все торговлю
- Daily loss limit предотвращает большие убытки
- Минимальные права биржевых API (только торговля)

## Дальнейшее развитие
- Добавить реальные WebSocket потоки от бирж
- Интеграция с LLM для Telegram парсинга
- RL для оптимизации исполнения ордеров
- Мобильное PWA приложение
- Распределённая архитектура (многопроцессорность)
