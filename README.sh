# Создание и запуск бота

## 1. Клонирование и настройка

git clone <your-repo-url> crypto-bot
cd crypto-bot

## 2. Виртуальное окружение

python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

## 3. Установка зависимостей

pip install -r requirements.txt

## 4. Настройка .env

cp .env.example .env
# Отредактируйте .env с вашими настройками

## 5. Генерация ключа шифрования (обязательно!)

python -c "from src.utils.crypto import generate_encryption_key; print(generate_encryption_key())"
# Вставьте полученный ключ в ENCRYPTION_KEY в .env

## 6. Инициализация базы данных

# Для SQLite (по умолчанию):
python -c "from src.db.session import init_db; import asyncio; asyncio.run(init_db())"

# Для PostgreSQL:
# Убедитесь, что PostgreSQL запущен и настроен в .env
python -c "from src.db.session import init_db; import asyncio; asyncio.run(init_db())"

## 7. Запуск бота

# Запуск бота (в foreground)
python -m src.main

# Или с веб-интерфейсом в отдельном терминале:
uvicorn src.web.api:app --host 0.0.0.0 --port 8000 --reload

## 8. Docker запуск

docker-compose up -d
# или для разработки:
docker-compose up

## 9. Проверка статуса

curl http://localhost:8000/health
curl http://localhost:8000/status

## 10. Веб-интерфейс

Откройте http://localhost:8000/docs для Swagger API
Откройте http://localhost:8000/ для главной страницы (по умолчанию)

## Команды管理

# Остановка бота
# В(":q") в консоли бота, или Ctrl+C

# Сброс БД (только для тестов!)
python -c "from src.db.session import drop_db; import asyncio; asyncio.run(drop_db())"

# Ручное переобучение ML
curl -X POST http://localhost:8000/ml/retrain

# Смена режима на real
curl -X POST http://localhost:8000/trading-mode -H "Content-Type: application/json" -d '"real"'

# Пауза торговли
curl -X POST http://localhost:8000/risk/pause

# Возобновление торговли
curl -X POST http://localhost:8000/risk/resume
