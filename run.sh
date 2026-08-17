# Скрипт локального развёртывания (для разработки)

#!/usr/bin/env bash

set -e

echo "🚀 CryptoBot Pro — локальный запуск"

# 0. Проверка зависимостей
command -v python3 &> /dev/null || { echo "❌ Python 3 не установлен"; exit 1; }

# 1. Виртуальное окружение
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 2. Установка зависимостей
if [ ! -f "$VENV_DIR/bin/pip" ] || [ requirements.txt -nt "$VENV_DIR/bin/pip" ]; then
    echo "📦 Установка зависимостей..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# 3. Настройка .env
if [ ! -f .env ]; then
    echo "📝 Создание .env из .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️ Отредактируйте .env перед запуском!"
    echo ""
fi

# 4. Инициализация БД
echo "🗄 Инициализация базы данных..."
python -c "from src.db.session import init_db; import asyncio; asyncio.run(init_db())" || \
python -m alembic upgrade head || \
echo "⚠️ База данных может быть уже инициализирована"

# 5. Запуск бота
echo ""
echo "🚀 Запуск бота..."
echo "   Paper режим: python -m src.main"
echo "   Web интерфейс: uvicorn src.web.api:app --reload"
echo ""
echo "Откройте http://localhost:8000/docs для API документации"
echo ""

# Если первый аргумент --run, запустить бота
if [ "$1" = "--run" ]; then
    python -m src.main
fi
