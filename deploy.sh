#!/bin/bash
# deploy.sh — автоматическое развёртывание CryptoBot Pro на VPS
# Использование: ./deploy.sh [--update] [--data-dir /path/to/data]

set -e

# === Конфигурация ===
VPS_HOST="${VPS_HOST:-your-vps-hostname.com}"
VPS_USER="${VPS_USER:-ubuntu}"
DEPLOY_DIR="/opt/cryptobot"
DATA_DIR="${DATA_DIR:-$DEPLOY_DIR/data}"
REPO_URL="https://github.com/trueit17-web/hermes_trade.git"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

# === Цвета ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# === Проверка аргументов ===

UPDATE_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --update)
            UPDATE_MODE=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            echo "Usage: $0 [--update] [--data-dir /path/to/data]"
            exit 1
            ;;
    esac
done

# === Основная функция ===

main() {
    info "Начало развёртывания CryptoBot Pro..."
    info "Цель: $VPS_USER@$VPS_HOST:$DEPLOY_DIR"

    if [ "$UPDATE_MODE" = true ]; then
        update_deployment
    else
        initial_deployment
    fi

    info "Развёртывание завершено!"
    info "Health check: curl http://$VPS_HOST:8000/health"
    info "Логи: ssh $VPS_USER@$VPS_HOST 'docker-compose -f $DEPLOY_DIR/docker-compose.yml logs -f bot'"
}

# === Первоначальное развёртывание ===

initial_deployment() {
    info "Шаг 1: Подключение к VPS и подготовка директории"

    ssh "$VPS_USER@$VPS_HOST" << 'SSH_EOF'
        sudo apt update && sudo apt install -y git python3-venv python3-pip docker.io docker-compose
        sudo systemctl enable docker
        sudo systemctl start docker
        sudo usermod -aG docker $(whoami)
        mkdir -p /opt/cryptobot
        cd /opt/cryptobot
        git clone $REPO_URL . || true
        git checkout main
    SSH_EOF

    info "Шаг 2: Создание .env файла"

    # Создаём .env из .env.example + кастомные настройки
    ssh "$VPS_USER@$VPS_HOST" "cd $DEPLOY_DIR && cp $ENV_EXAMPLE $ENV_FILE"

    info "Шаг 3: Генерация ключа шифрования"

    # Генерируем ключ на VPS и выводим его
    ENCRYPTION_KEY=$(ssh "$VPS_USER@$VPS_HOST" "cd $DEPLOY_DIR && python3 -c \"from src.utils.crypto import generate_encryption_key; print(generate_encryption_key())\"")
    info "Сгенерирован ключ шифрования: $ENCRYPTION_KEY"
    info "Добавьте ENCRYPTION_KEY=$ENCRYPTION_KEY в .env на VPS"

    info "Шаг 4: Запуск Docker Compose"

    ssh "$VPS_USER@$VPS_HOST" "cd $DEPLOY_DIR && docker-compose up -d"

    info "Шаг 5: Применение миграций"

    ssh "$VPS_USER@$VPS_HOST" "cd $DEPLOY_DIR && docker-compose exec bot alembic upgrade head || docker-compose exec bot python3 -c \"from src.db.session import init_db; import asyncio; asyncio.run(init_db())\""

    info "Шаг 6: Проверка статуса"

    sleep 5
    ssh "$VPS_USER@$VPS_HOST" "curl -s http://localhost:8000/health || echo 'Health check pending...'"

    info "✅ Развёртывание завершено!"
}

# === Обновление ===

update_deployment() {
    info "Шаг 1: Обновление кода с GitHub"

    ssh "$VPS_USER@$VPS_HOST" "cd $DEPLOY_DIR && git pull origin main"

    info "Шаг 2: Пересборка Docker образа"

    ssh "$VPS_USER@$VPS_HOST" "cd $DEPLOY_DIR && docker-compose build --no-cache"

    info "Шаг 3: Перезапуск сервисов"

    ssh "$VPS_USER@$VPS_HOST" "cd $DEPLOY_DIR && docker-compose up -d --force-recreate"

    info "Шаг 4: Применение миграций"

    # Раньше update_deployment() не запускал миграции вообще (в отличие от
    # initial_deployment()) — любой PR, добавляющий alembic-миграцию, ронял
    # бота на первом же запросе к новой таблице/колонке после --update,
    # пока кто-то не применял миграцию на VPS вручную. db должна быть уже
    # готова (docker-compose up -d ждёт service_healthy), поэтому применяем
    # миграции сразу после пересоздания контейнера bot, до проверки /health.
    ssh "$VPS_USER@$VPS_HOST" "cd $DEPLOY_DIR && docker-compose exec -T bot alembic upgrade head"

    info "Шаг 5: Проверка статуса"

    sleep 10
    ssh "$VPS_USER@$VPS_HOST" "curl -s http://localhost:8000/health"

    info "✅ Обновление завершено!"
}

# === Запуск ===

main "$@"
