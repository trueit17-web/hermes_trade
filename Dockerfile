FROM python:3.12-slim

WORKDIR /app

# Системные зависимости: libgomp1 нужен lightgbm (OpenMP рантайм), curl —
# для healthcheck в docker-compose.yml (curl -f http://localhost:8000/health).
# python:3.12-slim не содержит curl по умолчанию — без него healthcheck
# всегда падал бы с "command not found" независимо от реального состояния
# приложения (сам бот при этом отвечал нормально снаружи через nginx).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/
COPY alembic.ini .
COPY alembic/ ./alembic/
# Номер версии (см. CHANGELOG.md) — читается GET /system/metrics
# (src/web/system_metrics.py), чтобы по дашборду можно было однозначно
# отличить "новый код уже развернулся" от "докер собрал из кеша старый
# слой" (реальный инцидент при отладке редеплоя — см. CHANGELOG.md).
COPY VERSION .

# Создание директорий (data/ гитигнорится и монтируется как volume в docker-compose.yml)
RUN mkdir -p /app/data/logs /app/data/models

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Порт для веб-интерфейса
EXPOSE 8000

# Установка команды по умолчанию
CMD ["python", "-m", "src.main"]
