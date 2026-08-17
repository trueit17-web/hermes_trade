FROM python:3.12-slim

WORKDIR /app

# Системные зависимости (libgomp1 нужен lightgbm — OpenMP рантайм)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/

# Создание директорий (data/ гитигнорится и монтируется как volume в docker-compose.yml)
RUN mkdir -p /app/data/logs /app/data/models

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Порт для веб-интерфейса
EXPOSE 8000

# Установка команды по умолчанию
CMD ["python", "-m", "src.main"]
