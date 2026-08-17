"""Alembic environment setup."""
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Добавляем путь к src для импорта моделей
import os
import sys
from pathlib import Path

# Путь к корню проекта
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Импортируем Base и модели для autogenerate
from src.db.base import Base
from src.db import models  # noqa — импорт для регистрации моделей

# Загружаем конфигурацию Alembic
config = context.config

# Конфигурация логирования из alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Целевая мета-база — все модели из Base
target_metadata = Base.metadata


def get_url():
    """
    Получить URL подключения к БД.
    Сначала проверяем переменную окружения, затем fallback в alembic.ini.
    """
    from src.config import settings
    # Если DATABASE_URL установлен в settings, используем его
    if settings.database_url and not settings.database_url.startswith('driver://'):
        return settings.database_url
    # Fallback к URL из alembic.ini
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """
    Запуск миграций в offline-режиме (без подключения к БД).
    Используется для генерации SQL скриптов.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Запуск миграций в online-режиме (с подключением к БД).
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration['sqlalchemy.url'] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
