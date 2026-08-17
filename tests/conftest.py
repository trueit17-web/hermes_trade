"""Общие фикстуры для тестов: изолированная тестовая БД."""
import asyncio
import os

import pytest

from src.config import settings

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_cryptobot.db")
settings.database_url = f"sqlite+aiosqlite:///{TEST_DB_PATH}"

from src.db.session import init_db  # noqa: E402  (после переопределения database_url)


@pytest.fixture(scope="session", autouse=True)
def _test_database():
    """Создать таблицы в тестовой БД перед прогоном тестов и удалить файл после."""
    os.makedirs(os.path.dirname(TEST_DB_PATH), exist_ok=True)
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    asyncio.run(init_db())
    yield

    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
