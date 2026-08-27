"""Общие фикстуры для тестов: изолированная тестовая БД."""
import asyncio
import os
from unittest.mock import AsyncMock, patch

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


@pytest.fixture(autouse=True)
def _no_real_polling_delay():
    """
    ExecutionEngine поллит биржу с реальными паузами (_fetch_confirmed_order,
    _fetch_fill_details_via_trades) — без этого автопатча каждый тест,
    который не оборачивает вызов в `patch(asyncio.sleep)` вручную, реально
    ждал бы секунды на каждый create_order()/close_real_position(), и суммарно
    раздувал бы прогон всего файла тестов с секунд до минут. Патчим глобально
    для всех тестов, а не точечно в каждом — новые тесты/код с поллингом не
    должны каждый раз добавлять свой locals patch, чтобы остаться быстрыми.
    """
    with patch("src.execution.executor.asyncio.sleep", new=AsyncMock()):
        yield
