"""Асинхронные сессии БД (SQLite + PostgreSQL)."""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from src.config import settings
from src.db.models import Base

logger = logging.getLogger(__name__)


def get_engine():
    """Создать асинхронный engine для БД."""
    db_url = settings.database_url
    logger.info(f"Инициализация БД: {db_url}")

    # Для SQLite нужны специальные настройки
    connect_args = {}
    if "sqlite" in db_url:
        connect_args = {"check_same_thread": False}

    engine = create_async_engine(
        db_url,
        echo=False,
        future=True,
        connect_args=connect_args,
    )
    return engine


def get_session_maker(engine=None):
    """Создать фабрику сессий."""
    if engine is None:
        engine = get_engine()
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


async def init_db(engine=None):
    """Создать все таблицы (для разработки, в production используйте Alembic)."""
    if engine is None:
        engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("БД инициализирована")


async def drop_db(engine=None):
    """Удалить все таблицы (для тестов)."""
    if engine is None:
        engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("БД очищена")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Контекстный менеджер для сессии БД."""
    engine = get_engine()
    async_session_maker = get_session_maker(engine)
    session: AsyncSession = async_session_maker()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Ошибка сессии БД: {e}")
        raise
    finally:
        await session.close()


async def get_session_without_commit() -> AsyncGenerator[AsyncSession, None]:
    """Сессия без авто-коммита (для read-only операций)."""
    engine = get_engine()
    async_session_maker = get_session_maker(engine)
    session: AsyncSession = async_session_maker()
    try:
        yield session
    finally:
        await session.close()
