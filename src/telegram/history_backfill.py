"""
Бэкафилл истории Telegram-канала — накопление данных под БУДУЩУЮ
ML-модель качества сигнала (см. HistoricalSignal в src/db/models.py).

НЕ участвует в live-исполнении/статистике канала (win rate, expectancy
sizing) — пишет в отдельную таблицу, тем же парсером (regex + LLM-
фолбэки), что и live-обработчик (_handler в channel_monitor.py), просто
на СТАРЫХ сообщениях вместо новых.

Сообщения-картинки без подписи (скриншоты сигналов) в этой версии
пропускаются — vision-парсинг на потоке из сотен исторических сообщений
упёрся бы в те же квоты LLM, что уже дают 429 на живом трафике; текстовые
сигналы всё равно составляют основной объём большинства каналов.
"""
import asyncio
import logging

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from src.db.models import HistoricalSignal
from src.db.session import get_session
from src.telegram.channel_monitor import (
    _resolve_channel_id_input,
    get_telegram_client,
    is_closed_trade_report,
    parse_telegram_signal,
)
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Один прогон на канал может занимать долго (до `limit` сообщений, каждое —
# потенциально несколько LLM-фолбэков с сетевыми задержками) — повторное
# нажатие кнопки "⬇ История" в дашборде до завершения предыдущего запуска
# запускает ВТОРОЙ параллельный прогон для ТОГО ЖЕ канала. Оба берут свой
# снимок existing_ids в начале и не видят вставок друг друга, поэтому оба
# рано или поздно пытаются вставить одно и то же historical_message_id —
# UniqueViolationError на этом INSERT (реальный инцидент, прод: серия
# "duplicate key value violates unique constraint uq_historical_signal_message"
# несколько минут подряд на одном канале). Лок на канал сериализует прогоны
# внутри одного процесса вместо гонки.
_channel_locks: dict[int, asyncio.Lock] = {}


async def backfill_channel_history(
    db_channel_id: int, channel_id: str, channel_config: dict, limit: int = 200,
) -> dict:
    """Сериализует прогоны для одного и того же канала (см. _channel_locks)
    и делегирует саму работу в _backfill_channel_history_impl."""
    lock = _channel_locks.setdefault(db_channel_id, asyncio.Lock())
    async with lock:
        return await _backfill_channel_history_impl(db_channel_id, channel_id, channel_config, limit)


async def _backfill_channel_history_impl(
    db_channel_id: int, channel_id: str, channel_config: dict, limit: int = 200,
) -> dict:
    """
    Подгрузить до `limit` сообщений канала СТАРШЕ уже сохранённых в
    HistoricalSignal для этого канала (при первом запуске — начиная с
    самых новых) — повторные вызовы сами продвигаются вглубь истории по
    MIN(telegram_message_id), без ручного управления курсором вызывающим
    кодом (см. docstring HistoricalSignal).

    channel_config — тот же словарь {"channel_id", "channel_title",
    "parser_config"}, что и в live-обработчике — нужен parse_telegram_signal
    для LLM-фолбэков.

    Возвращает сводку по ЭТОМУ запуску (или {"error": "..."} при сбое до
    начала обработки — нет Telegram-клиента, канал не резолвится, история
    недоступна): scanned/stored/already_had/closed_reports/parsed_ok/
    unparsed/reached_channel_start.
    """
    client = get_telegram_client()
    if client is None:
        return {"error": "Telegram клиент не инициализирован"}

    try:
        entity = await client.get_entity(_resolve_channel_id_input(channel_id))
    except Exception as e:
        return {"error": f"Не удалось найти канал {channel_id}: {e}"}

    async with get_session() as session:
        oldest_stored_id = (
            await session.execute(
                select(func.min(HistoricalSignal.telegram_message_id))
                .where(HistoricalSignal.channel_id == db_channel_id)
            )
        ).scalar_one_or_none()
        existing_ids = set(
            (
                await session.execute(
                    select(HistoricalSignal.telegram_message_id)
                    .where(HistoricalSignal.channel_id == db_channel_id)
                )
            ).scalars().all()
        )

    # offset_id=0 — специальное значение Telethon "с самого нового
    # сообщения"; оно же оказывается MIN(telegram_message_id) после первого
    # запуска, продвигая курсор строго в глубь истории на каждом следующем.
    offset_id = oldest_stored_id or 0
    try:
        messages = await client.get_messages(entity, limit=limit, offset_id=offset_id)
    except Exception as e:
        return {"error": f"Не удалось получить историю сообщений {channel_id}: {e}"}

    scanned = stored = already_had = closed_reports = parsed_ok = unparsed = 0
    for message in messages:
        scanned += 1
        if message.id in existing_ids:
            already_had += 1
            continue
        text = message.text or ""
        if not text:
            continue

        if is_closed_trade_report(text):
            status = "closed_report"
            parsed = None
            closed_reports += 1
        else:
            try:
                parsed = await parse_telegram_signal(text, channel_config)
            except Exception as e:
                logger.warning(f"Бэкафилл {channel_id}: ошибка парсинга сообщения {message.id}: {e}")
                parsed = None
            if parsed:
                status = "parsed"
                parsed_ok += 1
            else:
                status = "unparsed"
                unparsed += 1

        message_date = message.date.replace(tzinfo=None) if message.date else utcnow()
        try:
            async with get_session() as session:
                session.add(HistoricalSignal(
                    channel_id=db_channel_id,
                    telegram_message_id=message.id,
                    raw_message=text,
                    message_date=message_date,
                    parse_status=status,
                    parsed_pair=parsed.get("pair") if parsed else None,
                    parsed_side=parsed.get("side") if parsed else None,
                    parsed_entry=parsed.get("entry") if parsed else None,
                    parsed_sl=parsed.get("sl") if parsed else None,
                    parsed_tp=parsed.get("tp") if parsed else None,
                    parsed_take_profits=parsed.get("take_profits") if parsed else None,
                    parsed_leverage=parsed.get("leverage") if parsed else None,
                ))
                await session.commit()
        except IntegrityError:
            # Тот же message.id уже вставлен — при однопроцессном
            # деплое лок выше (_channel_locks) должен это исключать, но
            # если снимок existing_ids всё же устарел (например, строка
            # вставлена мимо этой функции), считаем сообщение уже
            # загруженным вместо того, чтобы обрывать весь оставшийся
            # прогон одной гонкой на INSERT.
            already_had += 1
            continue
        stored += 1

    return {
        "scanned": scanned,
        "stored": stored,
        "already_had": already_had,
        "closed_reports": closed_reports,
        "parsed_ok": parsed_ok,
        "unparsed": unparsed,
        # Меньше запрошенного limit пришло — история канала закончилась,
        # дальше бэкафиллить для него нечего.
        "reached_channel_start": len(messages) < limit,
    }
