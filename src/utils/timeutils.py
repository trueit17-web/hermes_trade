"""Единая точка для получения текущего времени в UTC.

datetime.utcnow() помечен как deprecated (Python 3.12+) и будет удалён.
Эта функция — прямая замена с идентичной семантикой: наивный (без tzinfo)
datetime в UTC, вычисленный через datetime.now(timezone.utc) без
предупреждения. Значение бит-в-бит совпадает со старым datetime.utcnow(),
поэтому её можно безопасно использовать и как default= для колонок
DateTime (без timezone=True) в src/db/models.py — тип колонок не меняется.
"""
from datetime import UTC, datetime


def utcnow() -> datetime:
    """Текущее время в UTC как наивный datetime (замена datetime.utcnow())."""
    return datetime.now(UTC).replace(tzinfo=None)


def utcnow_timestamp() -> float:
    """
    Текущее время как unix-epoch (секунды). НЕ то же самое, что
    utcnow().timestamp() — на наивном datetime .timestamp() трактует
    значение как локальное время, а не UTC, и даёт неверный результат
    вне контейнеров/машин с TZ=UTC.
    """
    return datetime.now(UTC).timestamp()
