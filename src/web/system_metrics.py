"""Системные показатели сервера/контейнера для панели логов веб-панели."""
import os
import time
from pathlib import Path

import psutil

from src.config import settings

_PROCESS = psutil.Process(os.getpid())
_PROCESS_START = time.time()

# VERSION лежит в корне репозитория и копируется в образ отдельной
# командой в Dockerfile (COPY VERSION .) — читаем один раз при импорте
# модуля, а не на каждый запрос: в образе файл неизменен до следующего
# редеплоя. Реальный мотив: инцидент, когда без явного номера версии в
# ответе редеплоя было невозможно быстро отличить "код уже обновился" от
# "docker собрал из кеша старый слой" — см. CHANGELOG.md.
_VERSION_PATH = Path(__file__).parent.parent.parent / "VERSION"
try:
    _VERSION = _VERSION_PATH.read_text().strip()
except OSError:
    _VERSION = "unknown"

# Первый вызов psutil.cpu_percent()/Process.cpu_percent() всегда возвращает
# 0.0 — им нужен предыдущий замер для сравнения. "Прогреваем" здесь при
# импорте модуля, а не при первом реальном запросе — иначе самый первый
# GET /system/metrics после старта бота всегда показывал бы 0% CPU.
psutil.cpu_percent(interval=None)
_PROCESS.cpu_percent(interval=None)


def get_system_metrics() -> dict:
    """
    Снимок системных показателей: CPU/память/диск всей машины (видимой
    контейнеру — в Docker это, как правило, показатели ХОСТА, а не только
    cgroup-лимита контейнера) плюс отдельно — сам процесс бота (его RSS,
    CPU% и аптайм). Диск считается для settings.data_dir (/app/data) —
    это единственный примонтированный volume (см. docker-compose.yml),
    там же логи и модели, поэтому именно его заполнение операционно важно
    отслеживать, а не корневую ФС контейнера.
    """
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(settings.data_dir))
    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        load1 = load5 = load15 = None

    with _PROCESS.oneshot():
        proc_rss = _PROCESS.memory_info().rss
        proc_cpu_percent = _PROCESS.cpu_percent(interval=None)

    return {
        "version": _VERSION,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count() or 0,
        "load_avg": {"1m": load1, "5m": load5, "15m": load15},
        "memory": {"used": vm.used, "total": vm.total, "percent": vm.percent},
        "disk": {"used": disk.used, "total": disk.total, "percent": disk.percent},
        "process": {
            "rss": proc_rss,
            "cpu_percent": proc_cpu_percent,
            "uptime_seconds": time.time() - _PROCESS_START,
        },
    }
