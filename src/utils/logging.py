"""Настройка логирования с поддержкой цветного вывода и структурированных логов."""
import json
import logging
import queue
from collections import deque
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from src.config import settings

console = Console()
logger = logging.getLogger(__name__)

_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
# Публичный алиас — используется также вне этого модуля (см. GET /logs в
# src/web/api.py, отдающий персистентную историю из LogEntry, а не только
# ring-буфер) для фильтрации "уровень и выше" тем же порядком уровней.
LEVEL_ORDER = _LEVEL_ORDER


class RingBufferHandler(logging.Handler):
    """Хранит последние N лог-записей в памяти для отображения в веб-панели.

    Одновременно кладёт ту же запись в _pending_db_records (без ограничения
    размера) — оттуда её периодически забирает фоновый flush-цикл
    (_flush_logs_to_db_loop в main.py) и пишет в LogEntry (см. src/db/models.py),
    чтобы полная история логов переживала рестарт процесса и не терялась при
    перезаписи ring-буфера (capacity здесь — всего 2000 записей, на активном
    боте перезаписывается за десятки минут)."""

    def __init__(self, capacity: int = 2000):
        super().__init__()
        self.records: deque = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            self.records.append(entry)
            _pending_db_records.put_nowait(entry)
        except Exception:
            pass


ring_buffer_handler = RingBufferHandler()

# Неограниченная по размеру очередь (в отличие от ring_buffer_handler.records,
# у которой maxlen=2000) — накапливает записи между проходами фонового
# flush-цикла (main.py: _flush_logs_to_db_loop), не теряя ни одной, даже если
# сам цикл временно отстал (например, БД была недоступна). queue.SimpleQueue,
# а не deque — logging.Handler.emit может вызываться из разных потоков
# (uvicorn/thread pool), нужна потокобезопасная put/get без явной блокировки.
_pending_db_records: queue.SimpleQueue = queue.SimpleQueue()


def drain_pending_log_records(max_items: int = 10000) -> list[dict]:
    """Забрать накопленные с прошлого вызова лог-записи (не более max_items за
    раз — защита от одной гигантской транзакции, если flush-цикл долго не
    запускался) для записи в БД. Не блокирует — если очередь опустела раньше
    max_items, просто возвращает то, что успело накопиться."""
    records = []
    for _ in range(max_items):
        try:
            records.append(_pending_db_records.get_nowait())
        except queue.Empty:
            break
    return records

# Логгеры, которые модульная система логирования настраивает явно (см.
# setup_logging ниже) плюс сторонние — показываются в чекбокс-фильтре
# веб-панели даже до того, как что-то от них реально залогировалось.
_KNOWN_LOGGER_FAMILIES = [
    "src.main", "src.data_ingest", "src.strategy", "src.risk",
    "src.execution", "src.ml", "src.telegram", "src.web",
    "uvicorn", "apscheduler",
]


def _logger_family(name: str) -> str:
    """Свернуть полное имя логгера (напр. src.execution.executor) до 'семейства' (src.execution)."""
    parts = name.split(".")
    if len(parts) > 1 and parts[0] == "src":
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def get_logger_families() -> list[str]:
    """Список 'семейств' логгеров для чекбокс-фильтра — известные + реально встретившиеся."""
    seen = {_logger_family(r["logger"]) for r in ring_buffer_handler.records}
    return sorted(seen | set(_KNOWN_LOGGER_FAMILIES))


def get_recent_logs(
    level: str | None = None,
    search: str | None = None,
    loggers: list[str] | None = None,
    limit: int = 200,
) -> list[dict]:
    """Отфильтровать записи из ring-буфера логов для веб-панели.

    loggers=None — без фильтра по модулю (показать всё); loggers=[...] —
    показать только записи, чьё 'семейство' логгера входит в список (пустой
    список фильтрует всё, что осмысленно соответствует "ни один модуль не выбран").
    """
    min_level = _LEVEL_ORDER.get((level or "").upper())
    result = []
    for r in ring_buffer_handler.records:
        if min_level is not None and _LEVEL_ORDER.get(r["level"], 0) < min_level:
            continue
        if loggers is not None and _logger_family(r["logger"]) not in loggers:
            continue
        if search and search.lower() not in r["message"].lower():
            continue
        result.append(r)
    return result[-limit:]

# Кастомная тема для Rich
trading_theme = Theme({
    "info": "dim cyan",
    "warning": "yellow bold",
    "error": "red bold",
    "trade": "green bold",
    "signal": "blue bold",
    "ml": "magenta bold",
    "risk": "dark_orange bold",
    "telegram": "purple bold",
    "coinglass": "blue italic",
})


class StructuredFormatter(logging.Formatter):
    """
    Форматтер, который умеет выводить структурированные логи (JSON)
    для машинного парсинга и человекочитаемые логи для консоли.
    """

    def __init__(self, *, structured: bool = False, use_rich: bool = True):
        super().__init__()
        self.structured = structured
        self.use_rich = use_rich

    def format(self, record: logging.LogRecord) -> str:
        if self.structured:
            return self._format_structured(record)
        return super().format(record)

    def _format_structured(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_obj["exception"] = str(record.exc_info[1])
        if hasattr(record, "extra_data") and record.extra_data:
            log_obj["data"] = record.extra_data
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(level: str | None = None):
    """
    Настроить логирование для всего приложения.
    level: строка уровня (INFO, DEBUG, WARNING, ERROR) или None для значения из settings.
    """
    log_level = level or settings.log_level.upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Очистка существующих обработчиков
    root_logger.handlers.clear()

    # Консоль с Rich
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False,
    )
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root_logger.addHandler(console_handler)

    # Файл логов (в data/logs/)
    try:
        from pathlib import Path
        log_dir = Path(__file__).parent.parent / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "cryptobot.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)
        logger.info(f"Логи пишутся в {log_file}")
    except Exception as e:
        logger.warning(f"Не удалось настроить файловое логирование: {e}")

    # Ring-буфер в памяти — источник для /logs в веб-панели
    ring_buffer_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(ring_buffer_handler)

    # Специальные логгеры для модулей
    for module_logger_name in ["src.data_ingest", "src.strategy", "src.risk",
                                "src.execution", "src.ml", "src.telegram", "src.web"]:
        module_logger = logging.getLogger(module_logger_name)
        module_logger.setLevel(numeric_level)
        module_logger.propagate = True


def log_trade(trade_id: int, symbol: str, side: str, pnl: float, pnl_pct: float,
              outcome: str, strategy_name: str = "") -> None:
    """Удобная функция для логирования сделки."""
    logger = logging.getLogger("src.execution")
    data = {
        "trade_id": trade_id,
        "symbol": symbol,
        "side": side,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "outcome": outcome,
        "strategy": strategy_name,
    }
    if outcome == "win":
        logger.info(f"✅ Сделка #{trade_id} | {symbol} {side} | +{pnl:.2f} ({pnl_pct:.2f}%)", extra={"extra_data": data})
    elif outcome == "loss":
        logger.warning(f"❌ Сделка #{trade_id} | {symbol} {side} | {pnl:.2f} ({pnl_pct:.2f}%)", extra={"extra_data": data})
    else:
        logger.info(f"🟰 Сделка #{trade_id} | {symbol} {side} | {pnl:.2f} ({pnl_pct:.2f}%)", extra={"extra_data": data})
