"""Настройка логирования с поддержкой цветного вывода и структурированных логов."""
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from src.config import settings

console = Console()
logger = logging.getLogger(__name__)

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_obj["exception"] = str(record.exc_info[1])
        if hasattr(record, "extra_data") and record.extra_data:
            log_obj["data"] = record.extra_data
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(level: Optional[str] = None):
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
