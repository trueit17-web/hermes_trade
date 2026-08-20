"""WebSocket расширения для веб-интерфейса."""
import logging

from src.event_bus import Event, event_bus
from src.web.api import ws_manager

logger = logging.getLogger(__name__)


async def broadcast_event(event: Event):
    """Отправить событие всем WebSocket клиентам."""
    await ws_manager.broadcast({
        "type": "event",
        "event_type": event.type,
        "timestamp": event.timestamp,
        "source": event.source,
        "payload": event.payload,
    })


def setup_websocket_broadcast():
    """Подключить broadcast событий к event_bus."""
    async def _on_event(event: Event):
        await broadcast_event(event)

    event_bus.subscribe_all(_on_event)
    logger.info("WebSocket broadcast подключён к event_bus")
