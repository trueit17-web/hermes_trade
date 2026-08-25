"""WebSocket расширения для веб-интерфейса."""
import dataclasses
import logging

from src.event_bus import Event, event_bus
from src.web.api import ws_manager

logger = logging.getLogger(__name__)


async def broadcast_event(event: Event):
    """
    Отправить событие всем WebSocket клиентам.

    Раньше в сообщение попадал только event.payload — у базового Event это
    осмысленно, но конкретные события (например TradeEvent) хранят свои
    данные в СОБСТВЕННЫХ типизированных полях (symbol, pnl, direction,
    outcome, is_opening...), а не в payload, который у них всегда None.
    Клиент получал пустое "payload": null без единого реального поля
    сделки. dataclasses.asdict разворачивает все поля датакласса, включая
    унаследованные от подклассов.
    """
    data = dataclasses.asdict(event)
    event_type = data.pop("type")
    await ws_manager.broadcast({
        "type": "event",
        "event_type": event_type,
        **data,
    })


def setup_websocket_broadcast():
    """Подключить broadcast событий к event_bus."""
    async def _on_event(event: Event):
        await broadcast_event(event)

    event_bus.subscribe_all(_on_event)
    logger.info("WebSocket broadcast подключён к event_bus")
