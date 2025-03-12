import asyncio
import logging
from typing import Callable, Optional

from moatless.events import BaseEvent
from moatless.eventbus.base import BaseEventBus
from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)


class LocalEventBus(BaseEventBus):
    """
    Local in-memory implementation of the event bus.
    """

    def __init__(self, storage: Optional[BaseStorage] = None):
        super().__init__(storage=storage)
        self.subscribers = []
        self._initialized = False

    async def initialize(self):
        """Initialize the event bus."""
        if self._initialized:
            return

        self._initialized = True
        logger.debug("LocalEventBus initialized")

    async def subscribe(self, callback: Callable):
        """
        Subscribe to events.

        Args:
            callback: A function to be called when an event is published
        """
        if callback not in self.subscribers:
            self.subscribers.append(callback)
            logger.debug(f"Added subscriber {callback.__name__}")

    async def unsubscribe(self, callback: Callable):
        """
        Unsubscribe from events.

        Args:
            callback: The function to unsubscribe
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.debug(f"Removed subscriber {callback.__name__}")

    async def publish(self, event: BaseEvent):
        """
        Publish an event.

        Args:
            event: The event to publish
        """
        await self._save_event(event)

        for callback in self.subscribers:
            try:
                logger.debug(f"Notifying subscriber {callback.__name__} for event {event.event_type}")
                if asyncio.iscoroutinefunction(callback):
                    await self._run_async_callback(callback, event)
                else:
                    callback(event)
            except Exception as e:
                logger.exception(f"Error in subscriber {callback.__name__}: {e}")

    async def _run_async_callback(self, callback: Callable, event: BaseEvent):
        """Run an async callback with error handling."""
        try:
            await callback(event)
        except Exception as e:
            logger.exception(f"Error in async subscriber {callback.__name__}: {e}")
