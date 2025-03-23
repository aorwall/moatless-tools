import abc
import asyncio
import logging
from typing import Callable, Optional, Type, List

from moatless.context_data import current_project_id, current_trajectory_id
from moatless.events import BaseEvent
from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)


class BaseEventBus(abc.ABC):
    """
    Abstract base class for event bus operations.

    This class defines the interface for event bus operations such as
    subscribing to and publishing events.
    """

    _instance = None

    @classmethod
    def get_instance(
        cls, eventbus_impl: Optional[Type["BaseEventBus"]] = None, storage: Optional[BaseStorage] = None, **kwargs
    ) -> "BaseEventBus":
        """
        Get or create the singleton instance of EventBus.

        Args:
            eventbus_impl: Optional eventbus implementation class to use, defaults to None
                           The actual default implementation is determined by the caller
            **kwargs: Arguments to pass to the eventbus implementation constructor

        Returns:
            The singleton EventBus instance
        """
        if cls._instance is None:
            from moatless.eventbus.local_bus import LocalEventBus

            impl_class = eventbus_impl or LocalEventBus
            cls._instance = impl_class(storage=storage, **kwargs)

        return cls._instance

    def __init__(self, storage: Optional[BaseStorage] = None):
        """Initialize the event bus base class."""
        self._storage = storage
        self._lock = asyncio.Lock()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Mainly useful for testing."""
        cls._instance = None

    @abc.abstractmethod
    async def initialize(self):
        """Initialize the event bus."""
        pass

    @abc.abstractmethod
    async def subscribe(self, callback: Callable):
        """
        Subscribe to events.

        Args:
            callback: A function to be called when an event is published
        """
        pass

    @abc.abstractmethod
    async def unsubscribe(self, callback: Callable):
        """
        Unsubscribe from events.

        Args:
            callback: The function to unsubscribe
        """
        pass

    @abc.abstractmethod
    async def publish(self, event: "BaseEvent"):
        """
        Publish an event.

        Args:
            event: The event to publish
        """
        pass

    async def _save_event(self, event: "BaseEvent") -> None:
        """
        Save an event using the storage system.

        Args:
            event: The event to save

        This implementation saves events in JSONL format, with one event per line,
        which is more efficient for sequential event logging.
        """
        if not self._storage:
            logger.warning("No storage system configured, cannot save event")
            return

        project_id = event.project_id or current_project_id.get()
        trajectory_id = event.trajectory_id or current_trajectory_id.get()

        if not project_id or not trajectory_id:
            logger.warning(f"Cannot save event without project_id and trajectory_id: {event}")
            return

        key = self.event_key(project_id, trajectory_id)

        try:
            async with self._lock:
                await self._storage.append(key, event.model_dump())
                logger.debug(f"Saved event {event.event_type} for {project_id}/{trajectory_id}")
        except Exception as e:
            logger.exception(f"Error saving event: {e}")

    async def read_events(self, project_id: str, trajectory_id: str) -> List[BaseEvent]:
        """
        Read events for a specific project and trajectory.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            A list of event dictionaries
        """

        key = self.event_key(project_id, trajectory_id)

        try:
            if not await self._storage.exists(key):
                return []

            event_dicts = await self._storage.read_lines(key)
            return [BaseEvent.model_validate(line) for line in event_dicts]

        except Exception as e:
            logger.exception(f"Error reading events: {e}")
            return []

    def event_key(self, project_id: str, trajectory_id: str) -> str:
        """
        Get the key for the events file.
        """
        return f"projects/{project_id}/trajs/{trajectory_id}/events.jsonl"
