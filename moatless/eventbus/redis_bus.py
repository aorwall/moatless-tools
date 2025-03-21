import asyncio
import json
import logging
import os
from typing import Any, Callable, List, Optional

import redis.asyncio as redis

from moatless.events import BaseEvent
from moatless.eventbus.base import BaseEventBus
from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)


class RedisEventBus(BaseEventBus):
    """
    Redis-based implementation of the event bus.

    This implementation uses Redis for pub/sub to distribute events across
    multiple instances and persists events to the storage system.
    """

    def __init__(self, redis_url: Optional[str] = None, storage: Optional[BaseStorage] = None):
        """
        Initialize the Redis event bus.

        Args:
            redis_url: The Redis URL to connect to
        """
        super().__init__(storage=storage)
        self.subscribers: List[Callable] = []

        self._redis_url = redis_url or os.environ.get("REDIS_URL")
        if not self._redis_url:
            raise ValueError("REDIS_URL environment variable not set")
        self._redis: redis.Redis = redis.from_url(self._redis_url)
        self._pubsub = self._redis.pubsub()
        self._listener_task: Optional[asyncio.Task] = None
        self._redis_available = False
        self._initialized = False

        logger.info(f"Initialized RedisEventBus with Redis URL: {self._redis_url}")

    async def _check_redis_available(self) -> None:
        """Check if Redis is available."""
        try:
            await self._redis.ping()
            self._redis_available = True
            logger.debug("Redis connection successful")
        except Exception as e:
            self._redis_available = False
            logger.warning(f"Redis connection failed: {e}")

    async def initialize(self) -> None:
        """Initialize the event bus, including Redis connections."""
        if self._initialized:
            return

        await self._check_redis_available()

        if self._redis_available:
            try:
                await self._start_redis_listener()
                logger.debug("RedisEventBus initialized with Redis")
            except Exception as e:
                logger.exception(f"Error initializing Redis event bus: {e}")
                self._redis_available = False
        else:
            logger.warning("Redis unavailable, events will be local only")

        self._initialized = True

    async def _handle_redis_messages(self) -> None:
        """Handle messages from Redis pub/sub."""
        if not self._pubsub:
            return

        try:
            await self._pubsub.subscribe("events")
            async for message in self._pubsub.listen():
                if isinstance(message, dict) and message.get("type") == "message":
                    # Redis returns bytes, so we need to decode
                    data = message.get("data", "")
                    data_str = data.decode("utf-8") if isinstance(data, bytes) else str(data)

                    try:
                        event_dict = json.loads(data_str)

                        event = BaseEvent.model_validate(event_dict)
                        logger.debug(
                            f"Received event {event.scope} {event.event_type} for trajectory {event.trajectory_id} and project {event.project_id}"
                        )

                        for callback in self.subscribers:
                            try:
                                logger.debug(f"Notifying subscriber {callback.__name__} for event {event.event_type}")
                                if asyncio.iscoroutinefunction(callback):
                                    await self._run_async_callback(callback, event)
                                else:
                                    callback(event)
                            except Exception as e:
                                logger.exception(f"Error in subscriber {callback.__name__}: {e}")
                    except Exception as e:
                        logger.exception(f"Error processing Redis message: {e}")
        except asyncio.CancelledError:
            if self._pubsub:
                await self._pubsub.unsubscribe("events")
            raise
        except Exception as e:
            logger.exception(f"Error in Redis pubsub listener: {e}")

    async def _start_redis_listener(self) -> None:
        """Start the Redis pub/sub listener task."""
        if self._redis_available and not self._listener_task:
            self._listener_task = asyncio.create_task(self._handle_redis_messages())
            logger.info("Started Redis pub/sub listener")

    async def _stop_redis_listener(self) -> None:
        """Stop the Redis pub/sub listener task."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
            logger.info("Stopped Redis pub/sub listener")

    async def subscribe(self, callback: Callable) -> None:
        """
        Subscribe to events.

        Args:
            callback: A function to be called when an event is published
        """
        if callback not in self.subscribers:
            self.subscribers.append(callback)
            logger.debug(f"Added subscriber {callback.__name__}")

    async def unsubscribe(self, callback: Callable) -> None:
        """
        Unsubscribe from events.

        Args:
            callback: The function to unsubscribe
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.debug(f"Removed subscriber {callback.__name__}")

    async def publish(self, event: BaseEvent) -> None:
        """
        Publish an event.

        Args:
            event: The event to publish
        """
        await self._save_event(event)

        try:
            event_data = event.model_dump(mode="json")
            serialized = json.dumps(event_data)
            await self._redis.publish("events", serialized)
            logger.debug(f"Published event to Redis: {event.event_type}")
        except Exception as e:
            logger.exception(f"Error publishing event to Redis: {e}")

    async def _run_async_callback(self, callback: Callable, event: BaseEvent) -> None:
        """Run an async callback with error handling."""
        try:
            await callback(event)
        except json.JSONDecodeError as jde:
            logger.error(f"JSON decoding error when processing event {event.event_type} for {event.project_id}: {jde}")
            logger.info(f"Event details: {event}")
        except Exception as e:
            logger.exception(f"Failed to send event {event} to {callback.__name__}")
