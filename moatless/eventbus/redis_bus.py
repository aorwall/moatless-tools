import asyncio
from opentelemetry import trace
import json
import logging
import os
import uuid
from typing import Callable, Dict, List, Optional

import redis.asyncio as redis

from moatless.context_data import current_project_id, current_trajectory_id
from moatless.eventbus.base import BaseEventBus
from moatless.events import BaseEvent
from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


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

        # Generate a unique worker ID for this process
        self._worker_id = f"worker-{os.getpid()}-{uuid.uuid4().hex[:8]}"

        # Create Redis client - we'll use a connection pool for better performance
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._redis_available = False
        self._initialized = False
        self._closing = False

        # Set up subscription channel with worker ID to prevent duplicate processing
        self._channel = f"events:{self._worker_id}"

        logger.info(f"Initialized RedisEventBus with Redis URL: {self._redis_url}, worker ID: {self._worker_id}")

    async def _setup_redis(self) -> None:
        """Set up Redis connection."""
        try:
            # Create connection pool with proper configuration
            self._redis = redis.Redis.from_url(
                self._redis_url,
                decode_responses=True,  # Auto-decode responses
            )
            self._pubsub = self._redis.pubsub()

            # Test connection
            await self._redis.ping()
            self._redis_available = True
            logger.debug("Redis connection successfully established")
        except Exception as e:
            self._redis_available = False
            logger.warning(f"Redis connection failed: {e}")

    async def _check_redis_available(self) -> None:
        """Check if Redis is available."""
        if not self._redis:
            await self._setup_redis()
            return

        try:
            await self._redis.ping()
            self._redis_available = True
            logger.debug("Redis connection successful")
        except Exception as e:
            self._redis_available = False
            logger.warning(f"Redis connection failed: {e}")
            # Try to reconnect
            await self._setup_redis()

    @tracer.start_as_current_span("RedisEventBus.initialize")
    async def initialize(self) -> None:
        """Initialize the event bus, including Redis connections."""
        if self._initialized or self._closing:
            return

        await self._setup_redis()
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
            # Subscribe to worker-specific channel to prevent duplicate event processing
            # in multi-worker setups, and also to global events channel
            await self._pubsub.subscribe(self._channel)
            await self._pubsub.subscribe("events:global")

            logger.info(f"Subscribed to channels: {self._channel} and events:global")

            async for message in self._pubsub.listen():
                if isinstance(message, dict) and message.get("type") == "message":
                    # Redis returns bytes, so we need to decode
                    data = message.get("data", "")

                    try:
                        event_dict = json.loads(data)
                        event = BaseEvent.model_validate(event_dict)

                        channel = message.get("channel", "")
                        logger.debug(
                            f"Received event {event.scope} {event.event_type} on channel {channel} "
                            f"for trajectory {event.trajectory_id} and project {event.project_id}"
                        )

                        for callback in self.subscribers:
                            with tracer.start_as_current_span("RedisEventBus._handle_redis_messages.callback"):
                                try:
                                    logger.debug(
                                        f"Notifying subscriber {callback.__name__} for event {event.event_type}"
                                    )
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
                await self._pubsub.unsubscribe(self._channel)
                await self._pubsub.unsubscribe("events:global")
            raise
        except Exception as e:
            logger.exception(f"Error in Redis pubsub listener: {e}")

    async def _start_redis_listener(self) -> None:
        """Start the Redis pub/sub listener task."""
        if self._redis_available and not self._listener_task and not self._closing:
            self._listener_task = asyncio.create_task(self._handle_redis_messages())
            logger.info(f"Started Redis pub/sub listener for worker {self._worker_id}")

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
        if event.project_id is None:
            event.project_id = current_project_id.get()

        if event.trajectory_id is None:
            event.trajectory_id = current_trajectory_id.get()

        # Save to storage first
        await self._save_event(event)

        # Then to Redis
        await self._save_event_to_redis(event)

        try:
            event_data = event.model_dump(mode="json")
            serialized = json.dumps(event_data)

            # Determine routing strategy based on event type
            # For system events, broadcast to all workers on global channel
            # For specific events, target the worker-specific channel
            if event.scope in ["system", "admin"]:
                # System events go to all workers
                await self._redis.publish("events:global", serialized)
                logger.debug(f"Published global event to Redis: {event.event_type}")
            else:
                # Regular events go to worker-specific channel
                # This ensures only one worker processes each event
                await self._redis.publish(self._channel, serialized)
                logger.debug(f"Published worker-specific event to Redis: {event.event_type}")
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

    async def _save_event_to_redis(self, event: BaseEvent) -> None:
        """
        Save an event to Redis.

        Args:
            event: The event to save
        """
        if not self._redis_available:
            logger.warning("Redis unavailable, cannot save event to Redis")
            return

        project_id = event.project_id
        trajectory_id = event.trajectory_id

        if not project_id or not trajectory_id:
            logger.warning(f"Cannot save event to Redis without project_id and trajectory_id: {event}")
            return

        key = f"moatless:events:{project_id}:{trajectory_id}"

        try:
            event_data = event.model_dump(mode="json")
            serialized = json.dumps(event_data)

            # Redis rpush is a coroutine in redis.asyncio
            await self._redis.rpush(key, serialized)
            logger.debug(f"Saved event to Redis: {event.event_type}")
        except Exception as e:
            logger.exception(f"Error saving event to Redis: {e}")

    async def _read_events_from_redis(self, project_id: str, trajectory_id: str) -> List[BaseEvent]:
        """
        Read events for a specific project and trajectory from Redis.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            A list of BaseEvent objects
        """
        if not self._redis_available:
            logger.warning("Redis unavailable, cannot read events from Redis")
            return []

        key = f"moatless:events:{project_id}:{trajectory_id}"

        try:
            # Redis lrange is a coroutine in redis.asyncio
            event_strings = await self._redis.lrange(key, 0, -1)
            events = []

            for event_str in event_strings:
                try:
                    # String already decoded due to decode_responses=True
                    event_dict = json.loads(event_str)
                    event = BaseEvent.model_validate(event_dict)
                    events.append(event)
                except Exception as e:
                    logger.exception(f"Error parsing event: {e}")

            logger.debug(f"Read {len(events)} events from Redis for {project_id}/{trajectory_id}")
            return events
        except Exception as e:
            logger.exception(f"Error reading events from Redis: {e}")
            return []

    async def read_events(self, project_id: str, trajectory_id: str) -> List[BaseEvent]:
        """
        Read events for a specific project and trajectory from Redis or fallback to storage.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            A list of BaseEvent objects
        """
        # Try to read from Redis first
        events = await self._read_events_from_redis(project_id, trajectory_id)

        # If no events in Redis or Redis is unavailable, fall back to storage
        if not events and self._storage:
            storage_events = await super().read_events(project_id, trajectory_id)
            # Convert the dict events to BaseEvent objects
            if storage_events:
                events = [BaseEvent.model_validate(event_dict) for event_dict in storage_events]

        return events

    async def close(self) -> None:
        """
        Close Redis connections and clean up resources.
        This prevents "Unclosed client session" and "Task was destroyed but it is pending" errors.
        """
        if self._closing:
            logger.debug("Redis event bus already closing, ignoring duplicate close call")
            return

        self._closing = True
        logger.info(f"Closing Redis event bus connections for worker {self._worker_id}")

        # Stop the Redis listener task first
        await self._stop_redis_listener()

        # Clean up Redis pubsub connection
        if self._pubsub:
            try:
                await self._pubsub.unsubscribe(self._channel)
                await self._pubsub.unsubscribe("events:global")
                await self._pubsub.close()
                logger.debug("Closed Redis pubsub connection")
            except Exception as e:
                logger.exception(f"Error closing Redis pubsub connection: {e}")

        # Close the Redis client connection
        if self._redis:
            try:
                await self._redis.close()
                logger.debug("Closed Redis client connection")
            except Exception as e:
                logger.exception(f"Error closing Redis client: {e}")

        # Clear subscribers to prevent memory leaks
        self.subscribers.clear()

        self._initialized = False
        self._redis_available = False
        logger.info(f"Redis event bus connections closed for worker {self._worker_id}")

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Support using the event bus as an async context manager."""
        await self.close()
