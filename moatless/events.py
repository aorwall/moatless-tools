import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import List, Any, Callable, Optional
import os

import aiofiles
from pydantic import BaseModel, Field, field_validator

from moatless.utils.moatless import get_moatless_trajectory_dir
from . import context_data

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

class BaseEvent(BaseModel):
    """Base class for all events"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scope: Optional[str] = None
    trajectory_id: Optional[str] = None
    project_id: Optional[str] = None
    event_type: str
    data: Optional[dict] = Field(default_factory=dict)

    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_datetime(cls, value):
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    def to_dict(self) -> dict:
        extra_data = super().model_dump(exclude_none=True, exclude={'timestamp', 'event_type', 'scope', 'trajectory_id', 'project_id', 'data'})
        event_data = {
            "timestamp": self.timestamp.isoformat(),
            "project_id": self.project_id,
            "trajectory_id": self.trajectory_id,
            "scope": self.scope,
            "event_type": self.event_type,
            "data": self.data
        }
        if extra_data:
            event_data['data'] = extra_data
        return event_data



class FlowEvent(BaseEvent):
    scope: str = "flow"


class FlowStartedEvent(FlowEvent):
    """Flow-specific event"""

    event_type: str = "started"

class FlowCompletedEvent(FlowEvent):
    """Flow-specific event"""

    event_type: str = "completed"

class FlowErrorEvent(FlowEvent):
    event_type: str = "error"


class EventBus:
    _instance = None
    _redis = None
    _redis_available = False

    def __init__(self):
        self._subscribers: List[Callable[[str, BaseEvent], None]] = []
        self._lock = asyncio.Lock()
        self._pubsub = None
        self._subscriber_tasks = set()
        self._redis_listener_task = None
        self._check_redis_available()

    def _check_redis_available(self):
        """Check if Redis package is available"""
        try:
            import redis.asyncio
            self._redis_available = True
            logger.info("Redis package is available, pub/sub will use Redis")
        except ImportError:
            self._redis_available = False
            logger.info("Redis package not available, using local pub/sub only")

    @classmethod
    def get_instance(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self):
        """Initialize Redis connection if available"""
        if not self._redis_available or self._redis is not None:
            return

        try:
            from redis.asyncio import Redis
            
            host = os.getenv('REDIS_HOST', 'localhost')
            port = int(os.getenv('REDIS_PORT', 6379))
            db = int(os.getenv('REDIS_DB', 0))
            
            self._redis = Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            self._pubsub = self._redis.pubsub()
            
            logger.info("Successfully initialized Redis connection")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis connection: {str(e)}")
            self._redis_available = False

    async def _handle_redis_messages(self):
        """Handle all Redis messages in a single task"""
        if not self._redis_available:
            return
        logger.info("Starting Redis listener")

        try:
            await self._pubsub.subscribe("events")
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    event_data = json.loads(message["data"])
                    event = BaseEvent(**event_data)
                    logger.info(f"Received event {event.scope} {event.event_type} for trajectory {event.trajectory_id} and project {event.project_id}")
                    
                    # Notify all subscribers
                    for callback in self._subscribers:
                        logger.debug(f"Notifying subscriber {callback.__name__} for event {event.event_type}")
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
        except asyncio.CancelledError:
            if self._pubsub:
                await self._pubsub.unsubscribe("events")
            raise
        except Exception as e:
            logger.exception(f"Error in Redis message handler: {str(e)}")

    async def _start_redis_listener(self):
        """Start the Redis listener if not already running"""
        if not self._redis_available or not self._subscribers:
            return
            
        if self._redis_listener_task is None or self._redis_listener_task.done():
            loop = asyncio.get_event_loop()
            self._redis_listener_task = loop.create_task(self._handle_redis_messages())
            self._subscriber_tasks.add(self._redis_listener_task)
            self._redis_listener_task.add_done_callback(self._subscriber_tasks.discard)

    async def _stop_redis_listener(self):
        """Stop the Redis listener if running and no subscribers"""
        if self._redis_listener_task and not self._subscribers:
            self._redis_listener_task.cancel()
            try:
                await self._redis_listener_task
            except asyncio.CancelledError:
                pass
            self._redis_listener_task = None

    async def subscribe(self, callback: Callable):
        """Subscribe to events"""
        self._subscribers.append(callback)
        logger.info(f"Added subscriber {callback.__name__}")
        # Start Redis listener if this is the first subscriber
        if len(self._subscribers) == 1:
            await self._start_redis_listener()

    async def unsubscribe(self, callback: Callable):
        """Unsubscribe from events"""
        logger.info(f"Removing subscriber {callback.__name__}")
        self._subscribers.remove(callback)
        logger.info(f"Unsubscribed from {len(self._subscribers)} events")
        # Stop Redis listener if no more subscribers
        if not self._subscribers:
            await self._stop_redis_listener()

    async def publish(self, event: BaseEvent):
        """Publish event to Redis (if available) and save to jsonl"""
        if self._redis_available and self._redis is None:
            await self.initialize()
            
        if not event.trajectory_id:
            event.trajectory_id = context_data.current_trajectory_id.get()
        if not event.project_id:
            event.project_id = context_data.current_project_id.get()

        logger.info(f"Publishing event [{event.scope}:{event.event_type}] for trajectory {event.trajectory_id} and project {event.project_id}")
        
        await self._save_event(event)
        
        if self._redis_available and self._redis:
            try:
                await self._redis.publish("events", json.dumps(event.to_dict()))
            except Exception as e:
                logger.warning(f"Failed to publish to Redis: {str(e)}")
        
        # Handle local subscribers
        await asyncio.gather(*[
            self._run_async_callback(callback, event) 
            for callback in self._subscribers
        ])

    async def _save_event(self, event: BaseEvent):
        """Thread-safe event saving to trajectory-specific events.jsonl"""
        try:
            traj_dir = get_moatless_trajectory_dir(project_id=event.project_id, trajectory_id=event.trajectory_id)
            events_path = traj_dir / 'events.jsonl'

            async with self._lock:
                async with aiofiles.open(events_path, mode='a', encoding='utf-8') as f:
                    await f.write(json.dumps(event.to_dict()) + '\n')
                    await f.flush()
        except Exception as e:
            logger.exception(f"Error saving event: {str(e)}")
            raise

    async def _run_async_callback(self, callback: Callable, event: BaseEvent):
        """Helper method to run a single async callback"""
        await callback(event)

    async def read_events(self, project_id: str, trajectory_id: str) -> List[BaseEvent]:
        """Read events from trajectory-specific events.jsonl"""
        traj_dir = get_moatless_trajectory_dir(project_id=project_id, trajectory_id=trajectory_id)
        events_path = traj_dir / 'events.jsonl'
        if not events_path.exists():
            return []
        async with aiofiles.open(events_path, mode='r', encoding='utf-8') as f:
            events = []
            async for line in f:
                event_data = json.loads(line)
                events.append(BaseEvent(**event_data))
        return events

# Initialize the singleton instance
event_bus = EventBus.get_instance()