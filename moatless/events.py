import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import List, Any, Callable, Optional
import os

import aiofiles
from pydantic import BaseModel

from moatless.utils.moatless import get_moatless_trajectory_dir
from . import context_data

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))


class BaseEvent(BaseModel):
    """Base class for all events"""

    event_type: str


class SystemEvent(BaseModel):
    """System event combining context with domain event"""

    run_id: str
    event_type: str
    event: dict[str, Any]


class EventData(BaseModel):
    """Base class for specific event data"""

    pass


class LoopStartData(BaseModel):
    """Loop-specific event"""

    event_type: str = "loop_started"
    initial_node_id: int


class FailureEvent(BaseModel):
    """Failure-specific event"""

    event_type: str = "failure"
    scope: str
    node_id: int
    error: str

class LoopCompletedData(BaseModel):
    """Loop-specific event"""

    event_type: str = "loop_completed"
    total_iterations: int
    total_cost: float
    final_node_id: int


class FlowStartedEvent(BaseEvent):
    """Flow-specific event"""

    event_type: str = "flow_started"

class FlowCompletedEvent(BaseEvent):
    """Flow-specific event"""

    event_type: str = "flow_completed"


class SystemEventType(Enum):
    LOOP_STARTED = "loop_started"
    LOOP_COMPLETED = "loop_completed"
    LOOP_ERROR = "loop_error"
    LOOP_ITERATION = "loop_iteration"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_ERROR = "agent_error"
    AGENT_ACTION_CREATED = "agent_action_created"
    AGENT_ACTION_EXECUTED = "agent_action_executed"



class EventBus:
    _instance = None
    _redis = None
    _redis_available = False

    def __init__(self):
        self._subscribers: List[Callable[[str, BaseEvent], None]] = []
        self._lock = asyncio.Lock()
        self._pubsub = None
        self._subscriber_tasks = set()
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
            
            # Create a single task for handling Redis messages
            if self._redis_available and not self._subscriber_tasks:
                loop = asyncio.get_event_loop()
                task = loop.create_task(self._handle_redis_messages())
                self._subscriber_tasks.add(task)
                task.add_done_callback(self._subscriber_tasks.discard)
                
            logger.info("Successfully initialized Redis connection")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis connection: {str(e)}")
            self._redis_available = False

    async def _handle_redis_messages(self):
        """Handle all Redis messages in a single task"""
        if not self._redis_available:
            return

        try:
            await self._pubsub.subscribe("events")
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    event_data = json.loads(message["data"])
                    event = BaseEvent(**event_data)
                    trajectory_id = event_data.get("trajectory_id")
                    project_id = event_data.get("project_id")
                    
                    # Notify all subscribers
                    for callback in self._subscribers:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(trajectory_id, project_id, event)
                        else:
                            callback(trajectory_id, project_id, event)
        except asyncio.CancelledError:
            if self._pubsub:
                await self._pubsub.unsubscribe("events")
            raise
        except Exception as e:
            logger.exception(f"Error in Redis message handler: {str(e)}")

    def subscribe(self, callback: Callable):
        """Subscribe to events"""
        logger.info(f"Subscribing to event: {callback.__name__}")
        self._subscribers.append(callback)
        logger.info(f"Subscribed to {len(self._subscribers)} events")

    def unsubscribe(self, callback: Callable):
        """Unsubscribe from events"""
        logger.info(f"Unsubscribing from event: {callback.__name__}")
        self._subscribers.remove(callback)
        logger.info(f"Unsubscribed from {len(self._subscribers)} events")

    async def publish(self, event: BaseEvent):
        """Publish event to Redis (if available) and save to jsonl"""
        if self._redis_available and self._redis is None:
            await self.initialize()
            
        trajectory_id = context_data.current_trajectory_id.get()
        current_project_id = context_data.current_project_id.get()
        
        logger.info(f"Publishing event: {event.event_type} for trajectory {trajectory_id} and project {current_project_id}")
        
        # Save to file
        await self._save_event(trajectory_id, current_project_id, event)
        
        # Prepare event data
        event_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trajectory_id": trajectory_id,
            "project_id": current_project_id,
            "event_type": event.event_type,
            "data": event.model_dump(exclude_none=True, exclude={'event_type'})
        }
        
        # Publish to Redis if available
        if self._redis_available and self._redis:
            try:
                await self._redis.publish("events", json.dumps(event_data))
            except Exception as e:
                logger.warning(f"Failed to publish to Redis: {str(e)}")
        
        # Handle local subscribers
        await asyncio.gather(*[
            self._run_async_callback(callback, trajectory_id, current_project_id, event) 
            for callback in self._subscribers
        ])

    async def _save_event(self, trajectory_id: str, project_id: str, event: BaseEvent):
        """Thread-safe event saving to trajectory-specific events.jsonl"""
        try:
            traj_dir = get_moatless_trajectory_dir(trajectory_id)
            events_path = traj_dir / 'events.jsonl'

            event_dict = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trajectory_id": trajectory_id,
                "project_id": project_id,
                "event_type": event.event_type,
                "data": event.model_dump(exclude_none=True, exclude={'event_type'})
            }

            async with self._lock:
                async with aiofiles.open(events_path, mode='a', encoding='utf-8') as f:
                    await f.write(json.dumps(event_dict) + '\n')
                    await f.flush()
        except Exception as e:
            logger.exception(f"Error saving event: {str(e)}")
            raise

    async def _run_async_callback(self, callback: Callable, trajectory_id: str | None, project_id: str | None, event: BaseEvent):
        """Helper method to run a single async callback"""
        await callback(trajectory_id, project_id, event)


# Initialize the singleton instance
event_bus = EventBus.get_instance()