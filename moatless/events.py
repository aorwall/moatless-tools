import asyncio
from datetime import datetime
from enum import Enum
import json
import logging
import os
from pathlib import Path
import aiofiles
from typing import Optional, List, Dict, Any, Callable
import contextvars

from moatless.utils.moatless import get_moatless_trajectory_dir
from . import context_data

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class LoopCompletedData(BaseModel):
    """Loop-specific event"""

    event_type: str = "loop_completed"
    total_iterations: int
    total_cost: float
    final_node_id: int


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

    def __init__(self):
        self._subscribers: List[Callable[[str, BaseEvent], None]] = []
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, callback: Callable):
        logger.info(f"Subscribing to event: {callback.__name__}")
        self._subscribers.append(callback)
        logger.info(f"Subscribed to {len(self._subscribers)} events")

    async def publish(self, event: BaseEvent):
        """Publish event, handling both sync and async subscribers and saving to jsonl"""
        trajectory_id = context_data.current_trajectory_id.get()
        
        logger.info(f"Publishing event: {event.event_type} for trajectory {trajectory_id} to {len(self._subscribers)} subscribers")
        
        # Save event to trajectory-specific events.jsonl
        await self._save_event(trajectory_id, event)
        
        # Notify subscribers
        await asyncio.gather(*[self._run_async_callback(callback, trajectory_id, event) for callback in self._subscribers])

    async def _save_event(self, trajectory_id: str, event: BaseEvent):
        """Thread-safe event saving to trajectory-specific events.jsonl"""
        traj_dir = get_moatless_trajectory_dir(trajectory_id)
        events_path = traj_dir / 'events.jsonl'

        event_dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "trajectory_id": trajectory_id,
            "event_type": event.event_type,
            "data": event.model_dump(exclude_none=True, exclude={'event_type'})
        }

        async with self._lock:
            async with aiofiles.open(events_path, mode='a', encoding='utf-8') as f:
                await f.write(json.dumps(event_dict) + '\n')

    async def _run_async_callback(self, callback: Callable, trajectory_id: str | None, event: BaseEvent):
        """Helper method to run a single async callback"""
        await callback(trajectory_id, event)


event_bus = EventBus.get_instance()