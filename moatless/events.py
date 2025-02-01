import asyncio
from datetime import datetime
from enum import Enum
import logging
from typing import Optional, List, Dict, Any, Callable

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

    @classmethod
    def get_instance(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, callback: Callable):
        logger.info(f"Subscribing to event: {callback.__name__}")
        self._subscribers.append(callback)
        logger.info(f"Subscribed to {len(self._subscribers)} events")

    async def publish(self, run_id: str, event: BaseEvent):
        """Publish event, handling both sync and async subscribers"""
        logger.info(f"Publishing event: {event.event_type} to {len(self._subscribers)} subscribers")
        await asyncio.gather(*[self._run_async_callback(callback, run_id, event) for callback in self._subscribers])

    async def _run_async_callback(self, callback: Callable, run_id: str, event: BaseEvent):
        """Helper method to run a single async callback"""
        logger.info(f"Running async callback: {callback.__name__}")
        await callback(run_id, event)


event_bus = EventBus.get_instance()