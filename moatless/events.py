import logging
from datetime import datetime, timezone
from typing import Optional, List, Callable, Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class BaseEvent(BaseModel):
    """Base class for all events"""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scope: Optional[str] = None
    trajectory_id: Optional[str] = None
    project_id: Optional[str] = None
    event_type: str
    data: Optional[dict] = Field(default_factory=dict)

    model_config = {"ser_json_timedelta": "iso8601", "json_encoders": {datetime: lambda dt: dt.isoformat()}}

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_datetime(cls, value):
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    @classmethod
    def from_dict(cls, data: dict) -> "BaseEvent":
        """Create an event from a dictionary."""
        return cls.model_validate(data)


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
