from typing import Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime

from moatless.events import BaseEvent


class AgentEvent(BaseEvent):
    """Base class for pure agent events"""

    agent_id: str
    node_id: int


class AgentStarted(AgentEvent):
    """Emitted when an agent starts processing"""

    event_type: str = "agent_started"


class AgentActionCreated(AgentEvent):
    """Emitted when an agent creates an action"""

    event_type: str = "agent_action_created"
    action_name: str
    action_params: Dict


class AgentActionExecuted(AgentEvent):
    """Emitted when an agent executes an action"""

    event_type: str = "agent_action_executed"
    action_name: str
    observation: Optional[str]
