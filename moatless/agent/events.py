from typing import Optional, Dict

from moatless.events import BaseEvent


class AgentEvent(BaseEvent):
    """Base class for pure agent events"""
    
    agent_id: str
    node_id: int
    action_name: Optional[str] = None
    action_params: Optional[Dict] = None


class AgentStarted(AgentEvent):
    """Emitted when an agent starts processing"""

    event_type: str = "agent_started"


class AgentActionCreated(AgentEvent):
    """Emitted when an agent creates an action"""

    event_type: str = "agent_action_created"


class AgentActionExecuted(AgentEvent):
    """Emitted when an agent executes an action"""

    event_type: str = "agent_action_executed"
