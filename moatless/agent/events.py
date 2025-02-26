from typing import Optional, Dict

from moatless.events import BaseEvent


class AgentEvent(BaseEvent):
    """Base class for pure agent events"""

    scope: str = "agent"
    agent_id: str
    node_id: int
    
class RunAgentEvent(AgentEvent):
    """Emitted when an agent starts processing"""

    event_type: str = "run"

class AgentErrorEvent(AgentEvent):
    """Emitted when an agent fails to create an action"""

    event_type: str = "error"
    error: str


class ActionEvent(AgentEvent):
    """Base class for action events"""

    scope: str = "action"
    action_name: Optional[str] = None
    action_params: Optional[Dict] = None

class ActionCreatedEvent(ActionEvent):
    """Emitted when an agent creates an action"""

    event_type: str = "created"


class ActionExecutedEvent(ActionEvent):
    """Emitted when an agent executes an action"""

    event_type: str = "executed"
