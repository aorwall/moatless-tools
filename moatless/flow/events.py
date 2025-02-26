from typing import Optional

from moatless.events import BaseEvent

class FlowEvent(BaseEvent):
    scope: str = "flow"

class FlowStartedEvent(FlowEvent):
    event_type: str = "started"

class FlowCompletedEvent(FlowEvent):
    event_type: str = "completed"
    finish_reason: Optional[int] = None

class FlowErrorEvent(FlowEvent):
    event_type: str = "error"
    node_id: Optional[int] = None
    error: str

class NodeEvent(BaseEvent):
    scope: str = "node"
    node_id: int

class NodeExpandedEvent(NodeEvent):
    event_type: str = "expanded"
    child_node_id: int

class NodeSelectedEvent(NodeEvent):
    event_type: str = "selected"
    previous_node_id: int

class FeedbackGeneratedEvent(NodeEvent):
    event_type: str = "feedback_generated"

class NodeRewardEvent(NodeEvent):
    event_type: str = "reward_generated"
    reward: float

class NodeRewardFailureEvent(NodeEvent):
    event_type: str = "reward_failure"
    error: str

