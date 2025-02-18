from moatless.events import BaseEvent, FailureEvent


class FlowStartedEvent(BaseEvent):
    event_type: str = "flow_started"

class FlowCompletedEvent(BaseEvent):
    event_type: str = "flow_completed"
    finish_reason: str | None = None

class FlowErrorEvent(BaseEvent):
    event_type: str = "flow_error"
    error: str

class NodeExpandedEvent(BaseEvent):
    event_type: str = "node_expanded"
    parent_node_id: int
    child_node_id: int

class NodeSelectedEvent(BaseEvent):
    """Node-specific event"""

    event_type: str = "node_selected"
    previous_node_id: int
    selected_node_id: int


class FeedbackGeneratedEvent(BaseEvent):
    """Node-specific event"""

    event_type: str = "feedback_generated"
    node_id: int

class NodeRewardEvent(BaseEvent):
    """Node-specific event"""

    event_type: str = "node_reward"
    node_id: int
    reward: float


class NodeRewardFailureEvent(FailureEvent):
    """Node-specific event"""

    scope: str = "reward"
    node_id: int
    error: str

