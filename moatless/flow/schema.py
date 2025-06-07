import logging
from datetime import datetime
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from moatless.artifacts.artifact import ArtifactHandler
from moatless.completion.stats import CompletionInvocation, Usage
from moatless.discriminator.base import BaseDiscriminator
from moatless.expander.expander import Expander
from moatless.feedback.base import BaseFeedbackGenerator
from moatless.node import Reward
from moatless.runner.runner import JobStatus
from moatless.selector.base import BaseSelector
from moatless.value_function.base import BaseValueFunction

logger = logging.getLogger(__name__)


class FlowStatus(str, Enum):
    """Enum for system status values."""

    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    ERROR = "error"


class TrajectoryEventDTO(BaseModel):
    """Data transfer object for trajectory events."""

    id: str
    scope: str
    event_type: str
    timestamp: float
    project_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class TrajectoryListItem(BaseModel):
    """List item for trajectories."""

    project_id: str
    trajectory_id: str
    status: str
    message: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    last_restart: Optional[datetime] = None
    cost: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage: Optional[Usage] = None


class StartTrajectoryRequest(BaseModel):
    """Request to start a trajectory."""

    agent_id: str
    model_id: str
    message: Optional[str] = None


class RetryTrajectoryRequest(BaseModel):
    """Request to retry a trajectory."""

    node_id: str


class ExecuteNodeRequest(BaseModel):
    """Request to execute a node."""

    node_id: int


class NodeDTO(BaseModel):
    """Node information in the tree."""

    nodeId: int
    children: list["NodeDTO"] = Field(default_factory=list, description="Children nodes of this node")
    executed: bool = Field(default=False, description="Whether this node has been executed")
    reward: Optional[Reward] = None
    userMessage: Optional[str] = None
    assistantMessage: Optional[str] = None
    completion: Optional[CompletionInvocation] = None
    thoughts: Optional[str] = None
    error: Optional[str] = None
    warnings: list[str] = []
    errors: list[str] = []
    usage: Optional[Usage] = None
    terminal: bool = Field(default=False, description="Whether this node is in a terminal state")
    allNodeErrors: list[str] = Field(
        default_factory=list,
        description="All errors from this node, including action steps and file context",
    )
    allNodeWarnings: list[str] = Field(
        default_factory=list,
        description="All warnings from this node, including action steps and file context",
    )


class TrajectoryResponseDTO(BaseModel):
    """Data transfer object for trajectory responses."""

    trajectory_id: str
    project_id: str
    status: FlowStatus
    job_status: Optional[JobStatus] = None
    resolved: Optional[bool] = None
    flow_id: Optional[str] = None
    agent_id: Optional[str] = None
    model_id: Optional[str] = None
    events: list[TrajectoryEventDTO] = Field(default_factory=list)
    nodes: list[NodeDTO] = Field(default_factory=list)
    usage: Optional[Usage] = None


class NodeDetails(BaseModel):
    """Details about a node."""

    node: NodeDTO
    children: list[NodeDTO] = Field(default_factory=list)


class ToolCall(BaseModel):
    """A tool call."""

    name: str
    arguments: dict[str, Any]


class CompletionOutput(BaseModel):
    """Output of a completion."""

    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class CompletionInputMessage(BaseModel):
    """Input of a completion."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class CompletionDTO(BaseModel):
    """Data transfer object for completions."""

    system_prompt: Optional[str] = None
    input: Optional[List[CompletionInputMessage]] = None
    output: Optional[CompletionOutput] = None
    original_input: Optional[dict] = None
    original_output: Optional[dict] = None
    error: Optional[str] = None
