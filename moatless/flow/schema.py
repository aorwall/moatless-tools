import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Literal, Optional, Union

from moatless.artifacts.artifact import ArtifactHandler
from moatless.completion.stats import CompletionInvocation, Usage
from moatless.discriminator.base import BaseDiscriminator
from moatless.expander import Expander
from moatless.feedback.base import BaseFeedbackGenerator
from moatless.node import Reward
from moatless.runner.runner import JobStatus
from moatless.selector.base import BaseSelector
from moatless.value_function.base import BaseValueFunction
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class FlowConfig(BaseModel):
    """Configuration for a tree search instance."""

    id: str = Field(..., description="Unique identifier for the flow")
    description: Optional[str] = Field(None, description="Optional description of the flow")

    flow_type: Literal["tree", "loop"] = Field(..., description="Type of flow - tree or loop")

    # Common fields for both types
    max_iterations: int = Field(100, description="Maximum number of iterations")
    max_cost: float = Field(4.0, description="Maximum cost allowed in USD")
    agent_id: Optional[str] = Field(None, description="ID of the agent to use")
    artifact_handlers: list[ArtifactHandler] = Field(
        default_factory=list, description="List of artifact handlers used by the flow"
    )

    # Tree-specific fields
    max_expansions: Optional[int] = Field(3, description="Maximum number of expansions per iteration")
    max_depth: Optional[int] = Field(20, description="Maximum depth of the tree")
    min_finished_nodes: Optional[int] = Field(None, description="Minimum number of finished nodes required")
    max_finished_nodes: Optional[int] = Field(None, description="Maximum number of finished nodes allowed")
    reward_threshold: Optional[float] = Field(None, description="Minimum reward threshold for accepting nodes")

    # Component references
    selector: Optional[BaseSelector] = None
    expander: Optional[Expander] = None
    value_function: Optional[BaseValueFunction] = None
    feedback_generator: Optional[BaseFeedbackGenerator] = None
    discriminator: Optional[BaseDiscriminator] = None

    def __str__(self) -> str:
        """Return a nice string representation of the flow config."""
        components = []
        if self.description:
            components.append(f"Description: {self.description}")

        components.extend(
            [f"Type: {self.flow_type}", f"Max iterations: {self.max_iterations}", f"Max cost: ${self.max_cost:.2f}"]
        )

        if self.flow_type == "tree":
            components.extend([f"Max expansions: {self.max_expansions}", f"Max depth: {self.max_depth}"])

            if self.min_finished_nodes:
                components.append(f"Min finished nodes: {self.min_finished_nodes}")
            if self.max_finished_nodes:
                components.append(f"Max finished nodes: {self.max_finished_nodes}")
            if self.reward_threshold:
                components.append(f"Reward threshold: {self.reward_threshold}")

        if self.agent_id:
            components.append(f"Agent: {self.agent_id}")

        if self.selector:
            components.append(f"Selector: {self.selector.__class__.__name__}")
        if self.expander:
            components.append(f"Expander: {self.expander.__class__.__name__}")
        if self.value_function:
            components.append(f"Value function: {self.value_function.__class__.__name__}")
        if self.feedback_generator:
            components.append(f"Feedback generator: {self.feedback_generator.__class__.__name__}")
        if self.discriminator:
            components.append(f"Discriminator: {self.discriminator.__class__.__name__}")

        for artifact_handler in self.artifact_handlers:
            components.append(f"Artifact handler: {artifact_handler.__class__.__name__}")

        return f"Flow Config '{self.id}':\n" + "\n".join(f"- {c}" for c in components)

    model_config = {
        "json_encoders": {
            BaseSelector: lambda v: v.model_dump(),
            BaseValueFunction: lambda v: v.model_dump(),
            BaseFeedbackGenerator: lambda v: v.model_dump(),
            BaseDiscriminator: lambda v: v.model_dump(),
            ArtifactHandler: lambda v: v.model_dump(),
        }
    }

    @model_validator(mode="before")
    @classmethod
    def validate_components(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict):
            data = data.copy()

            if "selector" in data and data["selector"]:
                data["selector"] = BaseSelector.model_validate(data["selector"])
            if "value_function" in data and data["value_function"]:
                data["value_function"] = BaseValueFunction.model_validate(data["value_function"])
            if "feedback_generator" in data and data["feedback_generator"]:
                data["feedback_generator"] = BaseFeedbackGenerator.model_validate(data["feedback_generator"])
            if "discriminator" in data and data["discriminator"]:
                data["discriminator"] = BaseDiscriminator.model_validate(data["discriminator"])
            if "artifact_handlers" in data and data["artifact_handlers"]:
                data["artifact_handlers"] = [
                    ArtifactHandler.model_validate(handler) for handler in data["artifact_handlers"]
                ]

        return data

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        data = super().model_dump(*args, **kwargs)

        if self.selector:
            selector_data = self.selector.model_dump()
            selector_data["selector_class"] = f"{self.selector.__class__.__module__}.{self.selector.__class__.__name__}"
            data["selector"] = selector_data

        if self.value_function:
            value_data = self.value_function.model_dump()
            value_data["value_function_class"] = (
                f"{self.value_function.__class__.__module__}.{self.value_function.__class__.__name__}"
            )
            data["value_function"] = value_data

        if self.feedback_generator:
            feedback_data = self.feedback_generator.model_dump()
            feedback_data["feedback_generator_class"] = (
                f"{self.feedback_generator.__class__.__module__}.{self.feedback_generator.__class__.__name__}"
            )
            data["feedback_generator"] = feedback_data

        if self.discriminator:
            discriminator_data = self.discriminator.model_dump()
            discriminator_data["discriminator_class"] = (
                f"{self.discriminator.__class__.__module__}.{self.discriminator.__class__.__name__}"
            )
            data["discriminator"] = discriminator_data

        if self.artifact_handlers:
            data["artifact_handlers"] = [handler.model_dump() for handler in self.artifact_handlers]

        return data


class FlowStatus(str, Enum):
    """Enum for system status values."""

    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    ERROR = "error"


class RunAttempt(BaseModel):
    """Information about a single run attempt"""

    attempt_id: int
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    status: str = "running"  # running, error, completed
    error: Optional[str] = None
    error_trace: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlowStatusInfo(BaseModel):
    """System status information"""

    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    status: FlowStatus = FlowStatus.CREATED
    error: Optional[str] = None
    error_trace: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    restart_count: int = Field(default=0)
    last_restart: Optional[datetime] = None
    run_history: list[RunAttempt] = Field(default_factory=list)
    current_attempt: Optional[int] = None

    def start_new_attempt(self) -> RunAttempt:
        """Start a new run attempt"""
        attempt = RunAttempt(attempt_id=len(self.run_history), metadata=self.metadata)
        self.run_history.append(attempt)
        self.current_attempt = attempt.attempt_id
        return attempt

    def get_current_attempt(self) -> Optional[RunAttempt]:
        """Get the current run attempt"""
        if self.current_attempt is not None:
            return self.run_history[self.current_attempt]
        return None

    def complete_current_attempt(
        self, status: str = "completed", error: Optional[str] = None, error_trace: Optional[str] = None
    ):
        """Complete the current attempt"""
        if attempt := self.get_current_attempt():
            attempt.finished_at = datetime.now(timezone.utc)
            attempt.status = status
            attempt.error = error
            attempt.error_trace = error_trace


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
    system_status: FlowStatusInfo
    job_status: JobStatus
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
