"""Schema for trajectory data."""

from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from moatless.artifacts.artifact import ArtifactChange
from moatless.completion.stats import Usage, CompletionInvocation
from moatless.flow.schema import NodeDTO


class UsageDTO(BaseModel):
    """Usage information for a completion."""

    completionCost: Optional[float] = None
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    cachedTokens: Optional[int] = None


class CompletionDTO(BaseModel):
    """Completion information."""

    type: str
    usage: Optional[UsageDTO] = None
    tokens: str
    retries: Optional[int] = None
    input: Optional[str] = None
    response: Optional[str] = None


class ObservationDTO(BaseModel):
    """Observation information."""

    message: Optional[str] = None
    summary: Optional[str] = None
    properties: dict[str, Any] = {}


class ActionDTO(BaseModel):
    """Action information."""

    name: str
    shortSummary: str
    thoughts: Optional[str] = None
    properties: dict[str, Any] = {}


class ActionStepDTO(BaseModel):
    """Represents a single action step."""

    thoughts: Optional[str] = None
    action: ActionDTO
    artifacts: list[ArtifactChange] = []

    observation: Optional[ObservationDTO] = None
    completion: Optional[CompletionDTO] = None
    warnings: list[str] = []
    errors: list[str] = []


class FileContextSpanDTO(BaseModel):
    """Represents a span in a file context."""

    span_id: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    tokens: Optional[int] = None
    pinned: bool = False


class FileContextFileDTO(BaseModel):
    """Represents a file in the file context."""

    file_path: str
    content: Optional[str] = None
    patch: Optional[str] = None
    spans: list[FileContextSpanDTO] = []
    show_all_spans: bool = False
    tokens: Optional[int] = None
    is_new: bool = False
    was_edited: bool = False


class UpdatedFileDTO(BaseModel):
    """Represents an updated file with its changes."""

    file_path: str
    status: str
    tokens: Optional[int] = None
    patch: Optional[str] = None


class FileContextDTO(BaseModel):
    """File context information."""

    testResults: Optional[list[dict[str, Any]]] = None
    patch: Optional[str] = None
    files: list[FileContextFileDTO] = []
    warnings: list[str] = []
    errors: list[str] = []
    updatedFiles: list[UpdatedFileDTO] = Field(
        default_factory=list,
        description="List of files that have been updated since the last context",
    )


class TestResultsSummaryDTO(BaseModel):
    """Summary of test results."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0


class TimelineItemType(str, Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    ARTIFACT = "artifact"
    COMPLETION = "completion"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ERROR = "error"
    WORKSPACE = "workspace"
    WORKSPACE_FILES = "workspace_files"
    WORKSPACE_CONTEXT = "workspace_context"
    WORKSPACE_TESTS = "workspace_tests"
    REWARD = "reward"


class TimelineItemDTO(BaseModel):
    """Represents a timeline item in the node."""

    label: str
    type: TimelineItemType
    content: dict[str, Any]


class RewardDTO(BaseModel):
    """Reward information."""

    value: int
    explanation: Optional[str] = None


# Tree View Schema Classes
class ItemType(str, Enum):
    """Type of tree item."""

    COMPLETION = "completion"
    THOUGHT = "thought"
    ACTION = "action"
    NODE = "node"


class BaseTreeItemDTO(BaseModel):
    """Base class for tree items."""

    id: str
    type: str
    label: str
    detail: Optional[str] = None
    time: Optional[str] = None


class CompletionTreeItemDTO(BaseTreeItemDTO):
    """Represents a completion item in the tree."""

    type: str = ItemType.COMPLETION
    tokens: Optional[int] = None
    nodeId: str
    parentId: Optional[str] = None


class ThoughtTreeItemDTO(BaseTreeItemDTO):
    """Represents a thought item in the tree."""

    type: str = ItemType.THOUGHT
    nodeId: str


class ActionTreeItemDTO(BaseTreeItemDTO):
    """Represents an action item in the tree."""

    type: str = ItemType.ACTION
    actionType: str
    actionIndex: int
    nodeId: str
    children: Optional[list[Union["CompletionTreeItemDTO", "ThoughtTreeItemDTO", "ActionTreeItemDTO"]]] = Field(
        default_factory=list
    )


class NodeTreeItemDTO(BaseTreeItemDTO):
    """Represents a node item in the tree."""

    type: str = ItemType.NODE
    timestamp: str
    parentNodeId: Optional[str] = None
    children: Optional[
        list[Union["NodeTreeItemDTO", "CompletionTreeItemDTO", "ThoughtTreeItemDTO", "ActionTreeItemDTO"]]
    ] = Field(default_factory=list)


class TreeItemDTO(BaseModel):
    """Tree representation of trajectory data."""

    items: list[NodeTreeItemDTO] = Field(default_factory=list)


class TrajectoryDTO(BaseModel):
    """Represents trajectory data from a file, focusing on metrics and performance data."""

    duration: Optional[float] = None
    error: Optional[str] = None
    iterations: Optional[int] = None
    completionCost: Optional[float] = None
    totalTokens: Optional[int] = None
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    cachedTokens: Optional[int] = None
    flags: list[str] = []
    failedActions: int = 0
    duplicatedActions: int = 0
    nodes: list[NodeDTO] = []
