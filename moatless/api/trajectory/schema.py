"""Schema for trajectory data."""

from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


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
    properties: Dict[str, Any] = {}
    expectCorrection: bool = False


class ActionDTO(BaseModel):
    """Action information."""

    name: str
    shortSummary: str
    thoughts: Optional[str] = None
    properties: Dict[str, Any] = {}


class ActionStepDTO(BaseModel):
    """Represents a single action step."""

    thoughts: Optional[str] = None
    action: ActionDTO
    observation: Optional[ObservationDTO] = None
    completion: Optional[CompletionDTO] = None
    warnings: List[str] = []
    errors: List[str] = []


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
    spans: List[FileContextSpanDTO] = []
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

    testResults: Optional[List[Dict[str, Any]]] = None
    patch: Optional[str] = None
    files: List[FileContextFileDTO] = []
    warnings: List[str] = []
    errors: List[str] = []
    updatedFiles: List[UpdatedFileDTO] = Field(
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


class TimelineItemDTO(BaseModel):
    """Represents a timeline item in the node."""
    label: str
    type: TimelineItemType
    content: Dict[str, Any]


class NodeDTO(BaseModel):
    """Node information in the tree."""

    nodeId: int
    executed: bool = Field(default=False, description="Whether this node has been executed")
    userMessage: Optional[str] = None
    assistantMessage: Optional[str] = None
    actionCompletion: Optional[CompletionDTO] = None
    actionSteps: List[ActionStepDTO] = []
    fileContext: Optional[FileContextDTO] = None
    error: Optional[str] = None
    warnings: List[str] = []
    errors: List[str] = []
    terminal: bool = Field(default=False, description="Whether this node is in a terminal state")
    allNodeErrors: List[str] = Field(
        default_factory=list,
        description="All errors from this node, including action steps and file context",
    )
    allNodeWarnings: List[str] = Field(
        default_factory=list,
        description="All warnings from this node, including action steps and file context",
    )
    testResultsSummary: Optional[TestResultsSummaryDTO] = None
    items: List[TimelineItemDTO] = Field(
        default_factory=list,
        description="Timeline items for this node"
    )


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
    flags: List[str] = []
    failedActions: int = 0
    duplicatedActions: int = 0
    nodes: List[NodeDTO] = []
