from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime


class InstanceItemDTO(BaseModel):
    instanceId: str
    status: Literal["pending", "running", "no_patch", "completed", "failed", "resolved", "error", "eval_error"]
    duration: Optional[float] = None
    resolved: Optional[bool] = None
    error: Optional[str] = None
    iterations: Optional[int] = None
    completionCost: Optional[float] = None
    totalTokens: Optional[int] = None
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    cachedTokens: Optional[int] = None
    resolutionRate: Optional[float] = None
    flags: List[str] = []
    failedActions: int = 0
    duplicatedActions: int = 0
    splits: List[str] = []

class EvaluationSettingsDTO(BaseModel):
    model: str
    temperature: float
    maxIterations: int
    responseFormat: str
    messageHistoryFormat: str
    maxCost: float

class EvaluationResponseDTO(BaseModel):
    name: str | None = None
    status: Literal["pending", "running", "completed", "error"]
    isActive: bool
    settings: EvaluationSettingsDTO
    startedAt: Optional[datetime]
    totalCost: float
    promptTokens: int
    completionTokens: int
    cachedTokens: int
    totalTokens: int
    totalInstances: int
    completedInstances: int
    errorInstances: int
    resolvedInstances: int
    failedInstances: int
    instances: List[InstanceItemDTO]
    evalToolsId: Optional[str] = None


class EvaluationListItemDTO(BaseModel):
    """Represents an evaluation item in the list view."""
    name: str
    status: Literal["pending", "running", "completed", "error"]
    model: str
    maxExpansions: int
    startedAt: Optional[datetime]
    totalInstances: int
    completedInstances: int
    errorInstances: int
    resolvedInstances: int
    isActive: bool
    date: Optional[datetime] = None  # For easier date-based sorting/filtering
    resolutionRate: float = 0.0  # resolved/total instances
    totalCost: float = 0.0
    promptTokens: int = 0
    completionTokens: int = 0
    cachedTokens: int = 0
    totalTokens: int = 0
    resolvedByDollar: float = 0.0  # resolved instances per dollar spent

class EvaluationListResponseDTO(BaseModel):
    """Response model for list evaluations endpoint."""
    evaluations: List[EvaluationListItemDTO]

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
    status: Literal["added_to_context", "updated_context", "modified"]
    tokens: Optional[int] = None
    patch: Optional[str] = None

class FileContextDTO(BaseModel):
    """File context information."""
    summary: str
    testResults: Optional[List[Dict[str, Any]]] = None
    patch: Optional[str] = None
    files: List[FileContextFileDTO] = []
    warnings: List[str] = []
    errors: List[str] = []
    updatedFiles: List[UpdatedFileDTO] = Field(
        default_factory=list,
        description="List of files that have been updated since the last context"
    )

class NodeDTO(BaseModel):
    """Node information in the tree."""
    nodeId: int
    userMessage: Optional[str] = None
    assistantMessage: Optional[str] = None
    actionCompletion: Optional[CompletionDTO] = None
    actionSteps: List[ActionStepDTO] = []
    fileContext: Optional[FileContextDTO] = None
    error: Optional[str] = None
    warnings: List[str] = []
    errors: List[str] = []
    terminal: bool = Field(
        default=False,
        description="Whether this node is in a terminal state (determined by last action step's observation)"
    )

class InstanceResponseDTO(BaseModel):
    """Response model for tree visualization endpoint."""
    nodes: List[NodeDTO]
    totalNodes: int
    instance: Optional[Dict[str, Any]] = None
    evalResult: Optional[Dict[str, Any]] = None
    status: str
    duration: Optional[float] = None
    resolved: Optional[bool] = None
    error: Optional[str] = None
    iterations: Optional[int] = None
    completionCost: Optional[float] = None
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    cachedTokens: Optional[int] = None
    totalTokens: Optional[int] = None
    resolutionRate: Optional[float] = None
    splits: List[str] = []
    flags: List[str] = []
