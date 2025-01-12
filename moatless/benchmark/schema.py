import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any, List

from pydantic import BaseModel, Field, ConfigDict

from moatless.agent.settings import AgentSettings
from moatless.benchmark.report import BenchmarkResult
from moatless.completion.completion import CompletionModel
from moatless.completion.model import Usage
from moatless.discriminator import Discriminator
from moatless.feedback.feedback import FeedbackGenerator
from moatless.schema import MessageHistoryType
from moatless.selector.selector import Selector
from moatless.value_function.base import ValueFunction


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, MessageHistoryType):
            return obj.value
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class TreeSearchSettings(BaseModel):
    max_expansions: int = Field(
        3,
        description="The maximum number of expansions of one state.",
    )

    max_iterations: int = Field(
        100,
        description="The maximum number of iterations to run the tree search.",
    )

    max_cost: float = Field(
        4,
        description="The maximum cost spent on tokens before finishing.",
    )

    min_finished_nodes: Optional[int] = Field(
        2,
        description="The minimum number of finished nodes to consider before finishing",
    )

    max_finished_nodes: Optional[int] = Field(
        3,
        description="The maximum number of finished nodes to consider before finishing",
    )

    reward_threshold: Optional[int] = Field(
        None,
        description="The min reward threshold to consider before finishing.",
    )

    max_depth: int = Field(
        20,
        description="The maximum depth for one trajectory in simulations.",
    )

    model: Optional[CompletionModel] = Field(
        default=None,
        description="The default model.",
    )

    agent_settings: AgentSettings = Field(
        ..., description="Settings for creating the agent"
    )

    selector: Optional[Selector] = Field(
        default=None, description="Custom selector for tree search"
    )

    value_function: Optional[ValueFunction] = Field(
        None,
        description="The value function to use for the tree search.",
    )

    discriminator: Optional[Discriminator] = Field(
        None,
        description="The discriminator to use for the tree search.",
    )

    feedback_generator: Optional[FeedbackGenerator] = Field(
        None,
        description="The feedback generator to use for the tree search.",
    )


class InstanceStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    COMPLETED = "completed"
    ERROR = "error"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class EvaluationEvent:
    """Event emitted by the evaluation process"""

    evaluation_name: str
    event_type: str
    data: Any


class EvaluationInstance(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value,
        }
    )

    instance_id: str = Field(description="Unique identifier for the instance")
    status: InstanceStatus = Field(
        default=InstanceStatus.PENDING, description="Current status of the instance"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the instance was created",
    )
    started_at: Optional[datetime] = Field(
        default=None, description="When evaluation started"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="When evaluation completed"
    )
    submission: Optional[str] = Field(default=None, description="The submitted patch")
    error: Optional[str] = Field(
        default=None, description="Error message if evaluation failed"
    )
    resolved: Optional[bool] = Field(
        default=None, description="Whether the instance was resolved"
    )
    iterations: Optional[int] = Field(default=None, description="Number of iterations")
    usage: Optional[Usage] = Field(
        default=None, description="Total cost of the instance"
    )

    duration: Optional[float] = Field(
        default=None, description="Time taken to evaluate in seconds"
    )

    def start(self):
        self.status = InstanceStatus.STARTED
        self.started_at = datetime.now(timezone.utc)

    def complete(
        self,
        submission: Optional[str] = None,
        resolved: Optional[bool] = None,
        benchmark_result: Optional[BenchmarkResult] = None,
    ):
        self.status = InstanceStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.submission = submission
        self.resolved = resolved
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()
        if benchmark_result:
            self.benchmark_result = benchmark_result

    def fail(self, error: str):
        self.status = InstanceStatus.ERROR
        self.completed_at = datetime.now(timezone.utc)
        self.error = error
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()


class Evaluation(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value,
        }
    )

    evaluations_dir: str = Field(description="Directory where evaluations are stored")
    evaluation_name: str = Field(description="Name of the evaluation")
    settings: TreeSearchSettings = Field(description="Tree search settings")
    start_time: Optional[datetime] = Field(
        default=None, description="When the evaluation started"
    )
    finish_time: Optional[datetime] = Field(
        default=None, description="When the evaluation finished"
    )
    status: EvaluationStatus = Field(
        default=EvaluationStatus.PENDING, description="Current status of the evaluation"
    )
    instances: List[EvaluationInstance] = Field(default_factory=list)

    def get_instance(self, instance_id: str) -> EvaluationInstance | None:
        return next(
            instance
            for instance in self.instances
            if instance.instance_id == instance_id
        )


class EvaluationDatasetSplit(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value,
        }
    )

    name: str = Field(
        description="Name of the evaluation split (e.g., 'train', 'test', 'validation')"
    )
    description: str = Field(description="Description of what this split represents")
    instance_ids: list[str] = Field(
        description="List of instance IDs that belong to this split"
    )
