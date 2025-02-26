import json
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any, List, Dict

from pydantic import BaseModel, Field, ConfigDict

from moatless.benchmark.report import BenchmarkResult
from moatless.completion.model import Usage
from moatless.discriminator.base import BaseDiscriminator
from moatless.events import BaseEvent
from moatless.feedback import BaseFeedbackGenerator
from moatless.schema import MessageHistoryType
from moatless.selector import BaseSelector
from moatless.value_function.base import BaseValueFunction

from moatless.runner.runner import JobStatus

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

    model_id: str = Field(..., description="The ID of the model to use for the evaluation.")
    agent_id: str = Field(..., description="The ID of the agent to use for the evaluation.")

    selector: Optional[BaseSelector] = Field(default=None, description="Custom selector for tree search")

    value_function: Optional[BaseValueFunction] = Field(
        None,
        description="The value function to use for the tree search.",
    )

    discriminator: Optional[BaseDiscriminator] = Field(
        None,
        description="The discriminator to use for the tree search.",
    )

    feedback_generator: Optional[BaseFeedbackGenerator] = Field(
        None,
        description="The feedback generator to use for the tree search.",
    )


class InstanceStatus(str, Enum):
    CREATED = "created"
    PENDING = "pending"
    SETTING_UP = "setting_up"
    RUNNING = "running"
    COMPLETED = "completed"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    ERROR = "error"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class EvaluationEvent(BaseEvent):
    """Event emitted by the evaluation process"""
    scope: str = "evaluation"
    data: Any


class EvaluationInstance(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value,
        }
    )

    instance_id: str = Field(description="Unique identifier for the instance")
    status: InstanceStatus = Field(default=InstanceStatus.CREATED, description="Current status of the instance")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the instance was created",
    )
    started_at: Optional[datetime] = Field(default=None, description="When the instance started")
    completed_at: Optional[datetime] = Field(default=None, description="When the instance completed")
    start_evaluating_at: Optional[datetime] = Field(default=None, description="When the instance started evaluating")
    evaluated_at: Optional[datetime] = Field(default=None, description="When instance was evaluated")
    job_status: Optional[JobStatus] = Field(default=None, description="Status of the instance's job", exclude=True)
    error_at: Optional[datetime] = Field(default=None, description="When instance encountered an error")
    submission: Optional[str] = Field(default=None, description="The submitted patch")
    error: Optional[str] = Field(default=None, description="Error message if instance failed")
    resolved: Optional[bool] = Field(default=None, description="Whether the instance was resolved")
    iterations: Optional[int] = Field(default=None, description="Number of iterations")
    reward: Optional[int] = Field(default=None, description="Reward of the instance")
    usage: Optional[Usage] = Field(default=None, description="Total cost of the instance")
    benchmark_result: Optional[BenchmarkResult] = Field(default=None, description="Benchmark result")

    duration: Optional[float] = Field(default=None, description="Time taken to evaluate in seconds")

    def start(self):
        self.status = InstanceStatus.RUNNING
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
            self.benchmark_result = benchmark_result.model_dump()

    def fail(self, error: str):
        self.status = InstanceStatus.ERROR
        self.completed_at = datetime.now(timezone.utc)
        self.error = error
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()

    def model_dump(self, *args, **kwargs) -> dict:
        data = super().model_dump(*args, **kwargs)
        if self.benchmark_result and isinstance(self.benchmark_result, BenchmarkResult):
            data["benchmark_result"] = self.benchmark_result.model_dump(**kwargs)
        return data

class Evaluation(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value,
        }
    )

    evaluation_name: str = Field(..., description="Name of the evaluation")
    dataset_name: str = Field(..., description="Name of the dataset")
        
    flow_id: str = Field(..., description="ID of the flow configuration to use for the evaluation")
    model_id: str = Field(..., description="ID of the model to use for the evaluation")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the evaluation was created",
    )
    
    started_at: Optional[datetime] = Field(default=None, description="When the evaluation started")
    completed_at: Optional[datetime] = Field(default=None, description="When the evaluation finished")
    status: EvaluationStatus = Field(default=EvaluationStatus.PENDING, description="Current status of the evaluation")
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")
    instances: List[EvaluationInstance] = Field(default_factory=list, description="Instances of the evaluation")

    def get_instance(self, instance_id: str) -> EvaluationInstance | None:
        for instance in self.instances:
            if instance.instance_id == instance_id:
                return instance
        return None

    def get_summary(self) -> dict:
        """Get a summary of evaluation status and instance counts."""
        completed = sum(1 for i in self.instances if i.status == InstanceStatus.COMPLETED)
        running = sum(1 for i in self.instances if i.status == InstanceStatus.RUNNING)
        evaluating = sum(1 for i in self.instances if i.status == InstanceStatus.EVALUATING)
        evaluated = sum(1 for i in self.instances if i.status == InstanceStatus.EVALUATED)
        pending = sum(1 for i in self.instances if i.status == InstanceStatus.PENDING)
        errors = sum(1 for i in self.instances if i.status == InstanceStatus.ERROR)

        return {
            "status": self.status,
            "error": self.error,
            "counts": {
                "completed": completed,
                "running": running,
                "evaluating": evaluating,
                "evaluated": evaluated,
                "pending": pending,
                "errors": errors,
                "total": len(self.instances)
            }
        }

    def model_dump(self, *args, **kwargs) -> dict:
        data = super().model_dump(*args, **kwargs)
        data["instances"] = [i.model_dump(**kwargs) for i in self.instances]
        return data

class EvaluationDatasetSplit(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value,
        }
    )

    name: str = Field(description="Name of the evaluation split (e.g., 'train', 'test', 'validation')")
    description: str = Field(description="Description of what this split represents")
    instance_ids: list[str] = Field(description="List of instance IDs that belong to this split")

