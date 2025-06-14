import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from moatless.completion.stats import Usage
from moatless.events import BaseEvent
from moatless.runner.runner import JobStatus
from moatless.schema import MessageHistoryType
from moatless.flow.flow import AgenticFlow
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, MessageHistoryType):
            return obj.value
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class ExecutionStatus(str, Enum):
    """Tracks the execution lifecycle of an instance"""

    CREATED = "created"  # Instance created but not started
    QUEUED = "queued"  # Job is queued in the runner
    RUNNING = "running"  # Job is actively running
    EVALUATING = "evaluating"  # Tests are being run
    COMPLETED = "completed"  # Execution finished (may have succeeded or failed)
    ERROR = "error"  # Execution encountered an error


class ResolutionStatus(str, Enum):
    """Tracks the resolution outcome of an instance"""

    PENDING = "pending"  # Not yet evaluated
    RESOLVED = "resolved"  # Problem was fully solved
    FAILED = "failed"  # Problem was not solved
    PARTIALLY_RESOLVED = "partially_resolved"  # Some test cases passed


class InstanceStatus(str, Enum):
    """Legacy status enum - kept for backward compatibility"""

    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    FAILED = "failed"
    ERROR = "error"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"  # TODO: Remove this
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


class EvaluationEvent(BaseEvent):
    """Event emitted by the evaluation process"""

    scope: str = "evaluation"
    data: Any = Field(default_factory=dict)


class EvaluationInstance(BaseModel):
    model_config = ConfigDict(ser_json_timedelta="iso8601")

    instance_id: str = Field(description="Unique identifier for the instance")

    # Execution tracking
    execution_status: ExecutionStatus = Field(
        default=ExecutionStatus.CREATED, description="Current execution state of the instance"
    )
    job_status: Optional[JobStatus] = Field(default=None, description="Status from the job runner")

    # Resolution tracking
    resolution_status: ResolutionStatus = Field(
        default=ResolutionStatus.PENDING, description="Resolution outcome of the instance"
    )

    # Legacy fields for backward compatibility
    status: InstanceStatus = Field(
        default=InstanceStatus.CREATED, description="[DEPRECATED] Use execution_status and resolution_status instead"
    )
    completed: bool = Field(default=False, description="[DEPRECATED] Use execution_status == COMPLETED instead")
    resolved: Optional[bool] = Field(default=None, description="[DEPRECATED] Use resolution_status instead")

    # Timestamps
    created_at: Optional[datetime] = Field(
        default=None,
        description="When the instance was created",
    )
    queued_at: Optional[datetime] = Field(default=None, description="When the instance was queued")
    started_at: Optional[datetime] = Field(default=None, description="When the instance started")
    completed_at: Optional[datetime] = Field(default=None, description="When the instance completed")
    start_evaluating_at: Optional[datetime] = Field(default=None, description="When the instance started evaluating")
    evaluated_at: Optional[datetime] = Field(default=None, description="When instance was evaluated")
    error_at: Optional[datetime] = Field(default=None, description="When instance encountered an error")

    # Results
    submission: Optional[str] = Field(default=None, description="The submitted patch")
    error: Optional[str] = Field(default=None, description="Error message if instance failed")
    iterations: Optional[int] = Field(default=None, description="Number of iterations")
    reward: Optional[int] = Field(default=None, description="Reward of the instance")
    usage: Optional[Usage] = Field(default=None, description="Total cost of the instance")
    benchmark_result: Optional[dict[str, Any]] = Field(default=None, description="Benchmark result")
    duration: Optional[float] = Field(default=None, description="Time taken to evaluate in seconds")
    last_event_timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the last event")
    resolved_by: Optional[int] = Field(default=None, description="Number of agents that have resolved the evaluation")

    issues: List[str] = Field(default_factory=list, description="Issues in the instance")

    @field_validator('created_at', 'queued_at', 'started_at', 'completed_at', 'start_evaluating_at', 'evaluated_at', 'error_at', 'last_event_timestamp', mode='before')
    @classmethod
    def validate_timestamps(cls, v):
        """Handle invalid timestamp values by keeping them as None to avoid parsing errors"""
        if v is None or v == "None" or (isinstance(v, str) and v.strip() == ""):
            return None
        return v

    def _calculate_duration(self, start_time: Optional[datetime], end_time: Optional[datetime]) -> Optional[float]:
        """Safely calculate duration between two datetime objects, handling timezone differences"""
        if not start_time or not end_time:
            return None
            
        # Ensure both datetimes have the same timezone awareness
        if start_time.tzinfo is None and end_time.tzinfo is not None:
            # start_time is naive, end_time is aware - assume start_time is UTC
            start_time = start_time.replace(tzinfo=timezone.utc)
        elif start_time.tzinfo is not None and end_time.tzinfo is None:
            # start_time is aware, end_time is naive - assume end_time is UTC
            end_time = end_time.replace(tzinfo=timezone.utc)
        elif start_time.tzinfo is None and end_time.tzinfo is None:
            # Both are naive - leave as is
            pass
        # If both are aware, they can be subtracted directly
        
        return (end_time - start_time).total_seconds()

    def start(self):
        """Mark instance as queued for execution"""
        self.execution_status = ExecutionStatus.QUEUED
        self.queued_at = datetime.now(timezone.utc)
        self._sync_legacy_status()

    def mark_running(self):
        """Mark instance as running"""
        self.execution_status = ExecutionStatus.RUNNING
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc)
        self._sync_legacy_status()

    def mark_evaluating(self):
        """Mark instance as being evaluated"""
        self.execution_status = ExecutionStatus.EVALUATING
        if not self.start_evaluating_at:
            self.start_evaluating_at = datetime.now(timezone.utc)
        self._sync_legacy_status()

    def complete(
        self,
        submission: Optional[str] = None,
        resolved: Optional[bool] = None,
        benchmark_result: Optional[dict[str, Any]] = None,
    ):
        """Mark the instance as completed"""
        self.execution_status = ExecutionStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)

        if submission:
            self.submission = submission

        if resolved is not None:
            self.resolved = resolved
            if resolved:
                self.resolution_status = ResolutionStatus.RESOLVED
            else:
                self.resolution_status = ResolutionStatus.FAILED

        # Use the safe duration calculation method
        if self.started_at and self.completed_at:
            self.duration = self._calculate_duration(self.started_at, self.completed_at)

        if benchmark_result:
            # Store as dict to avoid type issues
            self.benchmark_result = benchmark_result.model_dump()

        self._sync_legacy_status()

    def fail(self, error: str):
        """Mark instance as failed with error"""
        self.execution_status = ExecutionStatus.ERROR
        self.completed_at = datetime.now(timezone.utc)
        self.error = error
        self.error_at = datetime.now(timezone.utc)
        # Use the safe duration calculation method
        if self.started_at and self.completed_at:
            self.duration = self._calculate_duration(self.started_at, self.completed_at)
        self._sync_legacy_status()

    def set_resolution(self, resolved: bool, partially_resolved: bool = False):
        """Set the resolution status based on evaluation results"""
        self.resolved = resolved
        if resolved:
            self.resolution_status = ResolutionStatus.RESOLVED
        elif partially_resolved:
            self.resolution_status = ResolutionStatus.PARTIALLY_RESOLVED
        else:
            self.resolution_status = ResolutionStatus.FAILED
        self._sync_legacy_status()

    def _sync_legacy_status(self):
        """Sync the legacy status field based on new status fields"""
        # Map execution status to legacy status
        if self.execution_status == ExecutionStatus.CREATED:
            self.status = InstanceStatus.CREATED
        elif self.execution_status == ExecutionStatus.QUEUED:
            self.status = InstanceStatus.PENDING
        elif self.execution_status == ExecutionStatus.RUNNING:
            self.status = InstanceStatus.RUNNING
        elif self.execution_status == ExecutionStatus.EVALUATING:
            self.status = InstanceStatus.EVALUATING
        elif self.execution_status == ExecutionStatus.ERROR:
            self.status = InstanceStatus.ERROR
        elif self.execution_status == ExecutionStatus.COMPLETED:
            # If completed, use resolution status
            if self.resolution_status == ResolutionStatus.RESOLVED:
                self.status = InstanceStatus.RESOLVED
            elif self.resolution_status == ResolutionStatus.PARTIALLY_RESOLVED:
                self.status = InstanceStatus.PARTIALLY_RESOLVED
            elif self.resolution_status == ResolutionStatus.FAILED:
                self.status = InstanceStatus.FAILED
            else:
                self.status = InstanceStatus.COMPLETED

        # Sync completed flag
        self.completed = self.execution_status == ExecutionStatus.COMPLETED

    def is_running(self) -> bool:
        """Check if instance is currently running"""
        return self.execution_status in [ExecutionStatus.RUNNING, ExecutionStatus.EVALUATING]

    def is_finished(self) -> bool:
        """Check if instance execution is finished (completed or errored)"""
        return self.execution_status in [ExecutionStatus.COMPLETED, ExecutionStatus.ERROR]

    def is_evaluated(self) -> bool:
        """Check if instance has been evaluated (resolution status determined)"""
        return self.resolution_status != ResolutionStatus.PENDING

    def model_dump(self, *args, **kwargs) -> dict:
        data = super().model_dump(*args, **kwargs)
        if self.benchmark_result:
            data["benchmark_result"] = self.benchmark_result
        return data


class Evaluation(BaseModel):
    model_config = ConfigDict(ser_json_timedelta="iso8601")

    evaluation_name: str = Field(..., description="Name of the evaluation")
    dataset_name: str = Field(..., description="Name of the dataset")

    flow: Optional[AgenticFlow] = Field(default=None, description="Flow configuration to use for the evaluation")

    flow_id: Optional[str] = Field(default=None, description="ID of the flow configuration to use for the evaluation")
    model_id: Optional[str] = Field(default=None, description="[DEPRECATED] ID of the model to use for the evaluation")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the evaluation was created",
    )

    started_at: Optional[datetime] = Field(default=None, description="When the evaluation started")
    completed_at: Optional[datetime] = Field(default=None, description="When the evaluation finished")
    status: EvaluationStatus = Field(default=EvaluationStatus.PENDING, description="Current status of the evaluation")
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")
    instances: list[EvaluationInstance] = Field(default_factory=list, description="Instances of the evaluation")

    def get_instance(self, instance_id: str) -> EvaluationInstance | None:
        for instance in self.instances:
            if instance.instance_id == instance_id:
                return instance
        return None

    def get_summary(self) -> dict:
        """Get a summary of evaluation status and instance counts."""
        # Execution status counts
        created = sum(1 for i in self.instances if i.execution_status == ExecutionStatus.CREATED)
        queued = sum(1 for i in self.instances if i.execution_status == ExecutionStatus.QUEUED)
        running = sum(1 for i in self.instances if i.execution_status == ExecutionStatus.RUNNING)
        evaluating = sum(1 for i in self.instances if i.execution_status == ExecutionStatus.EVALUATING)
        completed = sum(1 for i in self.instances if i.execution_status == ExecutionStatus.COMPLETED)
        errors = sum(1 for i in self.instances if i.execution_status == ExecutionStatus.ERROR)

        # Resolution status counts
        resolved = sum(1 for i in self.instances if i.resolution_status == ResolutionStatus.RESOLVED)
        failed = sum(1 for i in self.instances if i.resolution_status == ResolutionStatus.FAILED)
        partially_resolved = sum(
            1 for i in self.instances if i.resolution_status == ResolutionStatus.PARTIALLY_RESOLVED
        )
        pending_resolution = sum(1 for i in self.instances if i.resolution_status == ResolutionStatus.PENDING)

        return {
            "status": self.status,
            "error": self.error,
            "execution_counts": {
                "created": created,
                "queued": queued,
                "running": running,
                "evaluating": evaluating,
                "completed": completed,
                "errors": errors,
                "total": len(self.instances),
            },
            "resolution_counts": {
                "pending": pending_resolution,
                "resolved": resolved,
                "failed": failed,
                "partially_resolved": partially_resolved,
            },
            # Legacy counts for backward compatibility
            "counts": {
                "completed": completed,
                "running": running,
                "evaluating": evaluating,
                "evaluated": resolved + failed + partially_resolved,
                "pending": queued,
                "errors": errors,
                "total": len(self.instances),
            },
        }

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> "Evaluation":
        if isinstance(obj, dict):
            obj = obj.copy()
            obj["instances"] = [EvaluationInstance.model_validate(i) for i in obj["instances"]]
            if "flow" in obj:
                obj["flow"] = AgenticFlow.from_dict(obj["flow"])
        return super().model_validate(obj, **kwargs)

    def model_dump(self, *args, **kwargs) -> dict:
        data = super().model_dump(*args, **kwargs)
        data["instances"] = [i.model_dump(**kwargs) for i in self.instances]
        if self.flow:
            data["flow"] = self.flow.model_dump()
        return data


class EvaluationDatasetSplit(BaseModel):
    model_config = ConfigDict(ser_json_timedelta="iso8601")

    name: str = Field(description="Name of the evaluation split (e.g., 'train', 'test', 'validation')")
    description: str = Field(description="Description of what this split represents")
    instance_ids: list[str] = Field(description="List of instance IDs that belong to this split")


class EvaluationStatusSummary(BaseModel):
    """Summary of instance statuses in an evaluation"""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    # Execution status counts
    created: int = Field(default=0, description="Number of created instances")
    queued: int = Field(default=0, description="Number of queued instances")
    running: int = Field(default=0, description="Number of running instances")
    evaluating: int = Field(default=0, description="Number of instances being evaluated")
    completed: int = Field(default=0, description="Number of completed instances")
    error: int = Field(default=0, description="Number of instances with errors")

    # Resolution status counts
    pending_resolution: int = Field(default=0, description="Number of instances pending resolution")
    resolved: int = Field(default=0, description="Number of resolved instances")
    failed: int = Field(default=0, description="Number of failed instances")
    partially_resolved: int = Field(default=0, description="Number of partially resolved instances")

    # Legacy counts for backward compatibility
    pending: int = Field(default=0, description="[DEPRECATED] Use queued instead")


class RepoStats(BaseModel):
    """Statistics for a specific repository"""

    repo: str = Field(description="Repository name")
    total_instances: int = Field(description="Total instances for this repo")
    resolved_instances: int = Field(description="Number of resolved instances")
    failed_instances: int = Field(description="Number of failed instances")
    solve_rate: float = Field(description="Solve rate as percentage")


class EvaluationStats(BaseModel):
    """Comprehensive statistics for an evaluation"""

    # Overall metrics
    success_rate: float = Field(description="Success rate as percentage")
    avg_iterations: float = Field(description="Average number of iterations")
    avg_cost: float = Field(description="Average cost per instance")
    avg_tokens: int = Field(description="Average total tokens per instance")
    solved_instances_per_dollar: float = Field(description="Number of solved instances per dollar")
    solved_percentage_per_dollar: float = Field(description="Solved percentage per dollar")

    # Counts and totals
    total_instances: int = Field(description="Total number of finished instances")
    resolved_instances: int = Field(description="Number of resolved instances")
    failed_instances: int = Field(description="Number of failed instances")

    # Token breakdown
    total_cost: float = Field(description="Total cost for all instances")
    total_prompt_tokens: int = Field(description="Total prompt tokens")
    total_completion_tokens: int = Field(description="Total completion tokens")
    total_cache_read_tokens: int = Field(description="Total cache read tokens")

    # Iteration ranges
    iteration_range_min: int = Field(description="Minimum iterations")
    iteration_range_max: int = Field(description="Maximum iterations")

    # Distribution data
    success_distribution: dict[str, int] = Field(description="Distribution of success vs failure")
    iterations_distribution: dict[str, int] = Field(description="Distribution of iterations by ranges")
    cost_distribution: dict[str, int] = Field(description="Distribution of costs by ranges")

    # Per-repo statistics
    repo_stats: list[RepoStats] = Field(description="Statistics per repository")


class EvaluationSummary(BaseModel):
    """Lightweight summary of an evaluation for listing purposes"""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    evaluation_name: str = Field(..., description="Name of the evaluation")
    dataset_name: str = Field(..., description="Name of the dataset")
    flow_id: Optional[str] = Field(default=None, description="ID of the flow configuration used")
    model_id: Optional[str] = Field(default=None, description="[DEPRECATED] ID of the model used")
    status: str = Field(..., description="Status of the evaluation")
    created_at: datetime = Field(..., description="When the evaluation was created")
    started_at: Optional[datetime] = Field(default=None, description="When the evaluation started")
    completed_at: Optional[datetime] = Field(default=None, description="When the evaluation finished")
    instance_count: int = Field(..., description="Total number of instances")
    status_summary: EvaluationStatusSummary = Field(
        default_factory=EvaluationStatusSummary, description="Summary of instance statuses"
    )
    total_cost: float = Field(default=0, description="Total cost of the evaluation")
    prompt_tokens: int = Field(default=0, description="Total prompt tokens used")
    completion_tokens: int = Field(default=0, description="Total completion tokens used")
    cached_tokens: int = Field(default=0, description="Total cached tokens used")
    resolved_count: int = Field(default=0, description="Number of resolved instances")
    failed_count: int = Field(default=0, description="Number of failed instances")

    @classmethod
    def from_evaluation(cls, evaluation: Evaluation) -> "EvaluationSummary":
        """Create a summary from a full evaluation object"""
        # Initialize counters for token usage
        total_cost = 0
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        resolved_count = 0
        failed_count = 0

        # Create status summary
        summary = EvaluationStatusSummary()

        for instance in evaluation.instances:
            # Count execution statuses
            if instance.execution_status == ExecutionStatus.CREATED:
                summary.created += 1
            elif instance.execution_status == ExecutionStatus.QUEUED:
                summary.queued += 1
                summary.pending += 1  # Legacy compatibility
            elif instance.execution_status == ExecutionStatus.RUNNING:
                summary.running += 1
            elif instance.execution_status == ExecutionStatus.EVALUATING:
                summary.evaluating += 1
            elif instance.execution_status == ExecutionStatus.COMPLETED:
                summary.completed += 1
            elif instance.execution_status == ExecutionStatus.ERROR:
                summary.error += 1

            # Count resolution statuses
            if instance.resolution_status == ResolutionStatus.PENDING:
                summary.pending_resolution += 1
            elif instance.resolution_status == ResolutionStatus.RESOLVED:
                summary.resolved += 1
                resolved_count += 1
            elif instance.resolution_status == ResolutionStatus.FAILED:
                summary.failed += 1
                failed_count += 1
            elif instance.resolution_status == ResolutionStatus.PARTIALLY_RESOLVED:
                summary.partially_resolved += 1

            # Add up token usage if usage exists
            if instance.usage:
                usage = instance.usage
                total_cost += usage.completion_cost or 0
                prompt_tokens += usage.prompt_tokens or 0
                completion_tokens += usage.completion_tokens or 0
                cached_tokens += usage.cache_read_tokens or 0

        return cls(
            evaluation_name=evaluation.evaluation_name,
            dataset_name=evaluation.dataset_name,
            flow_id=evaluation.flow_id,
            model_id=evaluation.model_id,
            status=evaluation.status.value,
            created_at=evaluation.created_at,
            started_at=evaluation.started_at,
            completed_at=evaluation.completed_at,
            instance_count=len(evaluation.instances),
            status_summary=summary,
            total_cost=total_cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            resolved_count=resolved_count,
            failed_count=failed_count,
        )
