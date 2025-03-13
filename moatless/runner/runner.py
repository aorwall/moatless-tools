from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    NOT_FOUND = "not_found"


class RunnerStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class RunnerInfo(BaseModel):
    runner_type: str
    status: RunnerStatus
    data: dict[str, Any]


class JobInfo(BaseModel):
    """Information about a job in RQ."""

    id: str
    status: JobStatus
    enqueued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    exc_info: Optional[str] = None


class EvaluationJobStatus(BaseModel):
    """Status of all jobs for an evaluation."""

    evaluation_name: str
    status: str
    instances: dict[str, dict[str, Any]]
    error: Optional[str] = None
    traceback: Optional[str] = None


class JobsCollection(BaseModel):
    """Collection of job IDs for an evaluation."""

    run_jobs: list[str] = Field(default_factory=list)
    eval_jobs: list[str] = Field(default_factory=list)
    active_jobs: list[str] = Field(default_factory=list)
    queued_jobs: list[str] = Field(default_factory=list)
    finished_jobs: list[str] = Field(default_factory=list)
    failed_jobs: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    traceback: Optional[str] = None


class CancellationResult(BaseModel):
    """Result of cancelling jobs for an evaluation."""

    evaluation_name: str
    cancelled_jobs: list[str] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)
    error: Optional[str] = None
    traceback: Optional[str] = None


class RetryResult(BaseModel):
    """Result of retrying a job for an instance."""

    instance_id: str
    requeued_jobs: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    traceback: Optional[str] = None


class RestartResult(BaseModel):
    """Result of restarting failed jobs for an evaluation."""

    evaluation_name: str
    status: str
    message: str
    error: Optional[str] = None
    traceback: Optional[str] = None


class JobsStatusSummary(BaseModel):
    """Summary of job status counts for a project."""

    project_id: str
    total_jobs: int = 0
    queued_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    canceled_jobs: int = 0
    pending_jobs: int = 0
    job_ids: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "pending": [],
            "queued": [],
            "running": [],
            "completed": [],
            "failed": [],
            "canceled": [],
        }
    )


class BaseRunner(ABC):
    _instance = None

    @classmethod
    def get_instance(cls, runner_impl: Type["BaseRunner"] = None, **kwargs) -> "BaseRunner":
        """
        Get or create the singleton instance of Runner.

        Args:
            runner_impl: Optional runner implementation class to use, defaults to AsyncioRunner
            **kwargs: Arguments to pass to the runner implementation constructor

        Returns:
            The singleton Runner instance
        """
        if cls._instance is None:
            # Import here to avoid circular imports
            from moatless.runner.asyncio_runner import AsyncioRunner

            # Use AsyncioRunner as default implementation
            impl_class = runner_impl or AsyncioRunner
            cls._instance = impl_class(**kwargs)

        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Mainly useful for testing."""
        cls._instance = None

    @abstractmethod
    async def start_job(self, project_id: str, trajectory_id: str, job_func: Callable | str) -> bool:
        """Start a job for the given project and trajectory."""
        pass

    @abstractmethod
    async def get_jobs(self, project_id: str | None = None) -> list[JobInfo]:
        """Get a list of jobs for the given project."""
        pass

    @abstractmethod
    async def cancel_job(self, project_id: str, trajectory_id: str | None = None) -> None:
        """Cancel a job for the given project and trajectory."""
        pass

    @abstractmethod
    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists for the given project and trajectory."""
        pass

    @abstractmethod
    async def retry_job(self, project_id: str, trajectory_id: str) -> bool:
        """Retry a job for the given project and trajectory."""
        pass

    @abstractmethod
    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        """Get the status of a job for the given project and trajectory."""
        pass

    @abstractmethod
    async def get_runner_info(self) -> RunnerInfo:
        """Get information about the runner."""
        pass

    @abstractmethod
    async def get_job_status_summary(self, project_id: str) -> JobsStatusSummary:
        """Get a summary of job statuses for the given project."""
        pass
