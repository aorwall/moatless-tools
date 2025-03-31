from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"  # Job created but not yet scheduled
    INITIALIZING = "initializing"  # Pod scheduled, containers being created/pulled/setup
    RUNNING = "running"  # Pod is running and executing the task
    COMPLETED = "completed"  # Job finished successfully
    FAILED = "failed"  # Job execution failed
    CANCELED = "canceled"  # Job was manually canceled
    NOT_STARTED = "not_started"  # Job not found or not started yet


class RunnerStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class RunnerInfo(BaseModel):
    runner_type: str
    status: RunnerStatus
    data: dict[str, Any]


class JobInfo(BaseModel):
    """Information about a job."""

    id: str
    status: JobStatus
    project_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    enqueued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = None


class JobsStatusSummary(BaseModel):
    """Summary of job status counts for a project."""

    project_id: str
    total_jobs: int = 0
    pending_jobs: int = 0
    initializing_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    canceled_jobs: int = 0
    job_ids: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "pending": [],
            "initializing": [],
            "running": [],
            "completed": [],
            "failed": [],
            "canceled": [],
        }
    )


class JobDetailSection(BaseModel):
    """A section of job details to display in the UI."""

    name: str
    display_name: str
    data: dict[str, Any] = Field(default_factory=dict)
    items: Optional[list[dict[str, Any]]] = None


class JobDetails(BaseModel):
    """Detailed information about a job."""

    id: str
    status: JobStatus
    project_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    enqueued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    sections: list[JobDetailSection] = Field(default_factory=list)
    error: Optional[str] = None
    raw_data: Optional[dict[str, Any]] = None


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
    async def start_job(
        self, project_id: str, trajectory_id: str, job_func: Callable, node_id: int | None = None
    ) -> bool:
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

    @abstractmethod
    async def get_job_details(self, project_id: str, trajectory_id: str) -> Optional[JobDetails]:
        """Get detailed information about a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobDetails object containing detailed information about the job if available
        """
        pass

    async def get_job_logs(self, project_id: str, trajectory_id: str) -> Optional[str]:
        """Get logs for a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            String containing the logs if available, None otherwise
        """
        # Default implementation returns None - each runner can override as needed
        return None
