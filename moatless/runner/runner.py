from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Type, List

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"  # Job created but not yet scheduled
    RUNNING = "running"  # Pod is running and executing the task
    COMPLETED = "completed"  # Job finished successfully
    FAILED = "failed"  # Job execution failed
    CANCELED = "canceled"  # Job was manually canceled
    STOPPED = "stopped"  # Job was running but stopped unexpectedly (e.g., container vanished)


class RunnerStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class RunnerInfo(BaseModel):
    runner_type: str
    status: RunnerStatus
    data: dict[str, Any] = Field(default_factory=dict)


class JobFunction(BaseModel):
    module: Optional[str]
    name: str


class JobInfo(BaseModel):
    """Information about a job."""

    id: str
    status: JobStatus
    project_id: str
    trajectory_id: str
    node_id: Optional[int] = None
    job_func: Optional[JobFunction] = None
    enqueued_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class JobsStatusSummary(BaseModel):
    """Summary of job status counts."""

    total_jobs: int = 0
    pending_jobs: int = 0
    initializing_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    canceled_jobs: int = 0


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
    def get_instance(cls, runner_impl: Type["BaseRunner"] | None = None, **kwargs) -> "BaseRunner":
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
        """Start a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run
            node_id: Optional node ID

        Returns:
            True if the job was started successfully, False otherwise
        """
        pass

    @abstractmethod
    async def get_jobs(self, project_id: str | None = None) -> list[JobInfo]:
        """Get a list of jobs.

        Args:
            project_id: The project ID, or None for all projects

        Returns:
            List of JobInfo objects
        """
        pass

    @abstractmethod
    async def cancel_job(self, project_id: str, trajectory_id: str | None = None) -> None:
        """Cancel a job or all jobs for a project.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID, or None for all jobs in the project
        """
        pass

    @abstractmethod
    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_job_status(self, project_id: str, trajectory_id: str) -> Optional[JobStatus]:
        """Get the status of a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobStatus enum representing the current status, or None if the job does not exist
        """
        pass

    @abstractmethod
    async def get_runner_info(self) -> RunnerInfo:
        """Get information about the runner.

        Returns:
            RunnerInfo object with information about the runner
        """
        pass

    async def get_job_status_summary(self) -> JobsStatusSummary:
        """Get a summary of job statuses for a project.

        Args:
            project_id: The project ID

        Returns:
            JobsStatusSummary object with counts of jobs in each state
        """
        # Default implementation, runners can override
        summary = JobsStatusSummary()

        jobs = await self.get_jobs()

        # Count jobs by status
        for job in jobs:
            summary.total_jobs += 1
            if job.status == JobStatus.PENDING:
                summary.pending_jobs += 1
            elif job.status == JobStatus.RUNNING:
                summary.running_jobs += 1
            elif job.status == JobStatus.COMPLETED:
                summary.completed_jobs += 1
            elif job.status == JobStatus.FAILED:
                summary.failed_jobs += 1
            elif job.status == JobStatus.CANCELED:
                summary.canceled_jobs += 1

        return summary

    async def get_job_details(self, project_id: str, trajectory_id: str) -> Optional[JobDetails]:
        """Get detailed information about a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobDetails object if available, None otherwise
        """
        # Default implementation returns None, concrete runners can override
        return None

    async def get_job_logs(self, project_id: str, trajectory_id: str) -> Optional[str]:
        """Get logs for a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            String containing the logs if available, None otherwise
        """
        # Default implementation returns None, concrete runners can override
        return None

    async def get_queue_size(self) -> int:
        """Get the current size of the job queue.

        Returns:
            The number of jobs in the queue
        """
        # Default implementation returns 0, concrete runners can override
        return 0

    async def reset_jobs(self, project_id: str | None = None) -> bool:
        """Reset all jobs or jobs for a specific project.

        Args:
            project_id: Optional project ID to reset jobs for

        Returns:
            True if jobs were reset successfully, False otherwise
        """
        # Default implementation returns False, concrete runners can override
        return False

    async def cleanup_job(self, project_id: str, trajectory_id: str) -> None:
        """Clean up resources after a job completes.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
        """
        # Default implementation does nothing, concrete runners can override
        pass
