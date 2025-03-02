from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional

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
            "queued": [],
            "running": [],
            "completed": [],
            "failed": [],
            "canceled": [],
            "pending": [],
        }
    )


class Runner(ABC):
    """Runner for managing jobs."""

    @abstractmethod
    async def start_job(self, project_id: str, trajectory_id: str) -> bool:
        pass

    @abstractmethod
    async def get_jobs(self, project_id: str | None = None) -> list[JobInfo]:
        pass

    @abstractmethod
    async def cancel_job(self, project_id: str, trajectory_id: str | None = None) -> None:
        pass

    @abstractmethod
    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        pass

    @abstractmethod
    async def retry_job(self, project_id: str, trajectory_id: str) -> bool:
        pass

    @abstractmethod
    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        pass

    @abstractmethod
    async def get_runner_info(self) -> RunnerInfo:
        pass

    @abstractmethod
    async def get_job_status_summary(self, project_id: str) -> JobsStatusSummary:
        pass
