from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List

from pydantic import BaseModel, Field
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunnerStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"

class RunnerInfo(BaseModel):
    runner_type: str
    status: RunnerStatus
    data: Dict[str, Any]

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
    instances: Dict[str, Dict[str, Any]]
    error: Optional[str] = None
    traceback: Optional[str] = None


class JobsCollection(BaseModel):
    """Collection of job IDs for an evaluation."""
    run_jobs: List[str] = Field(default_factory=list)
    eval_jobs: List[str] = Field(default_factory=list)
    active_jobs: List[str] = Field(default_factory=list)
    queued_jobs: List[str] = Field(default_factory=list)
    finished_jobs: List[str] = Field(default_factory=list)
    failed_jobs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    traceback: Optional[str] = None


class CancellationResult(BaseModel):
    """Result of cancelling jobs for an evaluation."""
    evaluation_name: str
    cancelled_jobs: List[str] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(default_factory=list)
    error: Optional[str] = None
    traceback: Optional[str] = None


class RetryResult(BaseModel):
    """Result of retrying a job for an instance."""
    instance_id: str
    requeued_jobs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    traceback: Optional[str] = None


class RestartResult(BaseModel):
    """Result of restarting failed jobs for an evaluation."""
    evaluation_name: str
    status: str
    message: str
    error: Optional[str] = None
    traceback: Optional[str] = None


class Runner(ABC):
    """Runner for managing jobs."""
    
    @abstractmethod
    async def start_job(self, project_id: str, trajectory_id: str) -> bool:
        pass
    
    @abstractmethod
    async def get_jobs(self, project_id: str | None = None) -> List[JobInfo]:
        pass
    
    @abstractmethod
    async def cancel_job(self, project_id: str, trajectory_id: str) -> None:
        pass
    
    @abstractmethod
    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        pass
    
    @abstractmethod
    async def retry_job(self, project_id: str, trajectory_id: str) -> None:
        pass
    
    @abstractmethod
    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        pass
    
    @abstractmethod
    async def get_runner_info(self) -> RunnerInfo:
        pass
