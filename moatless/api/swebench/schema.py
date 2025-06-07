from datetime import datetime
from token import OP
from typing import Optional

from moatless.completion.manager import BaseCompletionModel
from moatless.completion.stats import Usage
from moatless.flow.flow import AgenticFlow
from moatless.runner.runner import JobInfo, RunnerInfo, JobStatus
from pydantic import BaseModel, Field


class SWEBenchInstanceDTO(BaseModel):
    """Schema for a SWEBench instance"""

    instance_id: str = Field(..., description="Unique identifier for the instance")
    dataset: str = Field(..., description="Dataset name for the instance")
    repo: str = Field(..., description="Repository name for the instance")
    resolved_count: int = Field(..., description="Number of agents that have resolved this instance")
    file_count: int = Field(..., description="Number of files in expected_spans")


class FullSWEBenchInstanceDTO(SWEBenchInstanceDTO):
    """Full details schema for a SWEBench instance"""

    problem_statement: Optional[str] = Field(None, description="Problem statement for the instance")
    golden_patch: Optional[str] = Field(None, description="Golden patch for the instance")
    test_patch: Optional[str] = Field(None, description="Test patch for the instance")
    expected_spans: Optional[dict] = Field(None, description="Expected spans for the instance")
    test_file_spans: Optional[dict] = Field(None, description="Test file spans for the instance")

    base_commit: Optional[str] = Field(None, description="Base commit for the instance")
    fail_to_pass: Optional[list[str]] = Field(None, description="Tests that failed but should pass with the fix")
    pass_to_pass: Optional[list[str]] = Field(None, description="Tests that passed and should continue to pass")
    resolved_by: Optional[list[dict]] = Field(
        None, description="List of agents that resolved this instance with their approach"
    )


class SWEBenchValidationRequestDTO(BaseModel):
    """Schema for validation request"""

    instance_id: str = Field(..., description="ID of the instance to validate")
    model_id: str = Field(..., description="ID of the model to use")
    agent_id: str = Field(..., description="ID of the agent to use")
    max_iterations: int = Field(15, description="Maximum number of iterations")
    max_cost: Optional[float] = Field(1.0, description="Maximum cost of the validation")


class SWEBenchValidationResponseDTO(BaseModel):
    """Schema for validation response"""

    run_id: str = Field(..., description="Unique identifier for the validation")


class SWEBenchInstancesResponseDTO(BaseModel):
    """Response containing all available SWEBench instances"""

    instances: list[SWEBenchInstanceDTO] = Field(..., description="List of available instances")


class EvaluationStatusSummaryDTO(BaseModel):
    pending: int = 0
    running: int = 0
    evaluating: int = 0
    completed: int = 0
    error: int = 0
    resolved: int = 0
    failed: int = 0


class EvaluationRequestDTO(BaseModel):
    """Request for creating an evaluation"""

    flow_id: Optional[str] = None
    flow: Optional[dict] = None
    model_id: Optional[str] = Field(None, description="ID of the model config to use")
    litellm_model_name: Optional[str] = Field(None, description="LiteLLM model name to use")
    name: str
    num_concurrent_instances: int = 1
    dataset: Optional[str] = None
    instance_ids: Optional[list[str]] = None


class DatasetDTO(BaseModel):
    """DTO for dataset information"""

    name: str
    description: str
    instance_count: int


class DatasetsResponseDTO(BaseModel):
    """Response containing list of datasets"""

    datasets: list[DatasetDTO]


class RunnerResponseDTO(BaseModel):
    """Response containing runner status"""

    info: RunnerInfo
    jobs: list[JobInfo]


class RunnerStatsDTO(BaseModel):
    """Lightweight stats about the runner for the status bar"""

    runner_type: str = Field(..., description="Type of runner (kubernetes, asyncio, etc.)")
    status: str = Field(..., description="Current runner status")
    active_workers: int = Field(0, description="Number of available workers/nodes")
    total_workers: int = Field(0, description="Total number of workers/nodes")
    pending_jobs: int = Field(0, description="Number of pending jobs")
    running_jobs: int = Field(0, description="Number of running jobs")
    total_jobs: int = Field(0, description="Total number of jobs")
    queue_size: int = Field(0, description="Number of jobs in the queue")


class JobStatusSummaryResponseDTO(BaseModel):
    """Response DTO for job status summary."""

    project_id: str
    total_jobs: int
    queued_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    canceled_jobs: int
    pending_jobs: int


class CancelJobsResponseDTO(BaseModel):
    """Response DTO for canceling jobs."""

    project_id: str
    canceled_queued_jobs: int | None = None
    canceled_running_jobs: int | None = None
    message: str
