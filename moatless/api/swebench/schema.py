from datetime import datetime
from typing import List, Optional

from moatless.api.trajectory.schema import UsageDTO
from moatless.completion.model import Usage
from moatless.runner.runner import JobsCollection, RunnerInfo, RunnerStatus
from pydantic import BaseModel, Field

from moatless.evaluation.schema import Evaluation, EvaluationInstance, InstanceStatus
from moatless.flow.schema import FlowConfig
from moatless.completion.manager import ModelConfig
from moatless.runner.runner import JobInfo

class SWEBenchInstanceDTO(BaseModel):
    """Schema for a SWEBench instance"""

    instance_id: str = Field(..., description="Unique identifier for the instance")
    problem_statement: str = Field(..., description="Problem statement for the instance")
    resolved_count: int = Field(..., description="Number of agents that have resolved this instance")


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

    instances: List[SWEBenchInstanceDTO] = Field(..., description="List of available instances")


class EvaluationStatusSummaryDTO(BaseModel):
    pending: int = 0
    running: int = 0
    evaluating: int = 0
    completed: int = 0
    error: int = 0
    resolved: int = 0
    failed: int = 0

class EvaluationListItemDTO(BaseModel):
    evaluation_name: str
    dataset_name: str
    flow_id: str
    model_id: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    instance_count: int
    status_summary: Optional[EvaluationStatusSummaryDTO] = None
    total_cost: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    resolved_count: int = 0
    failed_count: int = 0

    @classmethod
    def from_evaluation(cls, evaluation: Evaluation) -> "EvaluationListItemDTO":
        """Build EvaluationListItemDTO from Evaluation object."""
        # Initialize counters for token usage
        total_cost = 0
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        resolved_count = 0
        failed_count = 0

        # Create status summary
        summary = EvaluationStatusSummaryDTO()
        
        for instance in evaluation.instances:
            # Get status using the helper function
            status = get_instance_status(instance)
            
            # Update status summary counters
            if status == "pending":
                summary.pending += 1
            elif status == "running":
                summary.running += 1
            elif status == "evaluating":
                summary.evaluating += 1
            elif status == "completed":
                summary.completed += 1
            elif status == "error":
                summary.error += 1
            
            # Count resolved/failed instances
            if instance.resolved is True:
                summary.resolved += 1
                resolved_count += 1
            elif instance.resolved is False:
                summary.failed += 1
                failed_count += 1

            # Add up token usage if benchmark_result exists
            if instance.benchmark_result:
                result = instance.benchmark_result
                total_cost += result.total_cost
                prompt_tokens += result.prompt_tokens
                completion_tokens += result.completion_tokens
                cached_tokens += result.cached_tokens

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
            failed_count=failed_count
        )

def get_instance_status(instance: EvaluationInstance) -> str:
    """Get the status of an instance, including resolved/failed states."""
    if instance.evaluated_at and instance.resolved:
        return "resolved"
    elif instance.evaluated_at and instance.resolved is False:
        return "failed"
    elif instance.job_status:
        return instance.job_status.value
    else:
        return instance.status.value

class EvaluationListResponseDTO(BaseModel):
    """Response containing list of evaluations"""
    evaluations: List[EvaluationListItemDTO] = Field(..., description="List of evaluations")

class EvaluationInstanceDTO(BaseModel):
    """DTO for evaluation instance details"""
    instance_id: str
    status: str
    job_status: Optional[str] = None
    resolved: Optional[bool] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    evaluated_at: Optional[datetime] = None
    error_at: Optional[datetime] = None
    resolved_by: Optional[int] = None
    reward: Optional[int] = None
    usage: Optional[Usage] = None

class EvaluationResponseDTO(BaseModel):
    """Response containing evaluation details"""
    evaluation_name: str
    dataset_name: str
    status: str
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    flow: FlowConfig
    model: ModelConfig
    instances: List[EvaluationInstanceDTO]

class EvaluationRequestDTO(BaseModel):
    """Request for creating an evaluation"""
    flow_id: str
    model_id: str
    name: str
    num_concurrent_instances: int = 1
    dataset: str
    max_iterations: int = 10
    max_expansions: int = 1

class DatasetDTO(BaseModel):
    """DTO for dataset information"""
    name: str
    description: str
    instance_count: int

class DatasetsResponseDTO(BaseModel):
    """Response containing list of datasets"""
    datasets: List[DatasetDTO]


class RunnerResponseDTO(BaseModel):
    """Response containing runner status"""
    info: RunnerInfo
    jobs: List[JobInfo]
