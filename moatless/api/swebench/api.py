"""API endpoints for SWEBench validation."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from moatless.completion.manager import get_model_config
from moatless.evaluation.manager import EvaluationManager
from moatless.evaluation.schema import Evaluation, EvaluationInstance
from moatless.evaluation.utils import get_moatless_dataset_splits, get_moatless_instance, get_moatless_instances
from moatless.flow.manager import FlowManager, get_flow_config
from moatless.flow.schema import TrajectoryResponseDTO

from .schema import (
    CancelJobsResponseDTO,
    DatasetDTO,
    DatasetsResponseDTO,
    EvaluationInstanceDTO,
    EvaluationListItemDTO,
    EvaluationListResponseDTO,
    EvaluationRequestDTO,
    EvaluationResponseDTO,
    JobStatusSummaryResponseDTO,
    RunnerResponseDTO,
    SWEBenchInstanceDTO,
    SWEBenchInstancesResponseDTO,
    get_instance_status,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/evaluations", response_model=EvaluationResponseDTO)
async def create_evaluation(request: EvaluationRequestDTO):
    """Create a new evaluation run for a dataset."""

    try:
        manager = await EvaluationManager.get_instance()
        evaluation = await manager.create_evaluation(
            dataset_name=request.dataset,
            flow_id=request.flow_id,
            model_id=request.model_id,
        )

        return map_to_evaluation_response(evaluation)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluations/{evaluation_name}", response_model=EvaluationResponseDTO)
async def get_evaluation(evaluation_name: str):
    """Get evaluation status and results."""
    manager = await EvaluationManager.get_instance()
    evaluation = await manager.get_evaluation(evaluation_name)

    if not evaluation:
        raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_name} not found")
    return map_to_evaluation_response(evaluation)


@router.get("/evaluations/{evaluation_name}/clone", response_model=EvaluationResponseDTO)
async def clone_evaluation(evaluation_name: str):
    """Clone an existing evaluation."""
    manager = await EvaluationManager.get_instance()
    evaluation = await manager.clone_evaluation(evaluation_name)

    if not evaluation:
        raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_name} not found")
    return map_to_evaluation_response(evaluation)


@router.get("/evaluations", response_model=EvaluationListResponseDTO)
async def list_evaluations():
    """List all evaluations with their status summaries."""
    try:
        manager = await EvaluationManager.get_instance()
        evaluations = await manager.list_evaluations()

        response_items = [EvaluationListItemDTO.from_evaluation(eval) for eval in evaluations]

        return EvaluationListResponseDTO(evaluations=response_items)

    except Exception as e:
        logger.exception(f"Failed to list evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluations/{evaluation_name}/start", response_model=EvaluationResponseDTO)
async def start_evaluation(evaluation_name: str):
    """Start an existing evaluation."""
    try:
        manager = await EvaluationManager.get_instance()
        evaluation = await manager.start_evaluation(evaluation_name=evaluation_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"Failed to start evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return map_to_evaluation_response(evaluation)


@router.post("/evaluations/{evaluation_name}/process", response_model=EvaluationResponseDTO)
async def process_evaluation_results(evaluation_name: str):
    """Process all instances in an evaluation to ensure results are in sync."""
    try:
        manager = await EvaluationManager.get_instance()
        evaluation = await manager.process_evaluation_results(evaluation_name=evaluation_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"Failed to process evaluation results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return map_to_evaluation_response(evaluation)


@router.get("/evaluations/{evaluation_name}/instances/{instance_id}", response_model=TrajectoryResponseDTO)
async def get_evaluation_instance(evaluation_name: str, instance_id: str):
    """Get a specific instance of an evaluation."""
    try:
        flow_manager = FlowManager.get_instance()
        return await flow_manager.get_trajectory(project_id=evaluation_name, trajectory_id=instance_id)
    except Exception as e:
        logger.exception(f"Failed to get evaluation instance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluations/{evaluation_name}/runner", response_model=RunnerResponseDTO)
async def get_runner_status(evaluation_name: str):
    """Get the status of the runner."""
    manager = await EvaluationManager.get_instance()
    return RunnerResponseDTO(
        info=await manager.runner.get_runner_info(), jobs=await manager.runner.get_jobs(evaluation_name)
    )


@router.get("/evaluations/{evaluation_name}/jobs/status", response_model=JobStatusSummaryResponseDTO)
async def get_evaluation_jobs_status(evaluation_name: str):
    """Get a summary of job statuses for an evaluation."""
    try:
        manager = await EvaluationManager.get_instance()
        summary = await manager.runner.get_job_status_summary(evaluation_name)
        return JobStatusSummaryResponseDTO(
            project_id=evaluation_name,
            total_jobs=summary.total_jobs,
            queued_jobs=summary.queued_jobs,
            running_jobs=summary.running_jobs,
            completed_jobs=summary.completed_jobs,
            failed_jobs=summary.failed_jobs,
            canceled_jobs=summary.canceled_jobs,
            pending_jobs=summary.pending_jobs,
        )
    except Exception as e:
        logger.exception(f"Failed to get job status summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluations/{evaluation_name}/jobs/cancel", response_model=CancelJobsResponseDTO)
async def cancel_evaluation_jobs(evaluation_name: str):
    """Cancel all jobs for an evaluation."""
    try:
        manager = await EvaluationManager.get_instance()

        # Get job status summary first to know what we're canceling
        summary = await manager.runner.get_job_status_summary(evaluation_name)

        # Cancel all jobs and set evaluation status to PAUSED
        await manager.cancel_evaluation(evaluation_name)

        return CancelJobsResponseDTO(
            project_id=evaluation_name,
            canceled_queued_jobs=summary.queued_jobs,
            canceled_running_jobs=summary.running_jobs,
            message=f"Successfully canceled {summary.queued_jobs + summary.running_jobs} jobs for evaluation {evaluation_name}",
        )
    except Exception as e:
        logger.exception(f"Failed to cancel jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluations/{evaluation_name}/instances/{instance_id}/start")
async def start_instance(evaluation_name: str, instance_id: str):
    """Start a specific instance in an evaluation."""
    manager = await EvaluationManager.get_instance()
    await manager.start_instance(evaluation_name, instance_id)


@router.post("/evaluations/{evaluation_name}/instances/{instance_id}/retry")
async def retry_instance(evaluation_name: str, instance_id: str):
    """Retry a specific instance in an evaluation that failed."""
    manager = await EvaluationManager.get_instance()
    await manager.retry_instance(evaluation_name, instance_id)


@router.get("/instances", response_model=SWEBenchInstancesResponseDTO)
async def list_instances(
    page: int = 1, limit: int = 20, sort_by: str = "instance_id", order: str = "asc", search: Optional[str] = None
):
    """List all available SWEBench instances with pagination, sorting, and search."""
    try:
        instances = get_moatless_instances()

        # Filter instances by search query
        if search:
            instances = {k: v for k, v in instances.items() if search.lower() in v["instance_id"].lower()}

        # Sort instances
        sorted_instances = sorted(instances.values(), key=lambda x: x[sort_by], reverse=(order == "desc"))

        # Apply pagination
        start = (page - 1) * limit
        end = start + limit
        paginated_instances = sorted_instances[start:end]

        return SWEBenchInstancesResponseDTO(
            instances=[
                SWEBenchInstanceDTO(
                    instance_id=instance["instance_id"],
                    problem_statement=instance["problem_statement"],
                    resolved_count=len(instance.get("resolved_by", [])),
                )
                for instance in paginated_instances
            ]
        )
    except Exception as e:
        logger.exception(f"Failed to list instances: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", response_model=DatasetsResponseDTO)
async def list_datasets():
    """List all available datasets with their descriptions and instance counts."""
    try:
        datasets = get_moatless_dataset_splits()
        return DatasetsResponseDTO(datasets=[DatasetDTO(**dataset) for dataset in datasets.values()])
    except Exception as e:
        logger.exception(f"Failed to list datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def get_resolved_by(instance: EvaluationInstance) -> int:
    moatless_instance = get_moatless_instance(instance.instance_id)
    return len(moatless_instance.get("resolved_by", []))


def map_to_evaluation_response(evaluation: Evaluation) -> EvaluationResponseDTO:
    flow = get_flow_config(evaluation.flow_id)
    model = get_model_config(evaluation.model_id)

    return EvaluationResponseDTO(
        evaluation_name=evaluation.evaluation_name,
        dataset_name=evaluation.dataset_name,
        status=evaluation.status.value,
        created_at=evaluation.created_at,
        started_at=evaluation.started_at,
        completed_at=evaluation.completed_at,
        flow=flow,
        model=model,
        instances=[
            EvaluationInstanceDTO(
                instance_id=instance.instance_id,
                status=get_instance_status(instance),
                error=instance.error,
                created_at=instance.created_at,
                started_at=instance.started_at,
                completed_at=instance.completed_at,
                evaluated_at=instance.evaluated_at,
                error_at=instance.error_at,
                resolved=instance.resolved,
                resolved_by=get_resolved_by(instance),
                reward=instance.reward,
                usage=instance.usage,
            )
            for instance in evaluation.instances
        ],
    )
