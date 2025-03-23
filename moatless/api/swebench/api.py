"""API endpoints for SWEBench validation."""

import asyncio
import logging
from typing import Optional
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, HTTPException

from moatless.completion.manager import ModelConfigManager
from moatless.evaluation.manager import EvaluationManager
from moatless.evaluation.schema import Evaluation, EvaluationInstance, EvaluationSummary
from moatless.evaluation.utils import get_moatless_dataset_splits, get_moatless_instance, get_moatless_instances
from moatless.flow.manager import FlowManager
from moatless.flow.schema import TrajectoryResponseDTO
from moatless.api.dependencies import get_flow_manager, get_model_manager, get_evaluation_manager

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
    FullSWEBenchInstanceDTO,
    SWEBenchValidationRequestDTO,
    SWEBenchValidationResponseDTO,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/evaluations", response_model=EvaluationResponseDTO)
async def create_evaluation(
    request: EvaluationRequestDTO,
    model_manager: ModelConfigManager = Depends(get_model_manager),
    flow_manager: FlowManager = Depends(get_flow_manager),
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager),
):
    """Create a new evaluation run for a dataset."""

    evaluation = await evaluation_manager.create_evaluation(
        dataset_name=request.dataset,
        flow_id=request.flow_id,
        model_id=request.model_id,
        instance_ids=request.instance_ids,
    )

    return map_to_evaluation_response(evaluation, model_manager, flow_manager)


@router.get("/evaluations/{evaluation_name}", response_model=EvaluationResponseDTO)
async def get_evaluation(
    evaluation_name: str,
    model_manager: ModelConfigManager = Depends(get_model_manager),
    flow_manager: FlowManager = Depends(get_flow_manager),
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager),
):
    """Get evaluation status and results."""
    evaluation = await evaluation_manager.get_evaluation(evaluation_name)

    if not evaluation:
        raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_name} not found")
    return map_to_evaluation_response(evaluation, model_manager, flow_manager)


@router.get("/evaluations/{evaluation_name}/clone", response_model=EvaluationResponseDTO)
async def clone_evaluation(
    evaluation_name: str,
    model_manager: ModelConfigManager = Depends(get_model_manager),
    flow_manager: FlowManager = Depends(get_flow_manager),
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager),
):
    """Clone an existing evaluation."""
    evaluation = await evaluation_manager.clone_evaluation(evaluation_name)

    if not evaluation:
        raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_name} not found")
    return map_to_evaluation_response(evaluation, model_manager, flow_manager)


@router.get("/evaluations")
async def list_evaluations(evaluation_manager: EvaluationManager = Depends(get_evaluation_manager)):
    """List all evaluations with their status summaries."""
    return await evaluation_manager.list_evaluation_summaries()


@router.post("/evaluations/{evaluation_name}/start", response_model=EvaluationResponseDTO)
async def start_evaluation(
    evaluation_name: str,
    model_manager: ModelConfigManager = Depends(get_model_manager),
    flow_manager: FlowManager = Depends(get_flow_manager),
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager),
):
    """Start an existing evaluation."""
    try:
        evaluation = await evaluation_manager.start_evaluation(evaluation_name=evaluation_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"Failed to start evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return map_to_evaluation_response(evaluation, model_manager, flow_manager)


@router.post("/evaluations/{evaluation_name}/process", response_model=EvaluationResponseDTO)
async def process_evaluation_results(
    evaluation_name: str,
    model_manager: ModelConfigManager = Depends(get_model_manager),
    flow_manager: FlowManager = Depends(get_flow_manager),
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager),
):
    """Process all instances in an evaluation to ensure results are in sync."""
    try:
        evaluation = await evaluation_manager.process_evaluation_results(evaluation_name=evaluation_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"Failed to process evaluation results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return map_to_evaluation_response(evaluation, model_manager, flow_manager)


@router.get("/evaluations/{evaluation_name}/instances/{instance_id}", response_model=TrajectoryResponseDTO)
async def get_evaluation_instance(
    evaluation_name: str, instance_id: str, flow_manager: FlowManager = Depends(get_flow_manager)
):
    """Get a specific instance of an evaluation."""
    try:
        return await flow_manager.get_trajectory(project_id=evaluation_name, trajectory_id=instance_id)
    except Exception as e:
        logger.exception(f"Failed to get evaluation instance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluations/{evaluation_name}/runner", response_model=RunnerResponseDTO)
async def get_runner_status(
    evaluation_name: str, evaluation_manager: EvaluationManager = Depends(get_evaluation_manager)
):
    """Get the status of the runner."""
    return RunnerResponseDTO(
        info=await evaluation_manager.runner.get_runner_info(),
        jobs=await evaluation_manager.runner.get_jobs(evaluation_name),
    )


@router.get("/evaluations/{evaluation_name}/jobs/status", response_model=JobStatusSummaryResponseDTO)
async def get_evaluation_jobs_status(
    evaluation_name: str, evaluation_manager: EvaluationManager = Depends(get_evaluation_manager)
):
    """Get a summary of job statuses for an evaluation."""
    try:
        summary = await evaluation_manager.runner.get_job_status_summary(evaluation_name)
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


@router.post("/evaluations/{evaluation_name}/jobs/cancel")
async def cancel_evaluation_jobs(
    evaluation_name: str, evaluation_manager: EvaluationManager = Depends(get_evaluation_manager)
):
    """Cancel all jobs for an evaluation."""
    try:
        # Get job status summary first to know what we're canceling
        summary = await evaluation_manager.runner.get_job_status_summary(evaluation_name)

        await evaluation_manager.cancel_evaluation(evaluation_name)

    except Exception as e:
        logger.exception(f"Failed to cancel jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluations/{evaluation_name}/instances/{instance_id}/start")
async def start_instance(
    evaluation_name: str, instance_id: str, evaluation_manager: EvaluationManager = Depends(get_evaluation_manager)
):
    """Start a specific instance in an evaluation."""
    await evaluation_manager.start_instance(evaluation_name, instance_id)


@router.post("/evaluations/{evaluation_name}/instances/{instance_id}/retry")
async def retry_instance(
    evaluation_name: str, instance_id: str, evaluation_manager: EvaluationManager = Depends(get_evaluation_manager)
):
    """Retry a specific instance in an evaluation that failed."""
    await evaluation_manager.retry_instance(evaluation_name, instance_id)


@router.get("/instances", response_model=SWEBenchInstancesResponseDTO)
async def list_instances(
    page: int = 1,
    limit: int = 20,
    sort_by: str = "instance_id",
    order: str = "asc",
    search: Optional[str] = None,
    dataset: Optional[str] = None,
    repo: Optional[str] = None,
    min_resolved: int = 0,
    max_resolved: int = 1000,
    min_files: int = 0,
    max_files: int = 1000,
):
    """List all available SWEBench instances with pagination, sorting, and search."""
    try:
        instances = get_moatless_instances()

        # Filter by search query, dataset, repo, resolved count, and file count
        filtered_instances = {}
        for k, v in instances.items():
            # Filter by search query (in instance_id and problem statement)
            if (
                search
                and search.lower() not in v["instance_id"].lower()
                and search.lower() not in v["problem_statement"].lower()
            ):
                continue

            # Filter by dataset
            if dataset and not any(d["dataset"] == dataset for d in v.get("datasets", [])):
                continue

            # Filter by repository
            if repo and repo.lower() not in v["repo"].lower():
                continue

            # Filter by resolved count
            resolved_count = len(v.get("resolved_by", []))
            if resolved_count < min_resolved or resolved_count > max_resolved:
                continue

            # Filter by file count (count only expected_spans)
            file_count = len(v.get("expected_spans", {}))
            if file_count < min_files or file_count > max_files:
                continue

            # Add to filtered instances
            filtered_instances[k] = v

        # Sort instances
        sorted_instances = sorted(filtered_instances.values(), key=lambda x: x[sort_by], reverse=(order == "desc"))

        # Apply pagination
        start = (page - 1) * limit
        end = start + limit
        paginated_instances = sorted_instances[start:end]

        return SWEBenchInstancesResponseDTO(
            instances=[
                SWEBenchInstanceDTO(
                    instance_id=instance["instance_id"],
                    repo=instance["repo"],
                    dataset=instance["dataset"],
                    resolved_count=len(instance.get("resolved_by", [])),
                    file_count=len(instance.get("expected_spans", {})),
                )
                for instance in paginated_instances
            ]
        )
    except Exception as e:
        logger.exception(f"Failed to list instances: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instances/{instance_id}", response_model=FullSWEBenchInstanceDTO)
async def get_instance(instance_id: str):
    """Fetch a specific SWEBench instance by ID."""
    try:
        instance = get_moatless_instance(instance_id)
        if not instance:
            raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")

        # Convert string test lists to actual lists if they exist
        fail_to_pass = instance.get("fail_to_pass")
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = eval(fail_to_pass)  # Safe because we control the data
            except:
                fail_to_pass = []

        pass_to_pass = instance.get("pass_to_pass")
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = eval(pass_to_pass)  # Safe because we control the data
            except:
                pass_to_pass = []

        return FullSWEBenchInstanceDTO(
            instance_id=instance["instance_id"],
            repo=instance["repo"],
            dataset=instance["dataset"],
            problem_statement=instance["problem_statement"],
            resolved_count=len(instance.get("resolved_by", [])),
            file_count=len(instance.get("expected_spans", {})),
            golden_patch=instance["golden_patch"],
            test_patch=instance["test_patch"],
            expected_spans=instance["expected_spans"],
            test_file_spans=instance["test_file_spans"],
            base_commit=instance.get("base_commit"),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            resolved_by=instance.get("resolved_by"),
        )
    except Exception as e:
        logger.exception(f"Failed to get instance {instance_id}: {str(e)}")
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


def map_to_evaluation_response(
    evaluation: Evaluation, model_manager: ModelConfigManager, flow_manager: FlowManager
) -> EvaluationResponseDTO:
    flow = flow_manager.get_flow_config(evaluation.flow_id)
    model = model_manager.get_model_config(evaluation.model_id)

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
                iterations=instance.iterations,
                resolved=instance.resolved,
                resolved_by=get_resolved_by(instance),
                reward=instance.reward,
                usage=instance.usage,
            )
            for instance in evaluation.instances
        ],
    )
