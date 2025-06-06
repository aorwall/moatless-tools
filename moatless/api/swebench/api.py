"""API endpoints for SWEBench validation."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from moatless.api.dependencies import get_flow_manager, get_model_manager, get_evaluation_manager
from moatless.completion.manager import ModelConfigManager
from moatless.evaluation.manager import EvaluationManager
from moatless.evaluation.schema import Evaluation, EvaluationInstance, EvaluationStats, EvaluationSummary
from moatless.evaluation.utils import get_moatless_dataset_splits, get_moatless_instance, get_moatless_instances
from moatless.flow.manager import FlowManager
from moatless.flow.schema import TrajectoryResponseDTO
from moatless.flow.flow import AgenticFlow
from moatless.flow.search_tree import SearchTree

from .schema import (
    DatasetDTO,
    DatasetsResponseDTO,
    EvaluationRequestDTO,
    JobStatusSummaryResponseDTO,
    RunnerResponseDTO,
    SWEBenchInstanceDTO,
    SWEBenchInstancesResponseDTO,
    FullSWEBenchInstanceDTO,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/evaluations", response_model=Evaluation)
async def create_evaluation(
    request: EvaluationRequestDTO,
    model_manager: ModelConfigManager = Depends(get_model_manager),
    flow_manager: FlowManager = Depends(get_flow_manager),
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager),
):
    """Create a new evaluation run for a dataset."""
    if request.dataset:
        logger.info(f"Creating evaluation for dataset {request.dataset}")
    else:
        logger.info(f"Creating evaluation for all datasets with {request.instance_ids}")
        
    if request.flow:
        flow = SearchTree.model_validate(request.flow)
    else:
        flow = None

    evaluation = await evaluation_manager.create_evaluation(
        dataset_name=request.dataset,
        flow_id=request.flow_id,
        flow_config=flow,
        instance_ids=request.instance_ids,
    )

    return evaluation


@router.get("/evaluations/{evaluation_name}", response_model=Evaluation)
async def get_evaluation(
    evaluation_name: str,
    sync: bool = False,
    model_manager: ModelConfigManager = Depends(get_model_manager),
    flow_manager: FlowManager = Depends(get_flow_manager),
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager),
):
    """Get evaluation status and results."""
    evaluation = await evaluation_manager.get_evaluation(evaluation_name, sync=sync)

    if not evaluation:
        raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_name} not found")
    
    # TODO: Just to set resolved_by for now
    for instance in evaluation.instances:
        get_resolved_by(instance)
    
    return evaluation


@router.get("/evaluations/{evaluation_name}/clone", response_model=Evaluation)
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
    return evaluation


@router.get("/evaluations/{evaluation_name}/stats", response_model=EvaluationStats)
async def get_evaluation_stats(
    evaluation_name: str, evaluation_manager: EvaluationManager = Depends(get_evaluation_manager)
):
    """Get comprehensive statistics for an evaluation."""
    stats = await evaluation_manager.get_evaluation_stats(evaluation_name)
    return stats


@router.get("/evaluations/{evaluation_name}/config")
async def get_evaluation_config(
    evaluation_name: str,
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager),
):
    """Get the config for an evaluation."""
    return await evaluation_manager.get_config(evaluation_name)


@router.put("/evaluations/{evaluation_name}/config")
async def update_evaluation_config(
    evaluation_name: str,
    config: dict,
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager),
):
    """Update the config for an evaluation."""
    return await evaluation_manager.update_config(evaluation_name, config)


@router.get("/evaluations")
async def list_evaluations(evaluation_manager: EvaluationManager = Depends(get_evaluation_manager)):
    """List all evaluations with their status summaries."""
    return await evaluation_manager.list_evaluation_summaries()


@router.post("/evaluations/{evaluation_name}/start", response_model=Evaluation)
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

    return evaluation


@router.post("/evaluations/{evaluation_name}/process", response_model=Evaluation)
async def process_evaluation_results(
    evaluation_name: str,
    model_manager: ModelConfigManager = Depends(get_model_manager),
    flow_manager: FlowManager = Depends(get_flow_manager),
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager),
):
    """Process all instances in an evaluation to ensure results are in sync."""
    evaluation = await evaluation_manager.process_evaluation_results(evaluation_name=evaluation_name)
    return evaluation


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


@router.post("/evaluations/{evaluation_name}/instances/{instance_id}/process", response_model=EvaluationInstance)
async def process_instance_trajectory_results(
    evaluation_name: str, 
    instance_id: str, 
    evaluation_manager: EvaluationManager = Depends(get_evaluation_manager)
):
    """Process trajectory results for a specific instance in an evaluation."""
    # Get the evaluation
    evaluation = await evaluation_manager.get_evaluation(evaluation_name)
    
    # Find the specific instance
    instance = None
    for eval_instance in evaluation.instances:
        if eval_instance.instance_id == instance_id:
            instance = eval_instance
            break
    
    if not instance:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found in evaluation {evaluation_name}")
    
    # Process trajectory results for this specific instance
    updated_instance = await evaluation_manager._process_trajectory_results(evaluation, instance, force_update=True)
    
    # Save the evaluation with updated instance
    await evaluation_manager.save_evaluation(evaluation)
    
    return updated_instance
    


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


def get_resolved_by(instance: EvaluationInstance):
    moatless_instance = get_moatless_instance(instance.instance_id)
    instance.resolved_by = len(moatless_instance.get("resolved_by", []))

