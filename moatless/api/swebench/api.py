"""API endpoints for SWEBench validation."""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from glob import glob
from typing import Optional

from fastapi import APIRouter, HTTPException

from moatless.api.trajectories.api import load_trajectory_events, load_trajectory_status
from moatless.api.trajectories.schema import TrajectoryResponseDTO
from moatless.api.trajectory.trajectory_utils import load_trajectory_from_file
from moatless.benchmark.evaluation_manager import EvaluationManager
from moatless.benchmark.evaluation_runner import EvaluationRunner
from moatless.benchmark.schema import EvaluationStatus
from moatless.benchmark.utils import get_moatless_dataset_split, get_moatless_dataset_splits, get_moatless_instances
from moatless.utils.moatless import get_moatless_trajectory_dir
from moatless.validation.code_flow_validation import CodeFlowValidation
from .schema import (
    SWEBenchInstanceDTO,
    SWEBenchInstancesResponseDTO,
    SWEBenchValidationRequestDTO,
    SWEBenchValidationResponseDTO,
    EvaluationListItemDTO,
    EvaluationListResponseDTO,
    EvaluationResponseDTO,
    EvaluationInstanceDTO,
    EvaluationRequestDTO,
    DatasetDTO,
    DatasetsResponseDTO,
    StartEvaluationRequestDTO,
    get_instance_status
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/instances", response_model=SWEBenchInstancesResponseDTO)
async def list_instances(page: int = 1, limit: int = 20, sort_by: str = 'instance_id', order: str = 'asc', search: Optional[str] = None):
    """List all available SWEBench instances with pagination, sorting, and search."""
    try:
        instances = get_moatless_instances()
        
        # Filter instances by search query
        if search:
            instances = {k: v for k, v in instances.items() if search.lower() in v["instance_id"].lower()}
        
        # Sort instances
        sorted_instances = sorted(
            instances.values(),
            key=lambda x: x[sort_by],
            reverse=(order == 'desc')
        )
        
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


@router.post("/validate", response_model=SWEBenchValidationResponseDTO)
async def validate_instance(request: SWEBenchValidationRequestDTO):
    """Start a new validation run."""
    try:
        # Generate a unique run ID using timestamp
        run_id = f"validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize the validator
        validator = CodeFlowValidation()
        
        try:
            await validator.start_code_loop(
                run_id=run_id,
                agent_id=request.agent_id,
                model_id=request.model_id,
                instance_id=request.instance_id,
                max_iterations=request.max_iterations
            )
        except Exception as e:
            logger.exception(f"Validation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

        return SWEBenchValidationResponseDTO(
            run_id=run_id
        )

    except Exception as e:
        logger.exception(f"Failed to start validation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets", response_model=DatasetsResponseDTO)
async def list_datasets():
    """List all available datasets with their descriptions and instance counts."""
    try:
        datasets = get_moatless_dataset_splits()
        return DatasetsResponseDTO(datasets=[
            DatasetDTO(**dataset) for dataset in datasets.values()
        ])
    except Exception as e:
        logger.exception(f"Failed to list datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluations", response_model=EvaluationResponseDTO)
async def create_evaluation(request: EvaluationRequestDTO):
    """Create a new evaluation run for a dataset."""
    try:
        dataset = get_moatless_dataset_split(request.dataset)
        if not dataset:
            raise HTTPException(status_code=400, detail=f"Dataset {request.dataset} not found")

        manager = EvaluationManager()
        
        evaluation = manager.create_evaluation(
            dataset_name=request.dataset,
            instance_ids=dataset["instance_ids"],
            flow_id=request.flow_id,
            model_id=request.model_id,
        )
        
        return EvaluationResponseDTO(
            evaluation_name=evaluation.evaluation_name,
            dataset_name=evaluation.dataset_name,
            status=evaluation.status.value,
            created_at=evaluation.created_at,
            started_at=evaluation.started_at,
            completed_at=evaluation.completed_at,
            flow_id=evaluation.flow_id,
            model_id=evaluation.model_id,
            instances=[
                EvaluationInstanceDTO(
                    instance_id=instance.instance_id,
                    status=get_instance_status(instance),
                    error=instance.error,
                    started_at=instance.started_at,
                    completed_at=instance.completed_at,
                    evaluated_at=instance.evaluated_at,
                    resolved=instance.resolved
                )
                for instance in evaluation.instances
            ]
        )
        
    except Exception as e:
        logger.exception(f"Failed to create evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluations/{evaluation_name}", response_model=EvaluationResponseDTO)
async def get_evaluation(evaluation_name: str):
    """Get evaluation status and results."""
    try:
        manager = EvaluationManager()
        evaluation = manager._load_evaluation(evaluation_name)
        
        if not evaluation:
            raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_name} not found")
            
        return EvaluationResponseDTO(
            evaluation_name=evaluation.evaluation_name,
            dataset_name=evaluation.dataset_name,
            status=evaluation.status.value,
            created_at=evaluation.created_at,
            started_at=evaluation.started_at,
            completed_at=evaluation.completed_at,
            flow_id=evaluation.flow_id,
            model_id=evaluation.model_id,
            instances=[
                EvaluationInstanceDTO(
                    instance_id=instance.instance_id,
                    status=get_instance_status(instance),
                    error=instance.error,
                    created_at=instance.created_at,
                    started_at=instance.started_at,
                    completed_at=instance.completed_at,
                    evaluated_at=instance.evaluated_at,
                    resolved=instance.resolved
                )
                for instance in evaluation.instances
            ]
        )
    except Exception as e:
        logger.exception(f"Failed to get evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluations", response_model=EvaluationListResponseDTO)
async def list_evaluations():
    """List all evaluations with their status summaries."""
    try:
        manager = EvaluationManager()
        evaluations = manager.list_evaluations()
        
        response_items = [
            EvaluationListItemDTO.from_evaluation(eval)
            for eval in evaluations
        ]
        
        return EvaluationListResponseDTO(evaluations=response_items)
        
    except Exception as e:
        logger.exception(f"Failed to list evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluations/{evaluation_name}/start", response_model=EvaluationResponseDTO)
async def start_evaluation(evaluation_name: str, request: StartEvaluationRequestDTO):
    """Start an existing evaluation."""
    try:
        manager = EvaluationManager()
        evaluation = manager._load_evaluation(evaluation_name)
        
        if not evaluation:
            raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_name} not found")
            
        if evaluation.status in [EvaluationStatus.RUNNING, EvaluationStatus.COMPLETED]:
            raise HTTPException(status_code=400, detail="Evaluation cannot be started in its current state")

        runner = EvaluationRunner(evaluation=evaluation, num_concurrent_instances=request.num_concurrent_instances)
        asyncio.create_task(runner.run_evaluation())
            
        evaluation.status = EvaluationStatus.RUNNING
        evaluation.started_at = datetime.now(timezone.utc)
        manager._save_evaluation(evaluation)
        
        return EvaluationResponseDTO(
            evaluation_name=evaluation.evaluation_name,
            dataset_name=evaluation.dataset_name,
            flow_id=evaluation.flow_id,
            model_id=evaluation.model_id,
            status=evaluation.status.value,
            started_at=evaluation.started_at,
            completed_at=evaluation.completed_at,
            instances=[
                EvaluationInstanceDTO(
                    instance_id=instance.instance_id,
                    status=instance.status.value,
                    error=instance.error,
                    started_at=instance.started_at,
                    completed_at=instance.completed_at
                )
                for instance in evaluation.instances
            ]
        )
    except Exception as e:
        logger.exception(f"Failed to start evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluations/{evaluation_name}/instances/{instance_id}", response_model=TrajectoryResponseDTO)
async def get_evaluation_instance(evaluation_name: str, instance_id: str):
    """Get a specific instance of an evaluation."""
    try:
        manager = EvaluationManager()
        trajectory_dir = get_moatless_trajectory_dir(instance_id, evaluation_name)
        trajectory_path = trajectory_dir / 'trajectory.json'
        if not trajectory_path.exists():
            raise HTTPException(status_code=404, detail="Trajectory not found")
        try:
            trajectory = load_trajectory_from_file(trajectory_path)
            status = "finished"
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Trajectory not found")

        system_status = load_trajectory_status(trajectory_dir)
        if system_status.status == "running":
            system_status.status = "stopped"
        
        status = system_status.status

        events = load_trajectory_events(trajectory_dir)

        return TrajectoryResponseDTO(
            status=status,
            system_status=system_status,
                agent_id=system_status.metadata.get("agent_id"),
                model_id=system_status.metadata.get("model_id"),
                events=events,
                **trajectory.model_dump()
            )
    except Exception as e:
        logger.exception(f"Failed to get evaluation instance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
