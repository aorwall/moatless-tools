"""API endpoints for SWEBench validation."""

import os
import logging
import json
from typing import Optional, List
import asyncio
from datetime import datetime, timezone
from glob import glob

from pydantic import create_model, BaseModel
from fastapi import APIRouter, HTTPException, Query
from moatless.benchmark.schema import EvaluationStatus, InstanceStatus
from moatless.benchmark.utils import get_moatless_instances
from moatless.benchmark.evaluation_runner import EvaluationRunner, TreeSearchSettings, Evaluation
from .schema import (
    SWEBenchInstanceDTO,
    SWEBenchInstancesResponseDTO,
    SWEBenchValidationRequestDTO,
    SWEBenchValidationResponseDTO
)
from moatless.validation.code_flow_validation import CodeFlowValidation
from moatless.benchmark.evaluation_manager import EvaluationManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Add new schema classes
class DatasetDTO(BaseModel):
    name: str
    description: str
    instance_count: int

class DatasetsResponseDTO(BaseModel):
    datasets: List[DatasetDTO]

class EvaluationRequestDTO(BaseModel):
    agent_id: str
    model_id: str
    num_workers: int = 1
    dataset: str
    max_iterations: int = 10
    max_expansions: int = 1

class EvaluationInstanceDTO(BaseModel):
    instance_id: str
    status: str
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class EvaluationResponseDTO(BaseModel):
    evaluation_name: str
    dataset_name: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    instances: List[EvaluationInstanceDTO]

class EvaluationStatusSummaryDTO(BaseModel):
    pending: int = 0
    started: int = 0
    completed: int = 0
    error: int = 0

class EvaluationListItemDTO(BaseModel):
    evaluation_name: str
    dataset_name: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    instance_count: int
    status_summary: Optional[EvaluationStatusSummaryDTO] = None

class EvaluationListResponseDTO(BaseModel):
    evaluations: List[EvaluationListItemDTO]


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
        datasets = []
        dataset_files = glob("datasets/*.json")
        
        for dataset_file in dataset_files:
            with open(dataset_file, 'r') as f:
                data = json.load(f)
                if "name" in data and "instance_ids" in data:
                    datasets.append(DatasetDTO(
                        name=data.get("name", ""),
                        description=data.get("description", ""),
                        instance_count=len(data.get("instance_ids", []))
                    ))
        
        return DatasetsResponseDTO(datasets=datasets)
    except Exception as e:
        logger.exception(f"Failed to list datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluations", response_model=EvaluationResponseDTO)
async def create_evaluation(request: EvaluationRequestDTO):
    """Create a new evaluation run for a dataset."""
    try:
        dataset_path = f"datasets/{request.dataset}_dataset.json"
        if not os.path.exists(dataset_path):
            logger.error(f"No dataset found on {dataset_path}")
            raise HTTPException(status_code=400, detail=f"Dataset {request.dataset} not found")
            
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        manager = EvaluationManager()
        
        evaluation_name = manager.create_evaluation(
            dataset_name=request.dataset,
            instance_ids=dataset["instance_ids"],
            agent_id=request.agent_id,
            model_id=request.model_id,
            num_workers=request.num_workers,
            max_iterations=request.max_iterations,
            max_expansions=request.max_expansions
        )

        evaluation = manager._load_evaluation(evaluation_name)
        
        return EvaluationResponseDTO(
            evaluation_name=evaluation.evaluation_name,
            dataset_name=evaluation.dataset_name,
            status=evaluation.status.value,
            created_at=evaluation.created_at,
            started_at=evaluation.started_at,
            completed_at=evaluation.completed_at,
            settings=evaluation.settings,
            num_workers=evaluation.num_workers,
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
        logger.exception(f"Failed to get evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluations", response_model=EvaluationListResponseDTO)
async def list_evaluations():
    """List all evaluations with their status summaries."""
    try:
        manager = EvaluationManager()
        evaluations = manager.list_evaluations()
        
        response_items = []
        for eval in evaluations:
            # Create status summary if evaluation is not finished
            status_summary = None
            if eval.status != EvaluationStatus.COMPLETED:
                summary = EvaluationStatusSummaryDTO()
                for instance in eval.instances:
                    if instance.status == InstanceStatus.PENDING:
                        summary.pending += 1
                    elif instance.status == InstanceStatus.STARTED:
                        summary.started += 1
                    elif instance.status == InstanceStatus.COMPLETED:
                        summary.completed += 1
                    elif instance.status == InstanceStatus.ERROR:
                        summary.error += 1
                status_summary = summary

            response_items.append(
                EvaluationListItemDTO(
                    evaluation_name=eval.evaluation_name,
                    dataset_name=eval.dataset_name,
                    status=eval.status.value,
                    started_at=eval.start_time,
                    completed_at=eval.finish_time,
                    instance_count=len(eval.instances),
                    status_summary=status_summary
                )
            )
        
        return EvaluationListResponseDTO(evaluations=response_items)
        
    except Exception as e:
        logger.exception(f"Failed to list evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluations/{evaluation_name}/start", response_model=EvaluationResponseDTO)
async def start_evaluation(evaluation_name: str):
    """Start an existing evaluation."""
    try:
        manager = EvaluationManager()
        evaluation = manager._load_evaluation(evaluation_name)
        
        if not evaluation:
            raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_name} not found")
            
        if evaluation.status in [EvaluationStatus.RUNNING, EvaluationStatus.COMPLETED]:
            raise HTTPException(status_code=400, detail="Evaluation cannot be started in its current state")

        evaluation.status = EvaluationStatus.RUNNING
        evaluation.started_at = datetime.now(timezone.utc)
        manager._save_evaluation(evaluation)
        
        runner = EvaluationRunner(evaluation=evaluation)
        asyncio.create_task(runner.run_evaluation(evaluation_name))
            
        return EvaluationResponseDTO(
            evaluation_name=evaluation.evaluation_name,
            dataset_name=evaluation.dataset_name,
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
