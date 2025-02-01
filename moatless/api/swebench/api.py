"""API endpoints for SWEBench validation."""

import os
import logging
import json
from typing import Optional
import asyncio
from datetime import datetime, timezone

from pydantic import create_model
from fastapi import APIRouter, HTTPException, Query
from moatless.benchmark.utils import get_moatless_instances
from .schema import (
    SWEBenchInstanceDTO,
    SWEBenchInstancesResponseDTO,
    SWEBenchValidationRequestDTO,
    SWEBenchValidationResponseDTO
)
from moatless.validation.code_flow_validation import CodeFlowValidation

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
