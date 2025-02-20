"""API endpoints for model configuration management."""

import logging

from fastapi import APIRouter, HTTPException

from moatless.config.model_config import (
    get_model_config,
    get_all_configs,
    update_model_config,
    reset_model_config,
)
from .schema import ModelConfigUpdateDTO, ModelsResponseDTO
from moatless.config.model_config import ModelConfig

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("", response_model=ModelsResponseDTO)
async def list_models() -> ModelsResponseDTO:
    """Get all model configurations"""
    configs = get_all_configs()
    return ModelsResponseDTO(models=configs)
    

@router.get("/{model_id}", response_model=ModelConfig)
async def read_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a specific model"""
    try:
        logger.info(f"Getting model config for {model_id}")
        return get_model_config(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{model_id}")
async def update_model_config_api(model_id: str, update: ModelConfig):
    """Update configuration for a specific model"""
    try:
        logger.info(f"Updating model config for {model_id} with {update.model_dump(exclude_none=True)}")
        update_model_config(model_id, update.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{model_id}/overrides", response_model=ModelConfig)
async def reset_model_config_api(model_id: str) -> ModelConfig:
    """Reset model configuration to defaults by removing overrides"""
    try:
        config = reset_model_config(model_id)
        config.id = model_id
        return config
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))