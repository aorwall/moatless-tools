"""API endpoints for model configuration management."""

import logging
from fastapi import APIRouter, HTTPException
from moatless.config.model_config import (
    get_model_config,
    get_all_configs,
    update_model_config,
    reset_model_config,
)
from .schema import ModelConfigDTO, ModelConfigUpdateDTO, ModelsResponseDTO

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("", response_model=ModelsResponseDTO)
async def list_models() -> ModelsResponseDTO:
    """Get all model configurations"""
    all_configs = get_all_configs()
    configs = [ModelConfigDTO(**config) for config in all_configs.values()]
    return ModelsResponseDTO(models=configs)
    

@router.get("/{model_id}", response_model=ModelConfigDTO)
async def read_model_config(model_id: str) -> ModelConfigDTO:
    """Get configuration for a specific model"""
    try:
        logger.info(f"Getting model config for {model_id}")
        config = get_model_config(model_id)
        return ModelConfigDTO(**config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{model_id}", response_model=ModelConfigDTO)
async def update_model_config_api(model_id: str, update: ModelConfigUpdateDTO) -> ModelConfigDTO:
    """Update configuration for a specific model"""
    try:
        logger.info(f"Updating model config for {model_id} with {update.model_dump(exclude_none=True)}")
        config = update_model_config(model_id, update.model_dump(exclude_none=True))
        config["id"] = model_id
        return ModelConfigDTO(**config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{model_id}/overrides", response_model=ModelConfigDTO)
async def reset_model_config_api(model_id: str) -> ModelConfigDTO:
    """Reset model configuration to defaults by removing overrides"""
    try:
        config = reset_model_config(model_id)
        config["id"] = model_id
        return ModelConfigDTO(**config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))