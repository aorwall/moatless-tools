"""API endpoints for model configuration management."""

import logging

from fastapi import APIRouter, HTTPException

from moatless.completion.base import LLMResponseFormat
from moatless.completion.manager import (
    ModelConfig,
    ModelTestResult,
    _manager,
    add_model_from_base,
    create_model,
    delete_model_config,
    get_all_base_configs,
    get_all_configs,
    get_base_model_config,
    get_model_config,
    update_model_config,
)
from moatless.schema import MessageHistoryType

from .schema import AddModelFromBaseDTO, BaseModelsResponseDTO, CreateModelDTO, ModelConfigUpdateDTO, ModelsResponseDTO

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=ModelsResponseDTO)
async def list_models() -> ModelsResponseDTO:
    """Get all model configurations"""
    configs = get_all_configs()
    return ModelsResponseDTO(models=configs)


@router.get("/base", response_model=BaseModelsResponseDTO)
async def list_base_models() -> BaseModelsResponseDTO:
    """Get all base model configurations"""
    configs = get_all_base_configs()
    return BaseModelsResponseDTO(models=configs)


@router.get("/{model_id}", response_model=ModelConfig)
async def read_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a specific model"""
    try:
        logger.info(f"Getting model config for {model_id}")
        return get_model_config(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/base/{model_id}", response_model=ModelConfig)
async def read_base_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a specific base model"""
    try:
        logger.info(f"Getting base model config for {model_id}")
        return get_base_model_config(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("", response_model=ModelConfig)
async def add_model(request: CreateModelDTO) -> ModelConfig:
    """Add a new model configuration"""
    try:
        logger.info(f"Creating new model {request.id} from scratch")
        # Convert string enums to proper enum types
        request_dict = request.model_dump()
        request_dict["response_format"] = LLMResponseFormat(request_dict["response_format"])
        request_dict["message_history_type"] = MessageHistoryType(request_dict["message_history_type"])
        model_config = ModelConfig(**request_dict)
        return create_model(model_config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/base", response_model=ModelConfig)
async def add_base_model(request: AddModelFromBaseDTO) -> ModelConfig:
    """
    Creating from a base model (when base_model_id is provided)
    """
    try:
        logger.info(f"Adding new model {request.new_model_id} from base {request.base_model_id}")
        updates = request.updates.model_dump() if request.updates else None
        return add_model_from_base(request.base_model_id, request.new_model_id, updates)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{model_id}", response_model=ModelConfig)
async def update_model(model_id: str, updates: ModelConfig) -> ModelConfig:
    """Update configuration for a specific model"""
    try:
        logger.info(f"Updating model config for {model_id}")
        return update_model_config(model_id, updates)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(model_id: str) -> None:
    """Delete a user model configuration"""
    try:
        logger.info(f"Deleting model config for {model_id}")
        delete_model_config(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{model_id}/test", response_model=ModelTestResult)
async def test_model_config(model_id: str) -> ModelTestResult:
    """Test if a model configuration works correctly.

    This endpoint attempts to:
    1. Create a completion model with the configuration
    2. Send a simple test message
    3. Validate the response format

    Returns detailed information about the test results including:
    - Success/failure status
    - Response time
    - Any error information
    - The model's response if available
    """
    try:
        logger.info(f"Testing model configuration for {model_id}")
        return await _manager.test_model_setup(model_id)
    except Exception as e:
        logger.exception(f"Error testing model {model_id}")
        raise HTTPException(status_code=500, detail=f"Failed to test model configuration: {str(e)}")
