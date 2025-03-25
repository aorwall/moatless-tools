"""API endpoints for model configuration management."""

import logging

from fastapi import APIRouter, HTTPException, Depends
from moatless.api.dependencies import get_model_manager
from moatless.completion.manager import ModelTestResult, ModelConfigManager, BaseCompletionModel

from .schema import ModelsResponseDTO

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("")
async def list_models(model_manager: ModelConfigManager = Depends(get_model_manager)):
    """Get all model configurations"""
    try:
        configs = model_manager.get_all_configs()
        configs = [model.model_dump() for model in configs]
        return {"models": configs}
    except Exception as e:
        logger.exception(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/base")
async def list_base_models(model_manager: ModelConfigManager = Depends(get_model_manager)):
    """Get all base model configurations"""
    try:
        configs = model_manager.get_all_base_configs()
        configs = [model.model_dump() for model in configs]
        return {"models": configs}
    except Exception as e:
        logger.exception(f"Error listing base models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list base models: {str(e)}")


@router.get("/{model_id}")
async def read_model_config(model_id: str, model_manager: ModelConfigManager = Depends(get_model_manager)):
    """Get configuration for a specific model"""
    try:
        logger.info(f"Getting model config for {model_id}")
        return model_manager.get_model_config(model_id).model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/base/{model_id}")
async def read_base_model_config(model_id: str, model_manager: ModelConfigManager = Depends(get_model_manager)):
    """Get configuration for a specific base model"""
    try:
        logger.info(f"Getting base model config for {model_id}")
        return model_manager.get_base_model_config(model_id).model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("")
async def add_model(request: dict, model_manager: ModelConfigManager = Depends(get_model_manager)):
    """Add a new model configuration"""
    try:
        logger.info(f"Creating new model {request['id']} from scratch")
        # Convert string enums to proper enum types

        model_config = BaseCompletionModel.model_validate(request)
        new_model = await model_manager.create_model(model_config)
        return new_model.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/base")
async def add_base_model(request: dict, model_manager: ModelConfigManager = Depends(get_model_manager)):
    """
    Creating from a base model (when base_model_id is provided)
    """
    try:
        logger.info(f"Adding new model {request['new_model_id']} from base {request['base_model_id']}")
        updates = request.get("updates")
        new_model = await model_manager.add_model_from_base(request["base_model_id"], request["new_model_id"], updates)
        return new_model.model_dump()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{model_id}")
async def update_model(model_id: str, request: dict, model_manager: ModelConfigManager = Depends(get_model_manager)):
    """Update configuration for a specific model"""
    try:
        logger.info(f"Updating model config for {model_id}")
        updated_model = await model_manager.update_model_config(model_id, BaseCompletionModel.model_validate(request))
        return updated_model.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(model_id: str, model_manager: ModelConfigManager = Depends(get_model_manager)):
    """Delete a user model configuration"""
    try:
        logger.info(f"Deleting model config for {model_id}")
        await model_manager.delete_model_config(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{model_id}/test")
async def test_model_config(model_id: str, model_manager: ModelConfigManager = Depends(get_model_manager)):
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
        result = await model_manager.test_model_setup(model_id)
        return ModelTestResult.model_validate(result) if isinstance(result, dict) else result
    except Exception as e:
        logger.exception(f"Error testing model {model_id}")
        raise HTTPException(status_code=500, detail=f"Failed to test model configuration: {str(e)}")
