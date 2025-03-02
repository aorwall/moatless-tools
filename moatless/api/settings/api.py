"""API endpoints for model configuration management."""

import logging

from fastapi import APIRouter, HTTPException

from moatless.actions.action import Action
from moatless.artifacts.artifact import ArtifactHandler
from moatless.feedback.base import BaseFeedbackGenerator
from moatless.flow.manager import create_flow_config, get_all_configs, get_flow_config, update_flow_config
from moatless.flow.schema import FlowConfig
from moatless.selector.base import BaseSelector
from moatless.value_function.base import BaseValueFunction

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/flows", response_model=list[FlowConfig])
async def list_flow_configs() -> list[FlowConfig]:
    """Get all flow configurations"""
    return get_all_configs()


@router.post("/flows", response_model=FlowConfig)
async def create_flow_config_api(config: FlowConfig) -> FlowConfig:
    """Create a new flow configuration"""
    try:
        logger.info(f"Creating flow config {config.id} with {config.model_dump(exclude_none=True)}")
        return create_flow_config(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/flows/{config_id}", response_model=FlowConfig)
async def read_flow_config(config_id: str) -> FlowConfig:
    """Get configuration for a specific flow"""
    try:
        logger.info(f"Getting flow config for {config_id}")
        return get_flow_config(config_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/flows/{config_id}")
async def update_flow_config_api(config_id: str, update: FlowConfig):
    """Update configuration for a specific flow"""
    try:
        if config_id != update.id:
            raise HTTPException(status_code=400, detail="Config ID in path must match config ID in body")

        logger.info(f"Updating flow config {config_id} with {update.model_dump(exclude_none=True)}")
        update_flow_config(update)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/components/selectors")
async def get_available_selectors():
    """Get all available selector components"""
    components = BaseSelector.get_available_components()
    return {name: comp.model_json_schema() for name, comp in components.items()}


@router.get("/components/value-functions")
async def get_available_value_functions():
    """Get all available value function components"""
    components = BaseValueFunction.get_available_components()
    response = {name: comp.model_json_schema() for name, comp in components.items()}
    return response


@router.get("/components/feedback-generators")
async def get_available_feedback_generators():
    """Get all available feedback generator components"""
    components = BaseFeedbackGenerator.get_available_components()
    return {name: comp.model_json_schema() for name, comp in components.items()}


@router.get("/components/artifact-handlers")
async def get_available_artifact_handlers():
    """Get all available artifact handler components"""
    components = ArtifactHandler.get_available_components()
    return {name: comp.model_json_schema() for name, comp in components.items()}


@router.get("/components/actions")
async def get_available_actions():
    """Get all available action components"""
    components = Action.get_available_components()

    return [action_class.get_action_schema() for action_class in components.values()]
