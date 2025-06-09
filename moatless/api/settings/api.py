"""API endpoints for model configuration management."""

import logging

from fastapi import APIRouter, HTTPException, Depends
from moatless.actions.action import Action
from moatless.discriminator.base import BaseDiscriminator
from moatless.expander.expander import Expander
from moatless.api.dependencies import get_flow_manager
from moatless.artifacts.artifact import ArtifactHandler
from moatless.completion.base import BaseCompletionModel
from moatless.feedback.base import BaseFeedbackGenerator
from moatless.flow.manager import FlowManager
from moatless.flow.flow import AgenticFlow
from moatless.message_history.base import BaseMemory
from moatless.selector.base import BaseSelector
from moatless.value_function.base import BaseValueFunction

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/flows", response_model=list[dict])
async def list_flow_configs(flow_manager: FlowManager = Depends(get_flow_manager)) -> list[dict]:
    """Get all flow configurations"""
    flows = await flow_manager.get_all_configs()
    flow_dicts = [flow.model_dump(exclude_none=True, exclude_unset=True) for flow in flows]
    return flow_dicts


@router.post("/flows")
async def create_flow_config_api(config: dict, flow_manager: FlowManager = Depends(get_flow_manager)):
    """
    Create a new flow configuration.
    """
    logger.info(f"Creating flow config {config['id']}")
    return await flow_manager.create_config(config)


@router.get("/flows/{config_id}", response_model=dict)
async def read_flow_config(config_id: str, flow_manager: FlowManager = Depends(get_flow_manager)) -> dict:
    """
    Get a specific flow configuration.
    """
    logger.info(f"Getting flow config for {config_id}")
    flow = await flow_manager.get_flow_config(config_id)
    return flow.model_dump(exclude_none=True, exclude_unset=True)


@router.put("/flows/{config_id}")
async def update_flow_config_api(config_id: str, update: dict, flow_manager: FlowManager = Depends(get_flow_manager)):
    """
    Update a flow configuration.
    """
    # Ensure the ID in the URL matches the ID in the request body
    if config_id != update["id"]:
        raise ValueError(f"ID in URL ({config_id}) doesn't match ID in request body ({update['id']})")

    await flow_manager.update_config(update)


@router.delete("/flows/{config_id}")
async def delete_flow_config_api(config_id: str, flow_manager: FlowManager = Depends(get_flow_manager)):
    """
    Delete a flow configuration.
    """
    logger.info(f"Deleting flow config {config_id}")
    await flow_manager.delete_config(config_id)
    return {"status": "success", "message": f"Flow configuration {config_id} deleted successfully"}


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


@router.get("/components/expanders")
async def get_available_expanders():
    """Get all available expander components"""
    components = Expander.get_available_components()
    return {name: comp.model_json_schema() for name, comp in components.items()}


@router.get("/components/memory")
async def get_available_memory():
    """Get all available memory components"""
    components = BaseMemory.get_available_components()
    return {name: comp.model_json_schema() for name, comp in components.items()}


@router.get("/components/actions")
async def get_available_actions():
    """Get all available action components"""
    components = Action.get_available_components()
    return {name: comp.model_json_schema() for name, comp in components.items()}


# Add a new generic endpoint to get components by type
@router.get("/components/{component_type}")
async def get_components_by_type(component_type: str):
    """Get all available components of a specific type

    Component types:
    - selectors
    - value-functions
    - feedback-generators
    - artifact-handlers
    - memory
    - actions
    - discriminators
    - expanders
    """
    component_map = {
        "selectors": BaseSelector,
        "value-functions": BaseValueFunction,
        "feedback-generators": BaseFeedbackGenerator,
        "artifact-handlers": ArtifactHandler,
        "memory": BaseMemory,
        "actions": Action,
        "completion_model": BaseCompletionModel,
        "discriminators": BaseDiscriminator,
        "expanders": Expander,
    }

    if component_type not in component_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid component type: {component_type}. Available types: {list(component_map.keys())}",
        )

    base_class = component_map[component_type]
    components = base_class.get_available_components()
    return {name: comp.model_json_schema() for name, comp in components.items()}


# Add endpoint to get all components at once
@router.get("/components")
async def get_all_components():
    """Get all available components grouped by type"""
    component_map = {
        "selectors": BaseSelector,
        "value-functions": BaseValueFunction,
        "feedback-generators": BaseFeedbackGenerator,
        "discriminators": BaseDiscriminator,
        "artifact-handlers": ArtifactHandler,
        "memory": BaseMemory,
        "actions": Action,
        "completion_model": BaseCompletionModel,
        "expanders": Expander,
    }

    result = {}
    for component_type, base_class in component_map.items():
        components = base_class.get_available_components()
        result[component_type] = {name: comp.model_json_schema() for name, comp in components.items()}

    return result
