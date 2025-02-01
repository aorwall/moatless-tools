"""API endpoints for agent configuration management."""

import logging
from fastapi import APIRouter, HTTPException
from moatless.actions.action import Action
from moatless.agent.agent import ActionAgent
from moatless.config.agent_config import (
    get_config,
    get_all_configs,
    create_config,
    update_config,
    delete_config,
)
from .schema import (
    AgentConfigDTO,
    AgentConfigUpdateDTO,
    AgentConfigsResponseDTO,
    ActionInfoDTO,
    ActionsResponseDTO,
)

logger = logging.getLogger(__name__)


def _convert_to_dto(config_id: str, config: ActionAgent) -> AgentConfigDTO:
    """Convert a config dict to a DTO."""
    actions = [action.name for action in config.actions]
    return AgentConfigDTO(id=config_id, system_prompt=config.system_prompt, actions=actions)


router = APIRouter()


@router.get("/available-actions", response_model=ActionsResponseDTO)
async def list_available_actions() -> ActionsResponseDTO:
    """Get all available actions"""
    try:
        actions = []
        for action in Action.get_available_actions():
            actions.append(
                ActionInfoDTO(
                    name=action["name"],
                    description=action["description"],
                    args_schema=action["args_schema"],
                )
            )
        return ActionsResponseDTO(actions=actions)
    except Exception as e:
        logger.exception(f"Failed to list actions")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=AgentConfigsResponseDTO)
async def list_agent_configs() -> AgentConfigsResponseDTO:
    """Get all agent configurations"""
    configs = []
    all_configs = get_all_configs()
    for config_id, config in all_configs.items():
        configs.append(_convert_to_dto(config_id, config))
    return AgentConfigsResponseDTO(configs=configs)


@router.get("/{config_id}", response_model=AgentConfigDTO)
async def read_agent_config(config_id: str) -> AgentConfigDTO:
    """Get configuration for a specific agent"""
    try:
        config = get_config(config_id)
        return _convert_to_dto(config_id, config)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{config_id}", response_model=AgentConfigDTO)
async def create_agent_config_api(config_id: str, config: AgentConfigUpdateDTO) -> AgentConfigDTO:
    """Create a new agent configuration"""
    try:
        logger.info(f"Creating agent config {config_id} with {config.model_dump(exclude_none=True)}")
        config_dict = config.model_dump(exclude_none=True)
        created = create_config(config_id, config_dict)
        return _convert_to_dto(config_id, created)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{config_id}", response_model=AgentConfigDTO)
async def update_agent_config_api(config_id: str, update: AgentConfigUpdateDTO) -> AgentConfigDTO:
    """Update configuration for a specific agent"""
    try:
        logger.info(f"Updating agent config {config_id} with {update.model_dump(exclude_none=True)}")
        updates = update.model_dump(exclude_none=True)
        updated = update_config(config_id, updates)
        return _convert_to_dto(config_id, updated)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{config_id}")
async def delete_agent_config_api(config_id: str):
    """Delete an agent configuration"""
    try:
        delete_config(config_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
