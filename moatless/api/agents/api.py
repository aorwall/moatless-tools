"""API endpoints for agent configuration management."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from moatless.actions.action import Action
from moatless.agent.agent import ActionAgent
from moatless.agent.manager import AgentConfigManager
from moatless.api.dependencies import get_agent_manager

from .schema import (
    ActionConfigDTO,
    ActionsResponseDTO,
    AgentConfigDTO,
    AgentConfigsResponseDTO,
    AgentConfigUpdateDTO,
)

logger = logging.getLogger(__name__)


def _convert_to_dto(agent_id: str, agent: ActionAgent) -> AgentConfigDTO:
    """Convert a config dict to a DTO."""
    actions = [
        ActionConfigDTO(title=action.name, properties=action.model_dump(exclude={"action_class"}))
        for action in agent.actions
    ]
    return AgentConfigDTO(agent_id=agent_id, system_prompt=agent.system_prompt, actions=actions, memory=agent.memory)


router = APIRouter()


@router.get("/available-actions", response_model=ActionsResponseDTO)
async def list_available_actions():
    """Get all available actions with their schema."""
    actions = Action.get_available_actions()
    return ActionsResponseDTO(actions=actions)


@router.get("", response_model=AgentConfigsResponseDTO)
async def list_agent_configs(agent_manager: AgentConfigManager = Depends(get_agent_manager)) -> AgentConfigsResponseDTO:
    """Get all agent configurations"""
    configs = []
    all_agents = agent_manager.get_all_agents()
    for agent in all_agents:
        configs.append(_convert_to_dto(agent.agent_id, agent))
    return AgentConfigsResponseDTO(configs=configs)


@router.get("/{agent_id}")
async def read_agent_config(agent_id: str, agent_manager: AgentConfigManager = Depends(get_agent_manager)):
    """Get configuration for a specific agent"""
    agent = agent_manager.get_agent(agent_id)
    return agent.model_dump(exclude_none=True)


@router.post("")
async def create_agent_config_api(config: dict, agent_manager: AgentConfigManager = Depends(get_agent_manager)) -> dict:
    """Create a new agent configuration"""
    # Manually validate the config
    agent_config = ActionAgent.model_validate(config)
    logger.info(f"Creating agent config with {len(agent_config.actions)} actions")

    created = await agent_manager.create_agent(agent_config)
    # Return a dict representation instead of the object
    return created.model_dump(exclude_none=True)


@router.put("/{agent_id}")
async def update_agent_config_api(
    agent_id: str, update: dict, agent_manager: AgentConfigManager = Depends(get_agent_manager)
):
    """Update configuration for a specific agent"""
    # Manually validate the update
    agent_update = ActionAgent.model_validate(update)
    logger.info(f"Updating agent config {agent_id} with {len(agent_update.actions)} actions")
    await agent_manager.update_agent(agent_update)
    return {"status": "success"}


@router.delete("/{agent_id}")
async def delete_agent_config_api(agent_id: str, agent_manager: AgentConfigManager = Depends(get_agent_manager)):
    """Delete an agent configuration"""
    await agent_manager.delete_agent(agent_id)
