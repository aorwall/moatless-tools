"""API endpoints for agent configuration management."""

import logging

from fastapi import APIRouter

from moatless.actions.action import Action
from moatless.agent.agent import ActionAgent
from moatless.config.agent_config import (
    create_agent,
    delete_agent,
    get_agent,
    get_all_agents,
    update_agent,
)
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
    return AgentConfigDTO(id=agent_id, system_prompt=agent.system_prompt, actions=actions)


router = APIRouter()


@router.get("/available-actions", response_model=ActionsResponseDTO)
async def list_available_actions():
    """Get all available actions with their schema."""
    actions = Action.get_available_actions()
    return ActionsResponseDTO(actions=actions)


@router.get("", response_model=AgentConfigsResponseDTO)
async def list_agent_configs() -> AgentConfigsResponseDTO:
    """Get all agent configurations"""
    configs = []
    all_agents = get_all_agents()
    for agent in all_agents:
        configs.append(_convert_to_dto(agent.agent_id, agent))
    return AgentConfigsResponseDTO(configs=configs)


@router.get("/{agent_id}", response_model=AgentConfigDTO)
async def read_agent_config(agent_id: str) -> AgentConfigDTO:
    """Get configuration for a specific agent"""
    agent = get_agent(agent_id)
    return _convert_to_dto(agent_id, agent)


@router.post("", response_model=AgentConfigDTO)
async def create_agent_config_api(config: AgentConfigDTO) -> AgentConfigDTO:
    """Create a new agent configuration"""
    logger.info(f"Creating agent config with {len(config.actions)} actions")

    agent = _create_agent(agent_id=config.id, action_configs=config.actions, system_prompt=config.system_prompt)
    created = create_agent(agent)
    return _convert_to_dto(config.id, created)


@router.put("/{agent_id}")
async def update_agent_config_api(agent_id: str, update: AgentConfigUpdateDTO):
    """Update configuration for a specific agent"""
    logger.info(f"Updating agent config {agent_id} with {len(update.actions)} actions")
    agent = _create_agent(agent_id=agent_id, action_configs=update.actions, system_prompt=update.system_prompt)
    update_agent(agent)


@router.delete("/{agent_id}")
async def delete_agent_config_api(agent_id: str):
    """Delete an agent configuration"""
    delete_agent(agent_id)


def _create_agent(agent_id: str, action_configs: list[ActionConfigDTO], system_prompt: str) -> ActionAgent:
    actions = []
    for action_config in action_configs:
        actions.append(Action.model_validate(action_config.properties))
    logger.info(f"Created actions: {actions}")
    return ActionAgent(agent_id=agent_id, actions=actions, system_prompt=system_prompt)
