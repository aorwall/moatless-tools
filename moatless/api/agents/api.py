"""API endpoints for agent configuration management."""

import logging
from typing import List
from fastapi import APIRouter, HTTPException
from moatless.actions.action import Action
from moatless.actions.schema import ActionSchema
from moatless.agent.agent import ActionAgent
from moatless.config.agent_config import (
    get_agent,
    get_all_agents,
    create_agent,
    update_agent,
    delete_agent,
)
from .schema import (
    ActionConfigDTO,
    AgentConfigDTO,
    AgentConfigUpdateDTO,
    AgentConfigsResponseDTO,
    ActionsResponseDTO,
)

logger = logging.getLogger(__name__)


def _convert_to_dto(agent_id: str, agent: ActionAgent) -> AgentConfigDTO:
    """Convert a config dict to a DTO."""
    actions = [ActionConfigDTO(action_class=action.get_class_name(), properties=action.model_dump(exclude={"action_class"})) for action in agent.actions]
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
    try:
        agent = get_agent(agent_id)
        return _convert_to_dto(agent_id, agent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("", response_model=AgentConfigDTO)
async def create_agent_config_api(config: AgentConfigDTO) -> AgentConfigDTO:
    """Create a new agent configuration"""
    try:
        logger.info(f"Creating agent config with {config.model_dump(exclude_none=True)}")

        agent = _create_agent(config.id, config.actions, config.system_prompt)
        created = create_agent(agent)
        return _convert_to_dto(config.id, created)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{agent_id}")
async def update_agent_config_api(agent_id: str, update: AgentConfigUpdateDTO):
    """Update configuration for a specific agent"""
    try:
        logger.info(f"Updating agent config {agent_id} with {update.model_dump(exclude_none=True)}")
        agent = _create_agent(agent_id, update.actions, update.system_prompt)
        update_agent(agent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{agent_id}")
async def delete_agent_config_api(agent_id: str):
    """Delete an agent configuration"""
    try:
        delete_agent(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _create_agent(agent_id: str, actions: List[ActionConfigDTO], system_prompt: str) -> ActionAgent:
    actions = []
    for action in actions:
        actions.append(Action(action_class=action.action_class, **action.properties))
    return ActionAgent(agent_id=agent_id, actions=actions, system_prompt=system_prompt)
