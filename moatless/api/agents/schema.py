from moatless.actions.schema import ActionSchema
from moatless.agent.agent import ActionAgent
from moatless.message_history.base import BaseMemory
from pydantic import BaseModel, Field


class AgentConfigsResponseDTO(BaseModel):
    """Response containing all agent configurations."""

    configs: list[dict] = Field(..., description="List of agent configurations")


class ActionsResponseDTO(BaseModel):
    """Response containing all available actions."""

    actions: list[ActionSchema] = Field(..., description="List of available actions")
