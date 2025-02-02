from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from moatless.actions.action import Action
from moatless.actions.schema import ActionSchema
from moatless.message_history import MessageHistoryGenerator


class ActionConfigDTO(BaseModel):
    action_class: str = Field(..., description="The class name of the action")
    properties: Dict = Field(..., description="The properties of the action")

class AgentConfigDTO(BaseModel):
    """Data transfer object for agent configuration."""

    id: str = Field(..., description="The ID of the configuration")
    system_prompt: str = Field(..., description="System prompt to be used for generating completions")
    actions: List[ActionConfigDTO] = Field(default_factory=list, description="List of action names to enable")


class AgentConfigUpdateDTO(BaseModel):
    """Data transfer object for agent configuration updates."""

    system_prompt: str | None = Field(None, description="System prompt to be used for generating completions")
    actions: List[ActionConfigDTO] = Field(default_factory=list, description="List of action names to enable")

class AgentConfigsResponseDTO(BaseModel):
    """Response containing all agent configurations."""

    configs: List[AgentConfigDTO] = Field(..., description="List of agent configurations")



class ActionsResponseDTO(BaseModel):
    """Response containing all available actions."""

    actions: List[ActionSchema] = Field(..., description="List of available actions")
