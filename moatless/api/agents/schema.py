from typing import Dict, List
from pydantic import BaseModel, Field
from moatless.actions.action import Action
from moatless.message_history import MessageHistoryGenerator


class AgentConfigDTO(BaseModel):
    """Data transfer object for agent configuration."""

    id: str = Field(..., description="The ID of the configuration")
    system_prompt: str = Field(..., description="System prompt to be used for generating completions")
    use_few_shots: bool = Field(False, description="Whether to use few-shot examples for generating completions")
    disable_thoughts: bool = Field(False, description="Whether to disable thoughts in the action")
    actions: List[str] = Field(default_factory=list, description="List of action names to enable")
    message_generator: Dict = Field(default_factory=dict, description="Configuration for message history generator")


class AgentConfigUpdateDTO(BaseModel):
    """Data transfer object for agent configuration updates."""

    system_prompt: str | None = Field(None, description="System prompt to be used for generating completions")
    use_few_shots: bool | None = Field(None, description="Whether to use few-shot examples for generating completions")
    disable_thoughts: bool | None = Field(None, description="Whether to disable thoughts in the action")
    actions: List[str] | None = Field(None, description="List of action names to enable")
    message_generator: Dict | None = Field(None, description="Configuration for message history generator")


class AgentConfigsResponseDTO(BaseModel):
    """Response containing all agent configurations."""

    configs: List[AgentConfigDTO] = Field(..., description="List of agent configurations")


class ActionInfoDTO(BaseModel):
    """Information about an available action."""

    name: str = Field(..., description="Name of the action")
    description: str = Field(..., description="Description from the action's docstring")
    args_schema: Dict = Field(..., description="Schema for the action's arguments")


class ActionsResponseDTO(BaseModel):
    """Response containing all available actions."""

    actions: List[ActionInfoDTO] = Field(..., description="List of available actions")
