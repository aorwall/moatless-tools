from pydantic import BaseModel, Field

from moatless.actions.schema import ActionSchema
from moatless.message_history.base import BaseMemory


class ActionConfigDTO(BaseModel):
    title: str = Field(..., description="The name of the action")
    properties: dict = Field(..., description="The properties of the action")


class AgentConfigDTO(BaseModel):
    """Data transfer object for agent configuration."""

    agent_id: str = Field(..., description="The ID of the configuration")
    system_prompt: str = Field(..., description="System prompt to be used for generating completions")
    actions: list[ActionConfigDTO] = Field(default_factory=list, description="List of action names to enable")
    memory: BaseMemory | None = Field(
        None, description="Message history generator to be used for generating completions"
    )


class AgentConfigUpdateDTO(BaseModel):
    """Data transfer object for agent configuration updates."""

    system_prompt: str | None = Field(None, description="System prompt to be used for generating completions")
    actions: list[ActionConfigDTO] = Field(default_factory=list, description="List of action names to enable")
    memory: BaseMemory | None = Field(
        None, description="Message history generator to be used for generating completions"
    )


class AgentConfigsResponseDTO(BaseModel):
    """Response containing all agent configurations."""

    configs: list[AgentConfigDTO] = Field(..., description="List of agent configurations")


class ActionsResponseDTO(BaseModel):
    """Response containing all available actions."""

    actions: list[ActionSchema] = Field(..., description="List of available actions")
