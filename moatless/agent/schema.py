
class AgentConfigDTO(BaseModel):
    """Data transfer object for agent configuration."""

    id: str = Field(..., description="The ID of the configuration")
    system_prompt: str = Field(..., description="System prompt to be used for generating completions")
    actions: List[ActionConfigDTO] = Field(default_factory=list, description="List of action names to enable")

