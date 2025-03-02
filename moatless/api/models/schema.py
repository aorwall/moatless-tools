from typing import Optional

from pydantic import BaseModel, Field

from moatless.completion.manager import ModelConfig


class ModelConfigUpdateDTO(BaseModel):
    """Schema for model configuration updates"""

    model: Optional[str] = Field(None, description="LiteLLM model identifier (e.g. deepseek-chat)")
    model_base_url: Optional[str] = Field(None, description="Base URL for the model API (optional)")
    model_api_key: Optional[str] = Field(
        None,
        description="API key for the model (optional, will use CUSTOM_LLM_API_KEY env var if not set)",
    )
    temperature: Optional[float] = Field(
        None,
        description="Temperature for model sampling - higher values make output more random, lower values more deterministic",
    )
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate in the response")
    timeout: Optional[float] = Field(None, description="Timeout in seconds for model requests")
    thoughts_in_action: Optional[bool] = Field(
        None,
        description="When using tool_call format, include thoughts in the tool call properties instead of content. Useful when model can't mix thoughts and function calls in content",
    )
    disable_thoughts: Optional[bool] = Field(
        None,
        description="Disable thought generation completely. Works better with reasoning models like Claude-1 and Deepseek R1",
    )
    merge_same_role_messages: Optional[bool] = Field(
        None,
        description="Merge consecutive messages with same role. Required for models that only support one message per role",
    )
    messageCache: Optional[bool] = Field(None, description="Whether to enable message caching")
    fewShotExamples: Optional[bool] = Field(
        None,
        description="Include few-shot examples of available actions in the prompt to improve tool usage",
    )
    responseFormat: str = Field(
        ...,
        description="Format for model responses - 'react' uses structured ReACT format with thought/action/params in XML/JSON, 'tool_call' uses function calling",
    )
    messageHistoryType: str = Field(
        ...,
        description="How message history is formatted in completions - 'messages' keeps full message list unchanged, 'react' uses ReACT format with optimized history to reduce tokens",
    )


class ModelsResponseDTO(BaseModel):
    """Response model for listing all models"""

    models: list[ModelConfig] = Field(..., description="List of model configurations")


class BaseModelsResponseDTO(BaseModel):
    """Response model for listing base models"""

    models: list[ModelConfig] = Field(..., description="List of base model configurations")


class AddModelFromBaseDTO(BaseModel):
    """Request model for adding a new model from a base model"""

    base_model_id: str = Field(..., description="ID of the base model to copy from")
    new_model_id: str = Field(..., description="ID for the new model")
    updates: Optional[ModelConfigUpdateDTO] = Field(None, description="Optional configuration updates to apply")


class CreateModelDTO(BaseModel):
    """Request model for creating a new model from scratch"""

    id: str = Field(..., description="ID for the new model")
    model: str = Field(..., description="LiteLLM model identifier (e.g. deepseek-chat)")
    model_base_url: Optional[str] = Field(None, description="Base URL for the model API (optional)")
    model_api_key: Optional[str] = Field(None, description="API key for the model (optional)")
    temperature: float = Field(0.7, description="Temperature for model sampling")
    max_tokens: int = Field(4096, description="Maximum number of tokens to generate")
    timeout: float = Field(120.0, description="Timeout in seconds for model requests")
    thoughts_in_action: bool = Field(False, description="Include thoughts in action steps")
    disable_thoughts: bool = Field(False, description="Disable thought generation completely")
    merge_same_role_messages: bool = Field(False, description="Merge consecutive messages with same role")
    message_cache: bool = Field(True, description="Enable caching of message responses")
    few_shot_examples: bool = Field(True, description="Include few-shot examples in prompts")
    response_format: str = Field(..., description="Format for model responses")
    message_history_type: str = Field(..., description="Type of message history to use")
