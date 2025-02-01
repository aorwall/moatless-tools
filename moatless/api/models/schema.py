from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from moatless.completion.base import LLMResponseFormat
from moatless.schema import MessageHistoryType


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


class ModelConfigDTO(BaseModel):
    """Full model configuration including base and specific settings"""

    id: str = Field(..., description="Unique identifier for the model")
    model: str = Field(..., description="Model identifier used in LiteLLM")
    model_base_url: Optional[str] = Field(None, description="Base URL for the model API")
    model_api_key: Optional[str] = Field(None, description="API key for the model")
    temperature: Optional[float] = Field(..., description="Temperature for model sampling")
    max_tokens: Optional[int] = Field(..., description="Maximum number of tokens to generate")
    timeout: float = Field(..., description="Timeout in seconds for model requests")
    thoughts_in_action: bool = Field(..., description="Whether to include thoughts in actions")
    disable_thoughts: bool = Field(..., description="Whether to disable thoughts completely")
    merge_same_role_messages: bool = Field(..., description="Whether to merge consecutive messages with same role")
    message_cache: bool = Field(..., description="Whether to enable message caching")
    few_shot_examples: bool = Field(..., description="Whether to use few-shot examples")
    response_format: LLMResponseFormat = Field(..., description="Format for model responses")
    message_history_type: MessageHistoryType = Field(..., description="Type of message history to use")

    class Config:
        """Pydantic config"""

        alias_generator = None
        populate_by_name = True


class ModelsResponseDTO(BaseModel):
    """Response model for listing all models"""

    models: List[ModelConfigDTO] = Field(..., description="List of model configurations")
