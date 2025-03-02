import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import litellm
import tenacity
from litellm import BadRequestError, RateLimitError
from opentelemetry import trace
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from moatless.completion.model import Completion, Usage
from moatless.completion.schema import (
    AllMessageValues,
    ChatCompletionCachedContent,
    ChatCompletionTextObject,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
    ResponseSchema,
)
from moatless.events import BaseEvent, event_bus
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError
from moatless.schema import MessageHistoryType
from moatless.telemetry import set_attribute

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class CompletionEvent(BaseEvent):
    scope: str = "completion"
    model: str


class CompletionRetryEvent(CompletionEvent):
    event_type: str = "retry"
    retry_count: int
    message: str


class CompletionRetryError(Exception):
    """Exception raised when a completion should be retried"""

    def __init__(
        self,
        message: str,
        retry_message: AllMessageValues | None = None,
        retry_messages: list[AllMessageValues] | None = None,
    ):
        super().__init__(message)
        if retry_message:
            self.retry_messages = [retry_message]
        elif retry_messages:
            self.retry_messages = retry_messages
        else:
            self.retry_messages = []


class LLMResponseFormat(str, Enum):
    TOOLS = "tool_call"
    JSON = "json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    REACT = "react"


class CompletionResponse(BaseModel):
    """Container for completion responses that can include multiple structured outputs and text"""

    structured_outputs: list[ResponseSchema] = Field(default_factory=list)
    text_response: Optional[str] = Field(default=None)
    completion: Optional[Completion] = Field(default=None)
    thoughts: Optional[list[dict]] = Field(default=None)
    flags: list[str] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        text: str | None = None,
        output: list[ResponseSchema] | ResponseSchema | None = None,
        completion: Completion | None = None,
    ) -> "CompletionResponse":
        if isinstance(output, ResponseSchema):
            outputs = [output]
        elif isinstance(output, list):
            outputs = output
        else:
            outputs = None

        return cls(text_response=text, structured_outputs=outputs, completion=completion)

    @property
    def structured_output(self) -> Optional[ResponseSchema]:
        """Get the first structured output"""
        if len(self.structured_outputs) > 1:
            ignored_outputs = [output.__class__.__name__ for output in self.structured_outputs[1:]]
            logger.warning(
                f"Multiple structured outputs found in completion response, returning {self.structured_outputs[0].__class__.__name__} and ignoring: {ignored_outputs}"
            )
        return self.structured_outputs[0] if self.structured_outputs else None


ValidationFunction = Callable[
    [list[ResponseSchema], Optional[str], list[str]], Awaitable[tuple[list[ResponseSchema], Optional[str], list[str]]]
]


class BaseCompletionModel(BaseModel, ABC):
    model_id: str = Field(..., description="The ID of the model")
    model: str = Field(..., description="The model to use for completion")
    temperature: Optional[float] = Field(0.0, description="The temperature to use for completion")
    max_tokens: Optional[int] = Field(2000, description="The maximum number of tokens to generate")
    timeout: float = Field(120.0, description="The timeout in seconds for completion requests")
    model_base_url: Optional[str] = Field(default=None, description="The base URL for the model API")
    model_api_key: Optional[str] = Field(default=None, description="The API key for the model", exclude=True)
    response_format: LLMResponseFormat = Field(..., description="The response format expected from the LLM")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata for the completion model")
    message_cache: bool = Field(
        default=True,
        description="Use prompt caching to cache the message history if the LLM supports it",
    )
    thoughts_in_action: bool = Field(
        default=False,
        description="Whether to include thoughts in the action or in the message",
    )
    disable_thoughts: bool = Field(
        default=False,
        description="Whether to disable to use thoughts at all.",
    )
    few_shot_examples: bool = Field(False, description="Whether to use few-shot examples for generating completions")
    message_history_type: MessageHistoryType = Field(
        default=MessageHistoryType.MESSAGES,
        description="The type of message history to use",
    )
    headers: dict[str, Any] = Field(default_factory=dict, description="Additional headers provided to LiteLLM")
    params: dict[str, Any] = Field(default_factory=dict, description="Additional parameters provided to LiteLLM")
    merge_same_role_messages: bool = Field(
        default=False,
        description="Whether to merge messages with the same role into a single message as this is required by models like Deepseek-R1",
    )

    _response_schema: Optional[list[type[ResponseSchema]]] = PrivateAttr(default=None)
    _system_prompt: Optional[str] = PrivateAttr(default=None)
    _few_shot_prompt: Optional[str] = PrivateAttr(default=None)
    _post_validation_fn: Optional[ValidationFunction] = PrivateAttr(default=None)

    _completion_params: Optional[dict[str, Union[str, dict, list]]] = None
    _initialized: bool = False

    def initialize(
        self,
        response_schema: Union[list[type[ResponseSchema]], type[ResponseSchema]],
        system_prompt: str | None = None,
        post_validation_fn: Optional[ValidationFunction] = None,
    ) -> None:
        """Initialize the completion model with response schema and system prompt.

        This method prepares the model for completions by setting up the response schema
        and system prompt, and preparing completion parameters.

        Args:
            response_schema: The schema(s) to validate responses against
            system_prompt: The system prompt to use for completions
            post_validation_fn: Optional function to run after _validate_completion for additional validation

        Raises:
            CompletionRuntimeError: If any schema is not a subclass of ResponseSchema
        """
        if isinstance(response_schema, list):
            schemas = response_schema
        else:
            schemas = [response_schema]

        if not schemas:
            raise CompletionRuntimeError("At least one response schema must be provided")

        if self.response_format == LLMResponseFormat.REACT and self.message_cache:
            self.message_cache = False

        # Validate all schemas are subclasses of ResponseSchema
        for schema in schemas:
            if not issubclass(schema, ResponseSchema):
                raise CompletionRuntimeError(f"Schema {schema.__name__} must be a subclass of ResponseSchema")

        if self._response_schema and self._response_schema != schemas:
            raise ValueError("Response schema cannot be changed after initialization")

        if self._system_prompt and self._system_prompt != system_prompt:
            raise ValueError("System prompt cannot be changed after initialization")

        self._response_schema = schemas
        self._completion_params = self._get_completion_params(self._response_schema)
        self._post_validation_fn = post_validation_fn

        if self.few_shot_examples:
            self._few_shot_prompt = self._generate_few_shot_examples()

        if system_prompt:
            self._system_prompt = self._prepare_system_prompt(system_prompt, self._response_schema)

        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    @tracer.start_as_current_span("BaseCompletionModel.create_completion")
    async def create_completion(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
    ) -> CompletionResponse:
        set_attribute("model", self.model)
        set_attribute("model_id", self.model_id)
        set_attribute("temperature", self.temperature)
        set_attribute("max_tokens", self.max_tokens)
        set_attribute("timeout", self.timeout)
        set_attribute("model_base_url", self.model_base_url)

        if not self._initialized:
            raise ValueError("Model must be initialized with response schema before creating completion")

        if not system_prompt and not self._system_prompt:
            raise ValueError("No system prompt provided")

        if system_prompt:
            system_prompt = self._prepare_system_prompt(system_prompt, self._response_schema)
        else:
            system_prompt = self._system_prompt

        prepared_messages = self._prepare_messages(messages, system_prompt or self._system_prompt)
        return await self._create_completion_with_retries(messages=prepared_messages)

    def _prepare_system_prompt(
        self,
        system_prompt: str,
        response_schema: Union[list[type[ResponseSchema]], type[ResponseSchema]],
    ) -> str:
        """Prepare the system prompt by adding format-specific instructions.

        This method can be overridden by subclasses to add format-specific instructions
        to the system prompt (e.g. JSON schema, tool descriptions, etc).

        Args:
            system_prompt: The base system prompt
            response_schema: The response schema to use for completion

        Returns:
            The modified system prompt with format-specific instructions
        """
        return system_prompt

    def _generate_few_shot_examples(self) -> str:
        """Generate few-shot examples in the model's format.

        Returns:
            Formatted few-shot examples string
        """
        if not self._response_schema:
            return ""

        few_shot_examples = []
        for schema in self._response_schema:
            if hasattr(schema, "get_few_shot_examples"):
                examples = schema.get_few_shot_examples()
                if examples:
                    few_shot_examples.extend(examples)

        if not few_shot_examples:
            return ""

        prompt = "\n\n# Examples\nExamples of how to use the available actions:\n\n"
        return prompt

    def _prepare_messages(self, messages: list[dict], system_prompt: str) -> list[dict]:
        """Prepare messages with system prompt and few-shot examples"""
        messages = messages.copy()

        if self._few_shot_prompt:
            system_prompt = system_prompt + "\n\n" + self._few_shot_prompt

        messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    def _get_completion_params(self, schema: list[type[ResponseSchema]]) -> dict[str, Union[str, dict, list]]:
        """Get format-specific parameters for the LLM API call.

        This method configures how the LLM should structure its response by providing
        format-specific parameters to the API call. These parameters ensure the LLM
        outputs in the correct format (JSON, ReAct, Tool calls, etc).

        Args:
            schema: The response schema to use for completion

        Returns:
            A dictionary of parameters to pass to the LLM API call. Common keys include:
            - response_format: Dict specifying the response structure
            - tools: List of available tools/functions
            - tool_choice: How tools should be selected
            - function_call: Function call configuration
            -
        """
        return {}

    @tracer.start_as_current_span("BaseCompletionModel._create_completion_with_retries")
    async def _create_completion_with_retries(
        self,
        messages: list[dict],
    ) -> CompletionResponse:
        """Execute completion with retries for validation errors"""
        retry_count = 0
        accumulated_usage = Usage()
        completion_response = None

        @tenacity.retry(
            retry=tenacity.retry_if_exception_type(CompletionRetryError),
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_fixed(0),
            reraise=True,
            before_sleep=lambda retry_state: logger.info(f"Retrying completion with {len(messages)} messages"),
        )
        async def _do_completion_with_validation():
            nonlocal retry_count, accumulated_usage, completion_response
            retry_count += 1

            # Execute completion and get raw response
            completion_response = await self._execute_completion(messages)

            # Track usage from this attempt regardless of validation outcome
            usage = Usage.from_completion_response(completion_response, self.model)
            if usage:
                accumulated_usage += usage
                set_attribute("prompt_tokens", usage.prompt_tokens)
                set_attribute("completion_tokens", usage.completion_tokens)
                set_attribute("cache_read_tokens", usage.cache_read_tokens)
                set_attribute("cache_write_tokens", usage.cache_write_tokens)
                set_attribute("completion_cost", usage.completion_cost)
            else:
                logger.warning(f"No usage found for completion response: {completion_response}")

            if not completion_response.choices or (
                not completion_response.choices[0].message.content
                and not completion_response.choices[0].message.tool_calls
            ):
                logger.error(f"Completion response is empty: {completion_response.model_dump_json(indent=2)}")
                raise CompletionRejectError(
                    "Completion response is empty",
                    messages=messages,
                    last_completion=completion_response,
                    accumulated_usage=accumulated_usage,
                )

            try:
                # Validate the response - may raise CompletionRetryError
                structured_outputs, text_response, flags, thoughts = await self._validate_completion(
                    completion_response=completion_response,
                )

                # Run post validation if provided and raise CompletionRetryError if it fails
                if self._post_validation_fn:
                    structured_outputs, text_response, flags = await self._post_validation_fn(
                        structured_outputs, text_response, flags
                    )
            except CompletionRetryError as e:
                await self._send_retry_event(retry_count, str(e))

                tool_call_id = None
                if (
                    hasattr(completion_response.choices[0].message, "thinking")
                    and completion_response.choices[0].message.thinking
                ):
                    content = []
                    content.extend(completion_response.choices[0].message.thinking)
                    if completion_response.choices[0].message.content:
                        content.append(
                            ChatCompletionTextObject(type="text", text=completion_response.choices[0].message.content)
                        )

                    message_dump = completion_response.choices[0].message.model_dump()
                    message_dump["content"] = content
                    messages.append(message_dump)
                else:
                    if completion_response.choices[0].message.tool_calls:
                        # TODO: Support multiple tool calls
                        tool_call_id = completion_response.choices[0].message.tool_calls[0].id

                    messages.append(completion_response.choices[0].message.model_dump())

                logger.warning(f"Post validation failed with retry messages: {e.retry_messages}")
                if e.retry_messages:
                    messages.extend(e.retry_messages)
                else:
                    if tool_call_id:
                        messages.append(
                            ChatCompletionToolMessage(role="tool", content=str(e), tool_call_id=tool_call_id)
                        )
                    else:
                        messages.append(ChatCompletionUserMessage(role="user", content=str(e)))
                raise e

            response_dict = completion_response.model_dump()

            completion = Completion.from_llm_completion(
                input_messages=messages,
                completion_response=response_dict,
                model=self.model,
                retries=retry_count,
                usage=accumulated_usage,  # Use accumulated usage here
                flags=flags,
            )

            return CompletionResponse(
                structured_outputs=structured_outputs or [],
                text_response=text_response,
                completion=completion,
                flags=flags or [],
                thoughts=thoughts,
            )

        try:
            return await _do_completion_with_validation()
        except CompletionRetryError as e:
            logger.warning(
                f"Completion failed after {retry_count} retries. Exception: {e}. Completion response: {completion_response}"
            )
            raise CompletionRejectError(
                f"Completion failed after {retry_count} retries. Exception: {e}",
                messages=messages,
                last_completion=completion_response.model_dump() if completion_response else None,
                accumulated_usage=accumulated_usage,
            ) from e
        except CompletionRejectError as e:
            raise e
        except CompletionRuntimeError as e:
            raise e
        except Exception as e:
            logger.error(f"Completion failed after {retry_count} retries. Exception: {e}. Type: {type(e)}")
            raise CompletionRuntimeError(
                f"Completion failed after {retry_count} retries. Exception: {e}. Type: {type(e)}",
                messages=messages,
                last_completion=completion_response.model_dump() if completion_response else None,
                accumulated_usage=accumulated_usage,
            ) from e

    @tracer.start_as_current_span("BaseCompletionModel._execute_completion")
    async def _execute_completion(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """Execute a single completion attempt with LiteLLM.

        This method:
        1. Makes the API call through LiteLLM
        2. Returns the raw response or raises appropriate exceptions

        Args:
            messages: The conversation history

        Returns:
            Raw completion response from the LLM

        Raises:
            CompletionRuntimeError: For provider errors
        """
        params = self._completion_params.copy()
        if self.params:
            params.update(self.params)

        if self.merge_same_role_messages:
            messages = self._merge_same_role_messages(messages)

        @retry(
            retry=tenacity.retry_if_not_exception_type((BadRequestError, CompletionRuntimeError)),
            wait=wait_exponential(multiplier=5, min=5, max=60),
            stop=stop_after_attempt(3),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"Rate limited by provider, retrying in {retry_state.next_action.sleep} seconds"
            ),
        )
        async def _do_completion_with_rate_limit_retry():
            try:
                if "claude-3-" in self.model:
                    self._inject_prompt_caching(messages)

                if self.model_base_url:
                    params["api_base"] = self.model_base_url
                if self.model_api_key:
                    params["api_key"] = self.model_api_key

                if self.headers:
                    params["headers"] = self.headers

                return await litellm.acompletion(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=messages,
                    metadata=self.metadata or {},
                    timeout=self.timeout,
                    **params,
                )

            except BadRequestError as e:
                if e.response:
                    response_text = e.response.text
                else:
                    response_text = None

                logger.exception(f"LiteLLM completion failed. Model: {self.model}, " f"Response: {response_text}")

                raise CompletionRuntimeError(message=str(e), messages=messages) from e
            except RateLimitError:
                raise  # Let tenacity handle the retry
            except Exception as e:
                logger.exception(f"LiteLLM completion failed. Model: {self.model}, " f"Error: {e}")

                raise CompletionRuntimeError(message=str(e), messages=messages) from e

        return await _do_completion_with_rate_limit_retry()

    def _get_schema_names(self):
        return [schema.__name__ for schema in self._response_schema] if self._response_schema else ["None"]

    def _inject_prompt_caching(self, messages: list[dict[str, str]]) -> None:
        """Set cache breakpoints for Claude 3.5 message history.

        This method:
        1. Marks the system prompt for caching
        2. Sets cache breakpoints for the 3 most recent turns

        Args:
            messages: The conversation history to inject caching into
        """
        if messages[0]["role"] == "system":
            messages[0]["cache_control"] = ChatCompletionCachedContent(type="ephemeral")

        if not self.message_cache:
            return

        breakpoints_remaining = 3
        for message in reversed(messages):
            try:
                if isinstance(message.get("content"), list):
                    content = next((m for m in message["content"] if m.get("type") == "text"), None)
                elif message.get("role") == "assistant" and not message.get("content") and message.get("tool_calls"):
                    content = message["tool_calls"][-1]
                else:
                    content = message

                if content:
                    if breakpoints_remaining:
                        content["cache_control"] = ChatCompletionCachedContent(type="ephemeral")
                        breakpoints_remaining -= 1
                    elif "cache_control" in content:
                        del content["cache_control"]

                content = None
            except Exception:
                logger.exception(f"Error injecting prompt caching on message: {message}")

    @abstractmethod
    async def _validate_completion(
        self, completion_response: Any
    ) -> tuple[list[ResponseSchema], Optional[str], list[str], list[dict]]:
        """Validate and transform the LLM's response into a structured format.

        This method is responsible for:
        1. Extracting the relevant content from the LLM response
        2. Validating it matches the expected format
        3. Converting it into structured data using the response schema
        4. Handling any format-specific validation rules

        Args:
            completion_response: The raw response from the LLM API

        Returns:
            Tuple of:
            - List of validated ResponseSchema instances
            - Optional text response string
            - List of flags indicating any special conditions
            - List of ThoughtBlock instances
        Raises:
            CompletionRejectError: If the response fails validation and should be retried
            CompletionRuntimeError: If the response indicates a fundamental problem
        """
        raise NotImplementedError

    async def _send_retry_event(self, retry_count: int, message: str):
        logger.info(f"Retrying completion for model: {self.model}, retry count: {retry_count}, message: {message}")
        event = CompletionRetryEvent(
            model=self.model,
            retry_count=retry_count,
            message=message,
        )
        await event_bus.publish(event)

    def clone(self, **kwargs) -> "BaseCompletionModel":
        """Create a copy of the completion model with optional parameter overrides."""
        model_data = self.model_dump()
        model_data.update(kwargs)
        return BaseCompletionModel.model_validate(model_data)

    def model_dump(self, **kwargs):
        dump = super().model_dump(**kwargs)
        if "model_api_key" in dump:
            dump["model_api_key"] = None
        if "response_format" in dump:
            dump["response_format"] = dump["response_format"].value
        if "message_history_type" in dump:
            dump["message_history_type"] = dump["message_history_type"].value
        return dump

    @classmethod
    def create(cls, response_format: LLMResponseFormat, **kwargs):
        if response_format == LLMResponseFormat.REACT:
            from moatless.completion.react import ReActCompletionModel

            return ReActCompletionModel(response_format=response_format, **kwargs)
        elif response_format == LLMResponseFormat.TOOLS:
            from moatless.completion.tool_call import ToolCallCompletionModel

            return ToolCallCompletionModel(response_format=response_format, **kwargs)
        elif response_format == LLMResponseFormat.JSON:
            from moatless.completion.json import JsonCompletionModel

            return JsonCompletionModel(response_format=response_format, **kwargs)
        else:
            raise ValueError(f"Unknown response format: {response_format}")

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and "response_format" in obj:
            if isinstance(obj["response_format"], str):
                obj["response_format"] = LLMResponseFormat(obj["response_format"])
            if isinstance(obj["message_history_type"], str):
                obj["message_history_type"] = MessageHistoryType(obj["message_history_type"])
            return cls.create(**obj)

        return obj

    @model_validator(mode="after")
    def set_api_key(self) -> "BaseCompletionModel":
        """Update the model with the API key from env vars if model base URL is set but API key is not"""
        if self.model_base_url and not self.model_api_key:
            self.model_api_key = os.getenv("CUSTOM_LLM_API_KEY")
        return self

    def _merge_same_role_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Merge consecutive messages with the 'user' role into a single message.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            List of merged messages where consecutive user messages are combined
        """
        if not messages:
            return messages

        merged = []
        current_content: list[str] = []

        for message in messages:
            if message["role"] == "user":
                # User message - accumulate content
                if isinstance(message["content"], list):
                    # Handle list of content blocks
                    for content_block in message["content"]:
                        if isinstance(content_block, dict) and content_block["type"] == "text":
                            current_content.append(content_block["text"])
                        else:
                            # For non-text content blocks, flush current content and add message as-is
                            if current_content:
                                merged.append(
                                    {
                                        "role": "user",
                                        "content": "\n".join(current_content),
                                    }
                                )
                                current_content = []
                            merged.append(message)
                            break
                else:
                    # String content
                    current_content.append(message["content"])
            else:
                # Non-user message - flush any accumulated user content first
                if current_content:
                    merged.append({"role": "user", "content": "\n".join(current_content)})
                    current_content = []

                # Add non-user message as-is
                merged.append(message)

        # Add final user message if exists
        if current_content:
            merged.append({"role": "user", "content": "\n".join(current_content)})

        return merged

    def __str__(self) -> str:
        """Return a nice string representation of the completion model."""
        components = []

        components.extend(
            [
                f"Model: {self.model}",
                f"Response format: {self.response_format.value}",
                f"Temperature: {self.temperature}",
                f"Max tokens: {self.max_tokens}",
                f"Timeout: {self.timeout}s",
            ]
        )

        if self.model_base_url:
            components.append(f"Base URL: {self.model_base_url}")

        if self.metadata:
            components.append(f"Metadata: {self.metadata}")

        components.extend(
            [
                f"Message cache: {self.message_cache}",
                f"Thoughts in action: {self.thoughts_in_action}",
                f"Disable thoughts: {self.disable_thoughts}",
                f"Few shot examples: {self.few_shot_examples}",
                f"Message history type: {self.message_history_type.value}",
                f"Merge same role messages: {self.merge_same_role_messages}",
            ]
        )

        if self._response_schema:
            schema_names = [schema.__name__ for schema in self._response_schema]
            components.append(f"Response schemas: {', '.join(schema_names)}")

        return f"Completion Model '{self.model}':\n" + "\n".join(f"- {c}" for c in components)
