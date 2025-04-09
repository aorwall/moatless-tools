import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, List, Optional, Union

import litellm
import tenacity
from litellm.exceptions import BadRequestError
from litellm.files.main import RateLimitError
from openai import APIConnectionError
from opentelemetry import trace
from pydantic import BaseModel, Field, PrivateAttr
from tenacity import retry, stop_after_attempt, wait_exponential

from moatless.completion.schema import (
    AllMessageValues,
    ChatCompletionCachedContent,
    ChatCompletionTextObject,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
    ResponseSchema,
)
from moatless.completion.stats import CompletionAttempt, CompletionInvocation
from moatless.component import MoatlessComponent
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


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
    thought: Optional[str] = Field(default=None, description="Thought process from ReAct or similar models")
    completion_invocation: Optional[CompletionInvocation] = Field(
        default=None, description="The complete invocation data including all attempts"
    )

    @property
    def structured_output(self) -> Optional[ResponseSchema]:
        """Get the first structured output"""
        if len(self.structured_outputs) > 1:
            ignored_outputs = [output.__class__.__name__ for output in self.structured_outputs[1:]]
            logger.warning(
                f"Multiple structured outputs found in completion response, returning {self.structured_outputs[0].__class__.__name__} and ignoring: {ignored_outputs}"
            )
        return self.structured_outputs[0] if self.structured_outputs else None

    @property
    def completion_attempts(self) -> List[CompletionAttempt]:
        """Get the completion attempts from the invocation (for backward compatibility)"""
        return self.completion_invocation.attempts if self.completion_invocation else []


ValidationFunction = Callable[
    [list[ResponseSchema], Optional[str]], Awaitable[tuple[list[ResponseSchema], Optional[str]]]
]


class BaseCompletionModel(MoatlessComponent, ABC):
    model_id: Optional[str] = Field(default=None, description="The ID of the model")
    model: str = Field(..., description="The model to use for completion")
    temperature: Optional[float] = Field(0.0, description="The temperature to use for completion")
    max_tokens: Optional[int] = Field(2000, description="The maximum number of tokens to generate")
    timeout: float = Field(120.0, description="The timeout in seconds for completion requests")
    model_base_url: Optional[str] = Field(default=None, description="The base URL for the model API")
    model_api_key: Optional[str] = Field(default=None, description="The API key for the model")
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

    @classmethod
    def get_component_type(cls) -> str:
        return "completion_model"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.completion"

    @classmethod
    def _get_base_class(cls) -> type:
        return BaseCompletionModel

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
            
        if self.model_base_url:
            self._completion_params["api_base"] = self.model_base_url

        if self.model_api_key:
            self._completion_params["api_key"] = self.model_api_key

        if self.headers:
            self._completion_params["headers"] = self.headers
        
        if self.params:
            self._completion_params.update(self.params)
        
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
        messages: list[AllMessageValues],
        system_prompt: str | None = None,
    ) -> CompletionResponse:
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
        completion_response = None
        invocation = CompletionInvocation(model=self.model)

        @tenacity.retry(
            retry=tenacity.retry_if_exception_type(CompletionRetryError),
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_fixed(0),
            reraise=True,
            before_sleep=lambda retry_state: logger.info(f"Retrying completion with {len(messages)} messages"),
        )
        async def _do_completion_with_validation():
            nonlocal retry_count, completion_response
            retry_count += 1

            with invocation:
                try:
                    completion_response = await self._execute_completion(messages, invocation)

                    if invocation.current_attempt:
                        invocation.current_attempt.update_from_response(completion_response, self.model)

                    if not completion_response.choices or (
                        not completion_response.choices[0].message.content
                        and not completion_response.choices[0].message.tool_calls
                    ):
                        logger.error(f"Completion response is empty: {completion_response.model_dump_json(indent=2)}")
                        if invocation.current_attempt:
                            invocation.current_attempt.success = False
                            invocation.current_attempt.failure_reason = "Empty response"
                        raise CompletionRejectError(
                            "Completion response is empty",
                            completion_invocation=invocation,
                        )

                    try:
                        # Validate the response - may raise CompletionRetryError
                        structured_outputs, text_response, thought = await self._validate_completion(
                            completion_response=completion_response,
                        )

                        # Run post validation if provided and raise CompletionRetryError if it fails
                        if self._post_validation_fn:
                            # Keep the thought value, only update structured_outputs and text_response
                            structured_outputs, text_response = await self._post_validation_fn(
                                structured_outputs, text_response
                            )

                        if invocation.current_attempt:
                            invocation.current_attempt.success = True

                    except CompletionRetryError as e:
                        if invocation.current_attempt:
                            invocation.current_attempt.success = False
                            invocation.current_attempt.failure_reason = str(e)

                        tool_call_id = None
                        
                        if completion_response.choices[0].message.tool_calls:
                            # TODO: Support multiple tool calls
                            tool_call_id = completion_response.choices[0].message.tool_calls[0].id

                        messages.append(completion_response.choices[0].message.model_dump())

                        if e.retry_messages:
                            logger.warning(f"Post validation failed with retry messages: {e.retry_messages}")
                            messages.extend(e.retry_messages)
                        else:
                            logger.warning(f"Post validation failed with retry message: {e}")
                            if tool_call_id:
                                messages.append(
                                    ChatCompletionToolMessage(role="tool", content=str(e), tool_call_id=tool_call_id)  # type: ignore
                                )
                            else:
                                messages.append(ChatCompletionUserMessage(role="user", content=str(e)))  # type: ignore
                        raise e
                except Exception as e:
                    # Any exception will be handled by the context manager
                    raise e

                final_response = CompletionResponse(
                    structured_outputs=structured_outputs or [],
                    text_response=text_response,
                    thought=thought,
                    completion_invocation=invocation,
                )
                return final_response

        try:
            result = await _do_completion_with_validation()
            return result
        except CompletionRetryError as e:
            logger.warning(
                f"Completion failed after {retry_count} retries. Exception: {e}. Completion response: {completion_response}"
            )
            error = CompletionRejectError(
                f"Completion failed after {retry_count} retries. Exception: {e}",
                completion_invocation=invocation,
            )
            raise error from e
        except CompletionRejectError as e:
            # Add invocation data if not already present
            if not hasattr(e, "completion_invocation") or not getattr(e, "completion_invocation"):
                setattr(e, "completion_invocation", invocation)
            raise e
        except CompletionRuntimeError as e:
            # Add invocation data if not already present
            if not hasattr(e, "completion_invocation") or not getattr(e, "completion_invocation"):
                setattr(e, "completion_invocation", invocation)
            raise e
        except Exception as e:
            logger.error(f"Completion failed after {retry_count} retries. Exception: {e}. Type: {type(e)}")
            error = CompletionRuntimeError(
                f"Completion failed after {retry_count} retries. Exception: {e}. Type: {type(e)}",
                completion_invocation=invocation,
            )
            raise error from e

    @tracer.start_as_current_span("BaseCompletionModel._execute_completion")
    async def _execute_completion(self, messages: list[dict[str, str]], invocation: CompletionInvocation) -> Any:
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
        
        if self.merge_same_role_messages:
            messages = self._merge_same_role_messages(messages)

        attempt_count = 0

        @retry(
            retry=tenacity.retry_if_not_exception_type((BadRequestError, CompletionRuntimeError)),
            wait=wait_exponential(multiplier=5, min=5, max=60),
            stop=stop_after_attempt(3),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"Request failed, retrying in {retry_state.next_action.sleep} seconds"
            ),
        )
        async def _do_completion_with_rate_limit_retry():
            nonlocal attempt_count
            attempt_count += 1

            with invocation:
                try:
                    if "claude-3-" in self.model:
                        self._inject_prompt_caching(messages)

                    response = await litellm.acompletion(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=messages,
                        metadata=self.metadata or {},
                        timeout=self.timeout,
                        **self._completion_params,
                    )

                    if invocation.current_attempt:
                        invocation.current_attempt.update_from_response(response, self.model)
                        invocation.current_attempt.success = True

                    return response

                except BadRequestError as e:
                    if invocation.current_attempt:
                        invocation.current_attempt.failure_reason = f"BadRequestError: {str(e)}"

                    if e.response:
                        response_text = e.response.text
                    else:
                        response_text = None

                    logger.exception(f"LiteLLM completion failed. Model: {self.model}, " f"Response: {response_text}")

                    error = CompletionRuntimeError(message=str(e), completion_invocation=invocation)
                    raise error from e

                except APIConnectionError as e:
                    if invocation.current_attempt:
                        invocation.current_attempt.failure_reason = f"APIConnectionError: {str(e)}"

                    logger.exception(f"API connection error: {e}")
                    raise  # Let tenacity handle the retry

                except RateLimitError as e:
                    if invocation.current_attempt:
                        invocation.current_attempt.failure_reason = f"RateLimitError: {str(e)}"

                    raise  # Let tenacity handle the retry

                except Exception as e:
                    if invocation.current_attempt:
                        invocation.current_attempt.failure_reason = f"Exception: {str(e)}"

                    logger.exception(f"LiteLLM completion failed. Model: {self.model}, " f"Error: {e}")

                    error = CompletionRuntimeError(message=str(e), completion_invocation=invocation)
                    raise error from e

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
                content_obj = None

                # Handle different message content types
                if isinstance(message.get("content"), list):
                    content_obj = next((m for m in message["content"] if m.get("type") == "text"), None)
                elif message.get("role") == "assistant" and not message.get("content") and message.get("tool_calls"):
                    content_obj = message["tool_calls"][-1]
                else:
                    content_obj = message

                # Only apply cache control to dictionary objects
                if content_obj and isinstance(content_obj, dict):
                    if breakpoints_remaining:
                        content_obj["cache_control"] = ChatCompletionCachedContent(type="ephemeral")
                        breakpoints_remaining -= 1
                    elif "cache_control" in content_obj:
                        # Delete cache control only from dictionary objects
                        del content_obj["cache_control"]
            except Exception:
                logger.exception(f"Error injecting prompt caching on message: {message}")

    @abstractmethod
    async def _validate_completion(
        self, completion_response: Any
    ) -> tuple[list[ResponseSchema], Optional[str], Optional[str]]:
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
            - Optional thought string (for ReAct-style models)
        Raises:
            CompletionRejectError: If the response fails validation and should be retried
            CompletionRuntimeError: If the response indicates a fundamental problem
        """
        raise NotImplementedError

    def clone(self, **kwargs) -> "BaseCompletionModel":
        """Create a copy of the completion model with optional parameter overrides."""
        model_data = self.model_dump()
        model_data.update(kwargs)
        return BaseCompletionModel.model_validate(model_data)

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
