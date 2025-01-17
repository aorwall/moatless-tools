import json
import logging
import os
from abc import ABC
from enum import Enum
from typing import Optional, List, Union, Any, Dict

import tenacity
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, Field, model_validator

from moatless.completion.model import Completion, Usage
from moatless.completion.schema import ResponseSchema, AllMessageValues, ChatCompletionCachedContent
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError

logger = logging.getLogger(__name__)


class CompletionRetryError(Exception):
    """Exception raised when a completion should be retried"""

    def __init__(self, message: str, retry_message: AllMessageValues):
        super().__init__(message)
        self.retry_message = retry_message


class LLMResponseFormat(str, Enum):
    TOOLS = "tool_call"
    JSON = "json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    REACT = "react"


class CompletionResponse(BaseModel):
    """Container for completion responses that can include multiple structured outputs and text"""

    structured_outputs: List[ResponseSchema] = Field(default_factory=list)
    text_response: Optional[str] = Field(default=None)
    completion: Optional[Completion] = Field(default=None)
    flags: List[str] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        text: str | None = None,
        output: List[ResponseSchema] | ResponseSchema | None = None,
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


class BaseCompletionModel(BaseModel, ABC):
    model: str = Field(..., description="The model to use for completion")
    temperature: float = Field(0.0, description="The temperature to use for completion")
    max_tokens: int = Field(2000, description="The maximum number of tokens to generate")
    timeout: float = Field(120.0, description="The timeout in seconds for completion requests")
    model_base_url: Optional[str] = Field(default=None, description="The base URL for the model API")
    model_api_key: Optional[str] = Field(default=None, description="The API key for the model", exclude=True)
    response_format: LLMResponseFormat = Field(..., description="The response format expected from the LLM")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata for the completion model")
    message_cache: bool = Field(
        default=True, description="Cache the message history in the prompt cache if the LLM supports it"
    )
    thoughts_in_action: bool = Field(
        default=False,
        description="Whether to include thoughts in the action or in the message",
    )
    disable_thoughts: bool = Field(
        default=False,
        description="Whether to disable to use thoughts at all.",
    )

    response_schema: Optional[List[type[ResponseSchema]]] = Field(
        default=None, description="The schema(s) used to validate responses", exclude=True
    )
    system_prompt: Optional[str] = Field(
        default=None, description="The system prompt to use for completion", exclude=True
    )

    _completion_params: Optional[Dict[str, Union[str, Dict, List]]] = None
    _initialized: bool = False

    def initialize(
        self,
        response_schema: Union[List[type[ResponseSchema]], type[ResponseSchema]],
        system_prompt: str,
    ) -> None:
        """Initialize the completion model with response schema and system prompt.

        This method prepares the model for completions by setting up the response schema
        and system prompt, and preparing completion parameters.

        Args:
            response_schema: The schema(s) to validate responses against
            system_prompt: The system prompt to use for completions

        Raises:
            CompletionRuntimeError: If any schema is not a subclass of ResponseSchema
        """
        if isinstance(response_schema, list):
            schemas = response_schema
        else:
            schemas = [response_schema]

        # Validate all schemas are subclasses of ResponseSchema
        for schema in schemas:
            if not issubclass(schema, ResponseSchema):
                raise CompletionRuntimeError(f"Schema {schema.__name__} must be a subclass of ResponseSchema")

        self.response_schema = schemas
        self._completion_params = self._get_completion_params(self.response_schema)
        self.system_prompt = self._prepare_system_prompt(system_prompt, self.response_schema)
        self._initialized = True

    def create_completion(
        self,
        messages: List[dict],
    ) -> CompletionResponse:
        if not self._initialized:
            raise ValueError(
                "Model must be initialized with response schema and system prompt before creating completion"
            )

        prepared_messages = self._prepare_messages(messages, self.system_prompt)
        return self._create_completion_with_retries(messages=prepared_messages)

    def _prepare_system_prompt(
        self, system_prompt: str, response_schema: Union[List[type[ResponseSchema]], type[ResponseSchema]]
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

    def _prepare_messages(self, messages: List[dict], system_prompt: str) -> List[dict]:
        """Prepare messages with system prompt"""
        messages = messages.copy()
        messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    def _get_completion_params(self, schema: type[ResponseSchema]) -> dict[str, Union[str, dict, list]]:
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

    def _create_completion_with_retries(
        self,
        messages: List[dict],
    ) -> CompletionResponse:
        """Execute completion with retries for validation errors"""
        retry_count = 0
        accumulated_usage = Usage()
        completion_response = None

        @tenacity.retry(
            retry=tenacity.retry_if_exception_type((CompletionRetryError)),
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_fixed(0),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying litellm completion after error: {retry_state.outcome.exception()}"
            ),
        )
        def _do_completion_with_validation():
            nonlocal retry_count, accumulated_usage, completion_response
            retry_count += 1

            # Execute completion and get raw response
            completion_response = self._execute_completion(messages)

            # Track usage from this attempt regardless of validation outcome
            usage = Usage.from_completion_response(completion_response, self.model)
            if usage:
                accumulated_usage += usage
            else:
                logger.warning(f"No usage found for completion response: {completion_response}")

            try:
                # Validate the response - may raise CompletionRejectError
                structured_outputs, text_response, flags = self._validate_completion(
                    completion_response=completion_response,
                )
            except CompletionRetryError as e:
                messages.append(completion_response.choices[0].message.model_dump())
                messages.append(e.retry_message)
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
            )

        try:
            return _do_completion_with_validation()
        except CompletionRetryError as e:
            logger.warning(
                f"Completion failed after {retry_count} retries. Exception: {e}. Completion response: {completion_response}"
            )
            raise CompletionRejectError(
                f"Completion failed after {retry_count} retries. Exception: {e}. Type: {type(e)}",
                messages=messages,
                last_completion=completion_response.model_dump() if completion_response else None,
                accumulated_usage=accumulated_usage,
            ) from e
        except Exception as e:
            logger.error(f"Completion failed after {retry_count} retries. Exception: {e}. Type: {type(e)}")
            raise CompletionRuntimeError(
                f"Completion failed after {retry_count} retries. Exception: {e}. Type: {type(e)}",
                messages=messages,
                last_completion=completion_response.model_dump() if completion_response else None,
                accumulated_usage=accumulated_usage,
            ) from e

    def _execute_completion(
        self,
        messages: List[Dict[str, str]],
    ) -> ModelResponse:
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
        import litellm
        from litellm import BadRequestError

        params = self._completion_params.copy()

        try:
            betas = []
            if "claude-3-5" in self.model:
                self._inject_prompt_caching(messages)
                betas.append("prompt-caching-2024-07-31")

            if "claude-3-5-sonnet" in self.model:
                betas.append("computer-use-2024-10-22")

            if betas:
                extra_headers = {"anthropic-beta": ",".join(betas)}
            else:
                extra_headers = None

            return litellm.completion(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
                metadata=self.metadata or {},
                timeout=self.timeout,
                api_base=self.model_base_url,
                api_key=self.model_api_key,
                extra_headers=extra_headers,
                **params,
            )

        except BadRequestError as e:
            logger.exception(
                f"LiteLLM completion failed. Model: {self.model}, "
                f"Response Schemas: {', '.join(self._get_schema_names())}, "
                f"Completion Params:\n{json.dumps(params, indent=2)}\n"
                f"Error: {e}"
            )
            raise CompletionRuntimeError(str(e), messages=messages) from e
        except Exception as e:
            logger.exception(
                f"LiteLLM completion failed. Model: {self.model}, "
                f"Response Schemas: {', '.join(self._get_schema_names())}, "
                f"Error: {e}"
            )
            raise CompletionRuntimeError(str(e), messages=messages) from e

    def _get_schema_names(self):
        return [schema.__name__ for schema in self.response_schema] if self.response_schema else ["None"]

    def _inject_prompt_caching(self, messages: List[Dict[str, str]]) -> None:
        """Set cache breakpoints for Claude 3.5 message history.

        This method:
        1. Marks the system prompt for caching
        2. Sets cache breakpoints for the 3 most recent turns

        Args:
            messages: The conversation history to inject caching into
        """
        if messages[0]["role"] == "system":
            messages[0]["cache_control"] = ChatCompletionCachedContent(type="ephemeral")

        breakpoints_remaining = 3
        for message in reversed(messages):
            if breakpoints_remaining:
                if isinstance(message["content"], list):
                    if breakpoints_remaining:
                        breakpoints_remaining -= 1
                        message["content"][-1]["cache_control"] = ChatCompletionCachedContent(type="ephemeral")
                else:
                    message["cache_control"] = ChatCompletionCachedContent(type="ephemeral")
                    breakpoints_remaining -= 1

    def _validate_completion(self, completion_response: Any) -> tuple[List[ResponseSchema], Optional[str], List[str]]:
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

    def model_dump(self, **kwargs):
        dump = super().model_dump(**kwargs)
        if "model_api_key" in dump:
            dump["model_api_key"] = None
        if "response_format" in dump:
            dump["response_format"] = dump["response_format"].value
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
            return cls.create(**obj)

        return obj

    @model_validator(mode="after")
    def set_api_key(self) -> "BaseCompletionModel":
        """Update the model with the API key from env vars if model base URL is set but API key is not"""
        if self.model_base_url and not self.model_api_key:
            self.model_api_key = os.getenv("CUSTOM_LLM_API_KEY")
        return self
