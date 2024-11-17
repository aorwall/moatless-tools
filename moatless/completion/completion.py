import json
import logging
import os
from enum import Enum
from textwrap import dedent
from typing import Optional, Union, List, Tuple, Any

import anthropic
import instructor
import litellm
import openai
import tenacity
from anthropic import Anthropic, AnthropicBedrock, NOT_GIVEN
from anthropic.types import ToolUseBlock, TextBlock
from anthropic.types.beta import (
    BetaToolUseBlock,
    BetaTextBlock,
    BetaMessageParam,
    BetaCacheControlEphemeralParam,
)
from litellm.exceptions import (
    BadRequestError,
    NotFoundError,
    AuthenticationError,
    APIError,
)
from litellm.types.utils import ModelResponse
from openai import AzureOpenAI, OpenAI, LengthFinishReasonError
from pydantic import BaseModel, Field, model_validator, ValidationError

from moatless.completion.model import Message, Completion, StructuredOutput
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError

logger = logging.getLogger(__name__)


class LLMResponseFormat(str, Enum):
    TOOLS = "tool_call"
    JSON = "json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    REACT = "react"


class CompletionModel(BaseModel):
    model: str = Field(..., description="The model to use for completion")
    temperature: float = Field(0.0, description="The temperature to use for completion")
    max_tokens: int = Field(
        2000, description="The maximum number of tokens to generate"
    )
    model_base_url: Optional[str] = Field(
        default=None, description="The base URL for the model API"
    )
    model_api_key: Optional[str] = Field(
        default=None, description="The API key for the model"
    )
    response_format: LLMResponseFormat = Field(
        LLMResponseFormat.TOOLS, description="The response format expected from the LLM"
    )
    stop_words: Optional[list[str]] = Field(
        default=None, description="The stop words to use for completion"
    )
    metadata: Optional[dict] = Field(
        default=None, description="Additional metadata for the completion model"
    )

    @model_validator(mode="after")
    def validate_response_format(self):
        if self.response_format == LLMResponseFormat.TOOLS:
            # Always use JSON response format for deepseek chat as it isn't reliable with tools
            if self.model == "deepseek/deepseek-chat":
                self.response_format = LLMResponseFormat.JSON
            else:
                try:
                    support_function_calling = litellm.supports_function_calling(
                        model=self.model
                    )
                except Exception as e:
                    support_function_calling = False

                if not support_function_calling:
                    logger.debug(
                        f"The model {self.model} doens't support function calling, set response format to JSON"
                    )
                    self.response_format = LLMResponseFormat.JSON

        return self

    @property
    def supports_anthropic_prompt_caching(self):
        return self.model.startswith("claude-3-5-")

    @property
    def supports_anthropic_computer_use(self):
        return "claude-3-5-sonnet-20241022" in self.model

    @property
    def use_anthropic_client(self):
        """Skip LiteLLM and use Anthropic's client for beta features"""
        return "claude-3-5" in self.model

    @property
    def use_openai_client(self):
        """Skip LiteLLm and use OpenAI's client for beta features"""
        return self.model in [
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini",
        ]

    def create_completion(
        self,
        messages: List[Message],
        system_prompt: str,
        response_model: List[type[StructuredOutput]]
        | type[StructuredOutput]
        | None = None,
    ) -> Tuple[StructuredOutput, Completion]:
        if not system_prompt:
            raise ValueError("System prompt is required")

        completion_messages = self._map_completion_messages(messages)
        completion_response = None
        try:
            if self.use_anthropic_client:
                action_args, completion_response = self._anthropic_completion(
                    completion_messages, system_prompt, response_model
                )
            elif response_model is None:
                completion_messages.insert(
                    0, {"role": "system", "content": system_prompt}
                )
                action_args, completion_response = self._litellm_text_completion(
                    completion_messages,
                )
            elif self.response_format == LLMResponseFormat.REACT and isinstance(
                response_model, list
            ):
                action_args, completion_response = self._litellm_react_completion(
                    completion_messages, system_prompt, response_model
                )
            elif self.use_openai_client:
                action_args, completion_response = self._openai_completion(
                    completion_messages, system_prompt, response_model
                )
            elif self.response_format == LLMResponseFormat.TOOLS:
                action_args, completion_response = self._litellm_tool_completion(
                    completion_messages, system_prompt, response_model
                )
            else:
                action_args, completion_response = self._litellm_completion(
                    completion_messages, system_prompt, response_model
                )
        except CompletionRejectError as e:
            raise e
        except Exception as e:
            if isinstance(e, APIError):
                logger.error(
                    f"Request failed. self.model: {self.model}, base_url: {self.model_base_url}. Model: {e.model}, Provider {e.llm_provider}. Litellm {e.litellm_debug_info}. Exception {e.message}"
                )
                if e.status_code >= 500:
                    raise CompletionRejectError(
                        f"Failed to create completion: {e}",
                        messages=completion_messages,
                        last_completion=completion_response,
                    ) from e

            else:
                logger.error(f"Failed to get completion response from litellm: {e}")

            raise CompletionRuntimeError(
                f"Failed to get completion response: {e}",
                messages=completion_messages,
                last_completion=completion_response,
            ) from e

        if completion_response:
            completion = Completion.from_llm_completion(
                input_messages=completion_messages,
                completion_response=completion_response,
                model=self.model,
            )
        else:
            completion = None

        if (
            "stop_reason" in completion.response
            and completion.response["stop_reason"] == "max_tokens"
        ):
            raise CompletionRejectError(
                f"Max tokens reached in completion response",
                messages=completion_messages,
                last_completion=completion_response,
            )

        return action_args, completion

    def create_text_completion(self, messages: List[Message], system_prompt: str):
        completion_messages = self._map_completion_messages(messages)

        if (
            self.supports_anthropic_computer_use
            or self.supports_anthropic_prompt_caching
        ):
            response, completion_response = self._anthropic_completion(
                completion_messages, system_prompt
            )
        else:
            completion_messages.insert(0, {"role": "system", "content": system_prompt})
            response, completion_response = self._litellm_text_completion(
                completion_messages
            )

        completion = Completion.from_llm_completion(
            input_messages=completion_messages,
            completion_response=completion_response,
            model=self.model,
        )

        return response, completion

    def _litellm_completion(
        self,
        messages: list[dict],
        system_prompt: str,
        structured_output: type[StructuredOutput] | list[type[StructuredOutput]],
    ) -> Tuple[StructuredOutput, ModelResponse]:
        if not structured_output:
            raise CompletionRuntimeError(f"Response model are required for completion")

        if isinstance(structured_output, list) and len(structured_output) > 1:
            avalabile_actions = [
                action for action in structured_output if hasattr(action, "name")
            ]
            if not avalabile_actions:
                raise CompletionRuntimeError(f"No actions found in {structured_output}")

            class TakeAction(StructuredOutput):
                action: Union[tuple(structured_output)] = Field(...)
                action_type: str = Field(
                    ..., description="The type of action being taken"
                )

                @model_validator(mode="before")
                def validate_action(cls, data: dict) -> dict:
                    action_type = data.get("action_type")
                    if not action_type:
                        return data

                    # Find the correct action class based on action_type
                    action_class = next(
                        (
                            action
                            for action in avalabile_actions
                            if action.name == action_type
                        ),
                        None,
                    )
                    if not action_class:
                        action_names = [action.name for action in avalabile_actions]
                        raise ValidationError(
                            f"Unknown action type: {action_type}. Available actions: {', '.join(action_names)}"
                        )

                    # Validate the action data using the specific action class
                    data["action"] = action_class.model_validate(data["action"])
                    return data

            response_model = TakeAction
        else:
            response_model = structured_output

        system_prompt += dedent(f"""\n# Response format
You must respond with only a JSON object that match the following json_schema:\n

{json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

Make sure to return an instance of the JSON, not the schema itself.""")

        messages.insert(0, {"role": "system", "content": system_prompt})

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(
                (APIError, BadRequestError, NotFoundError, AuthenticationError)
            ),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            try:
                completion_response = litellm.completion(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    api_base=self.model_base_url,
                    api_key=self.model_api_key,
                    stop=self.stop_words,
                    messages=messages,
                    response_format={"type": "json_object"},
                    metadata=self.metadata,
                )

                if isinstance(
                    completion_response.choices[0].message.content, BaseModel
                ):
                    assistant_message = completion_response.choices[
                        0
                    ].message.content.model_dump()
                else:
                    assistant_message = completion_response.choices[0].message.content

                messages.append({"role": "assistant", "content": assistant_message})

                response = response_model.from_response(
                    completion_response, mode=instructor.Mode.JSON
                )

                if hasattr(response, "action"):
                    return response.action, completion_response

                return response, completion_response

            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Completion attempt failed with error: {e}. Will retry."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"The response was invalid. Fix the errors, exceptions found\n{e}",
                    }
                )
                raise CompletionRejectError(
                    message=e.message,
                    last_completion=completion_response,
                    messages=messages,
                )
            except Exception as e:
                logger.error(f"Completion attempt failed with error: {e}. Will retry.")

                raise CompletionRuntimeError(
                    f"Failed to get completion response: {e}",
                )

        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()

    def _validate_react_format(self, response_text: str):
        # Split into lines and remove empty ones
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]

        # Count occurrences of each section
        thought_count = sum(1 for line in lines if line.startswith("Thought:"))
        action_count = sum(1 for line in lines if line.startswith("Action:"))
        action_input_count = sum(
            1 for line in lines if line.startswith("Action Input:")
        )

        # Check for multiple action blocks
        if thought_count > 1 or action_count > 1 or action_input_count > 1:
            raise ValueError(
                "You can only specify one action at a time. If you need to perform multiple actions, describe your next steps in the Thought section and execute them one at a time."
            )

        # Check if all sections exist
        if thought_count != 1 or action_count != 1 or action_input_count != 1:
            raise ValueError(
                "Response must have exactly one 'Thought:', 'Action:', and 'Action Input:' section"
            )

        # Find the starting lines for each section
        thought_line = next(
            (i for i, line in enumerate(lines) if line.startswith("Thought:")), -1
        )
        action_line = next(
            (i for i, line in enumerate(lines) if line.startswith("Action:")), -1
        )
        action_input_line = next(
            (i for i, line in enumerate(lines) if line.startswith("Action Input:")), -1
        )

        # Check if sections are in correct order
        if not (thought_line < action_line < action_input_line):
            raise ValueError("Sections must be in order: Thought, Action, Action Input")

    def _litellm_react_completion(
        self,
        messages: list[dict],
        system_prompt: str,
        actions: list[type[StructuredOutput]],
    ) -> Tuple[StructuredOutput, ModelResponse]:
        action_input_schemas_str = ""

        for action in actions:
            schema = action.model_json_schema()
            if "scratch_pad" in schema["properties"]:
                del schema["properties"]["scratch_pad"]

            action_input_schemas_str += f"\n * {action.name}: {json.dumps(schema)}"

        system_prompt += dedent(f"""\n# Response format

Use the following format:

Thought: You should always think about what to do
Action: The action to take
Action Input: The input to the action. Always use valid JSON format with double quotes for strings and 'null' for null values. For example: 'Action Input: {{"file_pattern": "path/to/file", "optional_field": null}}'

You have access to the following tools: {action_input_schemas_str}

Important: Do not include multiple Thought-Action-Observation blocks. Do not include code blocks or additional text outside of this format.
""")

        system_prompt += """\n
**Examples of How to Use the Response Format:**

**Correct Usage:**

Thought: I need to update the error message in the authentication function to be more descriptive.
Action: StringReplace
Action Input: {
  "path": "auth/validator.py",
  "old_str": "    if not user.is_active:\n        raise ValueError(\"Invalid user\")\n    return user",
  "new_str": "    if not user.is_active:\n        raise ValueError(f\"Invalid user: {username} is not active\")\n    return user"
}

**Incorrect Usage (Multiple Actions in One Response):**
Thought: I need to view the current implementation of the logger configuration and then update it to include a file handler.
Action: ViewCode
Action Input: {
  "files": [
    {
      "file_path": "utils/logger.py",
      "start_line": null,
      "end_line": null,
      "span_ids": ["configure_logger"]
    }
  ]
}
Action: StringReplace
Action Input: {
  "path": "utils/logger.py",
  "old_str": "logging.basicConfig(level=logging.INFO, format=\"%(levelname)s - %(message)s\")",
  "new_str": "logging.basicConfig(level=logging.INFO, format=\"%(asctime)s - %(levelname)s - %(message)s\", handlers=[logging.FileHandler(\"app.log\"), logging.StreamHandler()])"
}"""

        messages.insert(0, {"role": "system", "content": system_prompt})

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(
                (APIError, BadRequestError, NotFoundError, AuthenticationError)
            ),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            response_text, completion_response = self._litellm_text_completion(messages)

            try:
                self._validate_react_format(response_text)

                thought_start = response_text.find("Thought:")
                action_start = response_text.find("Action:")
                action_input_start = response_text.find("Action Input:")

                if (
                    thought_start == -1
                    or action_start == -1
                    or action_input_start == -1
                ):
                    raise ValueError("Missing Thought, Action or Action Input sections")

                thought = response_text[thought_start + 8 : action_start].strip()
                action = response_text[action_start + 7 : action_input_start].strip()
                action_input = response_text[action_input_start + 13 :].strip()

                if not action or not action_input:
                    raise ValueError("Missing Action or Action Input values")

                # Find the matching action class
                action_class = next((a for a in actions if a.name == action), None)
                if not action_class:
                    action_names = [a.name for a in actions]
                    raise ValueError(
                        f"Unknown action: {action}. Available actions: {', '.join(action_names)}"
                    )

                action_request = action_class.model_validate_json(action_input)
                action_request.scratch_pad = thought
                return action_request, completion_response

            except Exception as e:
                logger.warning(f"ReAct parsing failed: {e}. Response: {response_text}")

                messages.append({"role": "assistant", "content": response_text})

                if isinstance(e, ValidationError):
                    messages.append(
                        {
                            "role": "user",
                            "content": f"The action input JSON is invalid. Please fix the following validation errors:\n{e}",
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"The response was invalid. Please follow the exact format:\n\nThought: your reasoning\nAction: the action name\nAction Input: the JSON input\n\nError: {e}",
                        }
                    )

                raise CompletionRejectError(
                    message=str(e),
                    last_completion=completion_response,
                    messages=messages,
                )

        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()

    def _litellm_text_completion(
        self, messages: list[dict]
    ) -> Tuple[str, ModelResponse]:
        litellm.drop_params = True

        completion_response = litellm.completion(
            model=self.model,
            api_base=self.model_base_url,
            api_key=self.model_api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop_words,
            messages=messages,
            metadata=self.metadata,
        )
        return completion_response.choices[0].message.content, completion_response

    def _litellm_tool_completion(
        self,
        messages: list[dict],
        system_prompt: str,
        actions: list[type[StructuredOutput]],
        is_retry: bool = False,
    ) -> Tuple[StructuredOutput, ModelResponse]:
        litellm.drop_params = True
        messages.insert(0, {"role": "system", "content": system_prompt})

        tools = []
        for action in actions:
            tools.append(openai.pydantic_function_tool(action))

        completion_response = litellm.completion(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            api_base=self.model_base_url,
            api_key=self.model_api_key,
            stop=self.stop_words,
            tools=tools,
            tool_choice="auto",
            messages=messages,
            metadata=self.metadata,
        )

        tool_args, tool_name, retry_message = None, None, None
        if (
            not completion_response.choices[0].message.tool_calls
            and completion_response.choices[0].message.content
        ):
            if "```json" in completion_response.choices[0].message.content:
                logger.info(
                    f"Found no tool call but JSON in completion response, will try to parse"
                )

                try:
                    action_request = self.action_type().from_response(
                        completion_response, mode=instructor.Mode.TOOLS
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to parse JSON as tool call, will try to parse as JSON "
                    )

                    try:
                        action_request = self.action_type().from_response(
                            completion_response, mode=instructor.Mode.JSON
                        )
                    except Exception as e:
                        logger.exception(
                            f"Failed to parse JSON as tool call from completion response: {completion_response.choices[0].message.content}"
                        )
                        raise e

                return action_request, completion_response
            elif completion_response.choices[0].message.content.startswith("{"):
                tool_args = json.loads(completion_response.choices[0].message.content)

            if tool_args:
                if "name" in tool_args:
                    tool_name = tool_args.get("name")

                if "parameters" in tool_args:
                    tool_args = tool_args["parameters"]

        elif completion_response.choices[0].message.tool_calls[0]:
            tool_call = completion_response.choices[0].message.tool_calls[0]
            tool_args = json.loads(tool_call.function.arguments)
            tool_name = tool_call.function.name

        if not tool_args:
            if is_retry:
                logger.error(
                    f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}"
                )
                raise ValueError(f"No tool call in response from LLM.")

            logger.warning(
                f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}. Will retry"
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": completion_response.choices[0].message.content,
                }
            )
            if not retry_message:
                retry_message = "You must response with a tool call."
            messages.append({"role": "user", "content": retry_message})
            return self._litellm_tool_completion(messages, is_retry=True)

        action_request = self.action_type().from_tool_call(
            tool_args=tool_args, tool_name=tool_name
        )
        return action_request, completion_response

    def input_messages(
        self, content: str, completion: Completion | None, feedback: str | None = None
    ):
        messages = []
        tool_call_id = None

        if completion:
            messages = completion.input

            response_message = completion.response["choices"][0]["message"]
            if response_message.get("tool_calls"):
                tool_call_id = response_message.get("tool_calls")[0]["id"]
                last_response = {
                    "role": response_message["role"],
                    "tool_calls": response_message["tool_calls"],
                }
            else:
                last_response = {
                    "role": response_message["role"],
                    "content": response_message["content"],
                }
            messages.append(last_response)

            if response_message.get("tool_calls"):
                tool_call_id = response_message.get("tool_calls")[0]["id"]

        if tool_call_id:
            new_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        else:
            new_message = {
                "role": "user",
                "content": content,
            }

        if feedback:
            new_message["content"] += "\n\n" + feedback

        messages.append(new_message)
        return messages

    def _openai_completion(
        self,
        messages: list[dict],
        system_prompt: str,
        actions: List[type[StructuredOutput]] | None = None,
        response_format: type[StructuredOutput] | None = None,
        is_retry: bool = False,
    ):
        if os.getenv("AZURE_API_KEY"):
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_API_KEY"),
                api_version="2024-07-01-preview",
                azure_endpoint=os.getenv("AZURE_API_BASE"),
            )
        else:
            client = OpenAI()

        messages.insert(0, {"role": "system", "content": system_prompt})

        tools = []
        if actions:
            for action in actions:
                schema = action.openai_schema
                tools.append(
                    openai.pydantic_function_tool(
                        action, name=schema["name"], description=schema["description"]
                    )
                )

        try:
            if actions:
                completion_response = client.beta.chat.completions.parse(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=self.stop_words,
                    messages=messages,
                    # tool_choice="required",
                    tools=tools,
                    parallel_tool_calls=True,
                )
            else:
                completion_response = client.beta.chat.completions.parse(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=self.stop_words,
                    messages=messages,
                    response_format=response_format,
                )
        except LengthFinishReasonError as e:
            logger.error(
                f"Failed to parse completion response. Completion: {e.completion.model_dump_json(indent=2)}"
            )
            from moatless.actions.reject import Reject

            # TODO: Raise exception instead?
            return Reject(
                rejection_reason=f"Failed to generate action: {e}"
            ), e.completion

        if not actions:
            response = completion_response.choices[0].message.parsed
            return response, completion_response

        elif not completion_response.choices[0].message.tool_calls:
            if is_retry:
                logger.error(
                    f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}"
                )
                raise RuntimeError(f"No tool call in response from LLM.")

            logger.warning(
                f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}. Will retry"
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": completion_response.choices[0].message.content,
                }
            )
            messages.append(
                {"role": "user", "content": "You must response with a tool call."}
            )
            return self._openai_completion(messages, actions, response_format, is_retry)
        else:
            tool_call = completion_response.choices[0].message.tool_calls[0]
            action_request = tool_call.function.parsed_arguments
            return action_request, completion_response

    def _anthropic_completion(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        response_model: type[StructuredOutput]
        | List[type[StructuredOutput]]
        | None = None,
    ) -> Tuple[StructuredOutput | str, Any]:
        tools = []
        tool_choice = {"type": "any"}

        if not response_model:
            tools = NOT_GIVEN
            tool_choice = NOT_GIVEN
        else:
            if isinstance(response_model, list):
                actions = response_model
            elif response_model:
                actions = [response_model]
            else:
                actions = []

            for action in actions:
                if hasattr(action, "name") and action.name == "str_replace_editor":
                    tools.append(
                        {"name": "str_replace_editor", "type": "text_editor_20241022"}
                    )
                else:
                    schema = action.anthropic_schema

                    # Remove scratch pad field, use regular text block for thoughts
                    if "scratch_pad" in schema["input_schema"]["properties"]:
                        del schema["input_schema"]["properties"]["scratch_pad"]

                    tools.append(schema)

        system_message = {"text": system_prompt, "type": "text"}

        if "anthropic" in self.model:
            anthropic_client = AnthropicBedrock()
            betas = ["computer-use-2024-10-22"]
        else:
            anthropic_client = Anthropic()
            betas = ["computer-use-2024-10-22", "prompt-caching-2024-07-31"]
            _inject_prompt_caching(messages)
            system_message["cache_control"] = {"type": "ephemeral"}

        completion_response = None
        retry_message = None
        for i in range(2):
            if i > 0:
                logger.warning(
                    f"Retrying completion request: {retry_message} (attempt {i})"
                )

            try:
                completion_response = anthropic_client.beta.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=[system_message],
                    # tool_choice=tool_choice,
                    tools=tools,
                    messages=messages,
                    betas=betas,
                )
            except anthropic.BadRequestError as e:
                logger.error(
                    f"Failed to create completion: {e}. Input messages: {json.dumps(messages, indent=2)}"
                )
                last_completion = (
                    completion_response.model_dump() if completion_response else None
                )
                raise CompletionRuntimeError(
                    f"Failed to create completion: {e}",
                    last_completion=last_completion,
                    messages=messages,
                ) from e

            tool_call_id = None
            try:
                text = None
                if not actions:
                    return completion_response.content[0].text, completion_response
                for block in completion_response.content:
                    if isinstance(block, ToolUseBlock) or isinstance(
                        block, BetaToolUseBlock
                    ):
                        action = None

                        tool_call_id = block.id

                        if len(actions) == 1:
                            action = actions[0]
                        else:
                            for check_action in actions:
                                if check_action.openai_schema["name"] == block.name:
                                    action = check_action
                                    break

                        if not action:
                            raise ValueError(f"Unknown action {block.name}")

                        action_args = action.model_validate(block.input)

                        if (
                            hasattr(action_args, "scratch_pad")
                            and text
                            and not action_args.scratch_pad
                        ):
                            action_args.scratch_pad = text

                        # TODO: We only support one action at the moment
                        return action_args, completion_response
                    elif isinstance(block, TextBlock) or isinstance(
                        block, BetaTextBlock
                    ):
                        text = block.text
                        # Extract thoughts from <thoughts> tags if present
                        if "<thoughts>" in text and "</thoughts>" in text:
                            start = text.index("<thoughts>") + len("<thoughts>")
                            end = text.index("</thoughts>")
                            text = text[start:end].strip()
                    else:
                        logger.warning(f"Unexpected block {block}]")

                retry_message = f"You're an autonomous agent that can't communicate with the user. Please provide a tool call."
            except anthropic.APIError as e:
                if hasattr(e, "status_code"):
                    raise CompletionRuntimeError(
                        f"Failed to call Anthropic API. Status code: {e.status_code}, Response: {e.body}"
                    ) from e
                else:
                    raise CompletionRuntimeError(
                        f"Failed to call Anthropic API. {e}"
                    ) from e
            except Exception as e:
                retry_message = f"The request was invalid. Please try again. Error: {e}"

            response_content = [
                block.model_dump() for block in completion_response.content
            ]
            messages.append({"role": "assistant", "content": response_content})

            user_message = self._create_user_message(tool_call_id, retry_message)
            messages.append(user_message)

        raise CompletionRejectError(
            f"Failed to create completion: {retry_message}",
            messages=messages,
            last_completion=completion_response,
        )

    def _map_completion_messages(self, messages: list[Message]) -> list[dict]:
        tool_call_id = None
        completion_messages = []
        for i, message in enumerate(messages):
            if message.role == "user":
                user_message = self._create_user_message(tool_call_id, message.content)
                completion_messages.append(user_message)
                tool_call_id = None
            elif message.role == "assistant":
                if message.tool_call:
                    tool_call_id = message.tool_call_id
                    content = []
                    if self.use_anthropic_client:
                        tool_input = message.tool_call.input.copy()

                        # Scratch pad is provided as a message instead of part of the tool call
                        if "scratch_pad" in message.tool_call.input:
                            scratch_pad = tool_input["scratch_pad"]
                            del tool_input["scratch_pad"]
                            if scratch_pad:
                                content.append(
                                    {
                                        "type": "text",
                                        "text": f"<thoughts>\n{scratch_pad}\n</thoughts>",
                                    }
                                )

                        content.append(
                            {
                                "id": tool_call_id,
                                "input": tool_input,
                                "type": "tool_use",
                                "name": message.tool_call.name,
                            }
                        )
                        completion_messages.append(
                            {"role": "assistant", "content": content}
                        )
                    elif self.response_format in [
                        LLMResponseFormat.TOOLS,
                    ]:
                        completion_messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": message.tool_call.name,
                                            "arguments": json.dumps(
                                                message.tool_call.input
                                            ),
                                        },
                                    }
                                ],
                            }
                        )
                    else:
                        action_json = {
                            "action": message.tool_call.input,
                            "action_type": message.tool_call.name,
                        }
                        json_content = json.dumps(action_json, indent=2)

                        json_content = f"```json\n{json_content}\n```"

                        completion_messages.append(
                            {
                                "role": "assistant",
                                "content": json_content,
                            }
                        )

                else:
                    tool_call_id = None
                    completion_messages.append(
                        {"role": "assistant", "content": message.content}
                    )

        return completion_messages

    def _create_user_message(self, tool_call_id: str | None, content: str) -> dict:
        if tool_call_id and self.use_anthropic_client:
            return {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": tool_call_id,
                        "content": f"<observation>\n{content}\n</observation>",
                        "type": "tool_result",
                    }
                ],
            }
        elif tool_call_id and self.response_format in [LLMResponseFormat.TOOLS]:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        elif tool_call_id and self.response_format in [LLMResponseFormat.TOOLS]:
            return {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": tool_call_id,
                        "content": content,
                        "type": "tool_result",
                    }
                ],
            }

        else:
            return {"role": "user", "content": content}

    def _get_tool_call(self, completion_response) -> Tuple[str, dict]:
        if (
            not completion_response.choices[0].message.tool_calls
            and completion_response.choices[0].message.content
        ):
            if "```json" in completion_response.choices[0].message.content:
                content = completion_response.choices[0].message.content
                json_start = content.index("```json") + 7
                json_end = content.rindex("```")
                json_content = content[json_start:json_end].strip()
            elif completion_response.choices[0].message.content.startswith("{"):
                json_content = completion_response.choices[0].message.content
            else:
                return None, None

            tool_call = json.loads(json_content)
            return tool_call.get("name"), tool_call

        elif completion_response.choices[0].message.tool_calls:
            tool_call = completion_response.choices[0].message.tool_calls[0]
            tool_dict = json.loads(tool_call.function.arguments)
            return tool_call.function.name, tool_dict

        return None

    def model_dump(self, **kwargs):
        dump = super().model_dump(**kwargs)
        if "model_api_key" in dump:
            dump["model_api_key"] = None
        if "response_format" in dump:
            dump["response_format"] = dump["response_format"].value
        return dump

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and "response_format" in obj:
            obj["response_format"] = LLMResponseFormat(obj["response_format"])
        return super().model_validate(obj)

    @model_validator(mode="after")
    def set_api_key(self) -> "CompletionModel":
        """
        Update the model with the API key from en vars if model base URL is set but API key is not as we don't persist the API key.
        """
        if self.model_base_url and not self.model_api_key:
            self.model_api_key = os.getenv("CUSTOM_LLM_API_KEY")

        return self


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        # message["role"] == "user" and
        if isinstance(content := message["content"], list):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break
