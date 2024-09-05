import enum
import logging
import sys
import importlib
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Tuple
from copy import deepcopy
import json

import instructor
import litellm
import openai
from anthropic import Anthropic
from anthropic.types import ToolUseBlock
from instructor import OpenAISchema
from instructor.exceptions import InstructorRetryException
from instructor.utils import classproperty
from litellm import token_counter
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict, model_validator, Extra, ValidationError

from moatless.file_context import FileContext
from moatless.repository import FileRepository
from moatless.schema import (
    Completion,
    FileWithSpans, ValueFunctionResult, Usage
)
from moatless.settings import Settings
from moatless.utils.llm_utils import LLMResponseFormat, generate_call_id, response_format_by_model
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class ActionRequest(OpenAISchema):

    @property
    def action_name(self):
        return self.__class__.__name__

    @classproperty
    def openai_tool_schema(cls):
        return {
            "type": "function",
            "function": cls.openai_schema
        }

class TakeAction(ActionRequest):

    action: ActionRequest = Field(
        ...,
        description="The action to take",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_steps(cls, data: Any):
        if isinstance(data, dict) and "action_name" in data:
            for action in cls.available_actions():
                if action.__name__ == data["action_name"]:
                    data["action"] = action.model_validate(data["action"])
                    return data

        return data

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data["action_name"] = self.action.action_name
        data["action"] = self.action.model_dump(**kwargs)
        return data

    @classmethod
    def available_actions(cls) -> List[ActionRequest]:
        return []


class StateOutcome(BaseModel):
    trigger: Optional[str] = Field(
        default=None,
        description="Trigger to transition to the next state. If None, no transition is made.",
    )
    output: Optional[dict[str, Any]] = Field(
        default=None,
        description="Output data to be passed to the next state.",
    )
    retry_message: Optional[str] = Field(
        default=None, description="Message to use in retry."
    )

    @classmethod
    def retry(cls, retry_message: str):
        return cls(trigger="retry", retry_message=retry_message)

    @classmethod
    def finish(cls, output: dict[str, Any] | None = None):
        output = output or {}
        return cls(trigger="finish", output=output)

    @classmethod
    def reject(cls, message: str):
        return cls(trigger="reject", output={"message": message})

    @classmethod
    def transition(cls, trigger: str, output: dict[str, Any] | None = None):
        output = output or {}
        return cls(trigger=trigger, output=output)

    @classmethod
    def stay_in_state(cls, output: dict[str, Any]):
        return cls(output=output)


class Content(ActionRequest):
    content: str


class Message(BaseModel):
    role: str
    content: Optional[str] = None


class AssistantMessage(Message):
    role: str = "assistant"
    content: Optional[str] = None
    action: Optional[ActionRequest] = None


class UserMessage(Message):
    role: str = "user"
    content: Optional[str] = None



class State(ABC, BaseModel):
    id: int = Field(..., description="The unique identifier of the state")
    previous_state: Optional["State"] = Field(
        default=None, description="The state that led to this state"
    )
    next_states: List["State"] = Field(
        default_factory=list, description="The states this state transitioned to"
    )

    max_iterations: Optional[int] = Field(
        None, description="The maximum number of transitions to this state."
    )

    max_expansions: int = Field(
        default=3, description="The maximum number of times this state can be expanded."
    )

    visits: List[Visit] = Field(
        default_factory=list, description="The visits to the state in MCTS backpropagation"
    )
    value_function_result: Optional[ValueFunctionResult] = Field(
        default=None,
        description="The result of the value function during MCTS"
    )

    feedback: Optional[str] = Field(
        default=None, description="Feedback provided the prompt"
    )

    _workspace: Optional[Workspace] = PrivateAttr(None)
    _initial_message: Optional[str] = PrivateAttr(None)

    _executed: bool = PrivateAttr(False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, exclude={"previous_state", "next_states"}
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._workspace = data.get("_workspace")
        self._initial_message = data.get("_initial_message")

    @abstractmethod
    def execute(self, mocked_action_request: ActionRequest | None = None) -> StateOutcome:
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def trace_name(self):
        return f"{self.__class__.__name__}:{self.id}"

    @property
    def executed(self):
        return self._executed

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def file_repo(self) -> FileRepository:
        return self._workspace.file_repo

    @property
    def file_context(self) -> FileContext:
        return self._workspace.file_context

    @property
    def initial_message(self) -> str:
        return self._initial_message

    def create_file_context(
        self, files: list[FileWithSpans] = None, **kwargs
    ) -> FileContext:
        if files is None:
            files = []
        return self.workspace.create_file_context(files, **kwargs)

    @classmethod
    def required_fields(cls) -> set[str]:
        return set()

    def get_previous_states(
        self, state: Optional["State"] = None
    ) -> list["State"]:
        """
        Retrieves previous states of the same type as the given state.
        If no state is provided, it returns all previous states.

        Args:
            state (State | None): The state to filter by. If None, all previous states are returned.

        Returns:
            list: A list of previous states, filtered by type if a state is provided.
        """
        previous_states = []
        current_state = self

        while current_state and current_state.previous_state:
            current_state = current_state.previous_state
            if not state or isinstance(current_state, type(state)):
                previous_states.insert(0, current_state)

        logger.debug(
            f"Found {len(previous_states)} previous states of type {state.__class__.__name__ if state else 'all types'}"
        )

        return previous_states

    def __str__(self):
        return self.model_dump_json(exclude={"previous_state", "next_states"})

    @classmethod
    @model_validator(mode="before")
    def validate_previous_state(cls, obj):
        if isinstance(obj, dict) and "previous_state_id" in obj:
            obj = obj.copy()
            obj["previous_state"] = None
        return super().model_validate(obj)

    def model_dump(self, **kwargs):
        if "exclude" not in kwargs:
            kwargs["exclude"] = {"previous_state", "next_states"}

        data = super().model_dump(**kwargs)
        return data

    def clone(self) -> "State":
        new_state = self.__class__(**self.model_dump())
        if hasattr(self, "_workspace"):
            new_state._workspace = self._workspace
        if hasattr(self, "_initial_message"):
            new_state._initial_message = self._initial_message
        return new_state

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        if self.model_dump() != other.model_dump():
            return False
        return True


class NoopState(State):

    def execute(self, mocked_action_request: ActionRequest | None = None) -> StateOutcome:
        raise RuntimeError(f"{self.trace_name} is a NoopState that should not be executed")


class Finished(NoopState):
    message: Optional[str] = None
    output: dict[str, Any] | None = None


class Rejected(NoopState):
    message: Optional[str] = None


class Pending(NoopState):
    def __init__(self, **data):
        if "id" not in data:
            data["id"] = 0
        super().__init__(**data)


class ActionTransaction(BaseModel):
    request: ActionRequest
    response: Optional[StateOutcome] = None
    completion: Optional[Completion] = None

    def model_dump(self, **kwargs):
        data = {}
        data["request"] = self.request.model_dump(**kwargs)
        data["response"] = self.response.model_dump(**kwargs) if self.response else None

        if Settings.include_completions_in_trajectories:
            data["completion"] = self.completion.model_dump(**kwargs) if self.completion else None

        return data


class AgenticState(State):
    model: Optional[str] = Field(
        default=None, description="The model to use for completion"
    )
    temperature: float = Field(0.0, description="The temperature to use for completion")
    max_tokens: int = Field(
        2000, description="The maximum number of tokens to generate"
    )
    max_message_tokens: Optional[int] = Field(
        None, description="The maximum number of tokens in a single message, can be used as a sanity check"
    )

    include_message_history: bool = Field(
        default=False,
        description="The message history from previous initations should be included in the completion request",
    )
    max_iterations: Optional[int] = Field(
        None, description="The maximum number of transitions to this state."
    )

    _actions: List[ActionTransaction] = PrivateAttr(default_factory=list)
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    def execute(self, mocked_action_request: ActionRequest | None = None) -> StateOutcome:
        if self._executed:
            raise ValueError(f"State has already been executed")

        if mocked_action_request:
            action = mocked_action_request
            completion = None
        else:
            try:
                action, completion = self._next_action()
            except ValidationError as e:
                return StateOutcome.retry(f"Failed to parse request. Error: {e}")

            except InstructorRetryException as e:
                if e.last_completion:
                    logger.warning(
                        f"{self.trace_name}: Failed to get a valid complection response from the LLM for action request {self.action_type()}. Will set state as rejected. Error {e}. Last completion: {e.last_completion}."
                    )
                    # TODO: Throw an error to abort the flow?
                    return StateOutcome.reject(f"Failed to get a valid response from {self.__class__.__name__}.")
                else:
                    logger.error(
                        f"{self.trace_name}: Failed to get a valid completion response from the LLM for action request {self.action_type()}. Error {e}."
                    )
                    raise e

        logger.debug(f"{self.trace_name}: Received new action {action.action_name}.")
        if self.action_type() and not isinstance(action, self.action_type()):
            raise RuntimeError(
                f"Invalid action type {action.__class__.__name__}, expected type {self.action_type().__name__}"
            )

        response = self._execute_action(action)
        self._actions.append(
            ActionTransaction(request=action, response=response, completion=completion)
        )

        if response.trigger and response.trigger != "retry":
            self._executed = True

        return response

    @abstractmethod
    def _execute_action(self, action: ActionRequest) -> StateOutcome:
        raise NotImplementedError

    @abstractmethod
    def action_type(self) -> type[ActionRequest] | None:
        """
        The type of the action to expect in the completion response.
        If not set a content string is expected.
        """
        raise NotImplementedError

    def init(self) -> Optional[StateOutcome]:
        """
        Initalize the state before exectuting with an action provided wby the LLM.
        Returns a StateOutcome if the state should transition immediately.
        """
        pass

    def handle_action(
        self, action: ActionRequest, completion: Completion | None
    ) -> StateOutcome:
        if self._executed:
            raise ValueError(f"State has already been executed")

        if self.action_type() and not isinstance(action, self.action_type()):
            raise ValueError(
                f"Invalid action type {action.__class__.__name__}, expected type {self.action_type().__name__}"
            )

        response = self._execute_action(action)
        self._actions.append(
            ActionTransaction(request=action, response=response, completion=completion)
        )
        logger.info(f"Added action to {self.name}: {len(self._actions)}")

        if response.trigger and response.trigger != "retry":
            self._executed = True

        return response

    @property
    def actions(self) -> List[ActionTransaction]:
        return self._actions

    @property
    def last_action(self) -> ActionTransaction | None:
        return self._actions[-1] if self._actions else None

    @property
    def action_request(self) -> ActionRequest | None:
        return self._actions[-1].request if self._actions else None

    @property
    def response(self) -> StateOutcome | None:
        return self._actions[-1].response if self._actions else None

    @property
    def outcome(self) -> StateOutcome | None:
        return (
            self._actions[-1].response.output
            if self._actions and self._actions[-1].response
            else None
        )

    @property
    def completion(self) -> Completion | None:
        return (
            self._actions[-1].completion
            if self._actions and self._actions[-1].completion
            else None
        )

    @property
    def initial_message(self) -> str:
        return self._initial_message

    def _next_action(
        self,
    ) -> Tuple[ActionRequest, Completion | None]:
        messages = self._to_completion_messages()

        metadata = {}
        if self._metadata:
            metadata.update(self._metadata)

        metadata["generation_name"] = self.name

        tokens = token_counter(messages=messages[-1:])
        if self.max_message_tokens and tokens > self.max_message_tokens:
            raise ValueError(f"Too many tokens in the new message: {tokens}")

        logger.info(f"{self.trace_name}: Create completion with {len(messages)} messages to {self.model}")

        response_format = response_format_by_model(self.model)

        if (
            response_format == LLMResponseFormat.ANTHROPIC_TOOLS
            and self.action_type()
        ):
            try:
                anthropic_client = Anthropic()

                apply_cache_control(messages[0])
                # apply_cache_control(messages[-1])

                tools = []
                if hasattr(self.action_type(), "available_actions"):
                    for action in self.action_type().available_actions():
                        tools.append(action.anthropic_schema)
                else:
                    tools.append(self.action_type().anthropic_schema)

                completion_response = (
                    anthropic_client.beta.prompt_caching.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=messages[0]["content"],
                        tool_choice={
                            "type": "any"
                        },
                        tools=tools,
                        messages=messages[1:],
                    )
                )

                completion = Completion.from_llm_completion(
                    input_messages=messages,
                    completion_response=completion_response,
                    model=self.model,
                )

                try:
                    action_request = None
                    if hasattr(self.action_type(), "available_actions"):
                        for block in completion_response.content:
                            if isinstance(block, ToolUseBlock):
                                action = None
                                for available_action in self.action_type().available_actions():
                                    if available_action.__name__ == block.name:
                                        action = available_action
                                        break

                                if not action:
                                    raise ValueError(f"Unknown action {block.name}")

                                tool_action_request = action.model_validate(block.input)

                                action_request = self.action_type()(action=tool_action_request)

                                # TODO: We only support one action at the moment
                                break
                            else:
                                logger.warning(f"Unexpected block {block}]")
                    else:
                        action_request = self.action_type().from_response(
                            completion_response, mode=instructor.Mode.ANTHROPIC_TOOLS
                        )

                    if not action_request:
                        raise ValueError(f"Failed to parse action request from completion response. Completion: {completion_response}")
                except Exception as e:
                    logger.exception(f"Failed to parse action request from completion response. Completion: {completion_response}")
                    raise e

                return action_request, completion

            except Exception as e:
                logger.error(f"Failed to get completion response from anthropic: {e}")
                raise e

        if self.action_type() is None and self.model.startswith("claude"):
            anthropic_client = Anthropic()

            apply_cache_control(messages[0])
            # apply_cache_control(messages[-1])

            completion_response = (
                anthropic_client.beta.prompt_caching.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=messages[0]["content"],
                    messages=messages[1:],
                )
            )

            completion = Completion.from_llm_completion(
                input_messages=messages,
                completion_response=completion_response,
                model=self.model,
            )
            action_request = Content(
                content=completion_response.content[0].text
            )

            return action_request, completion

        if self.action_type() is None:
            completion_response = litellm.completion(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=self.stop_words(),
                metadata=metadata,
                messages=messages,
            )
            action_request = Content(
                content=completion_response.choices[0].message.content
            )

            completion = Completion.from_llm_completion(
                input_messages=messages,
                completion_response=completion_response,
                model=self.model,
            )
            return action_request, completion
        elif response_format == LLMResponseFormat.STRUCTURED_OUTPUT:
            client = OpenAI()

            tools = []
            if hasattr(self.action_type(), "available_actions"):
                for action in self.action_type().available_actions():
                    tools.append(openai.pydantic_function_tool(action))
            else:
                tools.append(openai.pydantic_function_tool(self.action_type()))

            completion_response = client.beta.chat.completions.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=self.stop_words(),
                messages=messages,
                tool_choice="required",
                tools=tools,
            )

            tool_call = completion_response.choices[0].message.tool_calls[0]
            if hasattr(self.action_type(), "available_actions"):
                tool_action_request = tool_call.function.parsed_arguments
                action_request = self.action_type()(action=tool_action_request)
            else:
                action_request = tool_call.function.parsed_arguments

            if not action_request:
                raise ValueError(
                    f"Failed to parse action request from completion response. Completion: {completion_response}")

            completion = Completion.from_llm_completion(
                input_messages=messages,
                completion_response=completion_response,
                model=self.model,
            )

            return action_request, completion

        elif response_format == LLMResponseFormat.TOOLS:
            tools = []
            if hasattr(self.action_type(), "available_actions"):
                for action in self.action_type().available_actions():
                    tools.append(action.openai_tool_schema)
            else:
                tools.append(self.action_type().openai_tool_schema)

            completion_response = litellm.completion(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=self.stop_words(),
                tools=tools,
                metadata=metadata,
                messages=messages,
            )

            try:
                action_request = None
                # TODO: We only support one action at the moment
                tool_call = completion_response.choices[0].message.tool_calls[0]

                if hasattr(self.action_type(), "available_actions"):
                    action = None
                    for available_action in self.action_type().available_actions():
                        if available_action.__name__ == tool_call.function.name:
                            action = available_action
                            break

                    if not action:
                        raise ValueError(f"Unknown action {tool_call.function.name}")

                    tool_action_request = action.model_validate_json(tool_call.function.arguments)
                    action_request = self.action_type()(action=tool_action_request)
                else:
                    action_request = self.action_type().model_validate_json(tool_call.function.arguments)

                if not action_request:
                    raise ValueError(
                        f"Failed to parse action request from completion response. Completion: {completion_response}")
            except Exception as e:
                logger.exception(
                    f"Failed to parse action request from completion response. Completion: {completion_response}")
                raise e

            if not action_request:
                raise ValueError(
                    f"Failed to parse action request from completion response. Completion: {completion_response}")

            completion = Completion.from_llm_completion(
                input_messages=messages,
                completion_response=completion_response,
                model=self.model,
            )

            return action_request, completion
        else:
            client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON, metadata=metadata)

            try:
                action_request, completion_response = (
                    client.chat.completions.create_with_completion(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stop=self.stop_words(),
                        response_model=self.action_type(),
                        metadata=metadata,
                        messages=messages,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to get completion response from litellm: {e}")
                raise e

            completion = Completion.from_llm_completion(
                input_messages=messages,
                completion_response=completion_response,
                model=self.model,
            )

            return action_request, completion

    def create_file_context(
        self, files: list[FileWithSpans] = None, **kwargs
    ) -> FileContext:
        if files is None:
            files = []
        return self.workspace.create_file_context(files, **kwargs)

    def init(self):
        """Initialization logic for the state."""
        pass

    def messages(self) -> list[Message]:
        return []

    def retries(self) -> int:
        retries = 0
        for action in reversed(self._actions):
            if action.response.trigger == "retry":
                retries += 1
            else:
                return retries

        return retries

    def retry_messages(self) -> list[Message]:
        messages: list[Message] = []

        for action in self._actions:
            if isinstance(action.request, Content):
                messages.append(
                    AssistantMessage(
                        content=action.request.content,
                    )
                )
            else:
                if hasattr(action.request, "action"):
                    action_request = action.request.action
                else:
                    action_request = action.request
                messages.append(AssistantMessage(action=action_request))

            if action.response.retry_message:
                messages.append(
                    UserMessage(
                        content=action.response.retry_message,
                    )
                )

        return messages

    def _to_completion_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt()}]

        response_format = response_format_by_model(self.model)

        tool_call_id = None
        state_messages = self.messages()
        for message in state_messages:
            if message.role == "user":
                if tool_call_id and response_format in [
                    LLMResponseFormat.TOOLS,
                    LLMResponseFormat.STRUCTURED_OUTPUT,
                ]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": message.content,
                        }
                    )
                elif (
                    tool_call_id
                    and response_format == LLMResponseFormat.ANTHROPIC_TOOLS
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "tool_use_id": tool_call_id,
                                    "content": message.content,
                                    "type": "tool_result",
                                }
                            ],
                        }
                    )
                else:
                    messages.append({"role": "user", "content": message.content})
            elif message.role == "assistant":
                if message.action:
                    tool_call_id = generate_call_id()
                    if response_format == LLMResponseFormat.ANTHROPIC_TOOLS:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "id": tool_call_id,
                                        "input": message.action.model_dump(),
                                        "type": "tool_use",
                                        "name": message.action.action_name,
                                    }
                                ],
                            }
                        )
                    elif response_format in [
                        LLMResponseFormat.TOOLS,
                        LLMResponseFormat.STRUCTURED_OUTPUT,
                    ]:
                        messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": message.action.action_name,
                                            "arguments": message.action.model_dump_json(
                                                exclude_none=True
                                            ),
                                        },
                                    }
                                ],
                            }
                        )
                    else:
                        json_content = message.action.model_dump_json(indent=2)

                        if self.model.startswith("deepseek"):
                            json_content = f"```json\n{json_content}\n```"

                        messages.append(
                            {
                                "role": "assistant",
                                "content": json_content,
                            }
                        )

                else:
                    tool_call_id = None
                    messages.append({"role": "assistant", "content": message.content})

        return messages

    def system_prompt(self) -> str:
        logger.warning(f"{self.name}:{self.id} System prompt not implemented")
        return ""

    def stop_words(self) -> list[str] | None:
        return None

    def total_usage(self) -> Usage:
        total_usage = Usage()
        for action in self._actions:
            if action.completion and action.completion.usage:
                total_usage += action.completion.usage
        return total_usage

    def total_cost(self):
        total_usage = self.total_usage()
        if total_usage:
            return total_usage.completion_cost
        else:
            return 0

    def model_dump(self, **kwargs):
        if "exclude" not in kwargs:
            kwargs["exclude"] = {"previous_state", "next_states"}

        data = super().model_dump(**kwargs)
        return data

    def __str__(self):
        return self.model_dump_json(exclude={"previous_state", "next_states"})

    def __eq__(self, other):
        if not isinstance(other, AgenticState):
            return NotImplemented
        if self.model_dump() != other.model_dump():
            return False
        return True


def get_state_class(name: str) -> type[AgenticState]:
    name = name.split(":")[0] # FIXME: Remove again soon...

    builtin_states = {
        "NoopState": NoopState,
        "Finished": Finished,
        "Rejected": Rejected,
        "Pending": Pending,
    }
    if name in builtin_states:
        return builtin_states[name]

    # If not a built-in state, try to import dynamically
    possible_modules = [
        "moatless.edit",
        "moatless.find",
    ]

    for module_name in possible_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, name):
                cls = getattr(module, name)
                if isinstance(cls, type) and issubclass(cls, State):
                    return cls
        except ImportError:
            logger.debug(f"Could not import module {module_name}")

    # If still not found, try sys.modules as a fallback
    for module in sys.modules.values():
        if hasattr(module, name):
            cls = getattr(module, name)
            if isinstance(cls, type) and issubclass(cls, State):
                return cls

    raise ValueError(f"State {name} not found")

def apply_cache_control(message: dict):
    content = message["content"]
    if type(content) is str:
        content = dict(
            type="text",
            text=content,
        )
    else:
        content = message["content"][0]

    content["cache_control"] = {"type": "ephemeral"}
    message["content"] = [content]
