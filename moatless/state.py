import logging
import sys
import importlib
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from copy import deepcopy

from pydantic import BaseModel, Field, PrivateAttr, ConfigDict, model_validator

from moatless.file_context import FileContext
from moatless.repository import FileRepository
from moatless.types import (
    ActionRequest,
    ActionResponse,
    ActionTransaction,
    FileWithSpans,
    Message, Content, AssistantMessage,
    Usage, UserMessage,
)
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class AgenticState(ABC, BaseModel):
    id: int = Field(..., description="The unique identifier of the state")
    previous_state: Optional["AgenticState"] = Field(
        default=None, description="The state that led to this state"
    )
    next_states: List["AgenticState"] = Field(
        default_factory=list, description="The states this state transitioned to"
    )
    model: Optional[str] = Field(
        default=None, description="The model to use for completion"
    )
    temperature: float = Field(0.0, description="The temperature to use for completion")
    max_tokens: int = Field(
        1000, description="The maximum number of tokens to generate"
    )
    include_message_history: bool = Field(
        default=False,
        description="The message history from previous initations should be included in the completion request",
    )
    max_iterations: Optional[int] = Field(
        None, description="The maximum number of transitions to this state."
    )

    _workspace: Optional[Workspace] = PrivateAttr(None)
    _initial_message: Optional[str] = PrivateAttr(None)

    _executed: bool = PrivateAttr(False)
    _actions: List[ActionTransaction] = PrivateAttr(default_factory=list) 

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        exclude={"previous_state", "next_states"}
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._workspace = data.get('_workspace')
        self._initial_message = data.get('_initial_message')

    def handle_action(self, action: ActionRequest, usage: Usage | None) -> ActionResponse:
        if self._executed:
            raise ValueError(f"State has already been executed")

        response = self._execute_action(action)
        self._actions.append(ActionTransaction(request=action, response=response, usage=usage))

        if response.trigger and response.trigger != "retry":
            self._executed = True

        return response

    @abstractmethod
    def _execute_action(self, action: ActionRequest) -> ActionResponse:
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def executed(self):
        return self._executed

    @property
    def last_action(self) -> Optional[ActionTransaction]:
        return self._actions[-1] if self._actions else None

    @property
    def response(self) -> Optional[ActionResponse]:
        return self._actions[-1].response if self._actions else None

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

    def init(self):
        """Initialization logic for the state."""
        pass

    def finish(self, message: str):
        # TODO!!
        logger.info(message)

    def messages(self) -> list[Message]:
        return []

    @classmethod
    def required_fields(cls) -> set[str]:
        return set()

    def get_previous_states(self, state: Optional["AgenticState"] = None) -> list["AgenticState"]:
        """
        Retrieves previous states of the same type as the given state.
        If no state is provided, it returns all previous states.

        Args:
            state (AgenticState | None): The state to filter by. If None, all previous states are returned.

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
                messages.append(AssistantMessage(action=action.request))

            if action.response.retry_message:
                messages.append(
                    UserMessage(
                        content=action.response.retry_message,
                    )
                )

        return messages

    def system_prompt(self) -> str:
        return ""

    def action_type(self) -> type[ActionRequest] | None:
        """
        The type of the action to expect in the completion response.
        If not set a content string is expected.
        """
        raise NotImplementedError

    def stop_words(self) -> list[str] | None:
        return None

    def model_dump(self, **kwargs):
        data = super().model_dump(exclude={"previous_state", "next_states"}, **kwargs)
        data["name"] = self.name
        data["previous_state_id"] = self.previous_state.id if self.previous_state else None
        return data

    @classmethod
    @model_validator(mode="before")
    def validate_previous_state(cls, values):
        if isinstance(obj, dict) and "previous_state_id" in obj:
            obj = obj.copy()
            obj["previous_state"] = None
        return super().model_validate(obj)

    def clone(self) -> "AgenticState":
        new_state = self.__class__(**self.model_dump())
        if hasattr(self, '_workspace'):
            new_state._workspace = self._workspace
        return new_state

    def total_cost(self):
        total_cost = 0
        for action in self._actions:
            if action.usage:
                total_cost += action.usage.completion_cost

        return total_cost
    
    def __eq__(self, other):
        if not isinstance(other, AgenticState):
            return NotImplemented
        if self.model_dump() != other.model_dump():
            return False
        return True


class NoopState(AgenticState):

    def _execute_action(self, action: ActionRequest):
        raise ValueError("NoopState cannot handle actions")


class Finished(NoopState):
    message: Optional[str] = None
    output: dict[str, Any] | None = None


class Rejected(NoopState):
    message: Optional[str] = None


class Pending(NoopState):
    def __init__(self, **data):
        if 'id' not in data:
            data['id'] = 0
        super().__init__(**data)


def get_state_class(name: str) -> type[AgenticState]:
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
                if isinstance(cls, type) and issubclass(cls, AgenticState):
                    return cls
        except ImportError:
            logger.debug(f"Could not import module {module_name}")

    # If still not found, try sys.modules as a fallback
    for module in sys.modules.values():
        if hasattr(module, name):
            cls = getattr(module, name)
            if isinstance(cls, type) and issubclass(cls, AgenticState):
                return cls

    raise ValueError(f"State {name} not found")