import logging
import sys
import importlib
from abc import ABC, abstractmethod
from typing import Any, Optional
from copy import deepcopy

from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

from moatless.file_context import FileContext
from moatless.repository import FileRepository
from moatless.types import (
    ActionRequest,
    ActionResponse,
    FileWithSpans,
    Message,
)
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class AgenticState(ABC, BaseModel):
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

    _loop: Optional["AgenticLoop"] = PrivateAttr(None)  # noqa: F821

    _executed: bool = PrivateAttr(False)
    _last_action: Optional[ActionRequest] = PrivateAttr(None)
    _response: Optional[ActionResponse] = PrivateAttr(None)

    # model_config = ConfigDict(extra='allow')

    def __init__(self, **data):
        super().__init__(**data)
        self._loop = None

    def handle_action(self, action: ActionRequest) -> ActionResponse:
        if self._executed:
            raise ValueError(f"State has already been executed")

        self._last_action = action
        response = self._execute_action(action)

        if response.trigger and response.trigger != "retry":
            self._executed = True
            self._response = response

        return response

    @abstractmethod
    def _execute_action(self, action: ActionRequest) -> ActionResponse:
        raise NotImplementedError

    def _set_loop(self, loop: "AgenticLoop"):  # noqa: F821
        self._loop = loop
        self.init()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def executed(self):
        return self._executed

    @property
    def last_action(self) -> Optional[ActionRequest]:
        return self._last_action

    @property
    def response(self) -> Optional[ActionResponse]:
        return self._response

    @property
    def loop(self) -> "AgenticLoop":  # noqa: F821
        assert self._loop is not None, "Loop has not been set"
        return self._loop

    @property
    def workspace(self) -> Workspace:
        return self.loop.workspace

    @property
    def file_repo(self) -> FileRepository:
        return self.workspace.file_repo

    @property
    def file_context(self) -> FileContext:
        return self.workspace.file_context

    def create_file_context(
        self, files: list[FileWithSpans] = None, **kwargs
    ) -> FileContext:
        if files is None:
            files = []
        return self.workspace.create_file_context(files, **kwargs)

    def init(self):
        """Initialization logic for the state."""
        pass

    def transition_to(self, new_state: "AgenticState"):
        self.loop.transition_to(new_state)

    def finish(self, message: str):
        # TODO!!
        logger.info(message)

    def messages(self) -> list[Message]:
        return []

    @classmethod
    def required_fields(cls) -> set[str]:
        return set()

    def retries(self) -> int:
        retries = 0
        for action in reversed(self.loop._current_transition.actions):
            if action.trigger == "retry":
                retries += 1
            else:
                return retries

        return retries

    def retry_messages(self):
        return self.loop.retry_messages(self)

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
        data = super().model_dump(**kwargs)
        return {"name": self.name, **data}

    def clone(self) -> "AgenticState":
        data = self.model_dump(exclude={"_executed", "_last_action", "_response"})
        new_state = self.__class__(**data)
        new_state._loop = self._loop
        return new_state

    def __eq__(self, other):
        if not isinstance(other, AgenticState):
            return NotImplemented

        if self.model_dump() != other.model_dump():
            return False

        if self._loop and other._loop:
            self_context = self._loop.workspace.file_context
            other_context = other._loop.workspace.file_context

            return self_context.model_dump() == other_context.model_dump()

        return True


class NoopState(AgenticState):
    def __init__(self, **data):
        super().__init__(**data)

    def _execute_action(self, action: ActionRequest):
        raise ValueError("NoopState cannot handle actions")


class Finished(NoopState):
    message: Optional[str]

    output: dict[str, Any] | None = None

    def __init__(self, message: Optional[str] = None, **kwargs):
        super().__init__(message=message)
        self.output = kwargs


class Rejected(NoopState):
    message: str

    def __init__(self, message: str, **kwargs):
        super().__init__(message=message)


class Pending(NoopState):
    def __init__(self, **data):
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
