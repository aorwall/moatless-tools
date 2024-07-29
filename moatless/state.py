from abc import ABC, abstractmethod
from typing import Optional, List, Type, Any

from pydantic import BaseModel, Field, PrivateAttr

from moatless import Workspace, FileRepository
from moatless.file_context import FileContext
from moatless.types import (
    ActionRequest,
    ActionResponse,
    FileWithSpans,
    Message,
    AssistantMessage,
    UserMessage,
)
from copy import deepcopy
from inspect import signature
import inspect
from typing import Any, Dict


class AgenticState(ABC, BaseModel):
    include_message_history: bool = Field(
        default=False,
        description="The message history from previous initations should be included in the completion request",
    )
    model: str = Field(default="gpt-4o", description="The model to use for completion")
    temperature: float = Field(0.0, description="The temperature to use for completion")
    max_tokens: int = Field(
        1000, description="The maximum number of tokens to generate"
    )
    max_iterations: int = Field(
        6, description="The maximum number of transitions to this state."
    )

    _loop: Optional["AgenticLoop"] = PrivateAttr(None)

    def __init__(self, **data):
        super().__init__(**data)
        self._loop = None

    @abstractmethod
    def handle_action(self, action: ActionRequest) -> ActionResponse:
        raise NotImplementedError

    def _set_loop(self, loop: "AgenticLoop"):
        self._loop = loop
        if self._loop is not None:
            self.init()

    def __str__(self):
        return self.__class__.__name__

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def loop(self) -> "AgenticLoop":
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

    def create_file_context(self, files: List[FileWithSpans] = []) -> FileContext:
        return self.workspace.create_file_context(files)

    def init(self):
        """Initialization logic for the state."""
        pass

    def transition_to(self, new_state: "AgenticState"):
        self.loop.transition_to(new_state)

    def finish(self, message: str):
        # TODO!!
        print(message)

    def messages(self) -> list[Message]:
        return []

    @classmethod
    def required_fields(cls) -> set[str]:
        return set()

    def retries(self) -> int:
        retries = 0
        for action in reversed(self.trajectory.current_step.actions):
            if action.retry_message:
                retries += 1
            else:
                return retries

        return retries

    def retry_messages(self):
        return self.loop.retry_messages(self)

    def system_prompt(self) -> str:
        return ""

    def action_type(self) -> Optional[Type[ActionRequest]]:
        """
        The type of the action to expect in the completion response.
        If not set a content string is expected.
        """
        raise NotImplementedError

    @classmethod
    def copy_state(cls, state: 'AgenticState', loop: 'AgenticLoop') -> 'AgenticState':
        """Class method to create a deep copy of a state."""
        return state.deep_copy(loop)
    
    def stop_words(self) -> Optional[List[str]]:
        return None


class NoopState(AgenticState):

    def __init__(self, **data):
        super().__init__(**data)

    def handle_action(self, action: ActionRequest):
        raise NotImplementedError


class Finished(NoopState):
    message: str

    output: Optional[dict[str, Any]] = None

    def __init__(self, message: str, **kwargs):
        super().__init__(message=message)
        self.output = kwargs


class Rejected(NoopState):
    message: str

    def __init__(self, message: str):
        super().__init__(message=message)


class Pending(NoopState):

    def __init__(self, **data):
        super().__init__(**data)


    # def __deepcopy__(self, memo=None) -> 'AgenticState':
    #     """Create a deep copy of the state while respecting Pydantic's structure and preserving workspace."""
    #     if memo is None:
    #         memo = {}

    #     def _copy_value(value: Any, memo: Dict[int, Any]) -> Any:
    #         if id(value) in memo:
    #             return memo[id(value)]
    #         if isinstance(value, (int, float, str, bool, type(None))):
    #             return value
    #         elif isinstance(value, list):
    #             new_list = []
    #             memo[id(value)] = new_list
    #             new_list.extend(_copy_value(item, memo) for item in value)
    #             return new_list
    #         elif isinstance(value, dict):
    #             new_dict = {}
    #             memo[id(value)] = new_dict
    #             for k, v in value.items():
    #                 new_dict[_copy_value(k, memo)] = _copy_value(v, memo)
    #             return new_dict
    #         elif isinstance(value, Workspace):
    #             # Don't deep copy the workspace, just return the same instance
    #             return value
    #         else:
    #             return deepcopy(value, memo)

    #     # Create a dictionary of field values
    #     field_values = {}
    #     for name, field in self.__fields__.items():
    #         if hasattr(self, name):
    #             value = getattr(self, name)
    #             field_values[name] = _copy_value(value, memo)

    #     # Create a new instance using the model_construct method
    #     new_state = self.__class__.model_construct(**field_values)

    #     # Copy any additional attributes that aren't Pydantic fields
    #     for name, value in self.__dict__.items():
    #         if name not in self.__fields__ and not name.startswith('_'):
    #             copied_value = _copy_value(value, memo)
    #             setattr(new_state, name, copied_value)

    #     # Store the new object in the memo dictionary
    #     memo[id(self)] = new_state

    #     return new_state