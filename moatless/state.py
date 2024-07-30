import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

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
    include_message_history: bool = Field(
        default=False,
        description="The message history from previous initations should be included in the completion request",
    )
    model: Optional[str] = Field(
        default=None, description="The model to use for completion"
    )
    temperature: float = Field(0.0, description="The temperature to use for completion")
    max_tokens: int = Field(
        1000, description="The maximum number of tokens to generate"
    )
    max_iterations: Optional[int] = Field(
        None, description="The maximum number of transitions to this state."
    )

    _loop: Optional["AgenticLoop"] = PrivateAttr(None)  # noqa: F821

    # model_config = ConfigDict(extra='allow')

    def __init__(self, **data):
        super().__init__(**data)
        self._loop = None

    @abstractmethod
    def handle_action(self, action: ActionRequest) -> ActionResponse:
        raise NotImplementedError

    def _set_loop(self, loop: "AgenticLoop"):  # noqa: F821
        self._loop = loop
        self.init()

    def __str__(self):
        return self.__class__.__name__

    @property
    def name(self):
        return self.__class__.__name__

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
            if action.retry_message:
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

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["name"] = self.name
        return data


class NoopState(AgenticState):
    def __init__(self, **data):
        super().__init__(**data)

    def handle_action(self, action: ActionRequest):
        raise NotImplementedError


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
