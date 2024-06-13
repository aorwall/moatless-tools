import json
import logging
from abc import abstractmethod, ABC
from typing import Optional, Callable, Type, Any, List

import instructor
import litellm
from langfuse.decorators import observe
from litellm import token_counter, completion_cost

from moatless import FileRepository, Workspace
from moatless.file_context import FileContext
from pydantic import BaseModel, PrivateAttr, ValidationError, Field

from moatless.loop.utils import generate_call_id
from moatless.settings import Settings
from moatless.trajectory import Trajectory
from moatless.types import ActionSpec, FileWithSpans, ActionRequest, Message, Content

logger = logging.getLogger("Loop")


class BaseState(ABC, BaseModel):
    include_message_history: bool = Field(
        default=False,
        description="The message history from previous initations should be included in the completion request",
    )
    model: str = Field(
        Settings.agent_model, description="The model to use for completion"
    )
    temperature: float = Field(0.0, description="The temperature to use for completion")
    max_tokens: int = Field(
        1000, description="The maximum number of tokens to generate"
    )

    _loop: Optional["Loop"] = PrivateAttr(None)

    def __init__(self, **data):
        super().__init__(**data)
        self._loop = None

    @abstractmethod
    def handle_action(self, action: ActionRequest):
        raise NotImplementedError

    def _set_loop(self, loop: "Loop"):
        self._loop = loop
        self.init()

    def __str__(self):
        return self.__class__.__name__

    @property
    def loop(self) -> "Loop":
        assert self._loop is not None, "Loop has not been set"
        return self._loop

    @property
    def workspace(self) -> Workspace:
        return self.loop.workspace

    @property
    def trajectory(self) -> Trajectory:
        return self.workspace.trajectory

    @property
    def file_repo(self) -> FileRepository:
        return self.workspace.file_repo

    @property
    def file_context(self) -> FileContext:
        return self.workspace.file_context

    def create_file_context(self, files: List[FileWithSpans]) -> FileContext:
        return self.workspace.create_file_context(files)

    def init(self):
        """Initialization logic for the state."""
        pass

    def transition_to(self, new_state: "BaseState"):
        self.loop.transition_to(new_state)

    def finish(self, message: str):
        # TODO!!
        print(message)

    def messages(self) -> list[Message]:
        return []

    def system_prompt(self) -> str:
        return ""

    def action_type(self) -> Optional[Type[BaseModel]]:
        raise NotImplementedError

    def stop_words(self):
        return []


class InitialState(BaseState):
    message: str = Field(..., description="The initial message to start the loop")

    def __init__(self, message: str, **data):
        super().__init__(message=message, **data)


class NoopState(BaseState):

    def __init__(self, **data):
        super().__init__(**data)

    def handle_action(self, action: ActionRequest):
        raise NotImplementedError


class Finished(NoopState):
    reason: str

    def __init__(self, reason: str):
        super().__init__(reason=reason)


class Rejected(NoopState):
    reason: str

    def __init__(self, reason: str):
        super().__init__(reason=reason)


class Pending(NoopState):

    def __init__(self, **data):
        super().__init__(**data)


class Response(BaseModel):
    message: str


class Loop:

    def __init__(
        self,
        initial_state: Type[InitialState],
        workspace: Workspace,
        mocked_actions: Optional[List[ActionRequest]] = None,
        max_cost: float = 0.25,
        max_transitions: int = 25,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the Loop instance.

        Args:
            model (Optional[str]): Optional LLM model name.
            mocked_actions (Optional[List[Message]]): Optional list of mocked actions.
            trajectory (Optional[Trajectory]): Optional trajectory instance.
            max_cost (float): Maximum allowed cost for the loop execution.
        """

        self._workspace = workspace
        self._mocked_actions = mocked_actions
        self._max_cost = max_cost
        self._max_transitions = max_transitions
        self._transition_count = 0

        self._message_history = []

        self._initial_state = initial_state
        self._state: BaseState = Pending()

        self._metadata = metadata

    @observe()
    def run(self, message: str):
        """
        Run the loop and handle exceptions and cost checking.
        """

        if self.is_running():
            raise Exception("Loop is already running.")

        self.transition_to(self._initial_state(message=message))

        while self.is_running():
            try:
                action = self._run_completion()
                logger.info(f"{self.state}: Received new message.")
                self.state.handle_action(action)
            except Exception as e:
                logger.warning(f"Failed to run loop. Error: {e}")
                raise

            if self._check_cost_exceeded():
                raise Exception(
                    "The search was aborted because the cost exceeded the limit."
                )

        if isinstance(self.state, Finished):
            return self.state.reason
        elif isinstance(self.state, Rejected):
            return self.state.reason

        raise Exception(f"Loop exited with unknown state {self.state}.")

    def is_running(self) -> bool:
        return not isinstance(self.state, NoopState)

    def _set_state_loop(self, state: BaseState):
        state._set_loop(self)

    def _is_initial_state(self):
        logger.info(
            f"{self.state.__class__.__name__} == {self._initial_state.__name__}"
        )
        return self.state.__class__.__name__ == self._initial_state.__name__

    def transition_to(self, new_state: BaseState):
        if (
            isinstance(new_state, Finished) or isinstance(new_state, Rejected)
        ) and not self._is_initial_state():
            logger.info(
                f"{self.state} is {new_state}. Will transition back to the intiial state {self._initial_state.__name__}."
            )
            new_state = self._initial_state(message=new_state.reason)

        logger.info(f"Transitioning from {self.state} to {new_state}")

        self._transition_count += 1
        if self._transition_count > self._max_transitions:
            self.transition_to(Rejected(reason="Number of transitions exceeded."))
            raise RuntimeError("Maximum number of transitions exceeded.")

        self.trajectory.new_transition(str(new_state), new_state.model_dump())

        self._state = new_state
        self._set_state_loop(self.state)

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [tool.openai_tool_spec() for tool in self.state.actions()]

    @property
    def state(self):
        return self._state

    @property
    def workspace(self):
        return self._workspace

    @property
    def trajectory(self):
        return self._workspace.trajectory

    def messages(self) -> list[dict]:
        return [
            message.model_dump(exclude_none=True) for message in self.state.messages()
        ]

    def _run_completion(self) -> ActionRequest:
        messages = [{"role": "system", "content": self.state.system_prompt()}]

        tool_call_id = None
        state_messages = self.state.messages()
        for message in state_messages:
            if message.role == "user":
                if tool_call_id:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": message.content,
                        }
                    )
                else:
                    messages.append({"role": "user", "content": message.content})
            elif message.role == "assistant":
                if message.action:
                    tool_call_id = generate_call_id()
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": message.action.__class__.__name__,
                                        "arguments": message.action.model_dump_json(),
                                    },
                                }
                            ],
                        }
                    )
                else:
                    tool_call_id = None
                    messages.append({"role": "assistant", "content": message.content})

        logger.info(f"{self.state} Create completion with {len(messages)} messages")

        if self._mocked_actions is not None:
            if len(self._mocked_actions) == 0:
                raise Exception("No more mocked responses available.")

            action = self._mocked_actions.pop(0)
            logger.info(
                f"{self.state} Return mocked response ({len(self._mocked_actions)} left)."
            )
            return action

        metadata: dict[str, Any] = {}
        if self._metadata:
            metadata.update(self._metadata)
        metadata["generation_name"] = str(self.state)

        tokens = token_counter(messages=messages[-1:])
        if tokens > Settings.max_message_tokens:
            raise ValueError(f"Too many tokens in the new message: {tokens}")

        if self.state.action_type() is None:
            completion_response = litellm.completion(
                model=self.state.model,
                max_tokens=self.state.max_tokens,
                temperature=self.state.temperature,
                stop=self.state.stop_words(),
                metadata=metadata,
                messages=messages,
            )
            action = Content(content=completion_response.choices[0].message.content)
        else:
            client = instructor.from_litellm(litellm.completion)
            action, completion_response = (
                client.chat.completions.create_with_completion(
                    model=self.state.model,
                    max_tokens=self.state.max_tokens,
                    temperature=self.state.temperature,
                    stop=self.state.stop_words(),
                    response_model=self.state.action_type(),
                    metadata=metadata,
                    messages=messages,
                )
            )

        cost = None
        if completion_response:
            try:
                cost = completion_cost(completion_response=completion_response)
            except Exception as e:
                logger.info(f"Error calculating completion cost: {e}")

        self.workspace.trajectory.save_action(action=action, completion_cost=cost)

        # self.trajectory.save_action(
        #    name=action.name(),
        #    input=action.model_dump()
        # )

        return action

    def _check_cost_exceeded(self) -> bool:
        """
        Check if the total cost has exceeded the maximum allowed cost.

        Returns:
            bool: True if the cost has exceeded, False otherwise.
        """

        total_cost = self.workspace.trajectory.total_cost()
        if total_cost > self._max_cost:
            logger.warning(f"Max cost reached ({self._max_cost}). Exiting.")
            self.state.transition_to(Rejected(reason="max_cost_reached"))
            return True

        return False
