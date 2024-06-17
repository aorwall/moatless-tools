import logging
import random
import string
from typing import Optional, Type, Any, List, Tuple, Callable

import instructor
import litellm
from litellm import token_counter, completion_cost, ModelResponse
from pydantic import BaseModel, Field

from moatless import Workspace
from moatless.state import (
    AgenticState,
    InitialState,
    NoopState,
    Finished,
    Rejected,
    Pending,
)
from moatless.types import Response, Message, AssistantMessage, UserMessage
from moatless.trajectory import Trajectory
from moatless.types import (
    ActionRequest,
    Content,
)

logger = logging.getLogger("Loop")


class Transition(BaseModel):
    trigger: str
    source: Type[AgenticState]
    dest: Type[AgenticState]
    required_fields: set[str] = Field(default_factory=set)


class Transitions:

    def __init__(
        self,
        initial_state: Type[InitialState],
        transitions: List[Transition],
        global_params: Optional[dict[str, Any]] = None,
        state_params: Optional[dict[Type[AgenticState], dict[str, Any]]] = None,
    ):
        self._initial_state = initial_state
        self._global_params = global_params or {}
        self._state_params = state_params or {}
        self._source_trigger_index: dict[tuple[Type[AgenticState], str], list] = {}

        for transition in transitions:
            if (
                transition.source,
                transition.trigger,
            ) not in self._source_trigger_index:
                self._source_trigger_index[(transition.source, transition.trigger)] = []
            self._source_trigger_index[(transition.source, transition.trigger)].append(
                transition
            )

    def find_transition_by_source_and_trigger(
        self, source: Type[AgenticState], trigger: str
    ) -> List[Transition]:
        return self._source_trigger_index.get((source, trigger), [])

    def initial_state(self, **data) -> InitialState:
        return self._initial_state(**self._global_params, **data)

    def next_state(
        self, source: AgenticState, trigger: str, data: dict[str, Any]
    ) -> Optional[AgenticState]:
        transitions = self.find_transition_by_source_and_trigger(
            source.__class__, trigger
        )
        for transition in transitions:
            if transition.required_fields.issubset(data.keys()):
                params = {}
                params.update(self._global_params)
                params.update(self._state_params.get(transition.dest, {}))
                return transition.dest(**params, **data)
        return None


class AgenticLoop:

    def __init__(
        self,
        transitions: Transitions,
        workspace: Workspace,
        mocked_actions: Optional[List[dict]] = None,
        verify_state_func: Optional[Callable] = None,
        max_cost: float = 0.25,
        max_transitions: int = 25,
        max_message_tokens: int = 16000,
        max_retries: int = 2,
        max_rejections: int = 2,
        metadata: Optional[dict[str, Any]] = None,
        trajectory_path: Optional[str] = None,
    ):
        """
        Initialize the Loop instance.

        Args:

        """

        self._workspace = workspace
        self._trajectory_path = trajectory_path

        self._mocked_actions = mocked_actions
        self._verify_state_func = verify_state_func

        self._max_cost = max_cost
        self._max_message_tokens = max_message_tokens
        self._max_transitions = max_transitions
        self._max_retries = max_retries
        self._max_rejections = max_rejections

        self._transition_count = 0
        self._rejections = 0

        self._transitions = transitions

        self._initial_message = ""
        self._state: AgenticState = Pending()

        self._metadata = metadata

    def run(self, message: str) -> Response:
        """
        Run the loop and handle exceptions and cost checking.
        """

        if self.is_running():
            raise Exception("Loop is already running.")

        self._trajectory = Trajectory(
            "AgenticLoop", initial_message=message, persist_path=self._trajectory_path
        )

        self.transition_to(self._transitions.initial_state())

        while self.is_running():
            try:
                self._run()
            except Exception as e:
                logger.warning(f"Failed to run loop. Error: {e}")
                raise

            if self.retries() > self._max_retries:
                logger.warning(f"Max retries reached ({self._max_retries}). Exiting.")
                self.trajectory.save_info({"error": "Max retries reached."})
                return Response(
                    status="rejected",
                    message="The loop was aborted because the number of retries exceeded the limit.",
                )

            total_cost = self._trajectory.total_cost()
            if total_cost > self._max_cost:
                logger.warning(f"Max cost reached ({total_cost} > {self._max_cost}). Exiting.")
                self.trajectory.save_info({"error": "Max cost reached."})
                raise RuntimeError(
                    "The loop was aborted because the cost exceeded the limit.",
                )

        if isinstance(self.state, Finished):
            return Response(status="finished", message=self.state.message)
        elif isinstance(self.state, Rejected):
            return Response(status="rejected", message=self.state.message)

        raise RuntimeError(f"Loop exited with unknown state {self.state}.")

    def is_running(self) -> bool:
        return not isinstance(self.state, NoopState)

    def _set_state_loop(self, state: AgenticState):
        state._set_loop(self)

    def retries(self) -> int:
        retries = 0
        for action in reversed(self.trajectory.current_step.actions):
            if action.retry_message:
                retries += 1
            else:
                return retries

        return retries

    def retry_messages(self, state: AgenticState) -> List[Message]:
        messages: list[Message] = []

        if self.trajectory.current_step.name != state.name:
            return messages

        current_step = self.trajectory.current_step
        for action in current_step.actions:
            if action.retry_message:
                if isinstance(action.action, Content):
                    messages.append(
                        AssistantMessage(
                            content=action.action.content,
                        )
                    )
                else:
                    messages.append(AssistantMessage(action=action.action))

                messages.append(
                    UserMessage(
                        content=action.retry_message,
                    )
                )

        return messages

    def transition_to(self, new_state: AgenticState):
        logger.info(f"Transitioning from {self.state} to {new_state}")

        self._transition_count += 1
        if self._transition_count > self._max_transitions:
            new_state = Rejected(message="Max transitions exceeded.")

        if self.trajectory.transition_count(new_state) > new_state.max_iterations:
            new_state = Rejected(message=f"Max iterations exceeded for state {new_state.name}.")

        self.trajectory.new_transition(new_state)

        self._state = new_state
        self._set_state_loop(self.state)

    @property
    def state(self):
        return self._state

    @property
    def workspace(self):
        return self._workspace

    @property
    def trajectory(self):
        return self._trajectory

    def _to_completion_messages(self) -> list[dict]:
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
                    tool_call_id = None
                    messages.append({"role": "assistant", "content": message.content})

        return messages

    def _run(self):
        if not self.is_running():
            logger.info("Loop is not running.")
            return

        action, completion_response = self._next_action()

        cost = None
        if completion_response:
            try:
                cost = completion_cost(completion_response=completion_response)
            except Exception as e:
                logger.info(f"Error calculating completion cost: {e}")

        logger.info(f"{self.state}: Received new action {action.action_name}.")
        response = self.state.handle_action(action)

        self._trajectory.save_action(
            action=action,
            output=response.output,
            retry_message=response.retry_message,
            completion_cost=cost,
        )

        if not response.trigger:
            logger.info(
                f"{self.state}: No transition found. Staying in the same state."
            )
            return

        if response.trigger == "retry":
            logger.info(f"{self.state}: Retry requested. {response.retry_message}")
            return

        if response.trigger == "rejected":
            self._rejections += 1
            next_state = Rejected(message=f"Got {self._rejections} rejections, aborting.")
        else:
            self._rejections = 0
            try:
                next_state = self._transitions.next_state(
                    source=self.state,
                    trigger=response.trigger,
                    data=response.output,
                )
            except Exception as e:
                logger.error(f"Failed to initiate next state with trigger {response.trigger} and output {response.output}")
                raise

        if not next_state:
            raise ValueError(
                f"No transition found for {self.state} with trigger {response.trigger}"
            )

        logger.info(f"{self.state}: Transitioning to {next_state.name}")
        self.transition_to(next_state)

    def _next_action(self) -> Tuple[ActionRequest, Optional[ModelResponse]]:
        messages = self._to_completion_messages()
        logger.info(f"{self.state} Create completion with {len(messages)} messages")

        if self._verify_state_func:
            self._verify_state_func(self.state)

        if self._mocked_actions is not None:
            if len(self._mocked_actions) == 0:
                raise Exception("No more mocked responses available.")

            action = self._mocked_actions.pop(0)
            if self.state.action_type():
                try:
                    logger.info(
                        f"{self.state} Return mocked response with type {self.state.action_type().__name__} ({len(self._mocked_actions)} left)."
                    )
                    return self.state.action_type().model_validate(action), None
                except Exception as e:
                    logger.error(f"Failed to parse {action} to {self.state.action_type().__name__} in state {self.state.name}")
                    raise
            elif "content" in action:
                logger.info(f"{self.state} Return mocked response ({len(self._mocked_actions)} left).")
                return Content(content=action["content"]), None
            else:
                raise ValueError(f"Mocked action {action} does not have 'content' field.")

        metadata = {}
        if self._metadata:
            metadata.update(self._metadata)
        metadata["generation_name"] = str(self.state)

        tokens = token_counter(messages=messages[-1:])
        if tokens > self._max_message_tokens:
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
            return Content(content=completion_response.choices[0].message.content), completion_response
        else:

            if "mixtral" in self.state.model:
                mode = instructor.Mode.MISTRAL_TOOLS
            else:
                mode = instructor.Mode.TOOLS

            client = instructor.from_litellm(litellm.completion, mode=mode)
            return (
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


def generate_call_id():
    prefix = "call_"
    chars = string.ascii_letters + string.digits
    length = 24

    random_chars = "".join(random.choices(chars, k=length))

    random_string = prefix + random_chars

    return random_string
