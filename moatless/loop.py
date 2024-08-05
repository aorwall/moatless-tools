import json
import logging
import os
import random
import string
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Type, Tuple
import subprocess

import instructor
import litellm
from anthropic import Anthropic
from litellm import completion_cost, cost_per_token, token_counter
from pydantic import BaseModel, Field, PrivateAttr

from moatless.repository import GitRepository
from moatless.state import (
    AgenticState,
    Finished,
    NoopState,
    Pending,
    Rejected,
    get_state_class,
)
from moatless.trajectory import Trajectory, TrajectoryTransition, TrajectoryAction
from moatless.transition_rules import TransitionRules
from moatless.types import (
    ActionRequest,
    AssistantMessage,
    Content,
    Message,
    Response,
    UserMessage,
)
from moatless.workspace import Workspace

logger = logging.getLogger("Loop")


class AgenticLoop:
    def __init__(
        self,
        transition_rules: TransitionRules,
        workspace: Workspace,
        input_data: dict[str, Any] | None = None,
        trajectory: Trajectory | None = None,
        mocked_actions: list[dict] | None = None,
        expected_states: list[Type[AgenticState]] | None = None,
        reset_mocks_at_state: Optional[str] = None,
        verify_state_func: Optional[Callable] = None,
        max_cost: float = 0.25,
        max_actions: int = 2,
        max_transitions: int = 25,
        max_message_tokens: Optional[int] = None,
        max_retries: int = 2,
        max_rejections: int = 2,
        instructor_mode: instructor.Mode | None = None,
        metadata: dict[str, Any] | None = None,
        trajectory_path: Optional[str] = None,
        prompt_log_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Loop instance.

        Args:

        """
        self._trajectory = trajectory

        self._workspace = workspace

        self._input_data = input_data

        if trajectory_path:
            parent_dir = os.path.dirname(trajectory_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
        self._trajectory_path = trajectory_path

        if prompt_log_dir and not os.path.exists(prompt_log_dir):
            os.makedirs(prompt_log_dir)
        self._prompt_log_dir = prompt_log_dir

        self._mocked_actions = mocked_actions

        if expected_states and not verify_state_func:

            def verify_state_func(state: AgenticState):
                nonlocal expected_states
                if not expected_states:
                    raise ValueError(
                        f"No more expected states, but got {state.__class__}"
                    )
                expected_state = expected_states.pop(0)
                if not (
                    state.name == expected_state or isinstance(state, expected_state)
                ):
                    raise ValueError(
                        f"Expected state {expected_state} but got {state.__class__.__name__}"
                    )

                self.log_info(f"Verified expected next state {expected_state}")

        self._verify_state_func = verify_state_func

        self._reset_mocks_at_state = reset_mocks_at_state

        self._max_cost = max_cost
        self._max_message_tokens = max_message_tokens
        self._max_transitions = max_transitions
        self._max_actions = max_actions
        self._max_retries = max_retries
        self._max_rejections = max_rejections
        self._instructor_mode = instructor_mode

        self._transition_count = 0
        self._rejections = 0

        self._transition_rules = transition_rules

        self._initial_message = ""
        self._transitions: dict[int, TrajectoryTransition] = {}
        self._current_transition: TrajectoryTransition | None = None

        self._metadata = metadata

        self._type = "standard"

        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_trajectory_file(cls, trajectory_path: str, **kwargs):
        trajectory = Trajectory.load(trajectory_path)
        transitions = trajectory.transitions
        workspace = Workspace.from_dict(trajectory.workspace)

        return cls(
            transition_rules=transitions,
            trajectory=trajectory,
            workspace=workspace,
            **kwargs,
        )

    def persist(self, trajectory_path: str):
        self.trajectory.persist(trajectory_path)

    def retry_from_transition(
        self,
        transition_id: int,
        state_params: dict[Type[AgenticState], Any] = None,
    ):
        self.clone_transition(transition_id)
        # TODO: I'm using only state params as an easy way test out changes. Need to think about a better way to do this.
        self._transition_rules.state_params.update(state_params)

        while not self.is_finished():
            self.run_until_transition()

        if isinstance(self.state, Finished):
            return Response(status="finished", message=self.state.message or "")
        elif isinstance(self.state, Rejected):
            return Response(status="rejected", message=self.state.message or "")

        raise RuntimeError(f"Loop exited with unknown state {self.state.name}.")

    def initialize_or_load_trajectory(self, message: Optional[str] = None) -> None:
        if not self._trajectory:
            self._trajectory = Trajectory(
                "MoatlessTools",
                initial_message=message,
                persist_path=self._trajectory_path,
                workspace=self._workspace,
                transition_rules=self._transition_rules,
            )
            pending_transition = self._create_transition(
                state=Pending(),
                snapshot=self._workspace.snapshot()
            )
            self._set_current_transition(pending_transition)
        else:
            for transition in self._trajectory.transitions:
                self.set_current_transition_from_dict(transition)
                self.workspace.restore_from_snapshot(transition.get("snapshot"))

            for transition_data in self._trajectory.transitions:
                transition = self._transitions[transition_data["id"]]
                if transition_data.get("parent_id"):
                    parent = self._transitions[transition_data["parent_id"]]
                    transition.parent = parent
                    parent.children.append(transition)

    def run(self, message: Optional[str] = None) -> Response:
        """
        Run the loop and handle exceptions and cost checking.
        """

        if self.is_running():
            raise Exception("Loop is already running.")

        self.initialize_or_load_trajectory(message)

        while not self.is_finished():
            self.run_until_transition()

        if isinstance(self.state, Finished):
            return Response(status="finished", message=self.state.message or "")
        elif isinstance(self.state, Rejected):
            return Response(status="rejected", message=self.state.message or "")

        raise RuntimeError(f"Loop exited with unknown state {self.state.name}.")

    def run_until_transition(self) -> TrajectoryTransition:
        while not self.is_finished():
            total_cost = self.total_cost()
            if total_cost > self._max_cost:
                logger.warning(
                    f"{self.transition_name}: Max cost reached ({total_cost} > {self._max_cost}). Exiting."
                )
                self.trajectory.save_info({"error": "Max cost reached."})
                raise RuntimeError(
                    "The loop was aborted because the cost exceeded the limit.",
                )
            else:
                self.log_info(
                    f"Running transition {len(self._transitions)}. Current total cost: {total_cost}"
                )

            try:
                transition = self._run()
                if transition:
                    return transition
            except Exception as e:
                logger.warning(
                    f"{self.transition_name}: Failed to run loop. Error: {e}"
                )
                raise

            if self.retries() > self._max_retries:
                logger.warning(
                    f"{self.transition_name}: Max retries reached ({self._max_retries}). Exiting."
                )
                self.trajectory.save_info({"error": "Max retries reached."})
                return self.transition_to(Rejected(message="Max retries reached."))

        raise RuntimeError("Loop exited without a transition.")

    def total_cost(self):
        total_cost = 0
        for step in self._transitions.values():
            for action in step.actions:
                if action.completion_cost:
                    total_cost += action.completion_cost

        return total_cost

    def is_running(self) -> bool:
        return not isinstance(self.state, NoopState)

    def is_finished(self) -> bool:
        return isinstance(self.state, (Finished, Rejected))

    def _set_state_loop(self, state: AgenticState):
        state._set_loop(self)

    def retries(self) -> int:
        retries = 0
        for action in reversed(self._current_transition.actions):
            if action.trigger == "retry":
                retries += 1
            else:
                return retries

        return retries

    def retry_messages(self, state: AgenticState) -> list[Message]:
        messages: list[Message] = []

        if self._current_transition.name != state.name:
            return messages

        for action in self._current_transition.actions:
            if action.trigger == "retry":
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

    def _set_current_transition(self, transition: TrajectoryTransition):
        self._current_transition = transition
        self._transitions[transition.id] = transition
        self._trajectory.set_current_transition_id(transition.id)

    def set_current_transition_from_dict(self, transition_data: dict):
        state_data = transition_data.get("state", {})
        name = state_data.get("name")
        try:
            state_class = get_state_class(name)
            state = state_class(**state_data)

            transition = TrajectoryTransition(
                id=transition_data["id"],
                state=state,
                snapshot=transition_data.get("snapshot"),
                actions=[
                    TrajectoryAction(**action) for action in transition_data["actions"]
                ],
                timestamp=datetime.fromisoformat(transition_data["timestamp"]),
            )

            self._set_current_transition(transition)
            self._set_state_loop(state)
            state.init()

        except Exception as e:
            logger.exception(f"Failed to load state {name}")
            raise e

    def set_current_transition(self, transition: TrajectoryTransition):
        self._set_current_transition(transition)

    def revert_to_transition(self, transition_id: int) -> TrajectoryTransition:
        transition = self._transitions.get(transition_id)
        if transition:
            self.log_info(f"Reverting to transition {transition_id}")
            self._set_current_transition(transition)
            self.workspace.restore_from_snapshot(transition.snapshot)
            return transition
        else:
            logger.warning(
                f"Tried to revert to transition {transition_id} but it does not exist. Existing transition ids: {self._transitions.keys()}"
            )
            raise ValueError(
                f"Could not revert to transition {transition_id} as it does not exist."
            )

    def _create_transition(
        self,
        state: AgenticState,
        snapshot: dict | None = None,
        parent: TrajectoryTransition | None = None,
    ):
        transition = TrajectoryTransition(
            id=len(self._transitions) + 1, state=state, snapshot=snapshot, parent=parent
        )
        self.trajectory.create_transition(transition)
        self._transitions[transition.id] = transition
        return transition

    def clone_current_transition(self):
        cloned_state = self.state.clone()
        cloned_transition = self._create_transition(
                state=cloned_state,
                snapshot=self._current_transition.snapshot,
                parent=self._current_transition.parent,
            )
        self._set_current_transition(cloned_transition)
        return cloned_transition

    def transition_to(self, new_state: AgenticState) -> TrajectoryTransition:
        self.log_info(f"Transitioning from {self.state.name} to {new_state.name}")

        if self.transition_count() > self._max_transitions:
            new_state = Rejected(message="Max transitions exceeded.")

        if (
            new_state.max_iterations
            and self.transition_count(new_state) > new_state.max_iterations
        ):
            new_state = Rejected(
                message=f"Max transitions exceeded for state {new_state.name}."
            )

        transition = self._create_transition(
            state=new_state,
            snapshot=self.workspace.snapshot(),
            parent=self._current_transition,
        )

        if self._current_transition:
            self._current_transition.children.append(transition)

        self._set_current_transition(transition)
        self._set_state_loop(new_state)

        return transition

    def transition_count(self, state: AgenticState | None = None) -> int:
        if not state:
            return len(self._transitions)

        return len(
            [t for t in self._transitions.values() if t.state.name == state.name]
        )

    def get_previous_transitions(self, state: AgenticState | None):
        previous_transitions = []
        parent_transition = self._current_transition.parent
        while parent_transition:
            if not state or parent_transition.state.name == state.name:
                previous_transitions.insert(0, parent_transition)

            parent_transition = parent_transition.parent

        self.log_info(
            f"Found {len(previous_transitions)} previous transitions for {state.name if state else 'all states'}"
        )

        return previous_transitions

    @property
    def state(self):
        return self._current_transition.state if self._current_transition else Pending()

    @property
    def workspace(self) -> Workspace:
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
                if tool_call_id and self.instructor_mode == instructor.Mode.TOOLS:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": message.content,
                        }
                    )
                elif (
                    tool_call_id
                    and self.instructor_mode == instructor.Mode.ANTHROPIC_TOOLS
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
                    if self.instructor_mode == instructor.Mode.ANTHROPIC_TOOLS:
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
                    elif self.instructor_mode == instructor.Mode.TOOLS:
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

                        if self.state.model.startswith("deepseek"):
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

    def _run(self) -> TrajectoryTransition | None:
        """
        Run the loop for one iteration.

        Returns:

        """
        if self.is_finished():
            self.log_info("Loop already finished.")
            return None

        if isinstance(self.state, Pending):
            logger.info("Initializing first state.")
            initial_state = self._transition_rules.create_initial_state(
                **(self._input_data or {})
            )
            return self.transition_to(initial_state)

        action, cost, input_tokens, output_tokens = self._next_action()

        self.log_info(f"Received new action {action.action_name}.")
        response = self.state.handle_action(action)

        self._current_transition.actions.append(
            TrajectoryAction(
                action=action,
                trigger=response.trigger,
                retry_message=response.retry_message,
                completion_cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        self.trajectory.update_transition(self._current_transition)

        if not response.trigger:
            self.log_info(
                f"{self.state.name}: No trigger in action response. Staying in the same state."
            )
            return None

        self.log_info(f"Received response with trigger {response.trigger}")

        if response.trigger == "retry":
            self.log_info(f"Retry requested. {response.retry_message}")
            return None

        try:
            next_state = self._transition_rules.next_state(
                source=self.state,
                trigger=response.trigger,
                data=response.output,
            )
        except Exception:
            logger.exception(
                f"{self.transition_name}: Failed to initiate next state with trigger {response.trigger} and output {response.output}"
            )
            raise

        if not next_state:
            raise ValueError(
                f"No transition found for {self.state.name} with trigger {response.trigger}"
            )

        if response.trigger == "rejected" and next_state.__class__ != Rejected:
            self._rejections += 1
            next_state = Rejected(
                message=f"Got {self._rejections} rejections, aborting."
            )
        else:
            self._rejections = 0

        return self.transition_to(next_state)

    @property
    def instructor_mode(self):
        if self._instructor_mode:
            return self._instructor_mode

        if "gpt" in self.state.model:
            return instructor.Mode.TOOLS

        if self.state.model.startswith("claude"):
            return instructor.Mode.ANTHROPIC_TOOLS

        if self.state.model.startswith("openrouter/anthropic/claude"):
            return instructor.Mode.TOOLS

        return instructor.Mode.JSON

    def _next_mock_action(
        self,
    ) -> ActionRequest | None:
        if not self._mocked_actions:
            return None

        if self._reset_mocks_at_state and self.state.name == self._reset_mocks_at_state:
            self.log_info(f"Resetting mocked actions at state {self.state.name}")
            self._mocked_actions = []
            return None

        action = self._mocked_actions.pop(0)

        if self.state.action_type():
            try:
                self.log_info(
                    f"Return mocked response with type {self.state.action_type().__name__} ({len(self._mocked_actions)} left)."
                )
                return self.state.action_type().model_validate(action)

            except Exception:
                logger.error(
                    f"{self.transition_name}: Failed to parse {action} to {self.state.action_type().__name__} in state {self.state.name}"
                )
                raise
        elif "content" in action:
            self.log_info(f"Return mocked response ({len(self._mocked_actions)} left).")
            return Content(content=action["content"])

        else:
            raise ValueError(f"Mocked action {action} does not have 'content' field.")

    def _next_action(
        self,
    ) -> tuple[ActionRequest, Optional[float], Optional[int], Optional[int]]:
        messages = self._to_completion_messages()
        self.log_info(f"Create completion with {len(messages)} messages")

        if self._verify_state_func:
            self._verify_state_func(self.state)

        mocked_action = self._next_mock_action()
        if mocked_action:
            return mocked_action, None, None, None

        metadata = {}
        if self._metadata:
            metadata.update(self._metadata)
        metadata["generation_name"] = self.state.name

        tokens = token_counter(messages=messages[-1:])
        if self._max_message_tokens and tokens > self._max_message_tokens:
            raise ValueError(f"Too many tokens in the new message: {tokens}")

        self.log_info(f"Do completion request to {self.state.model}")

        if self.state.model.startswith("claude") and self.state.action_type():
            try:
                anthropic_client = instructor.from_anthropic(
                    Anthropic(),
                    mode=self.instructor_mode,
                )

                action_request, completion_response = (
                    anthropic_client.chat.completions.create_with_completion(
                        model=self.state.model,
                        max_tokens=self.state.max_tokens,
                        temperature=self.state.temperature,
                        # stop=self.state.stop_words(),
                        response_model=self.state.action_type(),
                        messages=messages,
                    )
                )

                self.log_info(
                    f"Input tokens: {completion_response.usage.input_tokens}, Output tokens: {completion_response.usage.output_tokens}"
                )
                (
                    prompt_tokens_cost_usd_dollar,
                    completion_tokens_cost_usd_dollar,
                ) = cost_per_token(
                    model=self.state.model,
                    prompt_tokens=completion_response.usage.input_tokens,
                    completion_tokens=completion_response.usage.output_tokens,
                )
                _final_cost = (
                    prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar
                )
            except Exception as e:
                self._log_prompt(messages, error=traceback.format_exc())
                raise e

            self._log_prompt(messages, completion_response.content)
            return (
                action_request,
                _final_cost,
                completion_response.usage.input_tokens,
                completion_response.usage.output_tokens,
            )

        if self.state.action_type() is None:
            completion_response = litellm.completion(
                model=self.state.model,
                max_tokens=self.state.max_tokens,
                temperature=self.state.temperature,
                stop=self.state.stop_words(),
                metadata=metadata,
                messages=messages,
            )
            action_request = Content(
                content=completion_response.choices[0].message.content
            )
        else:
            client = instructor.from_litellm(
                litellm.completion, mode=self.instructor_mode
            )

            try:
                action_request, completion_response = (
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
            except Exception as e:
                self._log_prompt(messages, error=traceback.format_exc())
                raise e

        try:
            cost = completion_cost(
                completion_response=completion_response,
                model=self.state.model,
            )
        except Exception as e:
            self.log_info(f"Error calculating completion cost: {e}")
            cost = 0

        self._log_prompt(
            messages, [completion_response.choices[0].message.model_dump()], error=None
        )
        prompt_tokens = completion_response.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = completion_response.get("usage", {}).get(
            "completion_tokens", 0
        )
        return action_request, cost, prompt_tokens, completion_tokens

    def _log_prompt(
        self,
        messages: list[dict],
        completion: Any | None = None,
        error: Optional[str] = None,
    ):
        if not self._prompt_log_dir:
            return

        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        prompt_path = (
            f"{self._prompt_log_dir}/{self._current_transition.id}_{self.state.name}"
        )
        if self.retries() > 0:
            prompt_path += f"_retry_{self.retries()}"

        prompt_path += f"_{time_str}.md"

        with open(prompt_path, "w") as f:
            f.write("\n\n# Completion\n")

            f.write("\n\n## Input\n")
            for message in messages:
                f.write(f"\n\n### {message['role']}\n\n")

                if "content" in message:
                    if isinstance(message["content"], str):
                        f.write(message["content"])
                    elif isinstance(message["content"], list):
                        for content in message["content"]:
                            if isinstance(content, str):
                                f.write(content)
                            if isinstance(content, dict) and "content" in content:
                                f.write(content["content"])
                            else:
                                f.write(
                                    f"\n\n```json\n{json.dumps(content, indent=2)}\n```"
                                )
                elif isinstance(message.get("content"), list):
                    for block in message["content"]:
                        f.write(f"\n\n### {block['tool_use_id']}\n")
                        f.write(block["content"])
                else:
                    f.write(f"\n\n```json\n{json.dumps(message, indent=2)}\n```")

            if completion:
                f.write("\n\n## Output\n")

                if isinstance(completion, list):
                    for block in completion:
                        if isinstance(block, BaseModel):
                            block = block.model_dump()

                        if isinstance(block, dict):
                            if block.get("content"):
                                f.write(f"{block.get('content')}\n")
                            else:
                                f.write(f"```json\n{json.dumps(block, indent=2)}\n```")
                        else:
                            f.write(f"```json\n{json.dumps(block, indent=2)}\n```")
                else:
                    f.write(f"```json\n{json.dumps(completion, indent=2)}\n```")

            if error:
                f.write("\n\n# Error\n")
                f.write(f"\n```\n{error}\n```\n")

    def log_info(self, message: str):
        logger.info(f"{self.transition_name}: {message}")

    @property
    def transition_name(self):
        if self._current_transition:
            return (
                f"{self._current_transition.state.name}:{self._current_transition.id}"
            )
        else:
            return "No transition"


def generate_call_id():
    prefix = "call_"
    chars = string.ascii_letters + string.digits
    length = 24

    random_chars = "".join(random.choices(chars, k=length))

    random_string = prefix + random_chars

    return random_string
