import json
import logging
import os
import random
import string
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Type

import instructor
import litellm
from anthropic import Anthropic
from litellm import completion_cost, cost_per_token, token_counter
from pydantic import BaseModel, Field

from moatless.state import (
    AgenticState,
    Finished,
    NoopState,
    Pending,
    Rejected,
)
from moatless.trajectory import Trajectory, TrajectoryTransition, TrajectoryAction
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


class TransitionRule(BaseModel):
    trigger: str
    source: type[AgenticState]
    dest: type[AgenticState]
    required_fields: set[str] = Field(default_factory=set)
    excluded_fields: set[str] = Field(default_factory=set)


class TransitionRules:
    def __init__(
        self,
        initial_state: type[AgenticState],
        transition_rules: list[TransitionRule],
        global_params: dict[str, Any] | None = None,
        state_params: dict[type[AgenticState], dict[str, Any]] | None = None,
    ):
        self._initial_state = initial_state
        self._global_params = global_params or {}
        self._state_params = state_params or {}
        self._source_trigger_index: dict[tuple[type[AgenticState], str], list] = {}

        for transition_rule in transition_rules:
            if (
                transition_rule.source,
                transition_rule.trigger,
            ) not in self._source_trigger_index:
                self._source_trigger_index[
                    (transition_rule.source, transition_rule.trigger)
                ] = []
            self._source_trigger_index[
                (transition_rule.source, transition_rule.trigger)
            ].append(transition_rule)

    def find_transition_rule_by_source_and_trigger(
        self, source: type[AgenticState], trigger: str
    ) -> list[TransitionRule]:
        return self._source_trigger_index.get((source, trigger), [])

    def initial_state(self, **data) -> AgenticState:
        params = {}
        params.update(self._global_params)
        params.update(self._state_params.get(self._initial_state, {}))
        params.update(data)
        return self._initial_state(**params)

    def next_state(
        self, source: AgenticState, trigger: str, data: dict[str, Any]
    ) -> AgenticState | None:
        transition_ruless = self.find_transition_rule_by_source_and_trigger(
            source.__class__, trigger
        )
        for transition_rule in transition_ruless:
            if transition_rule.required_fields.issubset(data.keys()):
                params = {}
                params.update(self._global_params)
                params.update(self._state_params.get(transition_rule.dest, {}))

                if transition_rule.excluded_fields:
                    data = {
                        k: v
                        for k, v in data.items()
                        if k not in transition_rule.excluded_fields
                    }

                params.update(data)
                return transition_rule.dest(**params)
        return None


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
    for module in sys.modules.values():
        if hasattr(module, name):
            cls = getattr(module, name)
            if isinstance(cls, type) and issubclass(cls, AgenticState):
                return cls

    raise ValueError(f"State {name} not found")


class AgenticLoop:
    def __init__(
        self,
        transition_rules: TransitionRules,
        workspace: Workspace,
        trajectory: Trajectory | None = None,
        mocked_actions: list[dict] | None = None,
        reset_mocks_at_state: Optional[str] = None,
        verify_state_func: Optional[Callable] = None,
        max_cost: float = 0.25,
        max_transitions: int = 25,
        max_message_tokens: Optional[int] = None,
        max_retries: int = 2,
        max_rejections: int = 2,
        instructor_mode: instructor.Mode | None = None,
        metadata: dict[str, Any] | None = None,
        trajectory_path: Optional[str] = None,
        prompt_log_dir: Optional[str] = None,
    ):
        """
        Initialize the Loop instance.

        Args:

        """

        self._trajectory = trajectory

        self._workspace = workspace

        if trajectory_path:
            parent_dir = os.path.dirname(trajectory_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
        self._trajectory_path = trajectory_path

        if prompt_log_dir and not os.path.exists(prompt_log_dir):
            os.makedirs(prompt_log_dir)
        self._prompt_log_dir = prompt_log_dir

        self._mocked_actions = mocked_actions
        self._reset_mocks_at_state = reset_mocks_at_state
        self._verify_state_func = verify_state_func

        self._max_cost = max_cost
        self._max_message_tokens = max_message_tokens
        self._max_transitions = max_transitions
        self._max_retries = max_retries
        self._max_rejections = max_rejections
        self._instructor_mode = instructor_mode

        self._transition_count = 0
        self._rejections = 0

        self._transition_rules = transition_rules

        self._initial_message = ""
        self._transitions: dict[int, TrajectoryTransition] = {}
        self._current_state: AgenticState = Pending()
        self._current_transition: TrajectoryTransition | None = None

        self._metadata = metadata

    @classmethod
    def from_trajectory_file(
        cls, transitions: TransitionRules, trajectory_path: str, **kwargs
    ):
        trajectory = Trajectory.load(trajectory_path)
        workspace = Workspace.from_dict(trajectory.workspace)

        return cls(
            transition_rules=transitions,
            trajectory=trajectory,
            workspace=workspace,
            **kwargs,
        )

    def retry_from_transition(
        self,
        transition_id: int,
        state_params: dict[Type[AgenticState], Any] = None,
    ):
        self.revert_to_transition(transition_id)
        # TODO: I'm using only state params as an easy way test out changes. Need to think about a better way to do this.
        self._transition_rules._state_params.update(state_params)

        # TODO: DRY
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

            total_cost = self.total_cost()
            if total_cost > self._max_cost:
                logger.warning(
                    f"Max cost reached ({total_cost} > {self._max_cost}). Exiting."
                )
                self.trajectory.save_info({"error": "Max cost reached."})
                raise RuntimeError(
                    "The loop was aborted because the cost exceeded the limit.",
                )

        if isinstance(self.state, Finished):
            return Response(status="finished", message=self.state.message or "")
        elif isinstance(self.state, Rejected):
            return Response(status="rejected", message=self.state.message or "")

        raise RuntimeError(f"Loop exited with unknown state {self.state}.")

    def run(
        self, message: Optional[str] = None, input_data: dict[str, Any] | None = None
    ) -> Response:
        """
        Run the loop and handle exceptions and cost checking.
        """

        if self.is_running():
            raise Exception("Loop is already running.")

        if not self._trajectory:
            self._trajectory = Trajectory(
                "AgenticLoop",
                initial_message=message,
                persist_path=self._trajectory_path,
                workspace=self.workspace.dict(),
            )
            initial_state = self._transition_rules.initial_state(**input_data or {})
            self.transition_to(initial_state)
        else:
            for transition in self._trajectory.transitions:
                self.set_current_transition_from_dict(transition)
                self._transitions[self._current_transition.id] = (
                    self._current_transition
                )

            for transition_data in self._trajectory.transitions:
                transition = self._transitions[transition_data["id"]]
                if "parent_id" in transition_data:
                    parent = self._transitions[transition_data["parent_id"]]
                    transition.parent = parent
                    parent.children.append(transition)

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

            total_cost = self.total_cost()
            if total_cost > self._max_cost:
                logger.warning(
                    f"Max cost reached ({total_cost} > {self._max_cost}). Exiting."
                )
                self.trajectory.save_info({"error": "Max cost reached."})
                raise RuntimeError(
                    "The loop was aborted because the cost exceeded the limit.",
                )

        if isinstance(self.state, Finished):
            return Response(status="finished", message=self.state.message or "")
        elif isinstance(self.state, Rejected):
            return Response(status="rejected", message=self.state.message or "")

        raise RuntimeError(f"Loop exited with unknown state {self.state}.")

    def total_cost(self):
        total_cost = 0
        for step in self._transitions.values():
            for action in step.actions:
                if action.completion_cost:
                    total_cost += action.completion_cost

        return total_cost

    def is_running(self) -> bool:
        return not isinstance(self.state, NoopState)

    def _set_state_loop(self, state: AgenticState):
        state._set_loop(self)

    def retries(self) -> int:
        retries = 0
        for action in reversed(self._current_transition.actions):
            if action.retry_message:
                retries += 1
            else:
                return retries

        return retries

    def retry_messages(self, state: AgenticState) -> list[Message]:
        messages: list[Message] = []

        if self._current_transition.name != state.name:
            return messages

        for action in self._current_transition.actions:
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

    def set_current_transition_from_dict(self, transition_data: dict):
        state_data = transition_data.get("state", {})
        name = state_data.get("name")
        logger.info(f"Setting current transition to {name}")
        try:
            state_class = get_state_class(name)
            state = state_class(**state_data)
            self.workspace.restore_from_snapshot(transition_data.get("snapshot"))

            transition = TrajectoryTransition(
                id=transition_data["id"],
                state=state,
                snapshot=transition_data.get("snapshot"),
                actions=[
                    TrajectoryAction(**action) for action in transition_data["actions"]
                ],
                timestamp=datetime.fromisoformat(transition_data["timestamp"]),
            )

            state._set_loop(self)
            state.init()

            self._current_state = state
            self._current_transition = transition
        except Exception as e:
            logger.exception(f"Failed to load state {name}")
            raise e

    def set_current_transition(self, transition: TrajectoryTransition):
        self.workspace.restore_from_snapshot(transition.snapshot)
        self._current_state = transition.state
        self._current_transition = transition

    def revert_to_transition(self, transition_id: int):
        transition = self._transitions.get(transition_id)
        if transition:
            self.set_current_transition(transition)
        else:
            raise ValueError("Invalid state index for reversion")

    def transition_to(self, new_state: AgenticState):
        logger.info(f"Transitioning from {self.state} to {new_state}")

        if self.transition_count() > self._max_transitions:
            new_state = Rejected(message="Max transitions exceeded.")

        if (
            new_state.max_iterations
            and self.transition_count(new_state) > new_state.max_iterations
        ):
            new_state = Rejected(
                message=f"Max transitions exceeded for state {new_state.name}."
            )

        transition = TrajectoryTransition(
            state=new_state,
            snapshot=self.workspace.snapshot(),
            parent=self._current_transition,
        )

        if self._current_transition:
            self._current_transition.children.append(transition)

        transition = self.trajectory.create_transition(transition)

        self._transitions[transition.id] = transition
        self._current_transition = transition
        self._current_state = new_state
        self._set_state_loop(self.state)

    def transition_count(self, state: AgenticState | None = None) -> int:
        if not state:
            return len(self._transitions)
        return len(self.get_transitions(state.name))

    def get_transitions(self, name: str):
        logger.info(
            f"Getting transitions for {name} from {len(self._transitions)} transitions."
        )

        previous_transitions = []
        parent_transition = self._current_transition.parent
        while parent_transition:
            if not name or parent_transition.state.name == name:
                previous_transitions.insert(0, parent_transition)

            parent_transition = parent_transition.parent

        return previous_transitions

    @property
    def state(self):
        return self._current_state

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

    def _run(self):
        if not self.is_running():
            logger.info("Loop is not running.")
            return

        action, cost, input_tokens, output_tokens = self._next_action()

        logger.info(f"{self.state}: Received new action {action.action_name}.")
        response = self.state.handle_action(action)

        self._current_transition.actions.append(
            TrajectoryAction(
                action=action,
                output=response.output,
                retry_message=response.retry_message,
                completion_cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        self.trajectory.save_transition(self._current_transition)

        if not response.trigger:
            logger.info(
                f"{self.state}: No transition found. Staying in the same state."
            )
            return

        if response.trigger == "retry":
            logger.info(f"{self.state}: Retry requested. {response.retry_message}")
            return

        try:
            next_state = self._transition_rules.next_state(
                source=self.state,
                trigger=response.trigger,
                data=response.output,
            )
        except Exception:
            logger.error(
                f"Failed to initiate next state with trigger {response.trigger} and output {response.output}"
            )
            raise

        if not next_state:
            raise ValueError(
                f"No transition found for {self.state} with trigger {response.trigger}"
            )

        if response.trigger == "rejected" and next_state.__class__ != Rejected:
            self._rejections += 1
            next_state = Rejected(
                message=f"Got {self._rejections} rejections, aborting."
            )
        else:
            self._rejections = 0

        logger.info(f"{self.state}: Transitioning to {next_state.name}")
        self.transition_to(next_state)

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

    def _next_mock_action(self) -> ActionRequest | None:
        if not self._mocked_actions:
            return None, None, None, None

        if self._reset_mocks_at_state and self.state.name == self._reset_mocks_at_state:
            logger.info(f"Resetting mocked actions at state {self.state.name}")
            self._mocked_actions = []
            return None, None, None, None

        action = self._mocked_actions.pop(0)

        if "action" not in action:
            return None, None, None, None

        cost = action.get("completion_cost", 0)
        input_tokens = action.get("input_tokens", 0)
        output_tokens = action.get("output_tokens", 0)

        if self.state.action_type():
            try:
                logger.info(
                    f"{self.state} Return mocked response with type {self.state.action_type().__name__} ({len(self._mocked_actions)} left)."
                )
                return (
                    self.state.action_type().model_validate(action["action"]),
                    cost,
                    input_tokens,
                    output_tokens,
                )
            except Exception:
                logger.error(
                    f"Failed to parse {action} to {self.state.action_type().__name__} in state {self.state.name}"
                )
                raise
        elif "content" in action["action"]:
            logger.info(
                f"{self.state} Return mocked response ({len(self._mocked_actions)} left)."
            )
            return (
                Content(content=action["action"]["content"]),
                cost,
                input_tokens,
                output_tokens,
            )

        else:
            raise ValueError(f"Mocked action {action} does not have 'content' field.")

    def _next_action(
        self,
    ) -> tuple[ActionRequest, Optional[float], Optional[int], Optional[int]]:
        messages = self._to_completion_messages()
        logger.info(f"{self.state} Create completion with {len(messages)} messages")

        if self._verify_state_func:
            self._verify_state_func(self.state)

        mocked_action, cost, input_tokens, output_tokens = self._next_mock_action()
        if mocked_action:
            return mocked_action, cost, input_tokens, output_tokens

        metadata = {}
        if self._metadata:
            metadata.update(self._metadata)
        metadata["generation_name"] = str(self.state)

        tokens = token_counter(messages=messages[-1:])
        if self._max_message_tokens and tokens > self._max_message_tokens:
            raise ValueError(f"Too many tokens in the new message: {tokens}")

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

                logger.info(
                    f"{self.state.name}: Input tokens: {completion_response.usage.input_tokens}, Output tokens: {completion_response.usage.output_tokens}"
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
                model="claude-3-5-sonnet-20240620",
            )
        except Exception as e:
            logger.info(f"Error calculating completion cost: {e}")
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

        transition_no = self.transition_count()
        prompt_path = f"{self._prompt_log_dir}/{transition_no:02d}_{self.state.name}.md"

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

                for block in completion:
                    if isinstance(block, BaseModel):
                        block = block.model_dump()

                    if isinstance(block, dict):
                        if "content" in block:
                            f.write(f"{block.get('content')}\n")
                        else:
                            f.write(f"```json\n{json.dumps(block, indent=2)}\n```")
                    else:
                        f.write(f"```json\n{json.dumps(block, indent=2)}\n```")

            if error:
                f.write("\n\n# Error\n")
                f.write(f"\n```\n{error}\n```\n")


def generate_call_id():
    prefix = "call_"
    chars = string.ascii_letters + string.digits
    length = 24

    random_chars = "".join(random.choices(chars, k=length))

    random_string = prefix + random_chars

    return random_string
