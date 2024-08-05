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
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

from moatless.repository import GitRepository
from moatless.state import (
    AgenticState,
    Finished,
    NoopState,
    Pending,
    Rejected,
    get_state_class,
)
from moatless.trajectory import Trajectory, TrajectoryAction
from moatless.transition_rules import TransitionRule, TransitionRules
from moatless.types import (
    ActionRequest,
    AssistantMessage,
    Content,
    Message,
    Response,
    Usage,
    UserMessage,
)
from moatless.utils.llm_utils import instructor_mode_by_model
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
        self._current_state: AgenticState | None = None
        self._state_history: dict[int, AgenticState] = {}

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

    def initialize_or_load_trajectory(self, message: Optional[str] = None) -> None:
        if not self._trajectory:
            self._trajectory = Trajectory(
                "MoatlessTools",
                initial_message=message,
                persist_path=self._trajectory_path,
                workspace=self._workspace,
                transition_rules=self._transition_rules,
            )
            pending_state = Pending()
            self._state_history[pending_state.id] = pending_state
            self._set_current_state(pending_state)
        else:
            for state_data in self._trajectory.states:
                self.set_current_state_from_dict(state_data)
                self.workspace.restore_from_snapshot(state_data.get("snapshot"))

    def run(self, message: Optional[str] = None) -> Response:
        """
        Executes the entire loop until completion or termination.

        This method initializes the loop if it hasn't started, and then repeatedly
        calls run_until_transition() until the loop is finished. It handles the
        overall flow of the loop, including initialization and final state processing.

        Args:
            message (Optional[str]): An optional initial message to start the loop with.

        Returns:
            Response: An object containing the final status and message of the loop.
                The status will be either "finished" or "rejected".

        Raises:
            RuntimeError: If an unexpected state or condition occurs during execution.
                This includes cases where the loop is already running, exits with an 
                unknown state, or encounters other unexpected runtime conditions.

        Note:
            This method will continue running until a Finished or Rejected state is reached,
            or until an exception occurs. It's designed to be the main entry point for
            executing the entire loop process.
        """
        if self.is_running():
            raise RuntimeError("Loop is already running.")

        self.initialize_or_load_trajectory(message)

        while not self.is_finished():
            self._execute_state_until_transition()

        if isinstance(self.state, Finished):
            return Response(status="finished", message=self.state.message or "")
        elif isinstance(self.state, Rejected):
            return Response(status="rejected", message=self.state.message or "")

        raise RuntimeError(f"Loop exited with unknown state {self.state.name}.")

    def _execute_state_until_transition(self) -> AgenticState | None:
        """
        Executes the state until a transition to a new state occurs.

        This method executes the state, processing actions and handling
        state changes until one of the following conditions is met:
        1. A transition to a new state occurs
        2. Maximum cost, retries, or transitions are exceeded

        Returns:
            AgenticState: The new state after a transition occurs

        Raises:
            RuntimeError: If the loop exits without a transition or if the maximum cost is exceeded
            ValueError: If the maximum number of retries is reached
        """
        while not self.state.executed:
            total_cost = self.total_cost()
            if total_cost > self._max_cost:
                self.log_info(f"Max cost reached ({total_cost} > {self._max_cost}). Exiting.")
                self.trajectory.save_info({"error": "Max cost reached."})
                raise RuntimeError("The loop was aborted because the cost exceeded the limit.")

            self.log_info(f"Running transition {len(self._trajectory.states)}. Current total cost: {total_cost}")

            try:
                state = self._execute_state()
                if state:
                    return state
            except Exception as e:
                self.log_info(f"Failed to run loop. Error: {e}")
                raise

            if self.state.retries() > self._max_retries:
                self.log_info(f"Max retries reached ({self._max_retries}). Exiting.")
                self.trajectory.save_info({"error": "Max retries reached."})
                return self.transition_to(Rejected(message="Max retries reached."))

        raise RuntimeError("Loop exited without a transition.")

    def _execute_state(self) -> AgenticState | None:
        """
        Execute one iteration of the current state and handle potential transitions.

        Processes the next action, updates the trajectory, and determines if a state
        transition should occur based on the action's response.

        Returns:
            AgenticState | None: The next state if transitioning, or None if remaining in the current state.

        Raises:
            ValueError: 
        """
        if self.state.executed:
            raise ValueError("Tried to execute already executed state.")

        if isinstance(self.state, Pending):
            logger.info("Initializing first state.")
            trigger = "init"
            output = {}
            
        else:
            action, usage = self._next_action()

            self.log_info(f"Received new action {action.action_name}.")
            response = self.state.handle_action(action, usage)

            if not response.trigger:
                self.log_info(
                    f"{self.state.name}: No trigger in action response. Staying in the same state."
                )
                return None

            self.log_info(f"Received response with trigger {response.trigger}")

            if response.trigger == "retry":
                self.log_info(f"Retry requested. {response.retry_message}")
                return None
            
            trigger = response.trigger
            output = response.output
            

        transition_rule = self._transition_rules.get_next_rule(
            self.state,
            trigger,
            output,
        )
        if not transition_rule:
            raise RuntimeError(
                f"No transition rule found for {self.state.name} with trigger {response.trigger} and output {response.output}"
            )

        next_state = self._create_state(transition_rule, output)
        return self.transition_to(next_state)

    def _create_state(self, transition_rule: TransitionRule, output: dict) -> AgenticState:
        params = {}
        params.update(self._transition_rules.params(transition_rule))

        for k, v in output.items():
            if transition_rule.excluded_fields and k in transition_rule.excluded_fields:
                continue
    
            params[k] = v

        params["id"] = self.state_count() + 1

        next_state_type = transition_rule.dest
        if next_state_type not in [Finished, Rejected]:

            if self.state_count() >= self._max_transitions:
                self.log_info(f"Max transitions exceeded ({self._max_transitions}). Transitioning to Rejected.")
                next_state_type = Rejected
                params["message"] = "Max transitions exceeded."   
            if (
                params.get("max_iterations")
                and self.state_count(next_state_type) >= params["max_iterations"]
            ):
                self.log_info(f"Max iterations exceeded ({params['max_iterations']}). Transitioning to Rejected.")
                next_state_type = Rejected
                params["message"] = f"Max iterations exceeded ({params['max_iterations']})."
    
        self.log_info(f"Creating state {next_state_type.__name__} with params {params}")
        params["previous_state"] = self._current_state
        params["_workspace"] = self._workspace

        next_state = next_state_type.model_validate(params)

        self._current_state.next_states.append(next_state)
        self._state_history[next_state.id] = next_state
        return next_state

    def total_cost(self):
        total_cost = 0
        for state in self._state_history.values():
            total_cost += state.total_cost()
        return total_cost

    def is_running(self) -> bool:
        return not isinstance(self.state, NoopState)

    def is_finished(self) -> bool:
        return isinstance(self.state, (Finished, Rejected))

    def _set_current_state(self, state: AgenticState):
        self._current_state = state
        self._trajectory.set_current_state(state)

    def set_current_state_from_dict(self, state_data: dict):
        name = state_data.get("name")
        try:
            state_class = get_state_class(name)
            state = state_class.model_validate(state_data)

            # Set previous_state if it exists
            if state_data.get("previous_state_id") is not None:
                state.previous_state = self._state_history.get(state_data["previous_state_id"])

            # Set next_states if they exist
            for next_state_id in state_data.get("next_state_ids", []):
                next_state = self._state_history.get(next_state_id)
                if next_state:
                    state.next_states.append(next_state)

            self._set_current_state(state)
            state.init()

        except Exception as e:
            logger.exception(f"Failed to load state {name}")
            raise e

    def revert_to_state(self, state_id: int) -> AgenticState:
        state = self._trajectory.get_state(state_id)
        if state:
            self.log_info(f"Reverting to state {state_id}")
            self._set_current_state(state)
            self.workspace.restore_from_snapshot(state.snapshot)
            return state
        else:
            logger.warning(f"Tried to revert to state {state_id} but it does not exist.")
            raise ValueError(f"Could not revert to state {state_id} as it does not exist.")

    def transition_to(self, new_state: AgenticState) -> AgenticState:
        self.log_info(f"Transitioning from {self.state.name} to {new_state.name}")

        self._trajectory.save_state(new_state)
        self._state_history[new_state.id] = new_state
        self._set_current_state(new_state)

        return new_state

    def _next_action(
        self,
    ) -> Tuple[ActionRequest, Usage | None]:
        messages = self._to_completion_messages()
        self.log_info(f"Create completion with {len(messages)} messages")

        if self._verify_state_func:
            self._verify_state_func(self.state)

        mocked_action = self._next_mock_action()
        if mocked_action:
            return mocked_action, None

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

            usage = Usage(
                completion_cost=_final_cost,
                completion_tokens=completion_response.usage.output_tokens,
                prompt_tokens=completion_response.usage.input_tokens,
            )

            return action_request, usage

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
        usage = Usage(
            completion_cost=cost,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )
        return action_request, usage
    
    def state_count(self, state: AgenticState | None = None) -> int:
        if not state:
            return len(self._state_history)

        return len(
            [s for s in self._state_history.values() if s.name == state.name]
        )

    @property
    def state(self):
        return self._current_state if self._current_state else Pending()

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

    @property
    def instructor_mode(self):
        if self._instructor_mode:
            return self._instructor_mode

        return instructor_mode_by_model(self.state.model)

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
            f"{self._prompt_log_dir}/{self._current_state.name}:{self._current_state.id}"
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
        if self._current_state:
            return f"{self._current_state.name}:{self._current_state.id}"
        else:
            return "No state"
