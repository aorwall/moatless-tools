import logging
import os
from collections.abc import Callable
from typing import Any, Optional, Type

from moatless.state import (
    AgenticState,
    Finished,
    NoopState,
    Pending,
    Content,
    Rejected,
    State, ActionRequest, Usage
)
from moatless.trajectory import Trajectory
from moatless.transition_rules import TransitionRule, TransitionRules
from moatless.schema import (
    Response
)
from moatless.utils.llm_utils import response_format_by_model, LLMResponseFormat
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class AgenticLoop:
    def __init__(
        self,
        transition_rules: TransitionRules,
        workspace: Workspace,
        input_data: dict[str, Any] | None = None,
        initial_message: str | None = None,
        trajectory: Trajectory | None = None,
        mocked_actions: list[dict] | None = None,
        continue_after_mocks: bool = False,
        expected_states: list[Type[State]] | None = None,
        reset_mocks_at_state: Optional[str] = None,
        verify_state_func: Optional[Callable] = None,
        max_cost: float = 0.25,
        max_transitions: int = 25,
        num_iterations: int = 40,
        max_expansions: int = 3,
        max_retries: int = 2,
        max_rejections: int = 2,
        metadata: dict[str, Any] | None = None,
        trajectory_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Loop instance.

        Args:

        """

        self._workspace = workspace

        self._input_data = input_data

        if trajectory_path:
            parent_dir = os.path.dirname(trajectory_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
        self._trajectory_path = trajectory_path

        if not trajectory:
            self._trajectory = Trajectory(
                "MoatlessTools",
                initial_message=initial_message,
                persist_path=self._trajectory_path,
                workspace=self._workspace,
                transition_rules=transition_rules,
            )
            pending_state = Pending()
            self._trajectory.save_state(pending_state)
            self._set_current_state(pending_state)
        else:
            self._trajectory = trajectory
            self._current_state = trajectory.get_current_state()

        self._trajectory.save_info(metadata or {})

        self._initial_message = initial_message

        if expected_states and not verify_state_func:

            def verify_state_func(state: State):
                nonlocal expected_states
                if not expected_states:
                    raise ValueError(
                        f"No more expected states, but got {state.__class__}"
                    )
                expected_state = expected_states.pop(0)
                if isinstance(expected_state, str):
                    if state.name != expected_state:
                        raise ValueError(
                            f"Expected state {expected_state} but got {state.__class__.__name__}"
                        )
                elif isinstance(expected_state, State) and not isinstance(
                    state, expected_state
                ):
                    raise ValueError(
                        f"Expected state {expected_state} but got {state.__class__.__name__}"
                    )

                self.log_info(f"Verified expected next state {expected_state}")

        self._verify_state_func = verify_state_func
        self._mocked_actions = mocked_actions
        self._continue_after_mocks = continue_after_mocks
        self._reset_mocks_at_state = reset_mocks_at_state

        self._max_cost = max_cost
        self._max_transitions = max_transitions
        self._max_retries = max_retries
        self._max_rejections = max_rejections

        # MCTS
        self._num_iterations = num_iterations
        self._max_expansions = max_expansions

        self._transition_count = 0
        self._rejections = 0

        self._transition_rules = transition_rules
        self._metadata = metadata

        self.kwargs = kwargs



    @classmethod
    def from_trajectory_file(cls, trajectory_path: str, **kwargs):
        trajectory = Trajectory.load(trajectory_path, **kwargs)
        return cls(
            transition_rules=trajectory.transitions,
            trajectory=trajectory,
            workspace=trajectory.workspace,
            trajectory_path=trajectory_path,
            **kwargs,
        )

    def persist(self, trajectory_path: str):
        self.trajectory.persist(trajectory_path)

    def run(self, message: Optional[str] = None) -> Response:
        """
        Executes the entire loop until completion or termination.

        This method initializes the loop if it hasn't started, and then repeatedly
        calls _execute_state_until_transition() until the loop is finished. It handles the
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

        # TODO: Move to always set this when the Loop is created instead
        if message:
            logger.warning(
                "Setting initial message in run is deprecated. Set in contructor."
            )
            self._initial_message = message
            self._trajectory._initial_message = message

        if not isinstance(self._current_state, Pending):
            self._trajectory.update_workspace_to_current_state()

        while not self.is_finished():
            self._execute_state_until_transition()

        if isinstance(self.state, Finished):
            return Response(status="finished", message=self.state.message or "")
        elif isinstance(self.state, Rejected):
            return Response(status="rejected", message=self.state.message or "")

        raise RuntimeError(f"Loop exited with unknown state {self.state.name}.")

    def _execute_state_until_transition(self) -> State | None:
        """
        This method executes the state until one of the following conditions is met:
        1. A transition to a new state occurs
        2. Maximum cost, retries, or transitions are exceeded

        Returns:
            State: The new state after a transition occurs

        Raises:
            RuntimeError: If the loop exits without a transition or if the maximum cost is exceeded
            ValueError: If the maximum number of retries is reached
        """

        if isinstance(self.state, (Rejected, Finished)):
            raise ValueError(
                f"The loop was aborted because a terminal state {self.state.name} was reached."
            )

        while not self.state.executed:
            total_cost = self.total_cost()
            if total_cost > self._max_cost:
                self.log_info(
                    f"Max cost reached ({total_cost} > {self._max_cost}). Exiting."
                )
                self.trajectory.save_info({"error": f"Max cost reached  ({total_cost} > {self._max_cost})."})
                rejected_state = self._create_state(
                    Rejected, {"message": "Max retries reached."}
                )
                return self.transition_to(rejected_state)

            self.log_info(
                f"Running transition {len(self._trajectory.states)}. Current total cost: {total_cost}"
            )

            try:
                state = self._execute_state()
                if state:
                    return state
            except Exception as e:
                self.log_info(f"Failed to run loop. Error: {e}")
                raise

            if self.state.retries() > self._max_retries:
                self.log_info(f"Max retries reached ({self._max_retries}) in {self.state.name}. Exiting.")
                self.trajectory.save_info({"error": f"Max retries reached in {self.state.name}."})
                rejected_state = self._create_state(
                    Rejected, {"message": "Max retries reached."}
                )
                return self.transition_to(rejected_state)

        raise RuntimeError("Loop exited without a transition.")

    def _execute_state(self) -> AgenticState | None:
        """
        Execute one iteration of the current state and handle potential transitions.

        Processes the next action, updates the trajectory, and determines if a state
        transition should occur based on the action's response.

        Returns:
            State | None: The next state if transitioning, or None if remaining in the current state.

        Raises:
            ValueError:
        """
        if isinstance(self.state, Pending):
            logger.info("Initializing first state.")
            trigger = "init"
            output = {}

        else:
            if self._verify_state_func:
                self._verify_state_func(self.state)

            if isinstance(self.state, AgenticState):
                outcome = self.state.init()
                if not outcome:
                    outcome = self.state.execute(mocked_action_request=self._next_mock_action())
            else:
                outcome = self.state.execute()

            self.log_info(f"Received response with trigger {outcome.trigger}")

            if outcome.trigger == "retry":
                self.log_info(f"Retry requested. {outcome.retry_message}")
                return None

            trigger = outcome.trigger
            output = outcome.output

        transition_rule = self._transition_rules.get_next_rule(
            self.state,
            trigger,
            output,
        )
        if not transition_rule:
            raise RuntimeError(
                f"No transition rule found for {self.state.name} with trigger {outcome.trigger} and output {outcome.output}"
            )

        next_state = self._create_state_from_transition_rule(transition_rule, output)
        return self.transition_to(next_state)

    def _create_state_from_transition_rule(
        self, transition_rule: TransitionRule, output: dict
    ) -> State:
        params = {}
        params.update(self._transition_rules.params(transition_rule))

        for k, v in output.items():
            if transition_rule.excluded_fields and k in transition_rule.excluded_fields:
                continue

            params[k] = v

        return self._create_state(transition_rule.dest, params)

    def _create_state(
        self, next_state_type: Type[State], params: dict
    ) -> State:
        if next_state_type not in [Finished, Rejected]:
            if len(self.state.get_previous_states()) >= self._max_transitions:
                self.log_info(
                    f"Max transitions exceeded ({self._max_transitions}). Transitioning to Rejected."
                )
                next_state_type = Rejected
                params["message"] = "Max transitions exceeded."
            if (
                params.get("max_iterations")
                and self.state.get_previous_states(self.state)
                > params["max_iterations"]
            ):
                self.log_info(
                    f"Max iterations exceeded ({params['max_iterations']}). Transitioning to Rejected."
                )
                next_state_type = Rejected
                params["message"] = (
                    f"Max iterations exceeded ({params['max_iterations']})."
                )

        logger.debug(
            f"{self.transition_name}: Creating state {next_state_type.__name__} with params {params}"
        )

        params["id"] = len(self._trajectory.transitions)
        try:
            next_state = next_state_type.model_validate(params)
            next_state.previous_state = self._current_state
            next_state._workspace = self._workspace
            next_state._initial_message = self._initial_message
        except Exception as e:
            logger.error(
                f"Failed to create state {next_state_type.__name__} with params {params}"
            )
            raise e

        self._trajectory.save_state(next_state)
        self._current_state.next_states.append(next_state)
        return next_state

    def total_usage(self):
        total_usage = Usage()
        for state in self._trajectory.transitions:
            if isinstance(state.state, AgenticState):
                total_usage += state.state.total_usage()
        return total_usage

    def total_cost(self):
        total_usage = self.total_usage()
        if total_usage:
            return total_usage.completion_cost
        else:
            return 0

    def is_running(self) -> bool:
        return not isinstance(self.state, NoopState)

    def is_finished(self) -> bool:
        return isinstance(self.state, (Finished, Rejected))

    def _set_current_state(self, state: State):
        self._current_state = state
        self._trajectory.set_current_state(state)

    def transition_to(self, new_state: State) -> State:
        self.log_info(f"Transitioning from {self.state.name} to {new_state.name}")

        self._trajectory.save_state(new_state)
        self._set_current_state(new_state)

        return new_state

    @property
    def state(self):
        return self._current_state

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def transition_count(self):
        return len(self._trajectory.transitions)

    def revert_to_state(self, state: State):
        self._set_current_state(state)
        self._trajectory.update_workspace_to_current_state()

    def clone_current_state(self):
        cloned_state = self.clone_state(self.state)
        self._set_current_state(cloned_state)
        return cloned_state

    def clone_state(self, state: State):
        cloned_state = state.clone()
        cloned_state.id = len(self._trajectory.transitions)
        cloned_state.previous_state = state.previous_state
        cloned_state.previous_state.next_states.append(cloned_state)
        self._trajectory.save_state(cloned_state)
        return cloned_state

    @property
    def instructor_mode(self) -> LLMResponseFormat | None:
        if self._instructor_mode:
            return self._instructor_mode

        return response_format_by_model(self.state.model)

    def _next_mock_action(
        self,
    ) -> ActionRequest | None:
        if self._mocked_actions is None:
            return None

        if len(self._mocked_actions) == 0 and self._continue_after_mocks:
            return None

        if self._reset_mocks_at_state and self.state.name == self._reset_mocks_at_state:
            self.log_info(f"Resetting mocked actions at state {self.state.name}")
            self._mocked_actions = []
            self._continue_after_mocks = True
            return None

        if not self._mocked_actions:
            raise ValueError(f"No more mocked actions left in state {self.state.name}")

        action = self._mocked_actions.pop(0)
        self.log_info(f"Return mocked response ({len(self._mocked_actions)} left).")

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

    def log_info(self, message: str):
        logger.info(f"{self.transition_name}: {message}")

    @property
    def transition_name(self):
        if self._current_state:
            return f"{self._current_state.name}:{self._current_state.id}"
        else:
            return "No state"
