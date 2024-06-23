import json
import logging
from typing import Optional, Any, List

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from moatless.state import AgenticState
from moatless.types import ActionRequest

logger = logging.getLogger(__name__)


class TrajectoryAction(BaseModel):
    action: ActionRequest
    retry_message: Optional[str] = None
    output: Optional[dict[str, Any]] = None
    completion_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def model_dump(self, **kwargs):
        dict = super().model_dump(**kwargs)
        action_dict = self.action.model_dump(**kwargs)
        dict["action"] = action_dict
        return dict


class TrajectoryTransition(BaseModel):
    state: Optional[AgenticState] = None
    actions: List[TrajectoryAction] = []

    @property
    def name(self):
        return self.state.name if self.state else None

    def model_dump(self, **kwargs):
        return {
            "name": self.name,
            "state": self.state.model_dump(**kwargs) if self.state else None,
            "actions": [action.model_dump(**kwargs) for action in self.actions],
        }


class Trajectory:

    def __init__(
        self,
        name: str,
        initial_message: Optional[str] = None,
        persist_path: Optional[str] = None,
    ):
        self._name = name
        self._persist_path = persist_path
        self._initial_message = initial_message

        self._transitions: list[TrajectoryTransition] = []
        self._current_transition: Optional[TrajectoryTransition] = None

        self._info: dict[str, Any] = {}

    @property
    def current_step(self):
        return self._current_transition

    @property
    def initial_message(self):
        return self._initial_message

    def get_transitions(self, name: str):
        logger.info(
            f"Getting transitions for {name} from {len(self._transitions)} transitions."
        )
        return [
            transition for transition in self._transitions if transition.name == name
        ]

    def transition_count(self, state: Optional[AgenticState] = None):
        if not state:
            return len(self._transitions)
        return len(self.get_transitions(state.name))

    def save_action(
        self,
        action: ActionRequest,
        output: Optional[dict[str, Any]] = None,
        retry_message: Optional[str] = None,
        completion_cost: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ):
        if self._current_transition:
            self._current_transition.actions.append(
                TrajectoryAction(
                    action=action,
                    output=output,
                    retry_message=retry_message,
                    completion_cost=completion_cost,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
            logger.info(
                f"Saving action {action.__class__.__name__} to {self._current_transition.name} ({len(self._current_transition.actions)} actions)"
            )

            self._maybe_persist()
        else:
            logger.warning(
                f"No current trajectory step to save action {action.model_dump_json()}."
            )

    def new_transition(self, state: AgenticState):
        if self._current_transition:
            self._transitions.append(self._current_transition)

        self._current_transition = TrajectoryTransition(state=state)
        self._maybe_persist()

    def save_info(self, info: dict):
        self._info = info
        self._maybe_persist()

    def to_dict(self, **kwargs):
        transition_dicts = [
            transition.model_dump(**kwargs) for transition in self._transitions
        ]
        if self._current_transition:
            transition_dicts.append(self._current_transition.model_dump(**kwargs))

        return {
            "name": self._name,
            "initial_message": self._initial_message,
            "transitions": transition_dicts,
            "info": self._info,
        }

    def total_cost(self):
        total_cost = 0
        for step in self._transitions:
            for action in step.actions:
                if action.completion_cost:
                    total_cost += action.completion_cost

        return total_cost

    def _maybe_persist(self):
        if self._persist_path:
            self.persist(self._persist_path)

    def persist(self, file_path: str):
        with open(f"{file_path}", "w") as f:
            f.write(
                json.dumps(
                    self.to_dict(exclude_none=True),
                    indent=2,
                    default=to_jsonable_python,
                )
            )
