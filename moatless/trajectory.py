import json
import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field
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
        data = super().model_dump(**kwargs)
        data["action"] = self.action.model_dump(**kwargs)
        return data


class TrajectoryTransition(BaseModel):
    id: int
    parent: Optional["TrajectoryTransition"] = None
    children: list["TrajectoryTransition"] = Field(default_factory=list)
    state: AgenticState | None = None
    snapshot: Optional[dict] = None
    actions: list[TrajectoryAction] = []
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        exclude = {"parent", "children"}

    @property
    def name(self):
        return self.state.name if self.state else None

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        if self.state:
            data["state"]["name"] = self.state.name
        data["actions"] = [action.model_dump(**kwargs) for action in self.actions]

        if self.parent:
            data["parent"] = self.parent.id

        return data


class Trajectory:
    def __init__(
        self,
        name: str,
        initial_message: Optional[str] = None,
        persist_path: Optional[str] = None,
        workspace: Optional[dict] = None,
    ):
        self._name = name
        self._persist_path = persist_path
        self._initial_message = initial_message
        self._workspace = workspace

        self._transitions: list[TrajectoryTransition] = []
        self._current_transition: TrajectoryTransition | None = None

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

    def transition_count(self, state: AgenticState | None = None):
        if not state:
            return len(self._transitions)
        return len(self.get_transitions(state.name))

    def save_action(
        self,
        action: ActionRequest,
        output: dict[str, Any] | None = None,
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

    def new_transition(self, state: AgenticState, snapshot: Optional[dict] = None):
        if self._current_transition:
            self._transitions.append(self._current_transition)

        transition = TrajectoryTransition(
            id=len(self._transitions),
            state=state,
            snapshot=snapshot,
            parent=self._current_transition,
        )

        if self._current_transition:
            self._current_transition.children.append(transition)

        self._current_transition = transition
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
            "workspace": self._workspace,
            "initial_message": self._initial_message,
            "transitions": transition_dicts,
            "info": self._info,
            "dummy_field": None,  # Add this line
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
