import json
import logging
from datetime import datetime
from typing import Any, Optional, List

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from moatless.workspace import Workspace
from moatless.transition_rules import TransitionRules
from moatless.state import AgenticState, get_state_class
from moatless.types import ActionRequest

logger = logging.getLogger(__name__)


class TrajectoryAction(BaseModel):
    action: ActionRequest
    trigger: Optional[str] = None
    retry_message: Optional[str] = None
    completion_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data["action"] = self.action.model_dump(**kwargs)
        return data


class TrajectoryTransition(BaseModel):
    id: int
    state: AgenticState
    snapshot: Optional[dict] = None
    actions: List[TrajectoryAction] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def name(self):
        return self.state.name if self.state else None

    def model_dump(self, **kwargs):
        data = {
            "id": self.id,
            "timestamp": self.timestamp,
            "state": self.state.model_dump(**kwargs) if self.state else None,
            "snapshot": self.snapshot,
            "actions": [action.model_dump(**kwargs) for action in self.actions],
        }

        if kwargs.get("exclude_none", False):
            data = {k: v for k, v in data.items() if v is not None}

        return data


class Trajectory:
    def __init__(
        self,
        name: str,
        initial_message: Optional[str] = None,
        persist_path: Optional[str] = None,
        workspace: Optional[Workspace] = None,
        transition_rules: Optional[TransitionRules] = None,
    ):
        self._name = name
        self._persist_path = persist_path
        self._initial_message = initial_message
        self._workspace = workspace.dict() if workspace else None
        self._transition_rules = transition_rules

        self._transitions: List[TrajectoryTransition] = []
        self._current_transition_id = 0

        self._info: dict[str, Any] = {}

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        if "transition_rules" in data:
            transition_rules = TransitionRules.model_validate(data["transition_rules"])
        else:
            transition_rules = None

        trajectory = cls(
            name=data["name"],
            initial_message=data["initial_message"],
            transition_rules=transition_rules,
        )
        trajectory._workspace = data["workspace"]
        trajectory._transitions = []
        for t in data["transitions"]:
            state_class = get_state_class(t["state"]["name"])
            t["state"] = state_class(**t["state"])
            trajectory._transitions.append(TrajectoryTransition(**t))

        return trajectory

    @property
    def initial_message(self):
        return self._initial_message

    @property
    def states(self) -> List[dict]:
        return [t.state.model_dump() for t in self._transitions]

    @property
    def transition_rules(self) -> TransitionRules:
        return self._transition_rules

    @property
    def workspace(self) -> dict[str, Any] | None:
        return self._workspace

    def set_current_state(self, state: AgenticState):
        self._current_transition_id = state.id
        self._maybe_persist()

    def save_state(self, state: AgenticState):
        for i, existing_transition in enumerate(self._transitions):
            if existing_transition.id == self._current_transition_id:
                self._transitions[i].state = state
                break
        else:
            transition = TrajectoryTransition(
                id=self._current_transition_id,
                state=state,
                snapshot=state.workspace.snapshot() if state.workspace else None,
            )   
            self._transitions.append(transition)

        self._maybe_persist()

    def get_state(self, state_id: int) -> AgenticState | None:
        for transition in self._transitions:
            if transition.id == state_id:
                return transition.state
        return None

    def save_info(self, info: dict):
        self._info = info
        self._maybe_persist()

    def get_mocked_actions(self) -> List[dict]:
        """
        Return a list of actions that can be used to mock the trajectory.
        """
        actions = []
        for transition in self._transitions:
            for action in transition.actions:
                actions.append(action.action.model_dump())
        return actions

    def get_expected_states(self) -> List[str]:
        """
        Return a list of expected states in the trajectory to use for verification when rerunning the trajectory.
        """
        return [transition.state.name for transition in self._transitions]

    def to_dict(self):
        return {
            "name": self._name,
            "transition_rules": self._transition_rules.model_dump(
                exclude_none=True, exclude_unset=True
            )
            if self._transition_rules
            else None,
            "workspace": self._workspace,
            "initial_message": self._initial_message,
            "current_transition_id": self._current_transition_id,
            "transitions": [t.model_dump(exclude_none=True, exclude_unset=True) for t in self._transitions],
            "info": self._info,
        }

    def _maybe_persist(self):
        if self._persist_path:
            self.persist(self._persist_path)

    def persist(self, file_path: str):
        with open(f"{file_path}", "w") as f:
            f.write(
                json.dumps(
                    self.to_dict(),
                    indent=2,
                    default=to_jsonable_python,
                )
            )