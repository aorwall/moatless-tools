import json
import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from moatless.workspace import Workspace
from moatless.transition_rules import TransitionRules
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
    id: Optional[int] = None
    parent: Optional["TrajectoryTransition"] = None
    children: list["TrajectoryTransition"] = Field(default_factory=list)
    state: AgenticState
    snapshot: Optional[dict] = None
    actions: list[TrajectoryAction] = []
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def name(self):
        return self.state.name if self.state else None

    def model_dump(self, **kwargs):
        data = {
            "id": self.id,
            "timestamp": self.timestamp,
            "parent_id": self.parent.id if self.parent else None,
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

        self._transitions: list[dict[str, Any]] = []

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
        trajectory._transitions = data["transitions"]
        trajectory._info = data["info"]

        return trajectory

    @property
    def initial_message(self):
        return self._initial_message

    @property
    def transitions(self) -> list[dict]:
        return sorted(self._transitions, key=lambda x: x["timestamp"])

    @property
    def transition_rules(self) -> TransitionRules:
        return self._transition_rules

    @property
    def workspace(self) -> dict[str, Any] | None:
        return self._workspace

    def create_transition(self, transition: TrajectoryTransition):
        transition.id = len(self._transitions) + 1
        self._transitions.append(
            transition.model_dump(exclude_none=True, exclude_unset=True)
        )
        self._maybe_persist()
        return transition

    def save_transition(self, transition: TrajectoryTransition):
        for i, t in enumerate(self._transitions):
            if t["id"] == transition.id:
                self._transitions[i] = transition.model_dump(
                    exclude_none=True, exclude_unset=True
                )
                self._maybe_persist()
                return

        raise ValueError(f"Transition with id {transition.id} not found")

    def save_info(self, info: dict):
        self._info = info
        self._maybe_persist()

    def get_mocked_actions(self) -> list[dict]:
        """
        Return a list of actions that can be used to mock the trajectory.
        """
        actions = []
        for transition in self._transitions:
            for action in transition["actions"]:
                actions.append(action["action"])
        return actions

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
            "transitions": self._transitions,
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
