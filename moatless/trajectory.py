import json
import logging
from datetime import datetime
import sys
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict
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
    id: Optional[int] = None
    parent: Optional["TrajectoryTransition"] = None
    children: list["TrajectoryTransition"] = Field(default_factory=list)
    state: Optional[AgenticState] = None
    snapshot: Optional[dict] = None
    actions: list[TrajectoryAction] = []
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def name(self):
        return self.state.name if self.state else None

    def model_dump(self, **kwargs):
        data = super().model_dump(exclude={"parent", "children", "state"}, **kwargs)
        data["actions"] = [action.model_dump(**kwargs) for action in self.actions]
        data["state"] = self.state.model_dump(**kwargs) if self.state else None

        if self.parent:
            data["parent_id"] = self.parent.id

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

        self._transitions: list[dict[str, Any]] = []

        self._info: dict[str, Any] = {}

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        trajectory = cls(
            name=data["name"],
            initial_message=data["initial_message"],
            workspace=data["workspace"],
        )
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
    def workspace(self) -> dict[str, Any] | None:
        return self._workspace

    def create_transition(self, transition: TrajectoryTransition):
        transition.id = len(self._transitions) + 1
        self._transitions.append(transition.model_dump())
        self._maybe_persist()
        return transition

    def save_transition(self, transition: TrajectoryTransition):
        for i, t in enumerate(self._transitions):
            if t["id"] == transition.id:
                self._transitions[i] = transition.model_dump()
                self._maybe_persist()
                return

        raise ValueError(f"Transition with id {transition.id} not found")

    def save_info(self, info: dict):
        self._info = info
        self._maybe_persist()

    def to_dict(self):
        return {
            "name": self._name,
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
