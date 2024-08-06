import json
import logging
from datetime import datetime
from typing import Any, Optional, List

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from moatless.workspace import Workspace
from moatless.transition_rules import TransitionRules
from moatless.state import AgenticState, get_state_class
from moatless.types import ActionRequest, ActionTransaction, ActionResponse, Usage, Content

logger = logging.getLogger(__name__)



class TrajectoryState(BaseModel):
    id: int
    timestamp: datetime = Field(default_factory=datetime.now)
    snapshot: Optional[dict] = None
    state: AgenticState

    @property
    def name(self):
        return self.state.name if self.state else None

    def model_dump(self, **kwargs):
        data = {
            "id": self.id,
            "name": self.state.name,
            "timestamp": self.timestamp,
        }

        if self.snapshot:
            data["snapshot"] = self.snapshot

        if self.state.previous_state:
            data["previous_state_id"] = self.state.previous_state.id

        properties = self.state.model_dump(exclude={"previous_state", "next_states", "id"}, **kwargs) if self.state else None
        if properties:
            data["properties"] = properties

        if self.state._actions:
            data["actions"] = [a.model_dump(**kwargs) for a in self.state._actions]

        return data


class Trajectory:
    def __init__(
        self,
        name: str,
        workspace: Workspace,
        initial_message: Optional[str] = None,
        persist_path: Optional[str] = None,
        transition_rules: Optional[TransitionRules] = None,
    ):
        self._name = name
        self._persist_path = persist_path
        self._initial_message = initial_message
        self._workspace = workspace

        # Workaround to set to keep the current initial workspace state when loading an existing trajectory.
        # TODO: Remove this when we have a better way to handle this.
        self._initial_workspace_state = self._workspace.dict()

        self._transition_rules = transition_rules

        self._current_transition_id = 0
        self._transitions: dict[int, TrajectoryState] = {}

        self._info: dict[str, Any] = {}

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        if "transition_rules" in data:
            transition_rules = TransitionRules.model_validate(data["transition_rules"])
        else:
            transition_rules = None

        workspace = Workspace.from_dict(data["workspace"])
        trajectory = cls(
            name=data["name"],
            initial_message=data["initial_message"],
            transition_rules=transition_rules,
            workspace=workspace
        )

        trajectory._transitions = {}
        trajectory._current_transition_id = data.get("current_transition_id", 0)

        for t in data["transitions"]:
            state_class = get_state_class(t["name"])
            state_data = t["properties"]
            state_data["id"] = t["id"]
            state = state_class.model_validate(state_data)

            state._workspace = trajectory._workspace
            state._initial_message = trajectory._initial_message
            state._actions = []
            if "actions" in t:
                for a in t["actions"]:
                    try:
                        if state.action_type() is None:
                            request = Content.model_validate(a["request"])
                        else:
                            request = state.action_type().model_validate(a["request"])
                        response = ActionResponse.model_validate(a.get("response"))
                        if a.get("usage"):
                            usage = Usage.model_validate(a.get("usage"))
                        else:
                            usage = None
                        state._actions.append(ActionTransaction(request=request, response=response, usage=usage))
                    except Exception as e:
                        logger.exception(f"Error loading action for state {state.name}: {a}")
                        raise e

            trajectory_state = TrajectoryState(
                id=t["id"],
                timestamp=datetime.fromisoformat(t["timestamp"]),
                snapshot=t.get("snapshot"),
                state=state
            )

            trajectory._transitions[t["id"]] = trajectory_state

        # Set previous_state and next_states
        for t in data["transitions"]:
            try:
                current_state = trajectory._transitions[t["id"]].state
                if t.get("previous_state_id") is not None:
                    current_state.previous_state = trajectory._transitions.get(t["previous_state_id"]).state
            except KeyError as e:
                logger.exception(f"Missing key {e}, existing keys: {trajectory._transitions.keys()}")
                raise

        trajectory._info = data.get("info", {})

        logger.info(f"Loaded trajectory {trajectory._name} with {len(trajectory._transitions)} transitions")

        current_state = trajectory._transitions.get(trajectory._current_transition_id)
        trajectory.restore_from_snapshot(current_state)

        return trajectory

    @property
    def initial_message(self):
        return self._initial_message

    @property
    def states(self) -> List[dict]:
        return [t.state.model_dump() for t in self.transitions]

    @property
    def transition_rules(self) -> TransitionRules:
        return self._transition_rules

    @property
    def workspace(self) -> dict[str, Any] | None:
        return self._workspace

    @property
    def transitions(self) -> List[TrajectoryState]:
        return sorted(self._transitions.values(), key=lambda x: x.id)

    def set_current_state(self, state: AgenticState):
        self._current_transition_id = state.id
        self._maybe_persist()

    def get_current_state(self) -> AgenticState:
        return self._transitions.get(self._current_transition_id).state

    def restore_from_snapshot(self, state: TrajectoryState):
        if not state.snapshot:
            logger.info(f"restore_from_snapshot(state: {state.id}:{state.name}) No snapshot found")
            return

        logger.info(f"restore_from_snapshot(starte: {state.id}:{state.name}) Restoring from snapshot")

        if state.snapshot.get("repository"):
            self._workspace.file_repo.restore_from_snapshot(state.snapshot["repository"])

        if state.snapshot.get("file_context"):
            self._workspace.file_context.restore_from_snapshot(state.snapshot["file_context"])

    def save_state(self, state: AgenticState):
        if state.id in self._transitions:
            self._transitions[state.id].state = state
        else:
            transition = TrajectoryState(
                id=state.id,
                state=state,
                snapshot=state.workspace.snapshot() if state.workspace else None,
            )   
            self._transitions[state.id] = transition

        self._maybe_persist()

    def get_state(self, state_id: int) -> TrajectoryState | None:
        return self._transitions.get(state_id)

    def save_info(self, info: dict):
        self._info = info
        self._maybe_persist()

    def get_mocked_actions(self) -> List[dict]:
        """
        Return a list of actions that can be used to mock the trajectory.
        """
        actions = []

        for transition in self.transitions:
            for action in transition.state._actions:
                actions.append(action.request.model_dump())
        return actions

    def get_expected_states(self) -> List[str]:
        """
        Return a list of expected states in the trajectory to use for verification when rerunning the trajectory.
        """
        return [transition.state.name for transition in self.transitions[1:]]

    def to_dict(self):
        return {
            "name": self._name,
            "transition_rules": self._transition_rules.model_dump(
                exclude_none=True
            )
            if self._transition_rules
            else None,
            "workspace": self._initial_workspace_state,
            "initial_message": self._initial_message,
            "current_transition_id": self._current_transition_id,
            "transitions": [t.model_dump(exclude_none=True) for t in self.transitions],
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