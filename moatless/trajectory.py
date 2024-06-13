import json
import logging
from typing import Optional, Any, List

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from moatless.types import ActionRequest, Message

logger = logging.getLogger(__name__)


class TrajectoryAction(BaseModel):
    action: ActionRequest
    completion_cost: Optional[float] = None


class TrajectoryStep(BaseModel):
    name: Optional[str] = None
    transition_input: Optional[dict[str, Any]] = None
    actions: List[TrajectoryAction] = []


class Trajectory:

    def __init__(self, name: str, input_data: Optional[dict[str, Any]] = None):
        self._name = name
        self._input_data = input_data
        self._output_data: Optional[dict[str, Any]] = None
        self._trajectory_steps: list[TrajectoryStep] = []

        self._current_trajectory_step: Optional[TrajectoryStep] = None
        self._info: dict[str, Any] = {}

        self._child_trajectorie: list[Trajectory] = []

    @property
    def current_step(self):
        return self._current_trajectory_step

    def get_steps(self, name: str):
        return [step for step in self._trajectory_steps if step.name == name]

    def save_action(
        self, action: ActionRequest, completion_cost: Optional[float] = None
    ):
        if self._current_trajectory_step:
            self._current_trajectory_step.actions.append(
                TrajectoryAction(action=action, completion_cost=completion_cost)
            )

    def save_error(self, error: str):
        if self._current_trajectory_step:
            self._current_trajectory_step.error = error

    def new_transition(self, name: str, transition_input: dict):
        if self._current_trajectory_step:
            self._trajectory_steps.append(self._current_trajectory_step)

        trajectory_step = TrajectoryStep(name=name, transition_input=transition_input)
        self._current_trajectory_step = trajectory_step

    def save_output(self, output: dict):
        self._output_data = output

    def save_info(self, info: dict):
        self._info = info

    def dict(self, **kwargs):
        steps = []
        for step in self._trajectory_steps:
            step_dict = step.dict(**kwargs)
            steps.append(step_dict)
        return {
            "name": self._name,
            "input": self._input_data,
            "steps": steps,
            "output": self._output_data,
            "info": self._info,
        }

    def total_cost(self):
        total_cost = 0
        for step in self._trajectory_steps:
            for action in step.actions:
                if action.completion_cost:
                    total_cost += action.completion_cost

        for child_traj in self._child_trajectorie:
            total_cost += child_traj.total_cost()

        return total_cost

    def persist(self, file_path: str):
        with open(f"{file_path}", "w") as f:
            f.write(
                json.dumps(
                    self.dict(exclude_none=True),
                    indent=2,
                    default=to_jsonable_python,
                )
            )
