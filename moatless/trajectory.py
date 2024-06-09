import json
import logging
from typing import Optional, Any

from litellm import completion_cost
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from moatless.types import ActionRequest

logger = logging.getLogger(__name__)


class TrajectoryAction(BaseModel):
    name: str
    input: Optional[dict[str, Any]] = None
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    trajectory: Optional[dict[str, Any]] = None


class TrajectoryStep(BaseModel):
    name: Optional[str] = None
    input: Optional[dict[str, Any]] = None
    thought: Optional[str] = None
    actions: list[TrajectoryAction] = []
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    completion_cost: Optional[float] = None


class Trajectory:

    def __init__(
        self,
        name: str,
        input_data: Optional[dict[str, Any]] = None,
        parent: Optional["Trajectory"] = None,
    ):
        self._name = name
        self._input_data = input_data
        self._output_data: Optional[dict[str, Any]] = None
        self._trajectory_steps: list[TrajectoryStep] = []
        self._current_trajectory_step: Optional[TrajectoryStep] = None
        self._parent = parent
        self._info: dict[str, Any] = {}

        self._child_trajectorie: list[Trajectory] = []

    def create_child_trajectory(
        self, name: str, input_data: Optional[dict[str, Any]] = None
    ):
        child_traj = Trajectory(name, input_data=input_data, parent=self)
        self._child_trajectorie.append(child_traj)
        return child_traj

    @property
    def current_step(self):
        return self._current_trajectory_step

    def new_step(
        self,
        name: Optional[str] = None,
        input: Optional[dict[str, Any]] = None,
        completion_response=None,
    ):

        cost = None
        if completion_response:
            try:
                cost = completion_cost(completion_response=completion_response)
            except Exception as e:
                logger.info(f"Error calculating completion cost: {e}")

        trajectory_step = TrajectoryStep(name=name, input=input, completion_cost=cost)

        self._trajectory_steps.append(trajectory_step)
        self._current_trajectory_step = trajectory_step

    def save_thought(self, thought: str):
        if self._current_trajectory_step:
            self._current_trajectory_step.thought = thought

    def save_action(
        self,
        name: str,
        input: ActionRequest | dict,
        output: Optional[dict] = None,
        error: Optional[str] = None,
        trajectory: Optional["Trajectory"] = None,
    ):
        action = TrajectoryAction(
            name=name,
            input=(
                input.dict(exclude_none=True)
                if isinstance(input, ActionRequest)
                else input
            ),
            trajectory=trajectory.dict(exclude_none=True) if trajectory else None,
            output=output,
            error=error,
        )

        if self._current_trajectory_step:
            self._current_trajectory_step.actions.append(action)

        return action

    def save_error(self, error: str):
        if self._current_trajectory_step:
            self._current_trajectory_step.error = error

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
            if step.completion_cost:
                total_cost += step.completion_cost

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
