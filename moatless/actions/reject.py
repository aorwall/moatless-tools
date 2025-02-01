from typing import Type, ClassVar

from pydantic import Field, ConfigDict

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.file_context import FileContext
from moatless.workspace import Workspace


class RejectArgs(ActionArguments):
    """Reject the task and explain why."""

    rejection_reason: str = Field(..., description="Explanation for rejection.")

    model_config = ConfigDict(title="Reject")

    def to_prompt(self):
        return f"Reject with reason: {self.rejection_reason}"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, RejectArgs)


class Reject(Action):
    args_schema: ClassVar[Type[ActionArguments]] = RejectArgs

    def execute(self, args: RejectArgs, file_context: FileContext | None = None):
        return Observation(message=args.rejection_reason, terminal=True)
