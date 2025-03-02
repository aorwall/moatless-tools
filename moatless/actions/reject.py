from typing import ClassVar, Type

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.file_context import FileContext


class RejectArgs(ActionArguments):
    """Reject the task and explain why."""

    rejection_reason: str = Field(..., description="Explanation for rejection.")

    model_config = ConfigDict(title="Reject")

    def to_prompt(self):
        return f"Reject with reason: {self.rejection_reason}"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, RejectArgs)


class Reject(Action):
    args_schema: ClassVar[type[ActionArguments]] = RejectArgs

    async def execute(self, args: RejectArgs, file_context: FileContext | None = None):
        return Observation(message="Rejected", terminal=True)
