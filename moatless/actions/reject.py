from typing import ClassVar

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

    is_terminal: bool = Field(default=True, description="Whether the action will finish the flow")

    async def execute(self, args: RejectArgs, file_context: FileContext | None = None):
        return Observation.create(message="Rejected", terminal=True)
