from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation
from moatless.file_context import FileContext
from moatless.workspace import Workspace


class MessageArgs(ActionArguments):
    """Respond with a message to the user."""

    message: str = Field(..., description="The message to send to the user.")

    class Config:
        title = "SendMessage"

    def to_prompt(self):
        return f"Message: {self.message}"

    def equals(self, other: "ActionArguments") -> bool:
        return other.message == self.message


class MessageAction(Action):
    """
    Action to just send a response to the user.
    """

    args_schema = MessageArgs

    def execute(
        self,
        args: MessageArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        return Observation()
