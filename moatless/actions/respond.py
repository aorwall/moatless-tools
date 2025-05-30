from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.file_context import FileContext
from moatless.workspace import Workspace


class RespondArgs(ActionArguments):
    """Respond with a message to the user."""

    message: str = Field(..., description="The message to send to the user.")

    model_config = ConfigDict(title="SendMessage")

    def to_prompt(self):
        return f"Message: {self.message}"


class Respond(Action):
    """
    Action to just send a response to the user.
    """

    args_schema = RespondArgs

    is_terminal: bool = Field(default=True, description="Whether the action will finish the flow")

    async def execute(
        self,
        args: RespondArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        return Observation.create(message=args.message)
