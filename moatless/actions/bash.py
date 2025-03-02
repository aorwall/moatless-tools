from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.file_context import FileContext


class BashArgs(ActionArguments):
    """
    Run commands in a bash shell
    * When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
    * You have access to a mirror of common linux and python packages via apt and pip.
    * State is persistent across command calls and discussions with the user.
    * To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
    * Please avoid commands that may produce a very large amount of output.
    """

    command: str = Field(..., description="The command to run")


class BashTool(Action):
    args_schema = BashArgs

    async def execute(self, args: BashArgs, file_context: FileContext | None = None) -> Observation:
        if not self.workspace:
            raise ValueError("Workspace is required to run commands")

        if not self.workspace.environment:
            raise ValueError("Environment is required to run commands")

        output = await self.workspace.environment.execute(args.command)
        return Observation.create(message=output)
