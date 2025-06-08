from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.environment import EnvironmentExecutionError
from moatless.file_context import FileContext


class RunPythonScriptArgs(ActionArguments):
    """
    Run a Python script file and return its output.

    The script file must exist in the repository and will be executed in the current repository
    directory with full access to the repository context. Since Python automatically adds the
    script's directory to sys.path, you can directly import any modules or packages from the
    repository root without needing to modify the Python path.
    """

    script_path: str = Field(..., description="Path to the Python script to execute")
    args: list[str] = Field(default=[], description="Command line arguments to pass to the script")
    timeout: int = Field(default=30, description="Timeout in seconds for script execution")

    model_config = ConfigDict(title="RunPythonScript")


class RunPythonScript(Action):
    """Action to run a Python script and capture its output."""

    args_schema = RunPythonScriptArgs

    async def execute(self, args: RunPythonScriptArgs, file_context: FileContext | None = None) -> Observation:
        """Execute a Python script and return its output."""
        if not self.workspace.environment:
            raise ValueError("Environment is required to run Python scripts")

        # Construct the Python command
        cmd_parts = ["python", args.script_path]
        if args.args:
            cmd_parts.extend(args.args)

        command = " ".join(cmd_parts)

        try:
            if file_context and file_context.shadow_mode:
                patch = file_context.generate_git_patch()
            else:
                patch = None

            output = await self.workspace.environment.execute(command, patch=patch, fail_on_error=True)

            return Observation.create(message=f"Python output:\n{output}")
        except EnvironmentExecutionError as e:
            return Observation.create(
                message=f"Python output:\n{e.stderr}", properties={"fail_reason": "execution_error"}
            )
