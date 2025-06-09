from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.file_context import FileContext
from pydantic import ConfigDict


class RunPythonCodeArgs(ActionArguments):
    """
    Execute Python code directly for debugging and exploration.

    This action is ideal for:
    - Debugging issues by running diagnostic code
    - Testing hypotheses about the codebase
    - Exploring data structures or API behavior
    - Performing quick calculations or validations

    The code runs in the current repository directory with full access to the repository context.
    Since Python automatically adds the script's directory to sys.path, you can directly import
    any modules or packages from the repository root without needing to modify the Python path.
    """

    code: str = Field(
        ..., description="Python code to execute. Write complete, runnable code including any necessary imports."
    )
    timeout: int = Field(default=30, description="Timeout in seconds for code execution")

    model_config = ConfigDict(title="RunPythonCode")


class RunPythonCode(Action):
    """Action to execute Python code directly for debugging and exploration purposes."""

    args_schema = RunPythonCodeArgs

    async def execute(self, args: RunPythonCodeArgs, file_context: FileContext | None = None) -> Observation:
        """Execute Python code and return its output."""
        if not self.workspace.environment:
            raise ValueError("Environment is required to run Python code")

        try:
            # Execute the code using the new environment method
            output = await self.workspace.environment.execute_python_code(args.code)

            # Extract first few lines of code for summary
            code_lines = args.code.strip().split("\n")
            if code_lines and code_lines[0]:
                code_preview = code_lines[0]
                if len(code_lines) > 1:
                    code_preview += "..."
            else:
                code_preview = "<empty code>"

            return Observation.create(
                message=f"Python output:\n{output}", summary=f"Executed Python code: {code_preview}"
            )
        except Exception as e:
            return Observation.create(
                message=f"Error executing code: {str(e)}", properties={"fail_reason": "execution_error"}
            )
