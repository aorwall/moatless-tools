from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.environment import EnvironmentExecutionError
from moatless.file_context import FileContext
from moatless.utils.tokenizer import count_tokens


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
    max_output_tokens: int = Field(default=2000, description="Maximum number of tokens to return in output")

    model_config = ConfigDict(title="RunPythonScript")


class RunPythonScript(Action):
    """Action to run a Python script and capture its output."""

    args_schema = RunPythonScriptArgs

    def _truncate_output_by_tokens(self, output: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> tuple[str, bool]:
        """
        Truncate output to fit within max_tokens limit.
        
        Returns:
            tuple: (truncated_output, was_truncated)
        """
        if not output:
            return output, False
            
        token_count = count_tokens(output, model)
        if token_count <= max_tokens:
            return output, False
        
        # Binary search to find the right truncation point
        lines = output.split('\n')
        if len(lines) <= 1:
            # If it's a single line, truncate by characters
            chars_per_token = len(output) / token_count
            target_chars = int(max_tokens * chars_per_token * 0.9)  # 90% to be safe
            truncated = output[:target_chars]
            return truncated, True
        
        # Binary search on lines
        left, right = 0, len(lines)
        best_result = ""
        
        while left < right:
            mid = (left + right + 1) // 2
            partial_output = '\n'.join(lines[:mid])
            
            if count_tokens(partial_output, model) <= max_tokens:
                best_result = partial_output
                left = mid
            else:
                right = mid - 1
        
        return best_result, True

    async def execute(self, args: ActionArguments, file_context: FileContext | None = None) -> Observation:
        """Execute a Python script and return its output."""
        if not isinstance(args, RunPythonScriptArgs):
            raise ValueError("Expected RunPythonScriptArgs")
            
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

            # Truncate output if it exceeds max_output_tokens
            truncated_output, was_truncated = self._truncate_output_by_tokens(output, args.max_output_tokens)
            
            message = f"Python output:\n{truncated_output}"
            properties = {}
            if was_truncated:
                message += f"\n\n[Output truncated at {args.max_output_tokens} tokens. Please revise the script to show less output if you need to see more details.]"
                properties["fail_reason"] = "truncated"

            return Observation.create(message=message, properties=properties)
        except EnvironmentExecutionError as e:
            # Also truncate error output
            truncated_error, was_truncated = self._truncate_output_by_tokens(e.stderr, args.max_output_tokens)
            
            message = f"Python output:\n{truncated_error}"
            properties = {"fail_reason": "execution_error"}
            if was_truncated:
                message += f"\n\n[Error output truncated at {args.max_output_tokens} tokens. Please revise the script to show less output if you need to see more details.]"
                # Add truncated as additional fail reason for errors
                properties["fail_reason"] = "execution_error_truncated"
            
            return Observation.create(message=message, properties=properties)
