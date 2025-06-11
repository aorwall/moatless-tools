import re
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
        
        # Try to preserve as much content as possible while staying under the limit
        lines = output.split('\n')
        
        if len(lines) <= 1:
            # Single line - use more aggressive character-based truncation
            chars_per_token = len(output) / token_count
            # Use 95% of the limit to be conservative but not overly so
            target_chars = int(max_tokens * chars_per_token * 0.95)
            truncated = output[:target_chars]
            return truncated, True
        
        # Multi-line - use binary search to find optimal line truncation point
        left, right = 0, len(lines)
        best_result = ""
        best_line_count = 0
        
        while left <= right:
            mid = (left + right) // 2
            partial_output = '\n'.join(lines[:mid])
            partial_tokens = count_tokens(partial_output, model)
            
            if partial_tokens <= max_tokens:
                best_result = partial_output
                best_line_count = mid
                left = mid + 1
            else:
                right = mid - 1
        
        # If we couldn't fit any complete lines, fall back to character truncation of first few lines
        if best_line_count == 0 and lines:
            # Take first few lines and truncate by characters
            first_few_lines = '\n'.join(lines[:min(3, len(lines))])
            if count_tokens(first_few_lines, model) > max_tokens:
                # Even first few lines are too much, truncate by characters
                chars_per_token = len(first_few_lines) / count_tokens(first_few_lines, model)
                target_chars = int(max_tokens * chars_per_token * 0.95)
                return first_few_lines[:target_chars], True
            else:
                return first_few_lines, True
        
        return best_result, True

    def _strip_ansi_codes(self, text: str) -> str:
        """
        Strip ANSI color codes and terminal sequences from text.
        
        This removes common ANSI escape sequences including:
        - Color codes (\033[31m, \033[0m, etc.)
        - Cursor movement (\033[K, \r, etc.)
        - Bold, underline, and other formatting
        """
        if not text:
            return text
            
        # ANSI escape sequence pattern - matches \033[ or \x1b[ followed by any characters until 'm'
        # Also matches common terminal control characters like \r
        ansi_pattern = r'\033\[[0-9;]*[mK]|\x1b\[[0-9;]*[mK]|\r'
        return re.sub(ansi_pattern, '', text)

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

            # Strip ANSI codes from output
            clean_output = self._strip_ansi_codes(output)

            # Truncate output if it exceeds max_output_tokens
            truncated_output, was_truncated = self._truncate_output_by_tokens(clean_output, args.max_output_tokens)
            
            message = f"Python output:\n{truncated_output}"
            properties = {}
            if was_truncated:
                message += f"\n\n[Output truncated at {args.max_output_tokens} tokens. Please revise the script to show less output if you need to see more details.]"
                properties["fail_reason"] = "truncated"

            return Observation.create(message=message, properties=properties)
        except EnvironmentExecutionError as e:
            # Strip ANSI codes from error output
            clean_error = self._strip_ansi_codes(e.stderr)
            
            # Also truncate error output
            truncated_error, was_truncated = self._truncate_output_by_tokens(clean_error, args.max_output_tokens)
            
            message = f"Python output:\n{truncated_error}"
            properties = {"fail_reason": "execution_error"}
            if was_truncated:
                message += f"\n\n[Error output truncated at {args.max_output_tokens} tokens. Please revise the script to show less output if you need to see more details.]"
                # Add truncated as additional fail reason for errors
                properties["fail_reason"] = "execution_error_truncated"
            
            return Observation.create(message=message, properties=properties)
