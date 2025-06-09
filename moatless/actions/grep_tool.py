import logging
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
)
from moatless.completion.schema import FewShotExample
from moatless.environment.base import EnvironmentExecutionError
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class GrepToolArgs(ActionArguments):
    """
    Search file contents using regular expressions.

    - Searches file contents using regular expressions
    - Filter files by pattern with the include parameter (eg. "*.js", "*.{ts,tsx}")
    - Supports path-based include patterns (eg. "**/requests/**/*.py")
    - Use this tool when you need to find files containing specific patterns
    - Start with broad search patterns and refine as needed for large codebases
    - For more targeted searches, use specific regex patterns with file filtering
    """

    pattern: str = Field(
        ...,
        min_length=1,
        description="The regex pattern to search for in file contents. Supports full regex syntax.",
    )

    include: Optional[str] = Field(
        None,
        description="Optional glob pattern to filter files (e.g. '*.py', '*.{ts,tsx}', '**/path/to/*.js')",
    )

    max_results: int = Field(
        100,
        description="Maximum number of results to return",
    )

    model_config = ConfigDict(title="GrepTool")

    def to_prompt(self):
        include_str = f" in files matching '{self.include}'" if self.include else ""
        return f"Search for regex pattern '{self.pattern}'{include_str}"

    def short_summary(self) -> str:
        param_str = f"pattern='{self.pattern}'"
        if self.include:
            param_str += f", include='{self.include}'"
        if self.max_results != 100:
            param_str += f", max_results={self.max_results}"
        return f"{self.name}({param_str})"

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Find all function definitions in JavaScript files",
                action=GrepToolArgs(
                    thoughts="I'll search for function definitions in JavaScript files using a regex pattern.",
                    pattern=r"function\s+\w+\s*\(",
                    include="*.js",
                    max_results=10,
                ),
            ),
            FewShotExample.create(
                user_input="Find all error logging statements in the codebase",
                action=GrepToolArgs(
                    thoughts="I'll search for error logging patterns across all files.",
                    pattern=r"log.*Error|console\.error",
                    include=None,
                    max_results=10,
                ),
            ),
            FewShotExample.create(
                user_input="Find all timeout or decode errors in the requests library",
                action=GrepToolArgs(
                    thoughts="I'll search for TimeoutError and DecodeError in the requests library files.",
                    pattern=r"TimeoutError|DecodeError",
                    include="**/requests/**/*.py",
                    max_results=20,
                ),
            ),
            FewShotExample.create(
                user_input="Find API endpoints in the controllers directory",
                action=GrepToolArgs(
                    thoughts="I'll search for API route definitions in the controllers directory.",
                    pattern=r"@\w+\.route\(|app\.get\(|app\.post\(",
                    include="src/controllers/**/*.js",
                    max_results=15,
                ),
            ),
        ]


class GrepTool(Action):
    """
    Fast content search tool using regular expressions.
    """

    args_schema = GrepToolArgs

    async def execute(
            self,
            args: ActionArguments,
            file_context: FileContext | None = None,
    ) -> Observation:
        if file_context is None:
            raise ValueError("File context must be provided to execute the grep tool action.")

        if not isinstance(args, GrepToolArgs):
            raise ValueError("Invalid arguments type for GrepTool")

        if not self._workspace:
            raise RuntimeError("No workspace set")

        if not self.workspace.environment:
            raise RuntimeError("No environment set")

        local_env = self.workspace.environment
        patch = file_context.generate_git_patch() if file_context.shadow_mode else None

        try:
            # Build the grep command
            grep_cmd = self._build_grep_command(args)
            logger.info(f"Executing grep command: {grep_cmd}")

            # Execute the grep command
            output = await self._execute_grep(grep_cmd, local_env, patch, args)

            # Process the grep output
            return self._process_grep_output(output, args)

        except EnvironmentExecutionError as e:
            # Return detailed error information to the LLM
            logger.error(f"Environment execution error: {e}")

            # Check for specific error types
            if e.return_code == 2 or e.return_code == 127:
                error_str = str(e)
                if (
                        ("not found" in error_str and "/bin/sh" in error_str)
                        or ("syntax error" in error_str.lower())
                        or ("command not found" in error_str.lower())
                ):
                    return Observation.create(
                        message=f"Shell command parsing error. Command: {grep_cmd}\nError: {e.stderr or str(e)}\nThis may be due to special characters in the regex pattern '{args.pattern}'. Please try escaping special characters or using a simpler pattern.",
                        properties={
                            "fail_reason": "shell_parsing_error",
                            "pattern": args.pattern,
                            "return_code": e.return_code,
                            "command": grep_cmd,
                            "stderr": e.stderr,
                        },
                    )

            # Generic environment execution error
            return Observation.create(
                message=f"Error executing grep command: {grep_cmd}\nError: {e.stderr or str(e)}",
                properties={
                    "fail_reason": "grep_execution_error",
                    "return_code": e.return_code,
                    "command": grep_cmd,
                    "stderr": e.stderr,
                },
            )

        except Exception as e:
            logger.error(f"Error in grep tool: {str(e)}")
            return Observation.create(
                message=f"Unexpected error in grep tool: {str(e)}",
                properties={
                    "fail_reason": "general_error",
                    "error_details": str(e),
                },
            )

    def _build_grep_command(self, args: GrepToolArgs) -> str:
        """Build the grep command based on the arguments."""
        # Base grep options
        grep_opts = [
            "grep",
            "-r",  # recursive
            "-n",  # show line numbers
            "-H",  # always show filename
            "--color=never",  # no color codes
            "-E",  # extended regex
            # Exclude common directories and files
            "--exclude-dir=.git",
            "--exclude-dir=.venv",
            "--exclude-dir=__pycache__",
            "--exclude-dir=node_modules",
            "--exclude=*.log",
            "--exclude=*.tmp",
        ]

        # Properly quote the pattern for shell
        quoted_pattern = shlex.quote(args.pattern)

        if not args.include:
            # No include pattern - search all files from current directory
            cmd = f"{' '.join(grep_opts)} {quoted_pattern} ."
        elif self._is_specific_file(args.include):
            # Specific file - grep just that file
            cmd = f"grep -n -H --color=never -E {quoted_pattern} {shlex.quote(args.include)}"
        else:
            # Use find to get matching files, then grep them
            # This approach is more reliable than mixing grep's --include with complex patterns
            find_pattern = self._convert_glob_to_find_pattern(args.include)

            # Build find command to get list of files
            find_cmd = f"find . -type f {find_pattern} -print0"

            # Pipe to xargs + grep for efficient processing
            # Using -print0 and xargs -0 handles filenames with spaces
            cmd = f"{find_cmd} | xargs -0 grep -n -H --color=never -E {quoted_pattern}"

        # Limit output lines
        return f"{cmd} | head -{args.max_results * 3}"

    def _is_specific_file(self, pattern: str) -> bool:
        """Check if the pattern refers to a specific file (no wildcards)."""
        return not any(char in pattern for char in ['*', '?', '[', ']'])

    def _convert_glob_to_find_pattern(self, glob_pattern: str) -> str:
        """Convert a glob pattern to find command options."""
        # Handle different glob patterns
        if glob_pattern.endswith('/'):
            # Directory search: path/to/dir/
            return f"-path './{glob_pattern}*'"
        elif '/' in glob_pattern:
            # Path-based pattern: **/requests/**/*.py or src/controllers/*.js
            # Convert ** to * for find's -path option
            find_pattern = glob_pattern.replace('**/', '*/').replace('/**', '/*')
            if not find_pattern.startswith('*'):
                find_pattern = './' + find_pattern
            return f"-path '{find_pattern}'"
        else:
            # Simple filename pattern: *.py, *.{ts,tsx}
            # Handle brace expansion manually
            if '{' in glob_pattern and '}' in glob_pattern:
                # Extract brace content: *.{ts,tsx} -> ts,tsx
                match = re.match(r'(.*)\{([^}]+)\}(.*)', glob_pattern)
                if match:
                    prefix, extensions, suffix = match.groups()
                    ext_list = extensions.split(',')
                    conditions = [f"-name '{prefix}{ext.strip()}{suffix}'" for ext in ext_list]
                    return f"\\( {' -o '.join(conditions)} \\)"
            return f"-name '{glob_pattern}'"

    async def _execute_grep(self, cmd: str, env, patch, args: GrepToolArgs) -> str:
        """Execute the grep command and handle common errors."""
        try:
            output = await env.execute(cmd, patch=patch)
            return output
        except EnvironmentExecutionError as e:
            # No matches found (grep returns exit code 1)
            if e.return_code == 1:
                return ""  # Empty output indicates no matches

            # For all other errors, we want to return this info to the LLM
            # so it can understand what went wrong
            raise e  # Re-raise to be handled in execute()

    def _process_grep_output(self, output: str, args: GrepToolArgs) -> Observation:
        """Process the raw grep output and format it nicely."""
        if not output.strip():
            msg = f"No matches found for regex pattern '{args.pattern}'"
            if args.include:
                msg += f" in files matching '{args.include}'"
            return Observation.create(message=msg)

        # Parse grep output lines
        matches = []
        file_matches = {}

        for line in output.strip().split('\n'):
            if not line.strip():
                continue

            # Skip any shell error messages
            if self._is_error_line(line):
                logger.debug(f"Skipping error line: {line}")
                continue

            # Parse the grep output line
            match_info = self._parse_grep_line(line)
            if not match_info:
                continue

            file_path = match_info['file_path']

            # Clean up file path
            if file_path.startswith('./'):
                file_path = file_path[2:]

            # Store match
            match_data = {
                'file_path': file_path,
                'line_num': match_info['line_num'],
                'content': match_info['content']
            }

            matches.append(match_data)

            # Group by file
            if file_path not in file_matches:
                file_matches[file_path] = []
            file_matches[file_path].append({
                'line_num': match_info['line_num'],
                'content': match_info['content']
            })

            # Stop if we've reached max results
            if len(matches) >= args.max_results:
                break

        # Format the output message
        message = self._format_output_message(matches, file_matches, args)

        return Observation.create(
            message=message,
            summary=f"Found {len(matches)} matches in {len(file_matches)} files",
            properties={
                "total_matches": len(matches),
                "total_files": len(file_matches),
                "matches": matches[:args.max_results],  # Ensure we don't exceed max
                "max_results": args.max_results,
                "pattern": args.pattern,
                "include": args.include,
            },
        )

    def _is_error_line(self, line: str) -> bool:
        """Check if a line is a shell error message."""
        error_indicators = [
            "/bin/sh:", "syntax error", "command not found",
            "find:", "grep:", "xargs:"  # Common command errors
        ]
        return any(indicator in line.lower() for indicator in error_indicators)

    def _parse_grep_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a grep output line into its components."""
        # Format: "file_path:line_num:content"
        parts = line.split(':', 2)

        if len(parts) < 3:
            logger.debug(f"Malformed grep line: {line}")
            return None

        try:
            file_path = parts[0]
            line_num = int(parts[1])
            content = parts[2].strip()

            return {
                'file_path': file_path,
                'line_num': line_num,
                'content': content
            }
        except ValueError as e:
            logger.debug(f"Error parsing grep line '{line}': {e}")
            return None

    def _format_output_message(self, matches: List[Dict], file_matches: Dict, args: GrepToolArgs) -> str:
        """Format the matches into a readable message."""
        message = f"Found {len(matches)} matches for regex pattern '{args.pattern}'"
        if args.include:
            message += f" in files matching '{args.include}'"
        message += "\n\n"

        # Display matches grouped by file
        for file_path, file_match_list in sorted(file_matches.items()):
            message += f"📄 {file_path}\n"

            for match in file_match_list:
                # Truncate very long lines for readability
                content = match['content']
                if len(content) > 200:
                    content = content[:197] + "..."
                message += f"    Line {match['line_num']}: {content}\n"

            message += "\n"

        # Add note if results were limited
        if len(matches) >= args.max_results:
            message += f"\nNote: Results limited to {args.max_results} matches. More matches may exist."

        return message
