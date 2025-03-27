from datetime import datetime
from typing import Optional

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
)
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext


class GrepToolArgs(ActionArguments):
    """
    Search file contents using regular expressions.

    - Fast content search tool that works with any codebase size
    - Searches file contents using regular expressions
    - Supports full regex syntax (eg. "log.*Error", "function\\s+\\w+", etc.)
    - Filter files by pattern with the include parameter (eg. "*.js", "*.{ts,tsx}")
    - Returns matching file paths sorted by modification time
    - Use this tool when you need to find files containing specific patterns
    - Start with broad search patterns and refine as needed for large codebases
    - For more targeted searches, use specific regex patterns with file filtering
    """

    pattern: str = Field(
        ...,
        description="The regex pattern to search for in file contents. Supports full regex syntax.",
    )

    include: Optional[str] = Field(
        None,
        description="Optional glob pattern to filter files (e.g. '*.py', '*.{ts,tsx}')",
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
                user_input="Find all TODO comments in Python files",
                action=GrepToolArgs(
                    thoughts="I'll search for TODO comments in Python files.",
                    pattern=r"#\s*TODO",
                    include="*.py",
                    max_results=10,
                ),
            ),
        ]


class GrepTool(Action):
    """
    Fast content search tool using regular expressions.
    """

    args_schema = GrepToolArgs

    async def _execute(
        self,
        args: GrepToolArgs,
        file_context: FileContext | None = None,
    ) -> str | None:
        if not file_context or not file_context._repo:
            raise RuntimeError("Repository not available for grep search.")

        try:
            # Use the new regex search method from FileRepository
            matches = await file_context._repo.find_regex_matches(
                regex_pattern=args.pattern, include_pattern=args.include, max_results=args.max_results
            )

            if not matches:
                return f"No matches found for regex pattern '{args.pattern}'"

            # Format the results for output
            message = f"Found {len(matches)} matches for regex pattern '{args.pattern}'"
            if args.include:
                message += f" in files matching '{args.include}'"
            message += "\n\n"

            # Group by file path for cleaner display
            files_grouped = {}
            for match in matches:
                file_path = match["file_path"]
                if file_path not in files_grouped:
                    mod_time = datetime.fromtimestamp(match["mod_time"]).strftime("%Y-%m-%d %H:%M:%S")
                    files_grouped[file_path] = {"mod_time": mod_time, "matches": []}

                files_grouped[file_path]["matches"].append({"line_num": match["line_num"], "content": match["content"]})

            for file_path, file_data in files_grouped.items():
                message += f"📄 {file_path}\n"

                for match in file_data["matches"]:
                    line_num = match["line_num"]
                    content = match["content"].strip()
                    message += f"    Line {line_num}: {content}\n"

                message += "\n"

            return message

        except Exception as e:
            return f"Error executing grep search: {str(e)}"
