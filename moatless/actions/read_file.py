import logging
from typing import Optional, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation, RewardScaleEntry
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class ReadFileArgs(ActionArguments):
    """Read specific lines from a file.

    This action allows you to read the contents of a file, either in its entirety or a specific range of lines.
    It's useful for examining code, configuration files, or any text file in the repository.

    The action will return at most 200 lines of content at a time. If more lines are requested,
    the content will be truncated and a note will be added indicating the truncation.

    Example usage:
    - Read entire file (first 200 lines): {"file_path": "src/main.py"}
    - Read specific lines: {"file_path": "src/main.py", "start_line": 10, "end_line": 20}
    - Read from line to end: {"file_path": "src/main.py", "start_line": 50}
    """

    file_path: str = Field(
        ...,
        description="The path to the file you want to read, relative to the repository root. For example: 'src/main.py'",
    )
    start_line: Optional[int] = Field(
        None,
        description="The first line number to include in the output (1-based indexing). If not specified, reading starts from the beginning of the file.",
    )
    end_line: Optional[int] = Field(
        None,
        description="The last line number to include in the output (inclusive). If not specified, reading continues until the end of the file or until reaching the 100-line limit.",
    )

    model_config = ConfigDict(title="ReadFile")

    @property
    def log_name(self):
        if self.start_line and self.end_line:
            return f"ReadFile({self.file_path} {self.start_line}-{self.end_line})"
        return f"ReadFile({self.file_path})"

    def to_prompt(self):
        prompt = f"Read file {self.file_path}"
        if self.start_line and self.end_line:
            prompt += f" lines {self.start_line}-{self.end_line}"
        return prompt

    def short_summary(self) -> str:
        return f"{self.name}(path={self.file_path})"


class ReadFile(Action):
    """An action that reads and returns the contents of a file.

    This action is designed to be used as a tool for examining file contents. It provides several features:
    - Reading entire files (up to 100 lines)
    - Reading specific line ranges
    - Reading from a specific line to the end
    - Automatic truncation with notification when content exceeds 100 lines
    - Optionally adding the read lines to the file context for future reference

    The action will return an Observation containing:
    - The file content in the message field
    - A summary of what was read
    - Error information if something went wrong

    Error cases handled:
    - File not found
    - Path is a directory
    - Invalid line numbers
    - Missing file context
    """

    args_schema = ReadFileArgs

    max_lines: int = Field(100, description="The maximum number of lines to read from the file.")

    async def execute(self, args: ReadFileArgs, file_context: FileContext | None = None) -> Observation:
        if file_context is None:
            raise ValueError("File context must be provided to execute the read action.")

        file = file_context.get_file(args.file_path)

        if not file:
            message = f"The requested file {args.file_path} is not found in the file repository."
            properties = {"fail_reason": "file_not_found"}
            return Observation.create(message=message, properties=properties)

        if self._repository.is_directory(args.file_path):
            message = f"The requested file {args.file_path} is a directory and not a file."
            properties = {"fail_reason": "is_directory"}
            return Observation.create(message=message, properties=properties)

        # Get the content of the file
        content = file.content
        lines = content.split("\n")

        # If no line range specified, return the first MAX_LINES
        if args.start_line is None and args.end_line is None:
            selected_lines = lines[: self.max_lines]
            content = "\n".join(selected_lines)

            file_context.add_line_span_to_context(
                args.file_path,
                1,  # 1-based indexing for start_line
                min(len(lines), self.max_lines),  # Don't exceed file length
            )

            return Observation.create(
                message=f"```{args.file_path}\n{content}\n```",
                summary=f"Read first {len(selected_lines)} lines from {args.file_path}",
            )

        # Validate line numbers
        if args.start_line is not None and args.start_line > len(lines):
            message = f"The requested start line {args.start_line} is greater than the number of lines in the file {len(lines)}."
            properties = {"fail_reason": "start_line_greater_than_file_length"}
            return Observation.create(message=message, properties=properties)

        # Get the requested lines
        start = args.start_line - 1 if args.start_line else 0  # Convert to 0-based index
        end = args.end_line if args.end_line else len(lines)  # end_line is inclusive

        # Calculate the actual end line, ensuring we don't exceed MAX_LINES
        actual_end = min(end, start + self.max_lines)
        selected_lines = lines[start:actual_end]

        # Add to file context if requested
        file_context.add_line_span_to_context(
            args.file_path,
            args.start_line or 1,  # Default to line 1 if start_line is None
            actual_end,
        )

        # Join the lines back together
        content = "\n".join(selected_lines)

        # If we had to truncate the selection, add a note
        truncation_note = ""
        if actual_end < end:
            truncation_note = f"\n\n... (truncated at {self.max_lines} lines)"

        line_info = f"lines {args.start_line or 1}-{actual_end}" if args.start_line or args.end_line else ""

        return Observation.create(
            message=f"```{args.file_path} {line_info}\n{content}{truncation_note}\n```",
            summary=f"Read lines {args.start_line or 1}-{actual_end} from {args.file_path}",
        )

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int | None = None) -> list[str]:
        criteria = [
            "File Selection Relevance: Assess whether the file being read is directly relevant to the task at hand.",
            "Line Range Appropriateness: Evaluate if the selected line range (if specified) contains the necessary information without excessive content.",
            "Information Extraction: Determine if the agent effectively extracts and utilizes the information gathered from reading the file.",
            "Avoiding Unnecessary Reads: Check if the agent avoids redundant reads of the same file sections.",
        ]
        return criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> list[RewardScaleEntry]:
        return [
            RewardScaleEntry(
                min_value=75,
                max_value=100,
                description="The file read is highly relevant, with an optimal line range selection that provides exactly the needed information for the task.",
            ),
            RewardScaleEntry(
                min_value=50,
                max_value=74,
                description="The file read is relevant with a reasonable line range, though some content may be unnecessary or some useful content may be missing.",
            ),
            RewardScaleEntry(
                min_value=25,
                max_value=49,
                description="The file read has some relevance but includes excessive unnecessary content or misses important sections.",
            ),
            RewardScaleEntry(
                min_value=0,
                max_value=24,
                description="The file read has minimal relevance to the task or reads an inappropriate amount of content.",
            ),
            RewardScaleEntry(
                min_value=-49,
                max_value=-1,
                description="The file read is irrelevant to the task, demonstrates misunderstanding of the codebase, or attempts to read non-existent files.",
            ),
        ]
