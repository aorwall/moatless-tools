import logging
from pathlib import Path

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.code_action_value_mixin import CodeActionValueMixin
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.schema import ActionArguments, Observation
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)

SNIPPET_LINES = 4


class InsertLinesArgs(ActionArguments):
    """
    Insert text at a specific line number in a file.

    Notes:
    * The text will be inserted AFTER the specified line number
    * Line numbers start at 1
    * The insert_line must be within the valid range of lines in the file
    * Proper indentation should be maintained in the inserted text
    """

    path: str = Field(..., description="Path to the file to edit")
    insert_line: int = Field(
        ...,
        description="Line number after which to insert the new text (indexing starts at 1)",
    )
    new_str: str = Field(..., description="Text content to insert at the specified line")

    model_config = ConfigDict(title="InsertLines")

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<insert_line>{self.insert_line}</insert_line>
<new_str>
{self.new_str}
</new_str>"""

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Add a new import statement at the beginning of the file",
                action=InsertLinesArgs(
                    thoughts="Adding import for datetime module",
                    path="utils/time_helper.py",
                    insert_line=1,
                    new_str="from datetime import datetime, timezone",
                ),
            ),
            FewShotExample.create(
                user_input="Add a new method to the UserProfile class",
                action=InsertLinesArgs(
                    thoughts="Adding a method to update user preferences",
                    path="models/user.py",
                    insert_line=15,
                    new_str="""    def update_preferences(self, preferences: dict) -> None:
        self._preferences.update(preferences)
        self._last_updated = datetime.now(timezone.utc)
        logger.info(f"Updated preferences for user {self.username}")""",
                ),
            ),
            FewShotExample.create(
                user_input="Add a new configuration option",
                action=InsertLinesArgs(
                    thoughts="Adding Redis configuration settings",
                    path="config/settings.py",
                    insert_line=25,
                    new_str="""REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'password': None
}""",
                ),
            ),
        ]


class InsertLine(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to insert text at a specific line in a file.
    """

    args_schema = InsertLinesArgs

    async def execute(
        self,
        args: InsertLinesArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        if args.path.startswith("/repo"):
            args.path = args.path[5:]
        if args.path.startswith("/"):
            args.path = args.path[1:]

        path = Path(args.path)

        if not file_context.file_exists(str(path)):
            return Observation.create(
                message=f"File {path} not found.",
                properties={"fail_reason": "file_not_found"},
            )

        context_file = file_context.get_context_file(str(path))
        if not context_file:
            return Observation.create(
                message=f"Could not get context for file: {path}",
                properties={"fail_reason": "context_error"},
            )

        if not context_file.lines_is_in_context(args.insert_line - 1, args.insert_line):
            return Observation.create(
                message=f"Line {args.insert_line} is not in the visible portion of file {path}. Please provide a line number within the visible code, use ViewCode to see the code.",
                properties={"fail_reason": "lines_not_in_context"},
            )

        file_text = context_file.content.expandtabs()
        new_str = args.new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if args.insert_line < 0 or args.insert_line > len(file_text_lines):
            return Observation.create(
                message=f"Invalid `insert_line` parameter: {args.insert_line}. It should be within the range of lines of the file: [0, {n_lines_file}]",
                properties={"fail_reason": "invalid_line_number"},
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = file_text_lines[: args.insert_line] + new_str_lines + file_text_lines[args.insert_line :]
        snippet_lines = (
            file_text_lines[max(0, args.insert_line - SNIPPET_LINES) : args.insert_line]
            + new_str_lines
            + file_text_lines[args.insert_line : args.insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        context_file.apply_changes(new_file_text)

        # Format the snippet with line numbers
        snippet_with_lines = "\n".join(
            f"{i + max(1, args.insert_line - SNIPPET_LINES + 1):6}\t{line}"
            for i, line in enumerate(snippet.split("\n"))
        )

        message = (
            f"The file {path} has been edited. Here's the result of running `cat -n` "
            f"on a snippet of the edited file:\n{snippet_with_lines}\n"
            "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). "
            "Edit the file again if necessary."
        )

        return Observation.create(message=message)
