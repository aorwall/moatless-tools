import logging
from pathlib import Path
from typing import List

from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.model import ActionArguments, Observation, FewShotExample
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.file import do_diff
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment

logger = logging.getLogger(__name__)

SNIPPET_LINES = 4


class InsertLineArgs(ActionArguments):
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
    new_str: str = Field(
        ..., description="Text content to insert at the specified line"
    )

    class Config:
        title = "InsertLines"


class InsertLine(Action, CodeModificationMixin):
    """
    Action to insert text at a specific line in a file.
    """

    args_schema = InsertLineArgs

    def __init__(
        self,
        runtime: RuntimeEnvironment | None = None,
        code_index: CodeIndex | None = None,
        repository: Repository | None = None,
        **data,
    ):
        super().__init__(**data)
        # Initialize mixin attributes directly
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_code_index", code_index)
        object.__setattr__(self, "_repository", repository)

    def execute(self, args: InsertLineArgs, file_context: FileContext) -> Observation:
        if args.path.startswith("/repo"):
            args.path = args.path[5:]
        if args.path.startswith("/"):
            args.path = args.path[1:]

        path = Path(args.path)

        if not file_context.file_exists(str(path)):
            return Observation(
                message=f"File {path} not found.",
                properties={"fail_reason": "file_not_found"},
            )

        context_file = file_context.get_context_file(str(path))
        if not context_file:
            return Observation(
                message=f"Could not get context for file: {path}",
                properties={"fail_reason": "context_error"},
            )

        file_text = context_file.content.expandtabs()
        new_str = args.new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if args.insert_line < 0 or args.insert_line > len(file_text_lines):
            return Observation(
                message=f"Invalid `insert_line` parameter: {args.insert_line}. It should be within the range of lines of the file: [0, {n_lines_file}]",
                properties={"fail_reason": "invalid_line_number"},
                expect_correction=True,
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[: args.insert_line]
            + new_str_lines
            + file_text_lines[args.insert_line :]
        )
        snippet_lines = (
            file_text_lines[max(0, args.insert_line - SNIPPET_LINES) : args.insert_line]
            + new_str_lines
            + file_text_lines[args.insert_line : args.insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        diff = do_diff(str(path), file_text, new_file_text)
        context_file.apply_changes(new_file_text)

        # Format the snippet with line numbers
        snippet_with_lines = "\n".join(
            f"{i + max(1, args.insert_line - SNIPPET_LINES + 1):6}\t{line}"
            for i, line in enumerate(snippet.split("\n"))
        )

        success_msg = (
            f"The file {path} has been edited. Here's the result of running `cat -n` "
            f"on a snippet of the edited file:\n{snippet_with_lines}\n"
            "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). "
            "Edit the file again if necessary."
        )

        observation = Observation(
            message=success_msg,
            properties={"diff": diff, "success": True},
        )

        if not self._runtime:
            return observation

        run_tests = RunTests(
            repository=self._repository,
            runtime=self._runtime,
            code_index=self._code_index,
        )
        test_observation = run_tests.execute(
            RunTestsArgs(
                scratch_pad=args.scratch_pad,
                test_files=[args.path],
            ),
            file_context,
        )

        observation.properties.update(test_observation.properties)
        observation.message += "\n\n" + test_observation.message

        return observation

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Add a new import statement at the beginning of the file",
                action=InsertLineArgs(
                    scratch_pad="Adding import for datetime module",
                    path="utils/time_helper.py",
                    insert_line=1,
                    new_str="from datetime import datetime, timezone",
                ),
            ),
            FewShotExample.create(
                user_input="Add a new method to the UserProfile class",
                action=InsertLineArgs(
                    scratch_pad="Adding a method to update user preferences",
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
                action=InsertLineArgs(
                    scratch_pad="Adding Redis configuration settings",
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
