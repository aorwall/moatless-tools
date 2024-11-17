import logging
from pathlib import Path
from typing import Literal, Optional, List

from pydantic import Field, PrivateAttr

from moatless.actions import RunTests
from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation, RetryException
from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.completion.model import ToolCall
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.file import do_diff
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)

Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]

SNIPPET_LINES: int = 4


class EditActionArguments(ActionArguments):
    """
    An filesystem editor tool that allows the agent to view, create, and edit files.
    """

    command: Command = Field(..., description="The edit command to execute")
    path: str = Field(..., description="The file path to edit")
    file_text: Optional[str] = Field(
        None, description="The text content for file creation"
    )
    view_range: Optional[List[int]] = Field(
        None, description="Range of lines to view [start, end]"
    )
    old_str: Optional[str] = Field(None, description="String to replace")
    new_str: Optional[str] = Field(None, description="Replacement string")
    insert_line: Optional[int] = Field(None, description="Line number for insertion")

    class Config:
        title = "str_replace_editor"

    def to_tool_call(self) -> ToolCall:
        return ToolCall(
            name=self.name, type="text_editor_20241022", input=self.model_dump()
        )


class ClaudeEditTool(Action):
    """
    An filesystem editor tool that allows the agent to view, create, and edit files.
    The tool parameters are defined by Anthropic and are not editable.
    """

    args_schema = EditActionArguments

    max_tokens_to_view: int = Field(
        2000, description="Max tokens to view in one command"
    )

    _runtime: RuntimeEnvironment | None = PrivateAttr(None)
    _code_index: CodeIndex | None = PrivateAttr(None)
    _repository: Repository | None = PrivateAttr(None)

    def __init__(
        self,
        runtime: RuntimeEnvironment | None = None,
        code_index: CodeIndex | None = None,
        repository: Repository | None = None,
        **data,
    ):
        super().__init__(**data)
        self._runtime = runtime
        self._code_index = code_index
        self._repository = repository

    def execute(
        self, args: EditActionArguments, file_context: FileContext
    ) -> Observation:
        # Claude tends to add /repo in the start of the file path.
        # TODO: Maybe we should add /repo as default on all paths?
        if args.path.startswith("/repo"):
            args.path = args.path[5:]

        # Remove leading `/` if present
        # TODO: Solve by adding /repo to all paths?
        if args.path.startswith("/"):
            args.path = args.path[1:]

        path = Path(args.path)

        validation_error = self.validate_path(file_context, args.command, path)
        if validation_error:
            return Observation(
                message=validation_error,
                properties={"fail_reason": "invalid_path"},
                expect_correction=True,
            )

        if args.command == "view":
            return self._view(file_context, path, args)
        elif args.command == "create":
            if not args.file_text:
                raise RetryException(
                    message="Parameter `file_text` is required for command: create",
                    action_args=args,
                )
            observation = self._create(file_context, path, args.file_text)
        elif args.command == "str_replace":
            if not args.old_str:
                raise RetryException(
                    message="Parameter `old_str` is required for command: str_replace",
                    action_args=args,
                )
            str_replace = StringReplace(
                runtime=self._runtime,
                code_index=self._code_index,
                repository=self._repository,
            )
            return str_replace.execute(
                StringReplaceArgs(
                    path=args.path,
                    old_str=args.old_str,
                    new_str=args.new_str or "",
                    scratch_pad=args.scratch_pad,
                ),
                file_context,
            )
        elif args.command == "insert":
            if args.insert_line is None:
                raise RetryException(
                    message="Parameter `insert_line` is required for command: insert",
                    action_args=args,
                )
            if args.new_str is None:
                raise RetryException(
                    message="Parameter `new_str` is required for command: insert",
                    action_args=args,
                )
            observation = self._insert(
                file_context, path, args.insert_line, args.new_str
            )
        else:
            raise RetryException(
                message=f"Unknown command: {args.command}",
                action_args=args,
            )

        if not observation.properties or not observation.properties.get("diff"):
            return observation

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

    def validate_path(
        self, file_context: FileContext, command: str, path: Path
    ) -> str | None:
        """
        Check that the path/command combination is valid.
        """
        # TODO: Check if its an absolute path?
        # if not path.is_absolute():
        #    suggested_path = Path("") / path
        #    return (
        #        f"The path {path} is not an absolute path, it should start with `/`. Maybe you meant {suggested_path}?"
        #    )

        # Check if path exists
        if not file_context.file_exists(str(path)) and command != "create":
            return f"The path {path} does not exist. Please provide a valid path."

        if file_context.file_exists(str(path)) and command == "create":
            return f"File already exists at: {path}. Cannot overwrite files using command `create`."

        # Check if the path points to a directory
        if file_context._repo.is_directory(str(path)):
            if command != "view":
                return f"The path {path} is a directory and only the `view` command can be used on directories"

        return None

    def _view(
        self, file_context: FileContext, path: Path, args: EditActionArguments
    ) -> Observation:
        context_file = file_context.get_context_file(str(path))
        if not context_file:
            return Observation(
                message=f"Could not get context for file: {path}",
                properties={"fail_reason": "context_error"},
            )

        file_content = context_file.content
        init_line = 1
        file_lines = file_content.split("\n")
        n_lines = len(file_lines)

        view_range = args.view_range
        if view_range:
            if len(view_range) != 2:
                raise RetryException(
                    message="Invalid view_range. It should be a list of two integers.",
                    action_args=args,
                )

            init_line, final_line = view_range

            if init_line < 1 or init_line > n_lines:
                raise RetryException(
                    message=f"Invalid view_range start line: {init_line}. Should be between 1 and {n_lines}",
                    action_args=args,
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])
        else:
            final_line = n_lines

        tokens = count_tokens(file_content)
        if tokens > self.max_tokens_to_view:
            view_context = FileContext(self._repository)
            view_context.add_file(str(path), show_all_spans=True)

            file_content = view_context.create_prompt(
                show_span_ids=True,
                show_outcommented_code=True,
                only_signatures=True,
                show_line_numbers=True,
            )

            raise RetryException(
                message=f"File {path} is too large ({tokens} tokens) to view in its entirety. Maximum allowed is {self.max_tokens_to_view} tokens. "
                f"Please specify a line range using view_range or spans with ViewCode to view specific parts of the file.\n"
                f"Here's a structure of the file {file_content}",
                action_args=args,
            )

        properties = {}
        added_spans = file_context.add_line_span_to_context(
            str(path), init_line, final_line
        )
        if not added_spans:
            properties["flag"] = "no_new_spans"

        message = self._make_output(file_content, f"{path}", init_line)

        return Observation(message=message, properties=properties)

    def _create(
        self, file_context: FileContext, path: Path, file_text: str
    ) -> Observation:
        if path.exists():
            return Observation(
                message=f"File already exists at: {path}",
                properties={"fail_reason": "file_exists"},
            )

        context_file = file_context.add_file(str(path))
        context_file.apply_changes(file_text)

        diff = do_diff(str(path), "", file_text)

        return Observation(
            message=f"File created successfully at: {path}",
            properties={"diff": diff, "success": True},
        )

    def _str_replace(
        self, file_context: FileContext, path: Path, old_str: str, new_str: str
    ) -> Observation:
        SNIPPET_LINES = 4

        context_file = file_context.get_context_file(str(path))
        if not context_file:
            return Observation(
                message=f"Could not get context for file: {path}",
                properties={"fail_reason": "context_error"},
            )

        file_content = context_file.content.expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs()

        if old_str == new_str:
            return Observation(
                message="The replacement string is the same as the original string. No changes were made.",
                properties={"fail_reason": "no_changes"},
            )

        occurrences = file_content.count(old_str)
        if occurrences == 0:
            new_str_occurrences = file_content.count(new_str)
            if new_str_occurrences > 0:
                return Observation(
                    message=f"New string '{new_str}' already exists in {path}. No changes were made.",
                    properties={"fail_reason": "string_already_exists"},
                )

            return Observation(
                message=f"String '{old_str}' not found in {path}",
                properties={
                    "fail_reason": "string_not_found",
                    "file_content": file_content,
                },
                expect_correction=True,
            )
        elif occurrences > 1:
            file_str = file_content
            lines = []
            pos = 0
            while True:
                pos = file_str.find(old_str, pos)
                if pos == -1:
                    break
                # Count newlines before this occurrence to get line number
                start_line = file_str.count("\n", 0, pos) + 1
                lines.append(start_line)
                pos += 1

            return Observation(
                message=f"Multiple occurrences of string found starting at lines {lines}",
                properties={
                    "fail_reason": "multiple_occurrences",
                    "file_content": file_content,
                },
                expect_correction=True,
            )

        properties = {
            "flags": [],
        }

        # Find the line numbers where new_str appears
        lines = file_content.split("\n")
        file_str = "\n".join(lines)  # Ensure consistent line endings

        start_pos = file_str.find(new_str)
        if start_pos != -1:
            # Count newlines before the match to get starting line number
            start_line = file_str.count("\n", 0, start_pos) + 1
            # Count newlines in old_str to get ending line number
            end_line = start_line + old_str.count("\n")

            properties["start_line"] = start_line
            properties["end_line"] = end_line

            # Check if these lines are in context
            if not context_file.lines_is_in_context(start_line, end_line):
                properties["flags"].append("lines_not_in_context")
                context_file.add_line_span(start_line, end_line)

        new_file_content = file_content.replace(old_str, new_str)

        diff = do_diff(str(path), file_content, new_file_content)
        if not diff:
            properties["fail_reason"] = "no_changes"
            return Observation(
                message=f"No changes made to the file {path}. Was the replacement string the same as the original string?",
                properties=properties,
            )

        context_file.apply_changes(new_file_content)

        # Create a snippet of the edited section
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        properties["success"] = True
        properties["diff"] = diff

        return Observation(
            message=success_msg,
            properties=properties,
        )

    def _insert(
        self, file_context: FileContext, path: Path, insert_line: int, new_str: str
    ) -> Observation:
        context_file = file_context.get_context_file(str(path))
        if not context_file:
            return Observation(
                message=f"Could not get context for file: {path}",
                properties={"fail_reason": "context_error"},
            )

        # Validate file exists and is not a directory
        if not file_context.file_exists(str(path)):
            return Observation(
                message=f"File {path} not found.",
                properties={"fail_reason": "file_not_found"},
            )
        file_text = context_file.content.expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > len(file_text_lines):
            return Observation(
                message=f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}",
                properties={"fail_reason": "invalid_line_number"},
                expect_correction=True,
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        diff = do_diff(str(path), file_text, new_file_text)

        context_file.apply_changes(new_file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

        return Observation(
            message=success_msg,
            properties={"diff": diff, "success": True},
        )

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ):
        """Generate output for the CLI based on the content of a file."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )

    def span_id_list(self, span_ids: set[str]) -> str:
        list_str = ""
        for span_id in span_ids:
            list_str += f" * {span_id}\n"
        return list_str


TRUNCATED_MESSAGE: str = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
MAX_RESPONSE_LEN: int = 16000


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )
