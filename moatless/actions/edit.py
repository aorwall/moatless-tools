import logging
from pathlib import Path
from typing import Literal, Optional, List

from pydantic import Field, PrivateAttr, field_validator

from moatless.actions import RunTests, CreateFile, ViewCode
from moatless.actions.action import Action
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.create_file import CreateFileArgs
from moatless.actions.model import ActionArguments, Observation, RetryException
from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.actions.view_code import ViewCodeArgs, CodeSpan
from moatless.completion import CompletionModel
from moatless.completion.model import ToolCall
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.file import do_diff
from moatless.repository.repository import Repository
from moatless.workspace import Workspace

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

    @field_validator("file_text")
    @classmethod
    def validate_file_text(cls, v, info):
        if info.data.get("command") == "create" and not v:
            raise ValueError("Parameter `file_text` is required for command: create")
        return v

    @field_validator("old_str")
    @classmethod
    def validate_old_str(cls, v, info):
        if info.data.get("command") == "str_replace" and not v:
            raise ValueError("Parameter `old_str` is required for command: str_replace")
        return v

    @field_validator("new_str")
    @classmethod
    def validate_new_str(cls, v, info):
        if info.data.get("command") == "str_replace" and v is None:
            raise ValueError(
                "Parameter `new_str` cannot be null for command: str_replace. Return an empty string if your intention was to remove old_str."
            )
        if info.data.get("command") == "insert" and v is None:
            raise ValueError("Parameter `new_str` is required for command: insert")
        return v

    @field_validator("insert_line")
    @classmethod
    def validate_insert_line(cls, v, info):
        if info.data.get("command") == "insert" and v is None:
            raise ValueError("Parameter `insert_line` is required for command: insert")
        return v

    @field_validator("view_range")
    @classmethod
    def validate_view_range(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("Invalid view_range. It should be a list of two integers.")
        return v

    @field_validator("command")
    @classmethod
    def validate_command(cls, v):
        valid_commands = {"view", "create", "str_replace", "insert", "undo_edit"}
        if v not in valid_commands:
            raise ValueError(f"Unknown command: {v}")
        return v

    class Config:
        title = "str_replace_editor"

    def to_tool_call(self) -> ToolCall:
        return ToolCall(
            name=self.name, type="text_editor_20241022", input=self.model_dump()
        )


class ClaudeEditTool(Action, CodeModificationMixin):
    """
    An filesystem editor tool that allows the agent to view, create, and edit files.
    The tool parameters are defined by Anthropic and are not editable.
    """

    args_schema = EditActionArguments

    max_tokens_to_view: int = Field(
        2000, description="Max tokens to view in one command"
    )

    _str_replace: StringReplace = PrivateAttr()
    _create_file: CreateFile = PrivateAttr()
    _repository: Repository | None = PrivateAttr(None)

    def __init__(
        self,
        code_index: CodeIndex | None = None,
        repository: Repository | None = None,
        completion_model: CompletionModel | None = None,
        **data,
    ):
        super().__init__(**data)
        object.__setattr__(self, "_code_index", code_index)
        object.__setattr__(self, "_repository", repository)
        object.__setattr__(self, "_completion_model", completion_model)

        self._str_replace = StringReplace(
            runtime=self._runtime,
            code_index=self._code_index,
            repository=self._repository,
        )
        self._create_file = CreateFile(
            runtime=self._runtime,
            code_index=self._code_index,
            repository=self._repository,
        )
        self._view_code = ViewCode(
            repository=self._repository, completion_model=completion_model
        )

    def execute(
        self,
        args: EditActionArguments,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
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
            return self._create_file.execute(
                CreateFileArgs(
                    path=args.path,
                    file_text=args.file_text,
                    thoughts=args.thoughts,
                ),
                file_context,
            )
        elif args.command == "str_replace":
            return self._str_replace.execute(
                StringReplaceArgs(
                    path=args.path,
                    old_str=args.old_str,
                    new_str=args.new_str or "",
                    thoughts=args.thoughts,
                ),
                file_context,
            )
        elif args.command == "insert":
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
            fail_on_not_found=False,
            repository=self._repository,
            runtime=self._runtime,
            code_index=self._code_index,
        )
        test_observation = run_tests.execute(
            RunTestsArgs(
                thoughts=args.thoughts,
                test_files=[args.path],
            ),
            file_context,
        )

        if test_observation:
            observation.message += f"\n\n{test_observation}"

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
        codespan = CodeSpan(file_path=str(path))

        view_range = args.view_range
        if view_range:
            codespan.start_line, codespan.end_line = view_range

        view_code_args = ViewCodeArgs(thoughts=args.thoughts, files=[codespan])
        return self._view_code.execute(view_code_args, file_context=file_context)

    def _create(
        self, file_context: FileContext, path: Path, file_text: str
    ) -> Observation:
        if file_context.file_exists(str(path)):
            return Observation(
                message=f"File already exists at: {path}",
                properties={"fail_reason": "file_exists"},
            )

        context_file = file_context.add_file(str(path))
        context_file.apply_changes(file_text)

        diff = do_diff(str(path), "", file_text)

        return Observation(
            message=f"File created successfully at: {path}",
            properties={"diff": diff},
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
            properties={"diff": diff},
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
