import logging
import re
from typing import List

from pydantic import Field, model_validator

from moatless.actions.action import Action
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.model import (
    ActionArguments,
    Observation,
    FewShotExample,
    RetryException,
)
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.repository.file import do_diff
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment

logger = logging.getLogger(__name__)

SNIPPET_LINES = 4


class StringReplaceArgs(ActionArguments):
    """
    Replace text in a file with exact string matching.

    Notes:
    * The old_str parameter must match EXACTLY one or more consecutive lines from the original file
    * Whitespace and indentation must match exactly
    * The old_str must be unique within the file - include enough surrounding context to ensure uniqueness
    * The new_str parameter contains the replacement text that will replace old_str
    * No changes will be made if old_str appears multiple times or cannot be found
    * Do not include line numbers in old_str or new_str - provide only the actual code content
    """

    path: str = Field(..., description="Path to the file to edit")
    old_str: str = Field(
        ...,
        description="Exact string from the file to replace - must match exactly, be unique, include proper indentation, and contain no line numbers",
    )
    new_str: str = Field(
        ...,
        description="New string to replace the old_str with - must use proper indentation and contain no line numbers",
    )

    @model_validator(mode="after")
    def validate_args(self) -> "StringReplaceArgs":
        if not self.path.strip():
            raise ValueError("path cannot be empty")
        if not self.old_str.strip():
            raise ValueError("old_str cannot be empty")

        def remove_line_numbers(text: str) -> str:
            lines = text.split("\n")
            # Pattern to match line numbers at start of line
            line_number_pattern = r"^\s*\d+"

            # Remove line numbers if found
            cleaned_lines = [re.sub(line_number_pattern, "", line) for line in lines]
            return "\n".join(cleaned_lines)

        self.old_str = remove_line_numbers(self.old_str)
        self.new_str = remove_line_numbers(self.new_str)

        return self

    class Config:
        title = "StringReplace"


class StringReplace(Action, CodeModificationMixin):
    """
    Action to replace strings in a file.
    """

    args_schema = StringReplaceArgs

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

    def execute(
        self, args: StringReplaceArgs, file_context: FileContext
    ) -> Observation:
        path_str = self.normalize_path(args.path)
        path, error = self.validate_file_access(path_str, file_context)
        if error:
            return error

        context_file = file_context.get_context_file(str(path))
        file_content = context_file.content.expandtabs()
        logger.info(f"Editing file {path}\n{file_content}")
        old_str = args.old_str.expandtabs()
        new_str = args.new_str.expandtabs()

        if old_str == new_str:
            return Observation(
                message=f"The old_str and new_str are the same. No changes were made.",
                properties={"fail_reason": "no_changes"},
            )

        # Use find_exact_matches instead of inline code
        exact_matches = find_exact_matches(old_str, file_content)

        if len(exact_matches) == 0:
            potential_matches = find_potential_matches(old_str, file_content)

            if len(potential_matches) == 1:
                match = potential_matches[0]
                match_content = match["content"]

                message = (
                    f"No changes were made. The provided old_str was not found, but a similar code block was found. "
                    f"To replace this code, the old_str must match exactly:\n\n```\n{match_content}\n```\n\n"
                )

                if match["diff_reason"] == "indentation_differs":
                    first_line_match = match_content.splitlines()[0]
                    first_line_old = old_str.splitlines()[0]
                    match_indent = len(first_line_match) - len(
                        first_line_match.lstrip()
                    )
                    provided_indent = len(first_line_old) - len(first_line_old.lstrip())

                    message += (
                        f"The content matches but the indentation is different. "
                        f"The actual code has {match_indent} spaces but your old_str has {provided_indent} spaces. "
                        f"Please update old_str to match the exact indentation shown above."
                    )
                elif match["diff_reason"] == "line_breaks_differs":
                    message += "The content matches but the line breaks are different. Please update old_str to match the exact line breaks shown above."

                raise RetryException(message, args)
            elif len(potential_matches) > 1:
                matches_info = "\n".join(
                    f"- Lines {m['start_line']}-{m['end_line']} ({m['diff_reason']})"
                    for m in potential_matches
                )
                raise RetryException(
                    message=f"Multiple potential matches found with different formatting:\n{matches_info}\nTry including more surrounding context to create a unique match.",
                    action_args=args,
                )

            # If no matches found at all
            new_str_occurrences = file_content.count(new_str)
            if new_str_occurrences > 0:
                return Observation(
                    message=f"New string '{new_str}' already exists in {path}. No changes were made.",
                    properties={"fail_reason": "string_already_exists"},
                )

            return Observation(
                message=f"String '{old_str}' not found in {path}",
                properties={"fail_reason": "string_not_found"},
                expect_correction=True,
            )
        elif len(exact_matches) > 1:
            matches_info = "\n".join(f"- Line {m['start_line']}" for m in exact_matches)
            return Observation(
                message=f"Multiple occurrences of string found:\n{matches_info}\nTry including more surrounding lines to create a unique match.",
                properties={"fail_reason": "multiple_occurrences"},
                expect_correction=True,
            )

        properties = {}

        match = exact_matches[0]
        start_line = match["start_line"] - 1  # Convert to 0-based index
        end_line = match["end_line"] - 1

        # Check if the lines to be modified are in context
        if not context_file.lines_is_in_context(start_line, end_line):
            properties["flags"] = ["lines_not_in_context"]
            logger.warning(
                f"Lines {start_line + 1}-{end_line + 1} are not in context for {path}"
            )

        new_file_content = file_content.replace(old_str, new_str)
        diff = do_diff(str(path), file_content, new_file_content)

        context_file.apply_changes(new_file_content)

        # Create a snippet of the edited section
        snippet_start_line = max(0, start_line - SNIPPET_LINES - 1)
        end_line = start_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[snippet_start_line:end_line])

        snippet_with_lines = self.format_snippet_with_lines(snippet, start_line + 1)

        success_msg = (
            f"The file {path} has been edited. Here's the result of running `cat -n` "
            f"on a snippet of {path}:\n{snippet_with_lines}\n"
            "Review the changes and make sure they are as expected. Edit the file again if necessary."
        )

        properties["diff"] = diff

        observation = Observation(
            message=success_msg,
            properties=properties,
        )

        return self.run_tests_and_update_observation(
            observation=observation,
            file_path=str(path),
            scratch_pad=args.scratch_pad,
            file_context=file_context,
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Update the error message in the validate_user method",
                action=StringReplaceArgs(
                    scratch_pad="Improving the error message to be more descriptive",
                    path="auth/validator.py",
                    old_str="""    if not user.is_active:
        raise ValueError("Invalid user")
    return user""",
                    new_str="""    if not user.is_active:
        raise ValueError(f"Invalid user: {username} does not meet the required criteria")
    return user""",
                ),
            ),
            FewShotExample.create(
                user_input="Update the logging configuration",
                action=StringReplaceArgs(
                    scratch_pad="Enhancing the logging configuration with more detailed format and file handler",
                    path="utils/logger.py",
                    old_str="""logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)""",
                    new_str="""logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)""",
                ),
            ),
            FewShotExample.create(
                user_input="Update the user validation logic",
                action=StringReplaceArgs(
                    scratch_pad="Adding email validation and password strength check",
                    path="auth/validator.py",
                    old_str="""def validate_user(username, password):
    if len(username) < 3:
        return False
    if len(password) < 8:
        return False
    return True""",
                    new_str="""def validate_user(username, password):
    if len(username) < 3 or not is_valid_email(username):
        return False
    if len(password) < 12 or not has_special_chars(password):
        return False
    if not has_numbers(password):
        return False
    return True""",
                ),
            ),
        ]


def normalize_indentation(s):
    return "\n".join(line.strip() for line in s.splitlines())


def normalize_line_breaks(s):
    # Remove all whitespace and line breaks
    return "".join(line.strip() for line in s.replace(" ", "").splitlines())


def find_potential_matches(old_str, new_content):
    matches = []
    content_lines = new_content.splitlines()
    old_str_lines = old_str.splitlines()
    window_size = len(old_str_lines)

    # Pre-compute normalized versions of old_str
    old_str_no_breaks = normalize_line_breaks(old_str)
    old_str_no_indent = normalize_indentation(old_str)

    # First pass: find indentation matches using fixed window size
    indentation_matches = []
    for start_idx in range(len(content_lines) - window_size + 1):
        window = "\n".join(content_lines[start_idx : start_idx + window_size])
        window_no_indent = normalize_indentation(window)

        if window_no_indent == old_str_no_indent:
            indentation_matches.append(
                {
                    "start_line": start_idx + 1,
                    "end_line": start_idx + window_size,
                    "content": window,
                    "diff_reason": "indentation_differs",
                }
            )

    # If we found indentation matches, return those only
    if indentation_matches:
        return indentation_matches

    # Second pass: find line break matches only if no indentation matches were found
    start_idx = 0
    while start_idx < len(content_lines):
        if not content_lines[start_idx].strip():
            start_idx += 1
            continue

        found_match = False
        for end_idx in range(start_idx + 1, min(start_idx + 5, len(content_lines) + 1)):
            window = "\n".join(content_lines[start_idx:end_idx])
            window_no_breaks = normalize_line_breaks(window)

            if window_no_breaks == old_str_no_breaks:
                matches.append(
                    {
                        "start_line": start_idx + 1,
                        "end_line": end_idx,
                        "content": window,
                        "diff_reason": "line_breaks_differ",
                    }
                )
                start_idx = end_idx  # Skip to the end of this window
                found_match = True
                break

        if not found_match:
            start_idx += 1  # Only increment by 1 if no match was found

    return matches


def find_exact_matches(old_str: str, file_content: str) -> list[dict]:
    """Find exact matches of old_str in file_content, preserving line numbers."""
    file_lines = file_content.splitlines()
    old_str_lines = old_str.splitlines()
    matches = []

    # Check each possible starting position in the file
    for i in range(len(file_lines) - len(old_str_lines) + 1):
        potential_match = "\n".join(file_lines[i : i + len(old_str_lines)])
        if potential_match == old_str:
            matches.append(
                {
                    "start_line": i + 1,
                    "end_line": i + len(old_str_lines),
                    "content": potential_match,
                    "diff_reason": "exact_match",
                }
            )

    return matches
