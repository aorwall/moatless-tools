import logging
import re
from typing import List

from pydantic import Field, model_validator

from moatless.actions.action import Action
from moatless.actions.code_action_value_mixin import CodeActionValueMixin
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.model import (
    ActionArguments,
    Observation,
    FewShotExample,
)
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.repository.file import do_diff
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)

SNIPPET_LINES = 4


class StringReplaceArgs(ActionArguments):
    """
    Applies a change to a file by replacing text with exact string matching.

    Notes:
    * The old_str parameter must match EXACTLY one or more consecutive lines from the original file
    * Whitespace and indentation must match exactly:
      - Use spaces for indentation, not tabs
      - Match the exact number of spaces from the original code
      - Do not modify the indentation pattern
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
            # Pattern to match: digits followed by exactly one tab, then any spaces
            line_number_pattern = r"^\s*(\d+)\t*"

            # First verify all lines start with a number and tab
            if not all(re.match(line_number_pattern, line) for line in lines):
                return text

            # Remove line numbers and tab while preserving remaining indentation
            cleaned_lines = []
            for line in lines:
                # Remove numbers and tab but keep remaining spaces
                cleaned_line = re.sub(line_number_pattern, "", line)
                cleaned_lines.append(cleaned_line)

            return "\n".join(cleaned_lines)

        # Remove trailing newlines and line numbers
        self.old_str = remove_line_numbers(self.old_str.rstrip("\n"))
        self.new_str = remove_line_numbers(self.new_str.rstrip("\n"))

        return self

    class Config:
        title = "StringReplace"

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<old_str>
{self.old_str}
</old_str>
<new_str>
{self.new_str}
</new_str>"""

    @classmethod
    def format_schema_for_llm(cls) -> str:
        return cls.format_xml_schema(
            {
                "path": "file/path.py",
                "old_str": "\nexact code to replace\n",
                "new_str": "\nreplacement code\n",
            }
        )

    def _short_str(self, str: str):
        str_split = str.split("\n")
        return str_split[0][:20]

    def short_summary(self) -> str:
        param_str = f'path="{self.path}"'
        return f"{self.name}({param_str})"


class StringReplace(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to replace strings in a file.
    """

    args_schema = StringReplaceArgs

    auto_correct_indentation: bool = Field(
        True,
        description="When True, automatically corrects indentation if all lines have the same indentation difference",
    )

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
        self,
        args: StringReplaceArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        path_str = self.normalize_path(args.path)
        path, error = self.validate_file_access(path_str, file_context)
        if error:
            return error

        context_file = file_context.get_context_file(str(path))
        file_content = context_file.content.expandtabs()
        old_str = args.old_str.expandtabs()
        new_str = args.new_str.expandtabs()

        if old_str == new_str:
            return Observation(
                message=f"The old_str and new_str are the same. No changes were made.",
                properties={"fail_reason": "no_changes"},
            )

        # Use find_exact_matches instead of inline code
        exact_matches = find_exact_matches(old_str, file_content)

        logger.info(f"Found {len(exact_matches)} exact matches")

        # Filter matches to only those in context
        in_context_exact_matches = [
            match
            for match in exact_matches
            if context_file.lines_is_in_context(match["start_line"], match["end_line"])
        ]

        # Set flag if we have exactly one in-context match but more total matches
        properties = {}
        if len(in_context_exact_matches) == 1 and len(exact_matches) > len(
            in_context_exact_matches
        ):
            properties["flags"] = ["targeted_in_context_replacement"]
            exact_matches = in_context_exact_matches

        if len(exact_matches) == 0:
            potential_matches = find_match_when_ignoring_indentation(
                old_str, file_content
            )

            if len(potential_matches) > 0:
                in_context_potential_matches = [
                    match
                    for match in potential_matches
                    if context_file.lines_is_in_context(
                        match["start_line"], match["end_line"]
                    )
                ]

                if len(in_context_potential_matches) == 1:
                    potential_matches = in_context_potential_matches

            # Handle auto-correction if enabled
            if (
                self.auto_correct_indentation
                and len(potential_matches) == 1
                and potential_matches[0].get("can_auto_correct")
            ):
                exact_matches = potential_matches
                indent_diff = potential_matches[0]["uniform_indent_diff"]

                # Adjust indentation in new_str
                new_str_lines = new_str.splitlines()
                adjusted_lines = []
                for line in new_str_lines:
                    if indent_diff < 0:
                        # Remove indentation if it exists
                        current_indent = len(line) - len(line.lstrip())
                        spaces_to_remove = min(current_indent, abs(indent_diff))
                        adjusted_lines.append(line[spaces_to_remove:])
                    else:
                        # Add indentation
                        adjusted_lines.append(" " * indent_diff + line)
                new_str = "\n".join(adjusted_lines)

                # Use the matched content as old_str
                old_str = exact_matches[0]["content"]

                # Fix argument for correct message history
                args.old_str = old_str
                args.new_str = new_str

                properties["flags"] = ["auto_corrected_indentation"]
            else:
                if not potential_matches:
                    potential_matches = find_potential_matches(old_str, file_content)

                # Filter potential matches to only those in context
                in_context_potential_matches = [
                    match
                    for match in potential_matches
                    if context_file.lines_is_in_context(
                        match["start_line"], match["end_line"]
                    )
                ]

                if (
                    len(potential_matches) > 0
                    and len(in_context_potential_matches) == 1
                ):
                    potential_matches = in_context_potential_matches

                if len(potential_matches) == 1:
                    match = potential_matches[0]
                    match_content = match["content"]
                    differences = match.get("differences", [])
                    differences_msg = "\n".join(f"- {diff}" for diff in differences)

                    message = (
                        f"No changes were made. The provided old_str was not found, but a similar code block was found. "
                        f"To replace this code, the old_str must match exactly:\n\n```\n{match_content}\n```\n\n"
                    )

                    if match["diff_reason"] == "indentation_differs":
                        differences_msg = "\n".join(
                            f"- {diff}" for diff in match.get("differences", [])
                        )
                        message += (
                            f"The content matches but the indentation is different.\n"
                            f"{differences_msg}\n\n"
                            f"Please update old_str to match the exact indentation shown above."
                        )
                    elif match["diff_reason"] == "line_breaks_differ":
                        message += f"Differences found:\n{differences_msg}\n\n"
                        message += "Please update old_str to match the exact line breaks and other special characters as shown above."

                    return Observation(
                        message=message,
                        properties={"flags": [match["diff_reason"]]},
                        expect_correction=True,
                    )
                elif len(potential_matches) > 1:
                    matches_info = "\n".join(
                        f"- Lines {m['start_line']}-{m['end_line']} ({m['diff_reason']}):\n```\n{m['content']}\n```"
                        for m in potential_matches
                    )
                    return Observation(
                        message=f"Multiple potential matches found with different formatting:\n{matches_info}\nTry including more surrounding context to create a unique match.",
                        properties={"flags": ["multiple_potential_occurrences"]},
                        expect_correction=True,
                    )

                # If no matches found at all
                new_str_occurrences = file_content.count(new_str)
                if new_str_occurrences > 0:
                    return Observation(
                        message=f"New string '{new_str}' already exists in {path}. No changes were made.",
                        properties={"fail_reason": "string_already_exists"},
                    )

                return Observation(
                    message=f"String '{old_str}' not found in {path}.\n\nRemember to write out the exact string you want to replace with the same indentation and no placeholders.",
                    properties={"fail_reason": "string_not_found"},
                    expect_correction=True,
                )
        elif len(exact_matches) > 1:
            matches_info = "\n".join(
                f"- Lines {m['start_line']}-{m['end_line']}:\n```\n{m['content']}\n```"
                for m in exact_matches
            )
            return Observation(
                message=f"Multiple occurrences of string found:\n{matches_info}\nTry including more surrounding lines to create a unique match.",
                properties={"flags": ["multiple_occurrences"]},
                expect_correction=True,
            )

        start_line = exact_matches[0]["start_line"]
        end_line = exact_matches[0]["end_line"]
        if not context_file.lines_is_in_context(start_line, end_line):
            properties["flags"] = ["lines_not_in_context"]
            logger.warning(
                f"Lines {start_line}-{end_line} are not in context for {path}"
            )

        # If we have exactly one in-context match and there are more total matches,
        # only replace the in-context occurrence to preserve other matches
        if "targeted_in_context_replacement" in properties.get("flags", []):
            match = in_context_exact_matches[0]
            # Count newlines up to the start of the match to find its position
            start_pos = 0
            for _ in range(match["start_line"] - 1):
                start_pos = file_content.find("\n", start_pos) + 1
            # Find the exact position of this occurrence
            start_pos = file_content.find(old_str, start_pos)
            # Replace only this occurrence
            logger.info(f"Do targeted replacement on line {start_pos}")

            new_file_content = (
                file_content[:start_pos]
                + new_str
                + file_content[start_pos + len(old_str) :]
            )
        else:
            new_file_content = file_content.replace(args.old_str, args.new_str)

        # Generate diff and apply changes
        diff = do_diff(str(path), file_content, new_file_content)
        context_file.apply_changes(new_file_content)

        # Create a snippet of the edited section
        snippet_start_line = max(0, start_line - SNIPPET_LINES - 1)
        end_line = start_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[snippet_start_line:end_line])

        snippet_with_lines = self.format_snippet_with_lines(snippet, start_line + 1)

        message = (
            f"The file {path} has been edited. Here's the result of running `cat -n` "
            f"on a snippet of {path}:\n{snippet_with_lines}\n"
            "Review the changes and make sure they are as expected. Edit the file again if necessary."
        )

        summary = f"The file {path} has been edited. Review the changes and make sure they are as expected. Edit the file again if necessary."

        properties["diff"] = diff

        observation = Observation(
            message=message,
            summary=summary,
            properties=properties,
        )

        test_summary = self.run_tests(
            file_path=str(path),
            file_context=file_context,
        )

        if test_summary:
            observation.message += f"\n\n{test_summary}"

        return observation

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Update the error message in the validate_user method",
                action=StringReplaceArgs(
                    thoughts="Improving the error message to be more descriptive",
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
                user_input="Update the user validation logic",
                action=StringReplaceArgs(
                    thoughts="Adding email validation and password strength check",
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
            FewShotExample.create(
                user_input="Add a new helper function to validate passwords",
                action=StringReplaceArgs(
                    thoughts="Adding a new function to check password complexity",
                    path="auth/validator.py",
                    old_str="""def validate_user(username, password):
    if len(username) < 3 or not is_valid_email(username):
        return False
    if len(password) < 12 or not has_special_chars(password):
        return False
    if not has_numbers(password):
        return False
    return True""",
                    new_str="""def is_password_complex(password: str) -> bool:
    has_uppercase = any(c.isupper() for c in password)
    has_lowercase = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)
    return all([has_uppercase, has_lowercase, has_digit, has_special])

def validate_user(username, password):
    if len(username) < 3 or not is_valid_email(username):
        return False
    if len(password) < 12 or not has_special_chars(password):
        return False
    if not has_numbers(password) or not is_password_complex(password):
        return False
    return True""",
                ),
            ),
            FewShotExample.create(
                user_input="Remove the deprecated logging configuration",
                action=StringReplaceArgs(
                    thoughts="Removing old logging setup that's no longer needed",
                    path="utils/logger.py",
                    old_str="""    # Legacy logging configuration
    if legacy_mode:
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(levelname)s] %(message)s",
            filename="legacy.log"
        )
        logger.warning("Using deprecated legacy logging configuration")
""",
                    new_str="",
                ),
            ),
            FewShotExample.create(
                user_input="Add a new test case for password validation with special characters",
                action=StringReplaceArgs(
                    thoughts="Adding a new test method for special character validation.",
                    path="tests/test_validator.py",
                    old_str="""def test_validate_user():
    # Test basic validation
    assert validate_user("bob@example.com", "password123") is True
    assert validate_user("alice@example.com", "short") is False
    
    # Test email format
    assert validate_user("invalid-email", "password123") is False

""",
                    new_str="""def test_validate_user():
    # Test basic validation
    assert validate_user("bob@example.com", "password123") is True
    assert validate_user("alice@example.com", "short") is False
    
    # Test email format
    assert validate_user("invalid-email", "password123") is False

def test_validate_password_special_chars():
    # Test passwords with special characters
    assert validate_user("bob@example.com", "Pass!@#123") is True
    assert validate_user("alice@example.com", "NoSpecialChars123") is False
    assert validate_user("carol@example.com", "!@#$%^&*(") is False  # No alphanumeric chars""",
                ),
            ),
        ]


def normalize_indentation(s):
    return "\n".join(line.strip() for line in s.splitlines())


def normalize_for_comparison(s):
    """
    Normalize a string for fuzzy comparison by removing most non-alphanumeric characters.
    Preserves backslashes, parentheses, curly braces, and % operator for string formatting.
    """
    # First, normalize line endings and remove empty lines
    s = "\n".join(line.strip() for line in s.splitlines() if line.strip())

    # Store removed characters for difference checking
    normalize_chars = r'["\'\s_=+,;]'
    removed_chars = set(re.findall(normalize_chars, s))

    # Normalize string by:
    # 1. Removing all whitespace and specified chars
    # 2. Converting to lowercase to make comparison case-insensitive
    # 3. Preserve backslashes, parentheses, curly braces and % operator
    normalized = s.lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r'["\'\s_=+,;]', "", normalized)

    return normalized, removed_chars


def find_match_when_ignoring_indentation(old_str, content):
    old_str_lines = old_str.splitlines()
    content_lines = content.splitlines()
    old_str_no_indent = normalize_indentation(old_str)

    window_size = len(old_str_lines)
    indentation_matches = []

    for start_idx in range(len(content_lines) - window_size + 1):
        window = "\n".join(content_lines[start_idx : start_idx + window_size])
        window_no_indent = normalize_indentation(window)

        if window_no_indent == old_str_no_indent:
            # Calculate indentation differences for each line
            differences = []
            indentation_diffs = set()

            for i, (old_line, window_line) in enumerate(
                zip(old_str_lines, content_lines[start_idx : start_idx + window_size])
            ):
                old_indent = len(old_line) - len(old_line.lstrip())
                window_indent = len(window_line) - len(window_line.lstrip())
                indent_diff = window_indent - old_indent
                indentation_diffs.add(indent_diff)

                if old_indent != window_indent:
                    differences.append(
                        f"Line {i+1}: expected {old_indent} spaces, found {window_indent} spaces"
                    )

            match_data = {
                "start_line": start_idx + 1,
                "end_line": start_idx + window_size,
                "content": window,
                "diff_reason": "indentation_differs",
                "differences": differences,
            }

            # If all lines have the same indentation difference, include it
            if len(indentation_diffs) == 1:
                match_data["uniform_indent_diff"] = indentation_diffs.pop()
                match_data["can_auto_correct"] = True

            indentation_matches.append(match_data)

    return indentation_matches


def find_potential_matches(old_str, new_content):
    matches = []
    content_lines = new_content.splitlines()
    if not content_lines:
        return matches

    # Pre-compute normalized versions of old_str
    old_str_normalized, old_str_chars = normalize_for_comparison(old_str)

    # Track processed lines to avoid overlapping matches
    processed_lines = set()

    start_idx = 0
    while start_idx < len(content_lines):
        if start_idx in processed_lines:
            start_idx += 1
            continue

        # Check if this line could start our match
        line_normalized, _ = normalize_for_comparison(content_lines[start_idx])
        if not line_normalized.strip() or not old_str_normalized.startswith(
            line_normalized
        ):
            start_idx += 1
            continue

        # Try increasing window sizes until we find a match
        for window_size in range(1, min(50, len(content_lines) - start_idx + 1)):
            # Skip if any line in the window was already processed
            if any(
                i in processed_lines for i in range(start_idx, start_idx + window_size)
            ):
                continue

            window = "\n".join(content_lines[start_idx : start_idx + window_size])
            window_normalized, window_chars = normalize_for_comparison(window)

            if old_str_normalized in window_normalized:
                # Mark all lines in this window as processed
                for i in range(start_idx, start_idx + window_size):
                    processed_lines.add(i)

                differences = []
                if window.count("\n") != old_str.count("\n"):
                    differences.append(
                        f"Line break count differs: found {window.count('\n') + 1} lines, "
                        f"expected {old_str.count('\n') + 1} lines"
                    )

                # Check for character differences
                added = window_chars - old_str_chars
                removed = old_str_chars - window_chars
                if added:
                    differences.append(
                        f"Additional characters found: {', '.join(sorted(added))}"
                    )
                if removed:
                    differences.append(
                        f"Missing characters: {', '.join(sorted(removed))}"
                    )

                matches.append(
                    {
                        "start_line": start_idx + 1,
                        "end_line": start_idx + window_size,
                        "content": window,
                        "diff_reason": "line_breaks_differ",
                        "differences": differences,
                    }
                )
                # Jump to next unprocessed line
                start_idx = start_idx + window_size
                break

        else:
            # No match found with any window size, move to next line
            start_idx += 1

    # If no matches found, try finding indentation-only differences
    if not matches:
        for i, line in enumerate(content_lines):
            if i in processed_lines:
                continue

            if normalize_indentation(line) == normalize_indentation(old_str):
                processed_lines.add(i)
                matches.append(
                    {
                        "start_line": i + 1,
                        "end_line": i + 1,
                        "content": line,
                        "diff_reason": "indentation_differs",
                    }
                )

    return matches


def find_exact_matches(old_str: str, file_content: str) -> list[dict]:
    """Find exact matches of old_str in file_content, preserving line numbers."""
    matches = []
    start_pos = 0

    while True:
        # Find the start position of the match
        start_pos = file_content.find(old_str, start_pos)
        if start_pos == -1:
            break

        # Count newlines before the match to get line number
        start_line = file_content.count("\n", 0, start_pos) + 1
        end_line = start_line + old_str.count("\n")

        matches.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "content": old_str,
                "diff_reason": "exact_match",
            }
        )

        # Move start_pos forward to find subsequent matches
        start_pos += len(old_str)

    return matches
