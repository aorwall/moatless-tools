import logging
from typing import Optional, Tuple, List, Dict

from litellm import completion
from pydantic import Field, PrivateAttr

from moatless.analytics import send_event
from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.codeblocks import CodeBlockTypeGroup, BlockPath
from moatless.codeblocks.module import Module
from moatless.codeblocks.parser.python import PythonParser
from moatless.coder.code_action import (
    CodeAgentFunction,
)
from moatless.coder.code_utils import extract_response_parts, CodePart, do_diff
from moatless.coder.prompt import CODER_SYSTEM_PROMPT, SEARCH_REPLACE_RESPONSE_FORMAT
from moatless.coder.types import FunctionResponse, Omissible
from moatless.file_context import FileContext
from moatless.session import Session
from moatless.settings import Settings

logger = logging.getLogger(__name__)


class SearchReplaceCode(CodeAgentFunction):
    """Update the code in one span. You can provide line numbers if only one part of the span should be updated."""

    instructions: str = Field(
        ..., description="Detailed instructions on how to update the code."
    )
    file_path: str = Field(description="The file path of the code to be updated.")
    start_line: int = Field(
        description="The start line of the code to be updated if just a part of the span should be updated.",
    )
    end_line: Omissible[int] = Field(
        description="The end line of the code to be updated if just a part of the span should be updated.",
        default=None
    )

    _max_retries: int = PrivateAttr(default=2)
    _max_tokens_for_span: int = PrivateAttr(default=500)

    _original_module: Module = PrivateAttr()

    _file_context: FileContext = PrivateAttr()
    _trace_id: str = PrivateAttr(default=None)
    _retries: int = PrivateAttr(default=0)
    _messages: List[Dict] = PrivateAttr(default_factory=list)

    _mock_response: Optional[str] = PrivateAttr(default=None)

    #def __init__(self, **kwargs):
    #    super().__init__(**kwargs)

    def call(self) -> FunctionResponse:
        original_file = self._file_context.get_file(self.file_path)
        if not original_file:
            logger.warning(f"File {self.file_path} not found in file context.")
            raise ValueError(f"{self.file_path} was not found in the file context.")

        self._original_module = original_file.module

        original_content = self._original_module.to_string()
        original_lines = original_content.split("\n")

        logger.info(f"Start line {self.start_line}: {original_lines[self.start_line - 1]}")

        pre_start_line_index = _get_pre_start_line_index(self.start_line, original_lines)
        post_end_line_index = _get_post_end_line_index(self.end_line, original_lines)
        lines_to_replace = original_lines[pre_start_line_index: post_end_line_index]

        file_context_content = self._file_context.create_prompt(exclude_comments=False)

        system_message = {
            "role": "system",
            "content": f"""{CODER_SYSTEM_PROMPT}
            
{SEARCH_REPLACE_RESPONSE_FORMAT}

# File context:
{file_context_content}
""",
        }

        user_message = {
            "role": "user",
            "content": self.instructions,
        }

        self._messages = [system_message, user_message]

        search_code = "\n".join(
            lines_to_replace
        )

        # TODO: Add a first line to the replace code?
        # replace_code = "\n".join(
        #    lines_to_replace[: 1]
        # )

        prepared_response = f"""{self.file_path}
<search>
{search_code}
</search>
<replace>
"""

        self._messages.append({
            "role": "assistant",
            "content": prepared_response
        })

        llm_response = completion(
            model=Settings.coder.coding_model,
            temperature=0.0,
            max_tokens=1500,
            messages=self._messages,
            stop=["</replace>"],
            metadata={
                "generation_name": "coder-write-code",
                "trace_name": "coder",
                "trace_id": self._trace_id,
                "session_id": Session.session_id,
                "tags": Session.tags,
            },
            mock_response=self._mock_response,
        )
        choice = llm_response.choices[0]
        self._messages.append(choice.message.dict())

        replacement_code = choice.message.content
        replacement_lines = replacement_code.split("\n")
        # Strip empty lines from the start and end
        while replacement_lines and replacement_lines[0].strip() == '':
            replacement_lines.pop(0)

        while replacement_lines and replacement_lines[-1].strip() == '':
            replacement_lines.pop()

        replacement_lines = remove_duplicate_lines(replacement_lines, original_lines[post_end_line_index:])

        updated_lines = original_lines[: pre_start_line_index] + replacement_lines + original_lines[post_end_line_index:]

        updated_content = "\n".join(updated_lines)
        diff = do_diff(self.file_path, original_content, updated_content)

        if diff:
            # TODO: Move this to let the agent decide if content should be saved
            self._file_context.save_file(self.file_path, updated_content)
            logger.info(f"Code updated and saved successfully.")
        else:
            logger.warning("No changes detected.")

        diff_lines = diff.split("\n")
        added_lines = [
            line
            for line in diff_lines
            if line.startswith("+") and not line.startswith("+++")
        ]
        removed_lines = [
            line
            for line in diff_lines
            if line.startswith("-") and not line.startswith("---")
        ]

        send_event(
            event="updated_code",
            properties={
                "added_lines": len(added_lines),
                "removed_lines": len(removed_lines),
                "file": self.file_path
            },
        )

        return FunctionResponse(message=diff)


def _get_pre_start_line_index(start_line: int, content_lines: List[str]) -> int:
    if start_line < 1 or start_line > len(content_lines):
        raise IndexError("start_line is out of range.")

    start_line_index = start_line - 1
    start_search_index = max(0, start_line_index - 1)
    end_search_index = max(0, start_line_index - 3)  # Search up to 3 lines back

    non_empty_indices = []

    for idx in range(start_search_index, end_search_index - 1, -1):
        if content_lines[idx].strip() != "":
            non_empty_indices.append(idx)

    # Check if any non-empty line was found within the search range
    if non_empty_indices:
        return non_empty_indices[-1]

    # If no non-empty lines were found, check the start_line itself
    if content_lines[start_line_index].strip() != "":
        return start_line_index

    # If the start_line is also empty, raise an exception
    raise ValueError("No non-empty line found within 3 lines above the start_line.")


def _get_post_end_line_index(end_line: int, content_lines: List[str]) -> int:
    if end_line < 1 or end_line > len(content_lines):
        raise IndexError("end_line is out of range.")

    end_line_index = end_line - 1
    start_search_index = min(len(content_lines) - 1, end_line_index + 1)
    end_search_index = min(len(content_lines) - 1, end_line_index + 3)  # Search up to 3 lines forward

    non_empty_indices = []

    for idx in range(start_search_index, end_search_index + 1):
        if content_lines[idx].strip() != "":
            non_empty_indices.append(idx)

    # Check if any non-empty line was found within the search range
    if non_empty_indices:
        return non_empty_indices[-1]

    # If no non-empty lines were found, check the end_line itself
    if content_lines[end_line_index].strip() != "":
        return end_line_index

    # If the end_line is also empty, raise an exception
    raise ValueError("No non-empty line found within 3 lines after the end_line.")


def remove_duplicate_lines(replacement_lines, original_lines):
    """
    Removes overlapping lines at the end of replacement_lines that match the beginning of original_lines.
    """
    if not replacement_lines or not original_lines:
        return replacement_lines

    max_overlap = min(len(replacement_lines), len(original_lines))

    for overlap in range(max_overlap, 0, -1):
        if replacement_lines[-overlap:] == original_lines[:overlap]:
            return replacement_lines[:-overlap]

    return replacement_lines
