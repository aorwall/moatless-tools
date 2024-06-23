import logging
from typing import Type, Optional, Tuple, List

from pydantic import PrivateAttr, Field, BaseModel

from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.codeblocks import CodeBlockTypeGroup, BlockSpan
from moatless.edit.prompt import CLARIFY_CHANGE_SYSTEM_PROMPT
from moatless.repository import CodeFile
from moatless.state import AgenticState, ActionResponse
from moatless.types import (
    FileWithSpans,
    ActionRequest,
    Message,
)
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger("ClarifyCodeChange")


class LineNumberClarification(ActionRequest):
    scratch_pad: str = Field(..., description="Thoughts on which lines to select")
    start_line: int = Field(
        ..., description="The start line of the code to be updated."
    )

    end_line: int = Field(..., description="The end line of the code to be updated.")
    reject: Optional[bool] = Field(
        None, description="Whether the request should be rejected."
    )


class ClarifyCodeChange(AgenticState):
    instructions: str
    file_path: str
    span_id: str

    start_line: Optional[int] = None
    end_line: Optional[int] = None

    max_tokens_in_edit_prompt: int = Field(
        500,
        description="The maximum number of tokens in a span to show the edit prompt.",
    )

    _file: Optional[CodeFile] = PrivateAttr(None)
    _span: Optional[BlockSpan] = PrivateAttr(None)
    _file_context_str: Optional[str] = PrivateAttr(None)

    def __init__(self, instructions: str, file_path: str, span_id: str, **data):
        super().__init__(
            instructions=instructions, file_path=file_path, span_id=span_id, **data
        )

    def init(self):
        self._file = self.file_repo.get_file(self.file_path)
        self._span = self._file.module.find_span_by_id(self.span_id)

        file_context = self.create_file_context(
            [FileWithSpans(file_path=self.file_path, span_ids=[self.span.span_id])]
        )

        # Include all function/class signatures if the block is a class
        if self.span.initiating_block.type == CodeBlockType.CLASS:
            for child in self.span.initiating_block.children:
                if (
                    child.type.group == CodeBlockTypeGroup.STRUCTURE
                    and child.belongs_to_span
                    and child.belongs_to_span.span_id != self._span.span_id
                ):
                    file_context.add_span_to_context(
                        file_path=self.file_path,
                        span_id=child.belongs_to_span.span_id,
                        tokens=1,
                    )  # TODO: Change so 0 can be set and mean "only signature"

        self._file_context_str = file_context.create_prompt(
            show_line_numbers=True,
            show_span_ids=False,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="... other code",
        )

    def handle_action(self, request: LineNumberClarification) -> ActionResponse:
        logger.info(
            f"{self}: Got line number clarification: {request.start_line} - {request.end_line}"
        )

        if request.reject:
            return ActionResponse.transition(
                trigger="reject", output={"message": request.scratch_pad}
            )

        retry_message = self._verify_line_numbers(request)
        if retry_message:
            return ActionResponse.retry(retry_message)

        if request.end_line - request.start_line < 4:
            start_line, end_line = self.get_line_span(
                request.start_line, request.end_line, self.max_tokens_in_edit_prompt
            )
        else:
            start_line, end_line = request.start_line, request.end_line

        if request.scratch_pad:
            self.instructions += "\n\n" + request.scratch_pad

        return ActionResponse.transition(
            trigger="edit_code",
            output={
                "instructions": self.instructions,
                "file_path": self.file_path,
                "span_id": self.span_id,
                "start_line": start_line,
                "end_line": end_line,
            },
        )

    @classmethod
    def required_fields(cls) -> set[str]:
        return {"instructions", "file_path", "span_id"}

    def action_type(self) -> Optional[Type[BaseModel]]:
        return LineNumberClarification

    @property
    def file(self) -> CodeFile:
        assert self._file is not None, "File has not been set"
        return self._file

    @property
    def span(self) -> BlockSpan:
        assert self._span is not None, "Span has not been set"
        return self._span

    def _verify_line_numbers(
        self, line_numbers: LineNumberClarification
    ) -> Optional[str]:
        logger.info(
            f"{self}: Verifying line numbers: {line_numbers.start_line} - {line_numbers.end_line}. "
            f"To span with line numbers: {self.span.start_line} - {self.span.end_line}"
        )

        if (
            line_numbers.start_line <= self.span.start_line
            and line_numbers.end_line >= self.span.end_line
        ):
            return f"The provided line numbers {line_numbers.start_line} - {line_numbers.end_line} covers the whole code span. You must specify line numbers of only lines you want to change."

        span_block = self.span.initiating_block

        # The LLM sometimes refer to only the lines of the class/function signature when it's intention is to edit lines
        if span_block.type.group == CodeBlockTypeGroup.STRUCTURE:
            last_block_content_line = span_block.children[0].start_line - 1

            logger.info(
                f"{self}: Checking if the line numbers only covers a class/function signature to "
                f"{self.span.initiating_block.path_string()} ({span_block.start_line} - {last_block_content_line})"
            )
            if (
                line_numbers.start_line == span_block.start_line
                and last_block_content_line >= line_numbers.end_line
                and self.span.initiating_block.sum_tokens()
                > self.max_tokens_in_edit_prompt
            ):
                clarify_msg = f"The line numbers {line_numbers.start_line} - {line_numbers.end_line} only covers to the signature of the {self.span.initiating_block.type.value}."
                logger.info(f"{self}: {clarify_msg}. Ask for clarification.")
                # TODO: Ask if this was intentional instead instructing the LLM
                return f"{clarify_msg}. You need to specify the exact part of the code that needs to be updated to fulfill the change."

        code_lines = self.file.content.split("\n")
        lines_to_replace = code_lines[
            line_numbers.start_line - 1 : line_numbers.end_line
        ]

        edit_block_code = "\n".join(lines_to_replace)

        tokens = count_tokens(edit_block_code)
        if tokens > self.max_tokens_in_edit_prompt:
            clarify_msg = f"Lines {line_numbers.start_line} - {line_numbers.end_line} has {tokens} tokens, which is higher than the maximum allowed {self.max_tokens_in_edit_prompt} tokens in completion"
            logger.info(f"{self} {clarify_msg}. Ask for clarification.")
            return f"{clarify_msg}. You need to specify the exact part of the code that needs to be updated to fulfill the change. If this is not possible you should reject the request."

        return None

    def system_prompt(self) -> str:
        return CLARIFY_CHANGE_SYSTEM_PROMPT

    def messages(self) -> list[Message]:
        messages = [
            Message(
                role="user",
                content=f"<instructions>\n{self.instructions}\n</instructions>\n<code>\n{self._file_context_str}\n</code>",
            )
        ]

        messages.extend(self.retry_messages())

        return messages

    def get_line_span(
        self,
        start_line: int,
        end_line: int,
        max_tokens: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the span that covers the lines from start_line to end_line
        """

        logger.info(
            f"Get span to change in {self.file_path} from {start_line} to {end_line}"
        )

        start_block = self.file.module.find_first_by_start_line(start_line)
        assert (
            start_block is not None
        ), f"No block found in {self.file_path} that starts at line {start_line}"

        if start_block.type.group == CodeBlockTypeGroup.STRUCTURE and (
            not end_line or start_block.end_line > end_line
        ):
            struture_block = start_block
        else:
            struture_block = start_block.find_type_group_in_parents(
                CodeBlockTypeGroup.STRUCTURE
            )

        assert (
            struture_block is not None
        ), f"No structure bock found for {start_block.path_string()}"

        if struture_block.sum_tokens() < max_tokens:
            logger.info(
                f"Return block [{struture_block.path_string()}] ({struture_block.start_line} - {struture_block.end_line}) with {struture_block.sum_tokens()} tokens that covers the provided line span ({start_line} - {end_line})"
            )
            return struture_block.start_line, struture_block.end_line

        if not end_line:
            end_line = start_line

        original_lines = self.file.content.split("\n")
        if struture_block.end_line - end_line < 5:
            logger.info(
                f"Set parent block [{struture_block.path_string()}] end line {struture_block.end_line} as it's {struture_block.end_line - end_line} lines from the end of the file"
            )
            end_line = struture_block.end_line
        else:
            end_line = _get_post_end_line_index(
                end_line, struture_block.end_line, original_lines
            )
            logger.info(f"Set end line to {end_line} from the end of the parent block")

        if start_line - struture_block.start_line < 5:
            logger.info(
                f"Set parent block [{struture_block.path_string()}] start line {struture_block.start_line} as it's {start_line - struture_block.start_line} lines from the start of the file"
            )
            start_line = struture_block.start_line
        else:
            start_line = _get_pre_start_line(
                start_line, struture_block.start_line, original_lines
            )
            logger.info(
                f"Set start line to {start_line} from the start of the parent block"
            )

        return start_line, end_line


def _get_pre_start_line(
    start_line: int, min_start_line: int, content_lines: List[str], max_lines: int = 4
) -> int:
    if start_line > len(content_lines):
        raise ValueError(
            f"start_line {start_line} is out of range ({len(content_lines)})."
        )

    if start_line - min_start_line < max_lines:
        return min_start_line

    start_line_index = start_line - 1
    start_search_index = max(0, start_line_index - 1)
    end_search_index = max(min_start_line, start_line_index - max_lines)

    non_empty_indices = []

    for idx in range(start_search_index, end_search_index - 1, -1):
        if content_lines[idx].strip() != "":
            non_empty_indices.append(idx)

    # Check if any non-empty line was found within the search range
    if non_empty_indices:
        return non_empty_indices[-1] + 1

    # If no non-empty lines were found, check the start_line itself
    if content_lines[start_line_index].strip() != "":
        return start_line_index + 1

    # If the start_line is also empty, raise an exception
    raise ValueError("No non-empty line found within 3 lines above the start_line.")


def _get_post_end_line_index(
    end_line: int, max_end_line: int, content_lines: List[str], max_lines: int = 4
) -> int:
    if end_line < 1 or end_line > len(content_lines):
        raise IndexError("end_line is out of range.")

    if max_end_line - end_line < max_lines:
        return max_end_line

    end_line_index = end_line - 1
    start_search_index = min(len(content_lines) - 1, end_line_index + 1)
    end_search_index = min(max_end_line - 1, end_line_index + max_lines)

    non_empty_indices = []

    for idx in range(start_search_index, end_search_index + 1):
        if content_lines[idx].strip() != "":
            non_empty_indices.append(idx)

    # Check if any non-empty line was found within the search range
    if non_empty_indices:
        return non_empty_indices[-1] + 1

    # If no non-empty lines were found, check the end_line itself
    if content_lines[end_line_index].strip() != "":
        return end_line_index + 1

    # If the end_line is also empty, raise an exception
    raise ValueError("No non-empty line found within 3 lines after the end_line.")
