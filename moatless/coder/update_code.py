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
from moatless.coder.prompt import CODER_SYSTEM_PROMPT, UPDATE_CODE_RESPONSE_FORMAT
from moatless.coder.types import FunctionResponse, Omissible
from moatless.file_context import FileContext
from moatless.session import Session
from moatless.settings import Settings

logger = logging.getLogger(__name__)


class UpdateCode(CodeAgentFunction):
    """Update the code in one span. You can provide line numbers if only one part of the span should be updated."""

    instructions: str = Field(
        ..., description="Detailed instructions on how to update the code."
    )
    file_path: str = Field(description="The file path of the code to be updated.")
    span_id: str = Field(description="The span id of the code to be updated.")
    start_line: Omissible[int] = Field(
        description="The start line of the code to be updated if just a part of the span should be updated.",
        default=None
    )
    end_line: Omissible[int] = Field(
        description="The end line of the code to be updated if just a part of the span should be updated.",
        default=None
    )

    _max_retries: int = PrivateAttr(default=2)
    _max_tokens_for_span: int = PrivateAttr(default=500)

    _start_index: Optional[int] = PrivateAttr(default=None)
    _end_index: Optional[int] = PrivateAttr(default=None)
    _original_block: CodeBlock = PrivateAttr()
    _original_module: Module = PrivateAttr()

    _file_context: FileContext = PrivateAttr()
    _trace_id: str = PrivateAttr(default=None)
    _retries: int = PrivateAttr(default=0)
    _messages: List[Dict] = PrivateAttr(default_factory=list)

    _mock_response: Optional[str] = PrivateAttr(default=None)

    def call(self) -> FunctionResponse:
        original_file = self._file_context.get_file(self.file_path)
        if not original_file:
            logger.warning(f"File {self.file_path} not found in file context.")
            return FunctionResponse(
                file_path=self.file_path,
                error=f"{self.file_path} was not found in the file context.",
            )

        self._original_module = original_file.module

        if self.start_line:
            logger.info(
                f"Updating code in file {self.file_path}, span {self.span_id} and lines {self.start_line}-{self.end_line}"
            )
        else:
            logger.info(f"Updating code in file {self.file_path}, span {self.span_id}")
        span = self._original_module.find_span_by_id(self.span_id)
        span_block = self._original_module.find_by_path(span.parent_block_path)

        if self.start_line is None:
            self._original_block = span_block
        else:
            if self.end_line is None:
                self.end_line = span_block.end_line

            if span_block.end_line == self.start_line - 1:
                logger.info(
                    f"Span {self.span_id} end line {span_block.endline} is one line before suggested start line "
                    f"{self.start_line}. Will change start line as it might just be a new addition"
                )
                self.start_line = span_block.end_line

            if (
                span_block.end_line < self.start_line
                or span_block.start_line > self.end_line
            ):
                self._fail(
                    error_type="line_not_in_block",
                    message=f"Line {self.start_line}-{self.end_line} is not within the block {span_block.identifier} on line"
                    f" {span_block.start_line}-{span_block.end_line}"
                )

            self._original_block, self._start_index, self._end_index = (
                self._find_block_that_has_lines(span_block)
            )

        span = self._original_module.find_span_by_id(self.span_id)
        if (
            span.tokens > self._max_tokens_for_span
            and self.start_line is None
            and self.end_line is None
        ):
            logger.warning(
                f"Span {self.span_id} is too big ({span.tokens} tokens). Try to narrow it down by defining a start and end line."
            )
            return self._fail(
                error_type="span_too_big",
                message=f"The span is too big, try to narrow it down by defining a start and end line",
            )

        task_instructions = self._task_instructions()

        file_context_content = self._file_context.create_prompt(exclude_comments=True)

        system_message = {
            "role": "system",
            "content": f"""{CODER_SYSTEM_PROMPT}

{UPDATE_CODE_RESPONSE_FORMAT}

# File context:
{file_context_content}
""",
        }

        instruction_message = {
            "role": "user",
            "content": task_instructions,
        }

        self._messages = [system_message, instruction_message]

        while self._retries < self._max_retries:
            response = self._dev_loop()
            if response:
                return response
            self._retries += 1

        return FunctionResponse(
            error=f"Failed to update code. No rejection reason provided."
        )

    def _dev_loop(self) -> Optional[FunctionResponse]:
        logger.info(f"Starting code update loop, retry {self._retries}")

        llm_response = completion(
            model=Settings.coder.coding_model,
            temperature=0.0,
            max_tokens=1500,
            messages=self._messages,
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

        extracted_parts = extract_response_parts(choice.message.content)

        changes = [part for part in extracted_parts if isinstance(part, CodePart)]

        thoughts = [part for part in extracted_parts if isinstance(part, str)]
        thoughts = "\n".join([thought for thought in thoughts])

        if thoughts:
            logger.info(f"Thoughts: {thoughts}")

        if not changes:
            if choice.finish_reason == "length":
                self._fail(
                    error_type="token_limit_exceeded",
                    message="No changed found, probably exeeded token limit.",
                )
                # TODO: Handle in a more graceful way than just aborting the flow
                raise ValueError("Token limit exceeded")
            else:
                return self._fail(
                    error_type="no_changes",
                    message=f"No code changes found in the updated code. {thoughts}"
                )

        if len(changes) > 1:
            logger.warning(
                f"Multiple code blocks found in response, ignoring all but the first one."
            )

        try:
            parser = PythonParser(apply_gpt_tweaks=True)  # FIXME
            updated_module = parser.parse(changes[0].content)
        except Exception as e:
            # TODO: Not sure if it's a good idea to retry on exceptions...
            rejection_reason = f"There was a syntax error in the code block. Please correct it and try again. Error: '{e}'"
            return self._reject(
                rejection_code="syntax_error", rejection_reason=rejection_reason
            )

        original_content = self._original_module.to_string()

        updated_block = self._find_updated_block(
            updated_module, self._original_block.full_path()
        )
        if not updated_block:
            blocks = updated_module.find_blocks_with_identifier(self._original_block.identifier)
            if len(blocks) == 1:
                if len(blocks[0].full_path()) < len(self._original_block.full_path()):
                    rejection_reason = f"Found {self._original_block.identifier} on the wrong level, expected it to be inside {self._original_block.full_path()}. Return the code as instructed in CODE TO UPDATE."
                    return self._reject(
                        rejection_code="block_on_wrong_level", rejection_reason=rejection_reason
                    )
            rejection_reason = (
                f"Couldn't find the expected block {self._original_block.identifier}. Return the code as instructed in CODE TO UPDATE. "
            )
            return self._reject(
                rejection_code="block_not_found", rejection_reason=rejection_reason
            )

        # TODO: Verify that only the expected code is returned

        error_blocks = updated_block.find_errors()
        if error_blocks:
            error_block_report = "\n\n".join(
                [f"```{block.content}```" for block in error_blocks]
            )
            rejection_reason = f"There are syntax errors in the updated file:\n\n{error_block_report}. Correct the errors and try again."
            return self._reject(
                rejection_code="syntax_error", rejection_reason=rejection_reason
            )

        if self._start_index is not None and self._end_index is not None:
            new_child_blocks = [
                child
                for child in updated_block.children
                if child.type not in [CodeBlockType.COMMENTED_OUT_CODE]
            ]

            if len(new_child_blocks) < (self._end_index - self._start_index / 2) and (
                (
                    new_child_blocks[0].type.group == CodeBlockTypeGroup.COMMENT
                    and new_child_blocks[0].type != CodeBlockType.COMMENT
                )
                or new_child_blocks[-1].type.group == CodeBlockTypeGroup.COMMENT
                and new_child_blocks[-1].type != CodeBlockType.COMMENT
            ):
                rejection_reason = (
                    "The updated code is only half the size of the original code. If this is expected, return "
                    "same response again. Otherwise write out the full code block, don't comment out any existing code!"
                )
                return self._reject(
                    rejection_code="suspected_incomplete_code",
                    rejection_reason=rejection_reason,
                )

            self._original_block.replace_children(
                self._start_index, self._end_index + 1, new_child_blocks
            )
        else:
            if not updated_block.is_complete():
                rejection_reason = f"The code is not fully implemented. Write out all code in {updated_block.path_string()} "
                return self._reject(
                    rejection_code="incomplete_code", rejection_reason=rejection_reason
                )

            original_index = self._original_block.parent.children.index(
                self._original_block
            )
            self._original_block.parent.replace_child(original_index, updated_block)

        updated_content = self._original_module.to_string()
        diff = do_diff(self.file_path, original_content, updated_content)

        if diff:
            # TODO: Move this to let the agent decide if content should be saved
            self._file_context.save_file(self.file_path)
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
                "file": self.file_path,
                "span_id": (self.span_id if self.span_id else None),
            },
        )

        return FunctionResponse(message=diff)

    def _task_instructions(self) -> str:
        if self._start_index is not None and self._end_index is not None:
            start_line = self._original_block.children[self._start_index].start_line
            end_line = self._original_block.children[self._end_index].end_line
        else:
            start_line = self._original_block.start_line
            end_line = self._original_block.end_line

        code = self._original_module.to_prompt(
            start_line=start_line,
            end_line=end_line,
            span_ids={self.span_id},
            show_outcommented_code=True,
            outcomment_code_comment="... other code",
        )

        code = code.strip()

        if self._original_block.type.group == CodeBlockTypeGroup.STRUCTURE:
            closest_structure_block = self._original_block
        else:
            closest_structure_block = self._original_block.find_type_group_in_parents(CodeBlockTypeGroup.STRUCTURE)

        code_instructions = ""
        if closest_structure_block.parent and closest_structure_block.parent.type == CodeBlockType.CLASS:
            code_instructions += f"You must keep the class signature for `{closest_structure_block.parent.identifier}`. "

        if closest_structure_block.type != CodeBlockType.MODULE:
            code_instructions += f"Only update the code inside the {closest_structure_block.type.value.lower()} `{closest_structure_block.identifier}`. "

        if (self._start_index is not None and self._end_index is not None) or not code_instructions:
            code_instructions += f"Only update the code that is not out commented. "

        return f"""# INSTRUCTIONS:
{self.instructions}

# CODE UPDATE INSTRUCTIONS:
{code_instructions}

# CODE TO UPDATE:
```python
{code}
```
"""

    def _find_updated_block(
        self, codeblock: CodeBlock, block_path: BlockPath
    ) -> Optional[CodeBlock]:
        updated_block = codeblock.find_by_path(block_path)
        if updated_block:
            return updated_block

        updated_block = codeblock.find_by_identifier(block_path[0])
        if not updated_block:
            logger.info(
                f"No code block found with id {block_path[0]}, will try to find a single block that isn't a placeholder"
            )
            blocks = [
                child
                for child in codeblock.children
                if child.type != CodeBlockType.COMMENTED_OUT_CODE
            ]
            if len(blocks) == 1:
                updated_block = blocks[0]
            else:
                logger.warning(
                    f"Found {len(blocks)} blocks that aren't placeholders, can't determine which one to update"
                )
                return None

        if not updated_block:
            return None

        if len(block_path) == 1:
            return updated_block
        elif len(block_path) > 1:
            return self._find_updated_block(updated_block, block_path[1:])
        else:
            return None

    def _find_block_by_identifer(self, codeblock: CodeBlock):
        blocks = self._original_module.find_blocks_with_identifier(codeblock.identifier)
        if len(blocks) == 1:
            return blocks[0]
        else:
            return None

    def _find_block_that_has_lines(
        self, codeblock: CodeBlock
    ) -> Tuple[CodeBlock, Optional[int], Optional[int]]:

        # If codeblock is smaller than min_tokens_for_split_span just return the whole block
        if codeblock.sum_tokens() < Settings.coder.min_tokens_for_split_span:
            return codeblock, None, None

        # TODO: Just to not generate else clauses etc out of context. Find a better solution on big compound statements?
        if any(
            [
                child.type == CodeBlockType.DEPENDENT_CLAUSE
                for child in codeblock.children
            ]
        ):
            return codeblock, None, None

        blocks_before_start_line = []
        tokens_before_start_line = 0
        blocks_in_span = []
        tokens_in_span = 0
        blocks_after_end_line = []
        tokens_after_end_line = 0

        for child in codeblock.children:
            if not (self.span_id and self.span_id in child.span_ids):
                continue

            # Go to block that covers all lines and is bigger than min tokens, check that block
            if (
                child.start_line <= self.start_line
                and child.end_line >= self.end_line
                and child.sum_tokens() > Settings.coder.min_tokens_for_split_span
                and child.children
            ):
                return self._find_block_that_has_lines(child)

            # If the block is before the start line or not in span, add it to the blocks_before_start_line list
            if child.end_line < self.start_line:
                if child.type == CodeBlockType.COMMENT and child.sum_tokens() > 25:
                    # cut off on large comments
                    blocks_before_start_line = []
                    tokens_before_start_line = 0
                else:
                    blocks_before_start_line.append(child)
                    tokens_before_start_line += child.sum_tokens()

            # If the block is after the end line, add it to the blocks_after_end_line list
            elif child.start_line > self.end_line:
                blocks_after_end_line.append(child)
                tokens_after_end_line += child.sum_tokens()

            # If the block is within the span, add it to the blocks_in_span list
            else:
                blocks_in_span.append(child)
                tokens_in_span += child.sum_tokens()

        # If tokens in span is higher than min_tokens, return the start and end index of the span
        if tokens_in_span > Settings.coder.min_tokens_for_split_span:
            start_index = codeblock.children.index(blocks_in_span[0])
            end_index = codeblock.children.index(blocks_in_span[-1])
            return codeblock, start_index, end_index

        # Loop to add blocks from before and after until min_tokens_for_split_span is exceeded
        additional_blocks = blocks_before_start_line[::-1] + blocks_after_end_line
        for block in additional_blocks:
            tokens_in_span += block.sum_tokens()
            blocks_in_span.append(block)

            if tokens_in_span > Settings.coder.min_tokens_for_split_span:
                break

        sorted_blocks = sorted(blocks_in_span, key=lambda x: x.start_line)
        start_index = codeblock.children.index(sorted_blocks[0])
        end_index = codeblock.children.index(sorted_blocks[-1])
        return codeblock, start_index, end_index

    def _fail(self, error_type: str, message: str):
        logger.warning(f"code_update_failed:{error_type}: {message}")
        send_event(
            event="code_update_failed",
            properties={"error_type": error_type, "retries": self._retries},
        )
        return FunctionResponse(error=f"Failed to update code. Error: {message}")

    def _reject(self, rejection_code: str, rejection_reason: str):
        logger.info(f"code_update_rejected:{rejection_code}: {rejection_reason}")
        send_event(
            event="code_update_rejected",
            properties={
                "error_type": rejection_code,
                "rejection_reason": rejection_reason,
                "retries": self._retries,
            },
        )

        self._messages.append(
            {
                "role": "user",
                "content": f"Sorry, I couldn't update the code. {rejection_reason}",
            }
        )

        return None
