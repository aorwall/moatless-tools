import json
import logging
from typing import Optional, Tuple

from instructor import OpenAISchema
from pydantic import Field

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.codeblocks import CodeBlockTypeGroup, BlockPath
from moatless.codeblocks.module import Module
from moatless.coder.code_action import CodeAction, respond_with_invalid_block
from moatless.coder.types import WriteCodeResult, CodeFunction
from moatless.file_context import FileContext
from moatless.settings import Settings

logger = logging.getLogger(__name__)


class UpdateCode(CodeFunction):
    """Update the code in one span. You can provide line numbers if only one part of the span should be updated."""

    instructions: str = Field(
        ..., description="Detailed instructions about what should be updated."
    )
    file_path: str = Field(description="The file path of the code to be updated.")
    span_id: str = Field(description="The span id of the code to be updated.")
    start_line: Optional[int] = Field(
        description="The start line of the code to be updated if just a part of the span should be updated.",
        default=None,
    )
    end_line: Optional[int] = Field(
        description="The end line of the code to be updated if just a part of the span should be updated.",
        default=None,
    )


class UpdateCodeAction(CodeAction):

    def __init__(self, file_context: FileContext = None):
        super().__init__(file_context)

    def _execute(
        self, task: UpdateCode, original_module: Module, updated_module: Module
    ) -> WriteCodeResult:
        # TODO: Refactor to not have to do this twice!
        original_block, start_index, end_index = self._find_block_and_span(
            original_module, task.span_id, task.start_line, task.end_line
        )

        updated_block = self._find_updated_block(
            updated_module, original_block.full_path()
        )
        if not updated_block:
            return respond_with_invalid_block(
                f"Couldn't find the expected block with path {task.block_path}",
                error_type="block_not_found",
            )

        # TODO: Verify that only the expected code is returned

        error_blocks = updated_block.find_errors()
        if error_blocks:
            logger.warning(
                f"Syntax errors found in updated block:\n{updated_block.to_string()}"
            )
            error_block_report = "\n\n".join(
                [f"```{block.content}```" for block in error_blocks]
            )
            return respond_with_invalid_block(
                f"There are syntax errors in the updated file:\n\n{error_block_report}. "
                f"Correct the errors and try again.",
                error_type="syntax_error",
            )

        if task.start_index is not None and task.end_index is not None:
            new_child_blocks = [
                child
                for child in updated_block.children
                if child.type not in [CodeBlockType.COMMENTED_OUT_CODE]
            ]

            if len(new_child_blocks) < (task.end_index - task.start_index / 2) and (
                (
                    new_child_blocks[0].type.group == CodeBlockTypeGroup.COMMENT
                    and new_child_blocks[0].type != CodeBlockType.COMMENT
                )
                or new_child_blocks[-1].type.group == CodeBlockTypeGroup.COMMENT
                and new_child_blocks[-1].type != CodeBlockType.COMMENT
            ):
                logger.warning(
                    f"Suspected placeholders in code block because of size difference. Expected {task.end_index - task.start_index} "
                    f"child blocks, received {len(new_child_blocks)}. {updated_block.to_string()}"
                )
                return respond_with_invalid_block(
                    "The updated code is only half the size of the original code. If this is expected, return same response again. Otherwise write out the full code block, don't comment out any existing code!",
                    error_type="suspected_incomplete_code",
                )

            original_block.children[start_index : end_index + 1] = new_child_blocks

        else:
            if not updated_block.is_complete():
                logger.warning(
                    f"Updated block isn't complete:\n{updated_block.to_string()}"
                )
                return respond_with_invalid_block(
                    f"The code is not fully implemented. Write out all code in the {updated_block.path_string()}",
                    error_type="incomplete_code",
                )
            original_index = original_block.parent.children.index(original_block)
            original_block.parent.children[original_index] = updated_block

        return WriteCodeResult()

    def _task_instructions(self, task: UpdateCode, module: Module) -> str:
        code_block, start_index, end_index = self._find_block_and_span(
            module, task.span_id, task.start_line, task.end_line
        )

        if start_index is not None and end_index is not None:
            start_line = code_block.children[start_index].start_line
            end_line = code_block.children[end_index].end_line
        else:
            start_line = code_block.start_line
            end_line = code_block.end_line

        code = module.to_prompt(
            start_line=start_line,
            end_line=end_line,
            show_outcommented_code=True,
            outcomment_code_comment="... other code",
        )

        code = code.strip()

        return f"""{task.instructions}

# Update the part of the implementation in {code_block.identifier}. Leave all placeholder comments as is.

```python
{code}
```"""

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

    def _find_block_and_span(
        self, module: Module, span_id: str, start_line: int, end_line: int
    ) -> Tuple[CodeBlock, Optional[int], Optional[int]]:
        span = module.find_span_by_id(span_id)
        span_block = module.find_by_path(span.parent_block_path)
        return self._find_block_that_has_lines(
            span_block, start_line, end_line, span_id
        )

    def _find_block_that_has_lines(
        self, codeblock: CodeBlock, start_line: int, end_line: int, span_id: str = None
    ) -> Tuple[CodeBlock, Optional[int], Optional[int]]:

        # Out of scope
        if codeblock.end_line < start_line or codeblock.start_line > end_line:
            return None, None, None

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
            if not (span_id and span_id in child.span_ids):
                continue

            # Go to block that covers all lines and is bigger than min tokens, check that block
            if (
                child.start_line <= start_line
                and child.end_line >= end_line
                and child.sum_tokens() > Settings.coder.min_tokens_for_split_span
            ):
                return self._find_block_that_has_lines(
                    child, start_line, end_line, span_id
                )

            # If the block is before the start line or not in span, add it to the blocks_before_start_line list
            if child.end_line < start_line:
                if child.type == CodeBlockType.COMMENT and child.sum_tokens() > 25:
                    # cut off on large comments
                    blocks_before_start_line = []
                    tokens_before_start_line = 0
                else:
                    blocks_before_start_line.append(child)
                    tokens_before_start_line += child.sum_tokens()

            # If the block is after the end line, add it to the blocks_after_end_line list
            elif child.start_line > end_line:
                blocks_after_end_line.append(child)
                tokens_after_end_line += child.sum_tokens()

            # If the block is within the span, add it to the blocks_in_span list
            else:
                blocks_in_span.append(child)
                tokens_in_span += child.sum_tokens()
                print(f"Add block in span: {child.identifier} ({child.type})")

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
