import logging
from typing import Optional, List, Tuple

from pydantic import BaseModel

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.codeblocks import BlockSpan, CodeBlockTypeGroup, BlockPath
from moatless.codeblocks.module import Module
from moatless.coder.code_utils import create_instruction_code_block
from moatless.coder.types import UpdateCodeTask
from moatless.coder.write_code import WriteCodeResult, respond_with_invalid_block
from moatless.settings import Settings
from moatless.types import Span

logger = logging.getLogger(__name__)


def update_code(
    original_module: Module,
    updated_module: Module,
    block_path: BlockPath,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> WriteCodeResult:
    original_block = original_module.find_by_path(block_path)
    updated_block = find_updated_block(updated_module, block_path)
    if not updated_block:
        return respond_with_invalid_block(
            f"Couldn't find the expected block with path {block_path}",
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

    if start_index is not None and end_index is not None:
        new_child_blocks = [
            child
            for child in updated_block.children
            if child.type not in [CodeBlockType.COMMENTED_OUT_CODE]
        ]

        if len(new_child_blocks) < (end_index - start_index / 2) and (
            (
                new_child_blocks[0].type.group == CodeBlockTypeGroup.COMMENT
                and new_child_blocks[0].type != CodeBlockType.COMMENT
            )
            or new_child_blocks[-1].type.group == CodeBlockTypeGroup.COMMENT
            and new_child_blocks[-1].type != CodeBlockType.COMMENT
        ):
            logger.warning(
                f"Suspected placeholders in code block because of size difference. Expected {end_index - start_index} "
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

    return WriteCodeResult(content=original_module.to_string())


def find_updated_block(
    codeblock: CodeBlock, block_path: BlockPath
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
        return find_updated_block(updated_block, block_path[1:])
    else:
        return None
