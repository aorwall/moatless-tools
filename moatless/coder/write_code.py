import logging
from typing import Optional, List

from pydantic import BaseModel

from moatless.analytics import send_event
from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.types import Span


logger = logging.getLogger(__name__)


class WriteCodeResult(BaseModel):
    content: Optional[str] = None
    error: Optional[str] = None
    change: Optional[str] = None


def write_code(
    original_block: CodeBlock,
    updated_block: CodeBlock,
    expected_block: CodeBlock = None,
    expected_span: Span = None,
    action: Optional[str] = None,
) -> WriteCodeResult:
    def respond_with_invalid_block(message: str):
        return WriteCodeResult(
            change=None,
            error=message,
        )

    block_label = expected_block.path_string() or "<>"
    if expected_span:
        block_label = (
            f"{block_label} ({expected_span.start_line}-{expected_span.end_line})"
        )

    if action == "delete":
        logger.info(
            f"Want to delete {block_label}, but can't because it's not supported yet"
        )
        # TODO: Delete block

        return WriteCodeResult(
            content=original_block.to_string(),
            change=None,
            error="Deleting blocks is not supported yet",
        )

    if action == "add":
        logger.info(
            f"Want to create {block_label}, but can't because it's not supported yet"
        )
        # TODO: Add block
        return WriteCodeResult(
            content=original_block.to_string(),
            change=None,
            error="Adding blocks is not supported yet",
        )

    if original_block.full_path() == expected_block.full_path():
        changed_block = updated_block
    else:
        changed_block = find_by_path_recursive(
            updated_block, expected_block.full_path()
        )

    if not changed_block:
        logger.warning(
            f"Couldn't find expected block {block_label} in content:\n{updated_block.to_string()}"
        )
        return respond_with_invalid_block(
            "The updated block should not contain multiple blocks."
        )

    error_blocks = changed_block.find_errors()
    if error_blocks:
        logger.warning(
            f"Syntax errors found in updated block:\n{updated_block.to_string()}"
        )
        error_block_report = "\n\n".join(
            [f"```{block.content}```" for block in error_blocks]
        )
        return respond_with_invalid_block(
            f"There are syntax errors in the updated file:\n\n{error_block_report}. "
            f"Correct the errors and try again."
        )

    if expected_block and not expected_span:
        if not changed_block.is_complete():
            logger.warning(
                f"Updated block isn't complete:\n{updated_block.to_string()}"
            )
            return respond_with_invalid_block(
                "The code is not fully implemented. Write out all code in the code block."
            )

        if len(changed_block.children) < len(expected_block.children) / 2 and (
            (
                changed_block.children[0].type == CodeBlockType.COMMENT
                and expected_block.children[0].type != CodeBlockType.COMMENT
            )
            or changed_block.children[-1].type == CodeBlockType.COMMENT
            and expected_block.children[-1].type != CodeBlockType.COMMENT
        ):
            logger.warning(
                f"Suspected placeholders in code block:\n{updated_block.to_string()}"
            )
            return respond_with_invalid_block(
                "The updated code is only half the size of the original code. Write out the full code block, don't comment out any existing code!"
            )

        original_block.replace_by_path(expected_block.full_path(), changed_block)
        logger.debug(f"Updated block: {expected_block.full_path()}")
    elif expected_span:
        start_idx, end_idx = _find_start_and_end_index(expected_block, expected_span)
        if start_idx is None:
            logger.warning(f"Couldn't find start index for span {expected_span}")
            return respond_with_invalid_block("Couldn't find start index for span")

        # TODO: Verify indentation etc

        del expected_block.children[start_idx:end_idx]
        expected_block.children[start_idx:start_idx] = changed_block.children
    else:
        original_block = changed_block
        logger.debug(f"Replaces full file")

    return WriteCodeResult(content=original_block.to_string())


def _find_start_and_end_index(block: CodeBlock, span: Span):
    start_index = None

    for i, child_block in enumerate(block.children):
        if child_block.start_line < span.start_line:
            continue

        if start_index is None and span.start_line >= child_block.start_line:
            start_index = i

        if span.end_line <= child_block.end_line:
            return start_index, i

    return None, None


def find_by_path_recursive(codeblock, block_path: List[str]):
    found = codeblock.find_by_path(block_path)
    if not found and len(block_path) > 1:
        return find_by_path_recursive(codeblock, block_path[1:])
    return found
