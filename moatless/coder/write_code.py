import logging
from typing import Optional, List

from pydantic import BaseModel

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.types import Span


logger = logging.getLogger(__name__)


class WriteCodeResult(BaseModel):
    content: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    change: Optional[str] = None


def write_code(
    original_file_block: CodeBlock,
    updated_file_block: CodeBlock,
    expected_span: Span = None,
    action: Optional[str] = None,
) -> WriteCodeResult:
    def respond_with_invalid_block(message: str, error_type: str = None):
        return WriteCodeResult(
            change=None,
            error_type=error_type,
            error=message,
        )

    if expected_span:
        if expected_span.block_path:
            expected_block = original_file_block.find_by_path(expected_span.block_path)
        else:
            expected_block = original_file_block.find_block_with_span(expected_span)

        if expected_block is None:
            logger.warning(
                f"Couldn't find expected block with path {expected_span.block_path}"
            )
            return respond_with_invalid_block(
                f"Couldn't find the expected block with path {expected_span.block_path}",
                error_type="block_not_found",
            )

        if action == "add":
            updated_parent_block = updated_file_block.find_by_path(
                expected_span.block_path
            )
            updated_child_blocks = [
                child
                for child in updated_parent_block.children
                if child.type
                not in [CodeBlockType.COMMENTED_OUT_CODE, CodeBlockType.COMMENT]
            ]

            if len(updated_child_blocks) == 1:
                changed_block = updated_child_blocks[0]
                logger.info(
                    f"Will add block {changed_block.path_string()} to {expected_block.path_string()} on line {expected_span.start_line}"
                )
            elif len(updated_child_blocks) > 1:
                logger.warning(
                    f"Found multiple child blocks in updated content for block path {expected_span.block_path}"
                )
                return respond_with_invalid_block(
                    f"Found multiple child blocks in updated content for block path {expected_span.block_path}",
                    error_type="multiple_blocks",
                )
            else:
                logger.warning(
                    f"Couldn't find expected block with path {expected_span.block_path}"
                )
                return respond_with_invalid_block(
                    f"Couldn't find the expected block with path {expected_span.block_path}",
                    error_type="block_not_found",
                )
        elif expected_block.full_path() == original_file_block.full_path():
            changed_block = updated_file_block
        else:
            changed_block = updated_file_block.find_by_path(expected_block.full_path())

    else:
        expected_block = original_file_block
        changed_block = updated_file_block

    if expected_block.path_string():
        block_label = (
            f"the {expected_block.type.lower()} {expected_block.path_string()}"
        )
        if expected_span:
            block_label += f" (L{expected_span.start_line}-L{expected_span.end_line})"
    elif expected_span:
        block_label = f"span L{expected_span.start_line}-L{expected_span.end_line}"
    else:
        block_label = "file"

    if action == "delete":
        logger.info(
            f"Want to delete {block_label}, but can't because it's not supported yet"
        )
        # TODO: Delete block

        return WriteCodeResult(
            content=original_file_block.to_string(),
            change=None,
            error_type="delete_not_supported",
            error="Deleting blocks is not supported yet",
        )

    if not changed_block:
        logger.warning(f"Couldn't find expected {block_label} in updated content.")

        return respond_with_invalid_block(
            f"Couldn't find the expected {block_label}. Did you return the full {expected_block.type.lower()}?"
        )

    if (
        action == "update"
        and expected_span.is_partial
        and (
            expected_span.start_line > expected_block.start_line
            or expected_span.end_line < expected_block.end_line
        )
    ):
        logger.info(
            f"Expected span ({expected_span.start_line}-{expected_span.end_line}) is starting or ending inside expected block ({expected_block.start_line}-{expected_block.end_line}, will set as 'update_span')"
        )
        action = "update_span"

    error_blocks = changed_block.find_errors()
    if error_blocks:
        logger.warning(
            f"Syntax errors found in updated block:\n{updated_file_block.to_string()}"
        )
        error_block_report = "\n\n".join(
            [f"```{block.content}```" for block in error_blocks]
        )
        return respond_with_invalid_block(
            f"There are syntax errors in the updated file:\n\n{error_block_report}. "
            f"Correct the errors and try again.",
            error_type="syntax_error",
        )

    if action in ["update", "add"]:
        if not changed_block.is_complete():
            logger.warning(
                f"Updated block isn't complete:\n{updated_file_block.to_string()}"
            )
            return respond_with_invalid_block(
                f"The code is not fully implemented. Write out all code in the {block_label}",
                error_type="incomplete_code",
            )

    if action == "update":
        no_of_changed_children = len(changed_block.children)
        no_of_expected_children = len(expected_block.children)
        if no_of_changed_children < no_of_expected_children / 2 and (
            (
                changed_block.children[0].type == CodeBlockType.COMMENT
                and expected_block.children[0].type != CodeBlockType.COMMENT
            )
            or changed_block.children[-1].type == CodeBlockType.COMMENT
            and expected_block.children[-1].type != CodeBlockType.COMMENT
        ):
            logger.warning(
                f"Suspected placeholders in code block because of size difference. Expected {no_of_expected_children} "
                f"child blocks, received {no_of_expected_children}. {updated_file_block.to_string()}"
            )
            return respond_with_invalid_block(
                "The updated code is only half the size of the original code. Write out the full code block, don't comment out any existing code!",
                error_type="suspected_incomplete_code",
            )

        if expected_block.full_path() == original_file_block.full_path():
            original_file_block = changed_block
            logger.debug(f"Updated full file")
        else:
            original_file_block.replace_by_path(
                expected_block.full_path(), changed_block
            )
            logger.debug(f"Updated block: {expected_block.full_path()}")

    if action in ["update_span", "add"]:
        start_idx, end_idx = _find_start_and_end_index(expected_block, expected_span)

        if action == "add":
            if start_idx is None:
                expected_block.append_child(changed_block)
            else:
                expected_block.insert_child(start_idx, changed_block)
        elif action == "update_span":
            if start_idx is None:
                logger.warning(f"Couldn't find start index for span {expected_span}")
                return respond_with_invalid_block(
                    f"Couldn't find start line for {block_label}",
                    error_type="incorrect_span",
                )

            # TODO: Verify indentation etc
            first_replaced = expected_block.children[start_idx]
            first_to_replace = changed_block.children[0]

            if first_replaced.content != first_to_replace.content:
                logger.warning(f"First line doesn't match for span {expected_span}")
                # TODO: Fail?

            first_to_replace.pre_lines = first_replaced.pre_lines
            first_to_replace.indentation = first_replaced.indentation

            expected_block.children[start_idx : end_idx + 1] = changed_block.children

    return WriteCodeResult(content=original_file_block.to_string())


def _find_start_and_end_index(block: CodeBlock, span: Span):
    start_index = None

    for i, child_block in enumerate(block.children):
        if child_block.start_line < span.start_line:
            continue

        if start_index is None and span.start_line >= child_block.start_line:
            start_index = i

        if span.end_line < child_block.start_line and i > 0:
            return start_index, i - 1

        if span.end_line <= child_block.end_line:
            return start_index, i

    return None, None


def find_by_path_recursive(codeblock, block_path: List[str]):
    found = codeblock.find_by_path(block_path)
    if not found and len(block_path) > 1:
        return find_by_path_recursive(codeblock, block_path[1:])
    return found
