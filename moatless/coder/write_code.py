import logging
from typing import Optional, List

from pydantic import BaseModel

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.codeblocks import BlockSpan, CodeBlockTypeGroup
from moatless.codeblocks.module import Module
from moatless.coder.code_utils import create_instruction_code_block
from moatless.settings import Settings
from moatless.types import Span


logger = logging.getLogger(__name__)


class WriteCodeResult(BaseModel):
    content: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    change: Optional[str] = None


def respond_with_invalid_block(message: str, error_type: str = None):
    return WriteCodeResult(
        change=None,
        error_type=error_type,
        error=message,
    )


def write_code(
    original_file_block: Module,
    updated_file_block: Module,
    span: BlockSpan = None,
    action: Optional[str] = None,
) -> WriteCodeResult:

    if span:
        if span.is_partial:
            expected_block = original_file_block.find_by_path(span.parent_block_path)
        else:
            expected_block = original_file_block.find_first_by_span_id(span.span_id)

        if action == "remove":
            return remove_span(original_file_block, span)

        if action == "add":
            return add_span(original_file_block, updated_file_block, span)

        if expected_block.full_path() == original_file_block.full_path():
            changed_block = updated_file_block
        else:
            changed_block = updated_file_block.find_by_path(expected_block.full_path())

    else:
        expected_block = original_file_block
        changed_block = updated_file_block

    if not expected_block:
        raise ValueError(
            f"No block found on path {span.block_paths[0]} in span {span.span_id}"
        )

    if expected_block.path_string():
        block_label = f"{expected_block.path_string()}"
    else:
        block_label = "file"

    if span.is_partial:
        logger.info(
            f"Expected span ({span}) is starting or ending inside expected block ({expected_block.start_line}-{expected_block.end_line}, will set as 'update_span')"
        )
        action = "update_span"
        block_label = span.span_id

    if not changed_block:
        logger.warning(f"Couldn't find expected {block_label} in updated content.")

        return respond_with_invalid_block(
            f"Couldn't find the expected {block_label}. Return the full code block like: \n```python\n{create_instruction_code_block(expected_block, span)}\n```",
        )

    if Settings.coder.debug_mode:
        logger.debug(
            f"Updating code with block: \n{changed_block.to_tree(show_spans=True, show_tokens=True)}"
        )

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

    if action == "update":
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
        if no_of_changed_children < (no_of_expected_children / 2) and (
            (
                changed_block.children[0].type.group == CodeBlockTypeGroup.COMMENT
                and expected_block.children[0].type != CodeBlockType.COMMENT
            )
            or changed_block.children[-1].type.group == CodeBlockTypeGroup.COMMENT
            and expected_block.children[-1].type != CodeBlockType.COMMENT
        ):
            logger.warning(
                f"Suspected placeholders in code block because of size difference. Expected {no_of_expected_children} "
                f"child blocks, received {no_of_changed_children}. {updated_file_block.to_string()}"
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

    if action == "update_span":
        start_idx, end_idx = _find_start_and_end_index(expected_block, span)
        if start_idx is None:
            logger.warning(f"Couldn't find start index for span {span.span_id}")
            return respond_with_invalid_block(
                f"Couldn't find start line for {block_label}",
                error_type="incorrect_span",
            )

        # TODO: Verify indentation etc
        new_child_blocks = [
            child
            for child in changed_block.children
            if child.type not in [CodeBlockType.COMMENTED_OUT_CODE]
        ]

        first_replaced = expected_block.children[start_idx]
        first_to_replace = new_child_blocks[0]

        if first_replaced.content != first_to_replace.content:
            logger.warning(f"First line doesn't match for span {span.span_id}")
            # TODO: Fail?

        first_to_replace.pre_lines = first_replaced.pre_lines
        first_to_replace.indentation = first_replaced.indentation

        expected_block.children[start_idx : end_idx + 1] = new_child_blocks

    return WriteCodeResult(content=original_file_block.to_string())


def add_span(original_block: Module, updated_file_block: Module, span: BlockSpan):
    last_block = original_block.find_last_by_span_id(span.span_id)

    last_parent_structure = last_block.find_type_group_in_parents(
        CodeBlockTypeGroup.STRUCTURE
    )

    updated_parent_block = updated_file_block.find_by_path(
        last_parent_structure.full_path()
    )

    if not updated_parent_block:
        last_parent_structure = last_parent_structure.parent

        print(
            f"Couldn't find parent block with path {last_parent_structure.full_path()}, will try {last_parent_structure.full_path()[:-1]}"
        )
        updated_parent_block = updated_file_block.find_by_path(
            last_parent_structure.full_path()
        )

        if not updated_parent_block:
            logger.warning(
                f"Couldn't find parent block with path {last_parent_structure.full_path()}"
            )
            return respond_with_invalid_block(
                f"Couldn't find the parent block with path {last_parent_structure.full_path()}",
                error_type="parent_block_not_found",
            )

    if Settings.coder.debug_mode:
        logger.debug(
            f"Adding new block: \n{updated_parent_block.to_tree(show_spans=True, show_tokens=True)}"
        )

    updated_child_blocks = [
        child
        for child in updated_parent_block.children
        if child.type not in [CodeBlockType.COMMENTED_OUT_CODE]
    ]

    if not updated_child_blocks:
        logger.warning(
            f"Couldn't find expected block with path {span.parent_block_path}"
        )
        return respond_with_invalid_block(
            f"Couldn't find the expected block with path {span.parent_block_path}",
            error_type="block_not_found",
        )

    start_index, end_index = _find_start_and_end_index(last_parent_structure, span)
    end_index = min(end_index + 1, len(last_parent_structure.children))

    for changed_block in updated_child_blocks:
        last_parent_structure.children.insert(end_index, changed_block)
        end_index += 1

    return WriteCodeResult(content=original_block.to_string())


def remove_span(original_block: Module, span: BlockSpan):
    first_path_in_span = span.block_paths[0]
    last_path_in_span = span.block_paths[-1]

    if len(first_path_in_span) < len(last_path_in_span):
        # TODO: Fails as only complete spans can be removed ATM
        logger.warning("")
        pass

    logger.info(
        f"Will remove span {span.span_id} from block path {first_path_in_span} to {last_path_in_span}"
    )

    if len(first_path_in_span) > 1:
        parent_block = original_block.find_by_path(first_path_in_span[:-1])
    else:
        parent_block = original_block

    remaining_children = []
    for child in parent_block.children:
        if not child.belongs_to_span or child.belongs_to_span.span_id != span.span_id:
            remaining_children.append(child)

    parent_block.children = remaining_children

    return WriteCodeResult(content=original_block.to_string())


def _find_start_and_end_index(block: CodeBlock, span: BlockSpan):
    start_index = None
    end_index = None

    in_span = False

    for i, child_block in enumerate(block.children):
        if not in_span and (
            not child_block.belongs_to_span
            or child_block.belongs_to_span.span_id != span.span_id
        ):
            continue

        if (
            not in_span
            and child_block.belongs_to_span
            and child_block.belongs_to_span.span_id == span.span_id
        ):
            in_span = True
            start_index = i

        if in_span and (
            not child_block.belongs_to_span
            or child_block.belongs_to_span.span_id != span.span_id
        ):
            return start_index, end_index
        elif in_span:
            end_index = i

    return start_index, end_index


def find_by_path_recursive(codeblock, block_path: List[str]):
    found = codeblock.find_by_path(block_path)
    if not found and len(block_path) > 1:
        return find_by_path_recursive(codeblock, block_path[1:])
    return found
