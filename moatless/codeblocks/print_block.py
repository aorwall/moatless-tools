from enum import Enum
from typing import List, Tuple

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.codeblocks import PathTree, NON_CODE_BLOCKS
from moatless.types import BlockPath

class BlockMarker(Enum):
    TAG = 1
    COMMENT = 2


def _create_placeholder(codeblock: CodeBlock) -> str:
    if codeblock.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
        return codeblock.create_commented_out_block("... other code").to_string()
    return ""


def _print_content(codeblock: CodeBlock, block_marker: BlockMarker = None) -> str:
    contents = ""

    if codeblock.pre_lines:
        contents += "\n" * (codeblock.pre_lines - 1)

        if codeblock.type in [CodeBlockType.FUNCTION, CodeBlockType.CLASS, CodeBlockType.CONSTRUCTOR]:
            if block_marker == BlockMarker.COMMENT:
                comment = codeblock.create_comment("block_path")
                contents += f"\n{codeblock.indentation}{comment}: {codeblock.path_string()}"
            elif block_marker == BlockMarker.TAG:
                contents += f"\n</block>\n{codeblock.indentation}\n<block id='{codeblock.path_string()}'>"

        for line in codeblock.content_lines:
            if line:
                contents += "\n" + codeblock.indentation + line
            else:
                contents += "\n"
    else:
        contents += codeblock.pre_code + codeblock.content

    return contents


def print_block(
        codeblock: CodeBlock,
        path_tree: PathTree = None,
        show_types: List[CodeBlockType] = None,
        block_marker: BlockMarker = None) -> str:
    content = ""
    if block_marker == BlockMarker.TAG:
        content += f"<block id='start'>\n\n"
    elif block_marker == BlockMarker.COMMENT:
        content += codeblock.create_comment_block("start").to_string()

    content += _print(codeblock, path_tree=path_tree, show_types=show_types, block_marker=block_marker)

    if block_marker == BlockMarker.TAG:
        content += "\n</block>"

    return content


def print_by_block_paths(codeblock: CodeBlock, block_paths: List[BlockPath], block_marker: BlockMarker = None) -> str:
    tree = PathTree()
    tree.extend_tree(block_paths)
    return print_block(codeblock, path_tree=tree, block_marker=block_marker)


def _print(
        codeblock: CodeBlock,
        path_tree: PathTree = None,
        show_everything: bool = False,
        show_types: List[CodeBlockType] = [],
        block_marker: BlockMarker = None) -> str:
    contents = _print_content(codeblock, block_marker=block_marker)

    has_outcommented_code = False
    for i, child in enumerate(codeblock.children):
        if show_everything:
            contents += _print(child, block_marker=block_marker, show_everything=True)
            continue

        child_tree = path_tree.child_tree(child.identifier) if path_tree else None
        if child_tree and child_tree.show:
            if has_outcommented_code and child.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
                contents += child.create_commented_out_block("... other code").to_string()
            contents += _print(codeblock=child, block_marker=block_marker, show_everything=True)
            has_outcommented_code = False
        elif child_tree or child.type in show_types:
            contents += _print(codeblock=child, path_tree=child_tree, show_types=show_types, block_marker=block_marker)
            has_outcommented_code = False
        elif child.type not in NON_CODE_BLOCKS:
            has_outcommented_code = True

    if codeblock.parent and has_outcommented_code and child.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
        contents += child.create_commented_out_block("... other code").to_string()

    return contents


def print_by_block_path(codeblock: CodeBlock, block_path: List[str] = None) -> str:
    contents = ""

    if codeblock.pre_lines:
        contents += "\n" * (codeblock.pre_lines - 1)
        for line in codeblock.content_lines:
            if line:
                contents += "\n" + codeblock.indentation + line
            else:
                contents += "\n"
    else:
        contents += codeblock.pre_code + codeblock.content

    has_outcommented_code = False
    for i, child in enumerate(codeblock.children):
        if not block_path or block_path[-1] == child.identifier:
            if has_outcommented_code and child.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
                contents += child.create_commented_out_block("... other code").to_string()
                has_outcommented_code = False
            contents += print_by_block_path(codeblock=child)
        elif len(block_path) > 1 and block_path[0] == child.identifier:
            contents += print_by_block_path(codeblock=child, block_path=block_path[1:])
        else:
            has_outcommented_code = True

    return contents


def _contains_block_paths(codeblock: CodeBlock, block_paths: List[List[str]]):
    return [block_path for block_path in block_paths
            if block_path[:len(codeblock.full_path())] == codeblock.full_path()]


def print_by_line_numbers(codeblock: CodeBlock, line_numbers: List[Tuple[int, int]]) -> str:
    contents = ""

    if codeblock.pre_lines:
        contents += "\n" * (codeblock.pre_lines - 1)
        for line in codeblock.content_lines:
            if line:
                contents += "\n" + codeblock.indentation + line
            else:
                contents += "\n"
    else:
        contents += codeblock.pre_code + codeblock.content

    has_outcommented_code = False

    for i, child in enumerate(codeblock.children):
        if child.type in [CodeBlockType.CLASS, CodeBlockType.FUNCTION, CodeBlockType.CONSTRUCTOR]:
            if has_outcommented_code and child.type not in [CodeBlockType.COMMENT,
                                                            CodeBlockType.COMMENTED_OUT_CODE]:
                contents += child.create_commented_out_block("... outcommented code").to_string()

            contents += print_by_line_numbers(child, line_numbers)
        elif _is_block_within_line_numbers(child, line_numbers):
            if has_outcommented_code and child.type not in [CodeBlockType.COMMENT,
                                                            CodeBlockType.COMMENTED_OUT_CODE]:
                contents += child.create_commented_out_block("... ooutcommented code").to_string()
            contents += print_by_line_numbers(child, line_numbers)
        else:
            has_outcommented_code = True

    if has_outcommented_code and child.type not in [CodeBlockType.COMMENT,
                                                    CodeBlockType.COMMENTED_OUT_CODE]:
        contents += child.create_commented_out_block("... outcommented code").to_string()

    return contents


def _is_block_within_line_numbers( codeblock: CodeBlock, line_numbers: List[Tuple[int, int]]) -> bool:
    for start_line, end_line in line_numbers:
        if start_line <= codeblock.start_line and end_line >= codeblock.end_line:
            return True
    return False


def print_with_blockpath_comments(codeblock: CodeBlock, field_name: str = "block_path") -> str:
    return _print_with_blockpath_comments(codeblock, field_name=field_name) + "\n</block>"


def _print_with_blockpath_comments(codeblock: CodeBlock, field_name: str = "block_path") -> str:
    contents = ""

    if not codeblock.parent:
        contents += f"<block id='start'>\n\n"

    if codeblock.pre_lines:
        contents += "\n" * (codeblock.pre_lines - 1)

        if codeblock.type in [CodeBlockType.FUNCTION, CodeBlockType.CLASS, CodeBlockType.CONSTRUCTOR]:
            contents += f"\n</block>\n{codeblock.indentation}\n<block id='{codeblock.path_string()}'>"

        for line in codeblock.content_lines:
            if line:
                contents += "\n" + codeblock.indentation + line
            else:
                contents += "\n"
    else:
        contents += codeblock.pre_code + codeblock.content

    for i, child in enumerate(codeblock.children):
        contents += _print_with_blockpath_comments(child, field_name=field_name)

    return contents
