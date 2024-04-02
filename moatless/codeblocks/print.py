from typing import List, Tuple

from moatless.codeblocks import CodeBlock, CodeBlockType


def _create_placeholder(codeblock: CodeBlock) -> str:
    if codeblock.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
        return codeblock.create_commented_out_block("... other code").to_string()
    return ""


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


def print_by_block_paths(codeblock: CodeBlock, block_paths: List[List[str]]) -> str:
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
        if not block_path and child.type not in [CodeBlockType.CLASS, CodeBlockType.FUNCTION,
                                                 CodeBlockType.CONSTRUCTOR]:
            if has_outcommented_code and child.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
                contents += child.create_commented_out_block("... other code").to_string()
                has_outcommented_code = False
                continue
            contents += self._to_context_string(codeblock=child, block_path=block_path)
        elif block_path and block_path[0] == child.identifier:
            if has_outcommented_code and child.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
                contents += child.create_commented_out_block("... other code").to_string()
                has_outcommented_code = False
            contents += self._to_context_string(codeblock=child, block_path=block_path[1:])
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
