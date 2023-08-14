from typing import List, Tuple, Optional

import tree_sitter_languages
from pydantic import BaseModel
from tree_sitter import Node


class CodeChunk(BaseModel):
    code: str
    type: str
    children: List["CodeChunk"] = []


def find_node(node: Node, node_type: str):
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def get_python_block_end(node: Node) -> Tuple[int, Optional[Node]]:
    block_node_types = [
        "function_definition", "class_definition", "if_statement",
        "for_statement", "while_statement", "try_statement", "with_statement"
    ]

    if node.type in block_node_types:
        block_node = find_node(node, "block")
        return block_node.start_byte if block_node else None, block_node

    return -1, None  # default return value if no match


def get_js_ts_block_end(node: Node) -> Tuple[int, Optional[Node]]:
    block_node_types = [
        "function_declaration", "function", "class_declaration", "if_statement",
        "for_statement", "while_statement", "do_statement", "try_statement", "switch_statement",
        "arrow_function"  # For TypeScript and modern JavaScript
    ]

    if node.type in block_node_types:
        block_node = find_node(node, "block")
        return block_node.start_byte if block_node else None, block_node

    return -1, None  # default return value if no match


def get_definition_end(node: Node) -> Tuple[int, Optional[Node]]:
    block_node_types = [
        "method_declaration",
        "constructor_declaration",
        "static_initializer",
        "instance_initializer",
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "synchronized_statement",
        "try_statement"
    ]

    switch_node_types = [
        "switch_statement"
    ]

    if node.type == "program":
        return node.start_byte, node
    elif node.type in block_node_types:
        block_node = find_node(node, "block")
        return block_node.start_byte if block_node else None, block_node
    elif node.type == "class_declaration":
        class_body_node = find_node(node, "class_body")
        return class_body_node.start_byte if class_body_node else None, class_body_node
    elif node.type in switch_node_types:
        switch_block_node = find_node(node, "switch_block")
        return switch_block_node.start_byte if switch_block_node else None, switch_block_node

    return node.end_byte, None


def parse_code(contents: str, node: Node):
    if node.prev_sibling:
        start_byte = node.prev_sibling.end_byte
    else:
        start_byte = node.start_byte

    end_byte, block = get_definition_end(node)

    children = []
    if block is None:
        code = contents[start_byte:end_byte]
    else:
        code = contents[start_byte:end_byte]
        for child in block.children:
            children.append(parse_code(contents, child))

    return CodeChunk(
        code=code,
        type=node.type,
        children=children
    )


def equal_nodes(original_nodes: List[Node], updated_nodes: List[Node], check_children = True):
    if len(original_nodes) != len(updated_nodes) or len(original_nodes) == 0:
        return False

    for i in range(0, len(original_nodes)):
        if original_nodes[i].children or updated_nodes[i].children:
            if check_children and not equal_nodes(original_nodes[i].children, updated_nodes[i].children, False):
                return False
        elif original_nodes[i].text.decode("utf8") != updated_nodes[i].text.decode("utf8"):
            return False
    return True


def find_next_equal_node(original_lines: list[Node], updated_lines: list[Node], start_original: int, start_updated: int):
    i = start_original

    while i < len(original_lines):
        j = start_updated
        while j < len(updated_lines):
            # TODO: Just check the first line...
            if original_lines[i].text.decode("utf8").split("\n")[0] == updated_lines[j].text.decode("utf8").split("\n")[0]:
                print("equal: ", original_lines[i].text.decode("utf8").split("\n")[0], "==",
                      updated_lines[j].text.decode("utf8").split("\n")[0])
                return i, j
            #elif FA equal_nodes(original_lines[i].children, updated_lines[j].children) and False: # FIXME
            #    return i, j
            j += 1
        i += 1

    return -1, -1


def merge_nodes(original_nodes: List[Node], updated_nodes: List[Node], original_code: str, updated_code: str, original_start: int = 0, updated_start: int = 0):
    merged_code = ""

    i = 0
    j = 0
    while i < len(original_nodes) and j < len(updated_nodes):
        if original_nodes[i].text.decode("utf8") == updated_nodes[j].text.decode("utf8"):
            if original_nodes[i].prev_sibling:
                start_byte = original_nodes[i].prev_sibling.end_byte
            else:
                start_byte = original_start
            merged_code += original_code[start_byte:original_nodes[i].end_byte]
            original_start = original_nodes[i].end_byte
            i += 1
            j += 1
        elif updated_nodes[j].type == "line_comment":
            j += 1
            orig_next, update_next = find_next_equal_node(original_nodes, updated_nodes, i, j)

            if orig_next == -1 or update_next == -1:
                print("Could not find next equal node")
                # TODO?
                return

            print("next lines")
            print("org " + str(i) + ": " + original_nodes[orig_next].text.decode("utf8"))
            print("upd " + str(j) + ": " + updated_nodes[update_next].text.decode("utf8"))

            if original_nodes[i].prev_sibling:
                start_byte = original_nodes[i].prev_sibling.end_byte
            else:
                start_byte = original_start

            merged_code += original_code[start_byte:original_nodes[orig_next].prev_sibling.end_byte]

            print("merged_code: ```", merged_code, "```")

            original_start = original_nodes[orig_next].start_byte

            i = orig_next
            if update_next > j:
                if original_nodes[i].prev_sibling:
                    start_byte = updated_nodes[j].prev_sibling.end_byte
                else:
                    start_byte = updated_nodes[j].start_byte
                merged_code += updated_code[start_byte : updated_nodes[update_next].prev_sibling.end_byte]
                print("merged_code: ", merged_code)

            j = update_next
        elif not updated_nodes[j].children and updated_nodes[j].text.decode("utf8") != original_nodes[i].text.decode("utf8"):
            if updated_nodes[j].prev_sibling:
                start_byte = updated_nodes[j].prev_sibling.end_byte
            else:
                start_byte = updated_nodes[j].start_byte
            merged_code += updated_code[start_byte: updated_nodes[j].end_byte]
            j += 1
            i += 1
        else:
            merged_code += merge_nodes(original_nodes[i].children, updated_nodes[j].children, original_code, updated_code, original_start)
            i += 1
            j += 1

    return merged_code

def merge_chunks(original_chunks: List[CodeChunk], updated_chunks: List[CodeChunk]):
    merged_code = ""
    merged_chunks = []

    i = 0
    j = 0
    while i < len(original_chunks) and j < len(updated_chunks):
        if original_chunks[i] == updated_chunks[j]:
            merged_chunks.append(original_chunks[i])
            i += 1
            j += 1
        elif updated_chunks[j].type == "line_comment":
            j += 1
            orig_next, update_next = find_next_equal_chunk(original_chunks, updated_chunks, i, j)

            if orig_next == -1 or update_next == -1:
                print("Could not find next equal node")
                # TODO?
                return

            merged_chunks.extend(original_chunks[i:orig_next])

            i = orig_next
            if update_next > j:
                merged_chunks.extend(updated_chunks[j:update_next])

            j = update_next
        elif not updated_chunks[j].children and updated_chunks[j] != original_chunks[i]:
            merged_chunks.append(updated_chunks[j])
            j += 1
            i += 1
        else:
            merged_chunks.extend(merge_chunks(original_chunks[i].children, updated_chunks[j].children))
            i += 1
            j += 1

    return merged_code

def fix_incomplete_response(original_content: str, updated_content: str, language: str):
    try:
        parser = tree_sitter_languages.get_parser(language)
    except Exception as e:
        print(
            f"Could not get parser for language {language}. Check "
            "https://github.com/grantjenks/py-tree-sitter-languages#license "
            "for a list of valid languages."
        )
        raise e

    original_tree = parser.parse(bytes(original_content, "utf8"))
    updated_tree = parser.parse(bytes(updated_content, "utf8"))

    if original_tree == updated_tree:
        return original_content

    if (
            not original_tree.root_node.children
            or original_tree.root_node.children[0].type == "ERROR"
    ):
        raise Exception("Original code snippet is invalid")

    if (
            not updated_tree.root_node.children
            or updated_tree.root_node.children[0].type == "ERROR"
    ):
        raise Exception("Updated code snippet is invalid")

    original_chunks = parse_code(original_content, original_tree.root_node)
    updated_chunks = parse_code(updated_content, updated_tree.root_node)

    return merge_nodes(original_tree.root_node.children, updated_tree.root_node.children, original_content, updated_content)
