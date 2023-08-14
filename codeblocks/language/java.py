from typing import Optional, List

from tree_sitter import Node

from codeblocks.codeblocks import CodeBlockType

class_node_types = [
    "annotation_type_declaration",
    "class_declaration",
    "enum_declaration",
    "interface_declaration",
    "record_declaration"
]

function_node_types = [
    "method_declaration",
    "constructor_declaration"
]

statement_node_types = [
    "static_initializer",
    "instance_initializer",
    "if_statement",
    "for_statement",
    "enhanced_for_statement",
    "while_statement",
    "do_statement",
    "synchronized_statement",
    "try_statement"
    "switch_expression"
]

block_delimiters = [
    "{",
    "}"
]


def get_java_block_type(node: Node) -> Optional[CodeBlockType]:
    if node.type == "program":
        return CodeBlockType.PROGRAM
    elif node.type in function_node_types:
        return CodeBlockType.FUNCTION
    elif node.type in class_node_types:
        return CodeBlockType.CLASS
    elif node.type in statement_node_types:
        return CodeBlockType.STATEMENT
    elif node.type in block_delimiters:
        return CodeBlockType.BLOCK_DELIMITER
    elif "comment" in node.type:
        return CodeBlockType.COMMENT
    else:
        return CodeBlockType.NONE


def find_block_node(node: Node):
    for child in node.children:
        if child.type.endswith("block") or child.type.endswith("body"):
            return child
    return None


def get_java_blocks(node: Node) -> List[Node]:
    if node.type == "program":
        for i, child in enumerate(node.children):
            if child.type == "package_declaration":
                if len(node.children) > i+1:
                    return node.children[i+1:]
        return node.children

    nodes = []
    block_node = find_block_node(node)
    if block_node:
        nodes.extend(block_node.children)

        next_sibling = block_node.next_sibling
        while next_sibling:
            nodes.append(next_sibling)
            next_sibling = next_sibling.next_sibling

    return nodes
