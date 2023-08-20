from typing import Optional, List

from tree_sitter import Node

from code_blocks import CodeBlockType, CodeBlock
from code_blocks.parser.parser import CodeParser, COMMENTED_OUT_CODE_KEYWORDS

block_node_types = [
    "function_definition", "class_definition", "if_statement",
    "for_statement", "while_statement", "try_statement", "with_statement",
    "expression_statement", "else_clause", "elif_clause"
]

child_block_types = ["ERROR", "block", "expression_statement"]

block_delimiters = [
    ":"
]

def _find_type(node: Node, type: str):
    for i, child in enumerate(node.children):
        if child.type == type:
            return i
    return None


def _find_delimiter_index(node: Node):
    for i, child in enumerate(node.children):
        if child.type == ":":
            return i
    return -1


class PythonParser(CodeParser):

    def __init__(self):
        super().__init__("python")

    def get_child_node_block_types(self):
        return child_block_types

    def get_block_node_types(self):
        return block_node_types

    def get_block_type(self, node: Node) -> Optional[CodeBlockType]:
        if node.type == "decorated_definition" and len(node.children) > 1:
            node = node.children[-1]
        if node.type == "module":
            return CodeBlockType.MODULE
        elif node.type == "function_definition":
            return CodeBlockType.FUNCTION
        elif node.type == "class_definition":
            return CodeBlockType.CLASS
        elif node.type in block_delimiters:
            return CodeBlockType.BLOCK_DELIMITER
        elif "comment" in node.type:
            comment = node.text.decode("utf8").strip()
            if comment.startswith("# ...") or any(keyword in comment.lower() for keyword in COMMENTED_OUT_CODE_KEYWORDS):
                return CodeBlockType.COMMENTED_OUT_CODE
            else:
                return CodeBlockType.COMMENT
        else:
            return CodeBlockType.CODE

    def get_child_nodes(self, node: Node) -> List[Node]:
        if node.type == "module":
            return node.children

        if node.type == "decorated_definition" and len(node.children) > 1:
            node = node.children[-1]

        if node.type == "assignment":
            delimiter = _find_type(node, "=")
            if delimiter:
                return node.children[delimiter + 1:]

        delimiter_index = _find_delimiter_index(node)
        if delimiter_index != -1:
            return node.children[delimiter_index:]
        else:
            return []
