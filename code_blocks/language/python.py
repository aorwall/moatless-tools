from typing import Optional, List

from tree_sitter import Node

from code_blocks import CodeBlockType
from code_blocks.language.language import Language

block_node_types = [
    "function_definition", "class_definition", "if_statement",
    "for_statement", "while_statement", "try_statement", "with_statement"
]

block_delimiters = [
    ":"
]


def _find_delimiter_index(node: Node):
    for i, child in enumerate(node.children):
        if child.type == ":":
            return i
    return -1


class Python(Language):

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
            if "..." in node.text.decode("utf8"):
                return CodeBlockType.COMMENTED_OUT_CODE
            else:
                return CodeBlockType.COMMENT
        else:
            return CodeBlockType.CODE

    def get_child_blocks(self, node: Node) -> List[Node]:
        if node.type == "module":
            return node.children

        if node.type == "decorated_definition" and len(node.children) > 1:
            node = node.children[-1]

        nodes = []
        delimiter_index = _find_delimiter_index(node)
        if delimiter_index != -1:
            for child in node.children[delimiter_index:]:
                if child.type == "block":
                    nodes.extend(child.children)
                else:
                    nodes.append(child)
        return nodes

    def comment(self, comment: str) -> str:
        return f"# {comment}"
