from typing import Optional, List

from tree_sitter import Node

from code_blocks.codeblocks import CodeBlockType
from code_blocks.language.language import Language, find_block_node

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


class Java(Language):

    def get_block_type(self, node: Node) -> Optional[CodeBlockType]:
        if node.type == "program":
            return CodeBlockType.MODULE
        elif node.type in function_node_types:
            return CodeBlockType.FUNCTION
        elif node.type in class_node_types:
            return CodeBlockType.CLASS
        elif node.type in statement_node_types:
            return CodeBlockType.STATEMENT
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

    def comment(self, comment: str) -> str:
        return f"// {comment}"
