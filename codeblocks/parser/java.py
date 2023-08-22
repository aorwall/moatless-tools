from typing import Optional, List

from tree_sitter import Node

from codeblocks.codeblocks import CodeBlockType
from codeblocks.parser.parser import CodeParser, _find_type

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


class JavaParser(CodeParser):

    def __init__(self):
        super().__init__("java")

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
        elif node.type == "import_declaration":
            return CodeBlockType.IMPORT
        elif "comment" in node.type:
            if "..." in node.text.decode("utf8"):
                return CodeBlockType.COMMENTED_OUT_CODE
            else:
                return CodeBlockType.COMMENT
        else:
            return CodeBlockType.CODE

    def get_block_delimiter_types(self):
        return block_delimiters

    def get_child_nodes(self, node: Node) -> List[Node]:
        if node.type == "program":
            for i, child in enumerate(node.children):
                if child.type == "package_declaration":
                    if len(node.children) > i+1:
                        return node.children[i+1:]
            return node.children

        nodes = []
        if node.type in ["local_variable_declaration", "field_declaration", "constant_declaration"] and node.children \
                and any(child.children for child in node.children):
            i, variable_declarator = _find_type(node, "variable_declarator")
            if variable_declarator and variable_declarator.children:
                delimiter, _ = _find_type(variable_declarator, "=")
                if delimiter:
                    return variable_declarator.children[delimiter+1:] + node.children[i+1:]

        if node.type == "variable_declarator":
            delimiter, _ = _find_type(node, "=")
            if delimiter:
                return node.children[delimiter + 1:]

        if node.type == "switch_rule":
            delimiter, _ = _find_type(node, "->")
            if delimiter:
                return node.children[delimiter + 1:]

        block_node = self.find_block_node(node)
        if block_node:
            nodes.extend(block_node.children)

            next_sibling = block_node.next_sibling
            while next_sibling:
                nodes.append(next_sibling)
                next_sibling = next_sibling.next_sibling

        return nodes
