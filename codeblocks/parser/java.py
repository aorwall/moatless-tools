from typing import Optional, Tuple

from tree_sitter import Node

from codeblocks.codeblocks import CodeBlockType
from codeblocks.parser.parser import CodeParser, find_type, find_nested_type

class_node_types = [
    "annotation_type_declaration",
    "class_declaration",
    "enum_declaration",
    "interface_declaration",
    "record_declaration"
]

function_node_types = [
    "method_declaration",
    "constructor_declaration",
    "compact_constructor_declaration"
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
    "try_statement",
    "try_with_resources_statement",
    "switch_expression"
]

block_delimiters = [
    "{",
    "}"
]


class JavaParser(CodeParser):

    def __init__(self):
        super().__init__("java")

    def get_block_definition(self, node: Node) -> Tuple[CodeBlockType, Optional[Node]]:
        if node.type == "program":
            # TODO: module_declaration?
            package_declaration = find_type(node, ["package_declaration"])
            if package_declaration:
                return CodeBlockType.MODULE, package_declaration.next_sibling
            return CodeBlockType.MODULE, node.children[0]

        if node.type in function_node_types:
            block_delimiter = find_nested_type(node, "{")
            return CodeBlockType.FUNCTION, block_delimiter

        if node.type in class_node_types:
            block_delimiter = find_nested_type(node, "{")
            return CodeBlockType.CLASS, block_delimiter

        if node.type in statement_node_types:
            block_delimiter = find_nested_type(node, "{")
            return CodeBlockType.STATEMENT, block_delimiter

        if node.type in block_delimiters:
            return CodeBlockType.BLOCK_DELIMITER, None

        if node.type == "import_declaration":
            return CodeBlockType.IMPORT, None

        if "comment" in node.type:
            if "..." in node.text.decode("utf8"):
                return CodeBlockType.COMMENTED_OUT_CODE, None
            else:
                return CodeBlockType.COMMENT, None

        if node.type in ["local_variable_declaration", "variable_declarator"]:
            identifier = find_nested_type(node, "identifier")
            return CodeBlockType.CODE, identifier.next_sibling

        if node.type == "expression_statement":
            return CodeBlockType.CODE, find_nested_type(node, "=")

        if node.type == "switch_rule":
            delimiter = find_type(node, ["->"])
            return CodeBlockType.CODE, delimiter

        if node.type == "block":
            block = node
        else:
            block = find_type(node, ["block"])
        if block:
            return CodeBlockType.CODE, block.children[0]

        if node.type == "ERROR":
            return CodeBlockType.ERROR, None

        return CodeBlockType.CODE, None
