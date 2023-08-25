from dataclasses import dataclass
from typing import Optional, List

from tree_sitter import Node

from codeblocks.codeblocks import CodeBlockType, CodeBlock

from codeblocks.parser.parser import CodeParser, find_nested_type, find_type

class_node_types = [
    "class_declaration",
    "abstract_class_declaration",
    "enum_declaration",
    "interface_declaration"
]

function_node_types = [
    "method_definition",
    "function_declaration"
]

statement_node_types = [
    "if_statement",
    "for_statement",
    "try_statement",
    "return_statement"
]

block_delimiters = [
    "{",
    "}",
    ";",
    "(",
    ")"
]


class JavaScriptParser(CodeParser):

    def __init__(self, language: str = "javascript"):
        super().__init__(language)

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

        elif node.type == "import_statement":
            return CodeBlockType.IMPORT
        elif "comment" in node.type:
            if "..." in node.text.decode("utf8"):
                return CodeBlockType.COMMENTED_OUT_CODE
            else:
                return CodeBlockType.COMMENT
        elif node.type == "lexical_declaration":
            arrow_func = find_nested_type(node, "arrow_function")
            if arrow_func:
                type_annotation = find_nested_type(node, "type_annotation")
                if type_annotation and type_annotation.start_byte < arrow_func.start_byte:
                    return CodeBlockType.CLASS
                else:
                    return CodeBlockType.FUNCTION

        return CodeBlockType.CODE

    def get_compound_node_types(self):
        return ["program"] + class_node_types + function_node_types + statement_node_types + ["jsx_element"]

    def get_child_node_block_types(self):
        return ["ERROR", "block", "statement_block", "object", "object_type", "expression_statement"]

    def get_block_delimiter_types(self):
        return block_delimiters

    def get_next_siblings(self, next_sibling: Node):
        nodes = []
        while next_sibling:
            nodes.append(next_sibling)
            next_sibling = next_sibling.next_sibling
        return nodes

    def find_block_node(self, node: Node):
        for child in node.children:
            if  child.type.endswith("body") or child.type.endswith("block") or child.type == "object_type":
                return child
        return None

    def find_arrow_func(self, node: Node):
        arrow_func = find_nested_type(node, "arrow_function")
        if arrow_func:
            arrow = find_nested_type(arrow_func, "=>")
            if arrow:
                block_delimiter = find_nested_type(arrow.next_sibling, "{")
                if block_delimiter:
                    return block_delimiter
                else:
                    return arrow.next_sibling
        return None

    def find_first_child(self, node: Node) -> Optional[Node]:
        if node.type in ["binary_expression"]:
            return None

        if node.type in ["expression_statement"]:
            node = node.children[0]

        if node.type in ["variable_declarator", "variable_declaration", "lexical_declaration", "call_expression", "new_expression", "type_alias_declaration"]:
            arrow_func = self.find_arrow_func(node)
            if arrow_func:
                return arrow_func

            delimiter = find_nested_type(node, "=")
            if delimiter:
                return delimiter
            else:
                end_delimiter = find_nested_type(node, ";")
                if end_delimiter:
                    return end_delimiter

        block = find_type(node, ["class_body", "enum_body", "statement_block", "object", "object_type", "object_pattern", "switch_body", "jsx_expression"])
        if block:
            delimiter = find_type(block, ["{"])
            if delimiter:
                return delimiter
            return block.children[0]

        call_func = find_type(node, ["call_expression"])
        if call_func:
            arrow_func = self.find_arrow_func(call_func);
            if arrow_func:
                return arrow_func

        parenthesized_expression = find_type(node, ["parenthesized_expression"])
        if parenthesized_expression:
            delimiter = find_type(parenthesized_expression, ["("])
            if delimiter:
                return delimiter
            return parenthesized_expression.children[0]

        if node.type in ["switch_case", "switch_default"]:
            delimiter = find_type(node, [":"])
            if delimiter:
                return delimiter

        if node.type in ["program", "statement_block"]:
            return node.children[0]

        if node.type in ["return_statement", "jsx_element", "call_expression", "assignment_expression"]:
            return node.children[1]


        return None

    def get_child_nodes(self, node: Node) -> List[Node]:
        if node.type == "program":
            return node.children
