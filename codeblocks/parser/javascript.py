from typing import Optional, Tuple

from tree_sitter import Node

from codeblocks.codeblocks import CodeBlockType
from codeblocks.parser.parser import CodeParser, find_nested_type, find_type

class_node_types = [
    "class_declaration",
    "abstract_class_declaration",
    "enum_declaration",
    "interface_declaration"
]

function_node_types = [
    "method_definition",
    "function_declaration",
    "abstract_method_signature"
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

    def get_block_definition(self, node: Node) -> Tuple[CodeBlockType, Optional[Node]]:
        if node.type == "program":
            return CodeBlockType.MODULE, node.children[0]

        if node.type in function_node_types:
            return CodeBlockType.FUNCTION, find_nested_type(node, "{")

        if node.type in class_node_types:
            return CodeBlockType.CLASS, find_nested_type(node, "{")

        if node.type in block_delimiters:
            return CodeBlockType.BLOCK_DELIMITER, None

        if node.type == "import_statement":
            return CodeBlockType.IMPORT, None

        if "comment" in node.type:
            if "..." in node.text.decode("utf8"):
                return CodeBlockType.COMMENTED_OUT_CODE, None
            else:
                return CodeBlockType.COMMENT, None

        if node.type in statement_node_types:
            block_type = CodeBlockType.STATEMENT
        else:
            block_type = CodeBlockType.CODE

        if node.type in ["binary_expression"]:
            return block_type, None

        if node.type in ["expression_statement"]:
            node = node.children[0]

        if node.type in ["variable_declarator", "variable_declaration", "lexical_declaration", "call_expression",
                         "new_expression", "type_alias_declaration"]:
            arrow_func = self.find_arrow_func(node)
            if arrow_func:
                if node.type == "lexical_declaration":
                    type_annotation = find_nested_type(node, "type_annotation")
                    if type_annotation and type_annotation.start_byte < arrow_func.start_byte:
                        return CodeBlockType.CLASS, arrow_func
                    else:
                        return CodeBlockType.FUNCTION, arrow_func
                else:
                    return block_type, arrow_func

            delimiter = find_nested_type(node, "=")
            if delimiter:
                return block_type, delimiter
            else:
                end_delimiter = find_nested_type(node, ";")
                if end_delimiter:
                    return block_type, end_delimiter

        if node.type in ["object"]:
            return block_type, find_type(node, ["{"]).next_sibling

        block = find_type(node,
                          ["class_body", "enum_body", "statement_block", "object_type", "object_pattern", "switch_body",
                           "jsx_expression"])
        if block:
            delimiter = find_type(block, ["{"])
            if delimiter:
                return block_type, delimiter
            return block_type, block.children[0]

        call_func = find_type(node, ["call_expression"])
        if call_func:
            arrow_func = self.find_arrow_func(call_func);
            if arrow_func:
                return block_type, arrow_func

        parenthesized_expression = find_type(node, ["parenthesized_expression"])
        if parenthesized_expression:
            delimiter = find_type(parenthesized_expression, ["("])
            if delimiter:
                return block_type, delimiter
            return block_type, parenthesized_expression.children[0]

        if node.type in ["switch_case", "switch_default"]:
            delimiter = find_type(node, [":"])
            if delimiter:
                return block_type, delimiter

        if node.type in ["statement_block"]:
            return block_type, node.children[0]

        if node.type in ["return_statement", "jsx_element", "call_expression", "assignment_expression"]:
            return block_type, node.children[1]

        return CodeBlockType.CODE, None
