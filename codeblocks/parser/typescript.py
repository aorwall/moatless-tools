from typing import Optional, List

from tree_sitter import Node

from codeblocks.codeblocks import CodeBlockType, CodeBlock
from codeblocks.parser.javascript import JavaScriptParser, block_delimiters

from codeblocks.parser.parser import CodeParser, _find_type, find_nested_type, find_type

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


class TypeScriptParser(JavaScriptParser):

    def __init__(self, language: str = "typescript"):
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
        return ["ERROR", "block", "statement_block", "object_type"]

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

    def find_first_child_(self, node: Node) -> Optional[Node]:
        if node.type in ["expression_statement"]:
            node = node.children[0]

        return super().find_first_child(node)
