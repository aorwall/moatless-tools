from typing import Optional, List

from tree_sitter import Node

from codeblocks.codeblocks import CodeBlockType, CodeBlock

from codeblocks.parser.parser import CodeParser, _find_type, find_nested_type

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
    "(",
    ")",
    ";"
]



class TypeScriptParser(CodeParser):

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

    def get_child_nodes(self, node: Node) -> List[Node]:
        if node.type == "program":
            for i, child in enumerate(node.children):
                if child.type == "package_declaration":
                    if len(node.children) > i+1:
                        return node.children[i+1:]
            return node.children

        nodes = []
        if node.type == "lexical_declaration":
            i, variable_declarator = _find_type(node, "variable_declarator")
            if variable_declarator and variable_declarator.children:
                arrow_func = find_nested_type(node, "arrow_function")
                if arrow_func:
                    delimiter, _ = _find_type(arrow_func, "=>")
                    return arrow_func.children[delimiter+1:] + self.get_next_siblings(variable_declarator.next_sibling)

                delimiter, _ = _find_type(variable_declarator, "=")
                if delimiter:
                    return variable_declarator.children[delimiter:] + node.children[i+1:]
                else:
                    end_delimiter, _ = _find_type(node, ";")
                    if end_delimiter:
                        return node.children[end_delimiter:]

        if node.type == "type_alias_declaration":
            delimiter, _ = _find_type(node, "=")
            if delimiter:
                return node.children[delimiter:]

        if node.type == "return_statement":
            if len(node.children) > 1 and node.children[1].type == "parenthesized_expression":
                return node.children[1].children + node.children[2:]
            else:
                return node.children[1:]

        if node.type in ["jsx_element", "call_expression"]:
            return node.children[1:]

        if node.type in ["parenthesized_expression", "jsx_expression"]:
            return node.children

        if node.type == "variable_declarator":
            delimiter, _ = _find_type(node, "=")
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
