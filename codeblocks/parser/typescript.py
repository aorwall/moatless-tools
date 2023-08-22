from typing import Optional, List

from tree_sitter import Node

from codeblocks.codeblocks import CodeBlockType, CodeBlock

from codeblocks.parser.parser import CodeParser, _find_type, find_block_node

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
    ")"
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
        else:
            return CodeBlockType.CODE

    def get_compound_node_types(self):
        return ["program"] + class_node_types + function_node_types + statement_node_types + ["jsx_element"]

    def get_child_node_block_types(self):
        return ["ERROR", "block"]

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
        if node.type == "lexical_declaration":
            i, variable_declarator = _find_type(node, "variable_declarator")
            if variable_declarator and variable_declarator.children:
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
            return node.children[1:]

        if node.type in ["jsx_element"]:
            return node.children[1:]

        if node.type in ["parenthesized_expression"]:
            return node.children

        if node.type == "variable_declarator":
            delimiter, _ = _find_type(node, "=")
            if delimiter:
                return node.children[delimiter + 1:]

        block_node = find_block_node(node)
        if block_node:
            nodes.extend(block_node.children)

            next_sibling = block_node.next_sibling
            while next_sibling:
                nodes.append(next_sibling)
                next_sibling = next_sibling.next_sibling

        return nodes
