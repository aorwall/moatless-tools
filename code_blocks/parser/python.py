from typing import Optional, List

from tree_sitter import Node

from code_blocks import CodeBlockType, CodeBlock
from code_blocks.parser.parser import CodeParser

block_node_types = [
    "function_definition", "class_definition", "if_statement",
    "for_statement", "while_statement", "try_statement", "with_statement",
    "expression_statement", "else_clause", "elif_clause"
]

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

    def __init__(self, gpt_mode: bool = False):
        super().__init__("python")
        self.gpt_mode = gpt_mode

    def _find_end_index(self, node: Node):
        if self.gpt_mode:
            for i, child in enumerate(node.children):
                if child.type in ["else_clause", "elif_clause"]:
                    return i
        return len(node.children)

    def get_block_node_types(self):
        return block_node_types

    def parse_code(self, content_bytes: bytes, node: Node, start_byte: int = 0) -> List[CodeBlock]:
        pre_code = content_bytes[start_byte:node.start_byte].decode(self.encoding)

        block_type = self.get_block_type(node)

        # if pre code have other chars than " " and "\n" then it is an error
        if pre_code.strip():
            block_type = CodeBlockType.ERROR

        child_nodes = self.get_child_blocks(node)

        children = []

        first_node = child_nodes[0] if child_nodes else None
        if first_node:
            end_byte = first_node.start_byte
            end_line = first_node.start_point[0] #TODO?
        else:
            end_byte = node.end_byte
            end_line = node.end_point[0]

        code = content_bytes[node.start_byte:end_byte].decode(self.encoding)

        for child in child_nodes:
            if child.type in ["ERROR", "block", "expression_statement"]:
                child_children = []
                if child.children:
                    for child_child in child.children:
                        child_children.extend(self.parse_code(content_bytes, child_child, start_byte=end_byte))
                        end_byte = child_child.end_byte
                if self._is_error(child):
                    children.append(CodeBlock(
                        type=CodeBlockType.ERROR,
                        tree_sitter_type=node.type,
                        start_line=node.start_point[0],
                        end_line=end_line,
                        pre_code=pre_code,
                        content=code,
                        children=child_children
                    ))
                else:
                    children.extend(child_children)
            else:
                children.extend(self.parse_code(content_bytes, child, start_byte=end_byte))
                end_byte = child.end_byte

        if not node.parent and child_nodes and child_nodes[-1].end_byte < node.end_byte:
            children.append(CodeBlock(
                type=CodeBlockType.SPACE,
                pre_code=content_bytes[child_nodes[-1].end_byte:node.end_byte].decode(self.encoding),
                content="",
        ))


        blocks = [CodeBlock(
            type=block_type,
            tree_sitter_type=node.type,
            start_line=node.start_point[0],
            end_line=end_line,
            pre_code=pre_code,
            content=code,
            children=children
        )]

        if child_nodes:
            next_sibling = child_nodes[-1].next_sibling
            while next_sibling:
                blocks.extend(self.parse_code(content_bytes, next_sibling, start_byte=end_byte))
                end_byte = next_sibling.end_byte
                next_sibling = next_sibling.next_sibling

        return blocks

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

        if node.type == "assignment":
            delimiter = _find_type(node, "=")
            if delimiter:
                return node.children[delimiter + 1:]


        nodes = []
        delimiter_index = _find_delimiter_index(node)
        if delimiter_index != -1:
            end_index = self._find_end_index(node)
            for child in node.children[delimiter_index:end_index]:
                nodes.append(child)

        return nodes
