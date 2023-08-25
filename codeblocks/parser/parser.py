from typing import Optional, List

import tree_sitter_languages
from tree_sitter import Node

from codeblocks.codeblocks import CodeBlock, CodeBlockType
from codeblocks.parser.comment import get_comment_symbol

commented_out_keywords = ["rest of the code", "existing code", "other code"]
child_block_types = ["ERROR", "block"]
module_types = ["program", "module"]


def _find_type(node: Node, type: str):
    for i, child in enumerate(node.children):
        if child.type == type:
            return i, child
    return None, None

def find_type(node: Node, types: List[str]):
    if node.type in types:
        return node
    for child in node.children:
        if child.type in types:
            return child
    return None

def find_nested_type(node: Node, type: str):
    if node.type == type:
        return node
    for child in node.children:
        found_node = find_nested_type(child, type)
        if found_node:
            return found_node
    return None


class CodeParser:

    def __init__(self, language: str, encoding: str = "utf8"):
        try:
            self.tree_parser = tree_sitter_languages.get_parser(language)
        except Exception as e:
            print(f"Could not get parser for language {language}.")
            raise e
        self.encoding = encoding
        self.language = language

    def find_block_node(self, node: Node):
        for child in node.children:
            if child.type.endswith("block") or child.type.endswith("body"):
                return child
        return None

    def get_block_type(self, node: Node) -> Optional[CodeBlockType]:
        if node.type in module_types:
            return CodeBlockType.MODULE
        elif node.type.startswith("function") or node.type.startswith("method"):
            return CodeBlockType.FUNCTION
        elif node.type.startswith("class"):
            return CodeBlockType.CLASS
        elif "_statement" in node.type:
            return CodeBlockType.STATEMENT
        elif node.type in self.get_block_delimiter_types():
            return CodeBlockType.BLOCK_DELIMITER
        elif "import" in node.type:
            return CodeBlockType.IMPORT
        elif "comment" in node.type:
            if self.is_commented_out_code(node):
                return CodeBlockType.COMMENTED_OUT_CODE
            else:
                return CodeBlockType.COMMENT
        else:
            return CodeBlockType.CODE

    def is_commented_out_code(self, node: Node):
        comment = node.text.decode("utf8").strip()
        return (comment.startswith(f"{get_comment_symbol(self.language)} ...") or
                any(keyword in comment.lower() for keyword in commented_out_keywords))


    def get_child_node_block_types(self):
        return child_block_types

    def get_block_delimiter_types(self):
        return ["{", "}", ":", "(", ")"]

    def get_compound_node_types(self) -> List[str]:
        return []

    def find_first_child(self, node: Node):
        if node.children:
            return node.children[0]
        return None

    def _find_delimiter_index(self, node: Node):
        for i, child in enumerate(node.children):
            if child.type == self.get_block_delimiter_types():
                return i
        return -1

    def _is_error(self, node: Node) -> bool:
        if node.type != "ERROR":
            return False
        if len(node.children) == 1 and node.children[0].type in self.get_compound_node_types():
            return False
        return True

    def parse_code(self, content_bytes: bytes, node: Node, start_byte: int = 0) -> CodeBlock:
        pre_code = content_bytes[start_byte:node.start_byte].decode(self.encoding)

        block_type = self.get_block_type(node)

        first_child = self.find_first_child(node)

        children = []

        if first_child:
            end_byte = first_child.start_byte
            end_line = node.end_point[0]
        else:
            end_byte = node.end_byte
            end_line = node.end_point[0]

        code = content_bytes[node.start_byte:end_byte].decode(self.encoding)

        next_node = first_child
        while next_node:
            children.append(self.parse_code(content_bytes, next_node, start_byte=end_byte))
            end_byte = next_node.end_byte
            if next_node.next_sibling:
                next_node = next_node.next_sibling
            else:
                next_node = self.get_parent_next(next_node, node)

        if not node.parent and node.end_byte > end_byte:
            children.append(CodeBlock(
                type=CodeBlockType.SPACE,
                pre_code=content_bytes[end_byte:node.end_byte].decode(self.encoding),
                start_line=end_line,
                end_line=node.end_point[0],
                content="",
        ))

        return CodeBlock(
            type=block_type,
            tree_sitter_type=node.type,
            start_line=node.start_point[0],
            end_line=end_line,
            pre_code=pre_code,
            content=code,
            children=children,
            language=self.language
        )

    def get_parent_next(self, node: Node, orig_node: Node):
        if node != orig_node:
            if node.next_sibling:
                return node.next_sibling
            else:
                return self.get_parent_next(node.parent, orig_node)
        return None

    def parse(self, content: str) -> CodeBlock:
        tree = self.tree_parser.parse(bytes(content, self.encoding))
        codeblock = self.parse_code(content.encode(self.encoding), tree.walk().node)
        codeblock.language = self.language
        return codeblock

