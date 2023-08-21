from typing import Optional, List

import tree_sitter_languages
from tree_sitter import Node

from codeblocks.codeblocks import CodeBlock, CodeBlockType

COMMENTED_OUT_CODE_KEYWORDS = ["rest of the code", "existing code", "other code"]

child_block_types = ["ERROR", "block"]

def _find_type(node: Node, type: str):
    for i, child in enumerate(node.children):
        if child.type == type:
            return i, child
    return None, None

class CodeParser:

    def __init__(self,
                 language: str,
                 encoding: str = "utf8"):

        try:
            self.tree_parser = tree_sitter_languages.get_parser(language)
        except Exception as e:
            print(f"Could not get parser for language {language}.")
            raise e
        self.encoding = encoding
        self.language = language

    def get_block_type(self, node: Node) -> Optional[CodeBlockType]:
        pass

    def get_child_nodes(self, node: Node) -> List[Node]:
        pass

    def get_child_node_block_types(self):
        return child_block_types

    def get_block_delimiter_types(self):
        return []

    def get_compound_node_types(self) -> List[str]:
        return []

    def _is_error(self, node: Node) -> bool:
        if node.type != "ERROR":
            return False
        if len(node.children) == 1 and node.children[0].type in self.get_compound_node_types():
            return False
        return True

    def parse_code(self, content_bytes: bytes, node: Node, start_byte: int = 0) -> CodeBlock:
        pre_code = content_bytes[start_byte:node.start_byte].decode(self.encoding)

        block_type = self.get_block_type(node)
        child_nodes = self.get_child_nodes(node)

        children = []

        first_node = child_nodes[0] if child_nodes else None
        if first_node:
            if first_node.prev_sibling:
                end_byte = first_node.prev_sibling.end_byte
                end_line = first_node.prev_sibling.end_point[0]
            else:
                end_byte = first_node.start_byte
                end_line = node.end_point[0]
        else:
            end_byte = node.end_byte
            end_line = node.end_point[0]

        code = content_bytes[node.start_byte:end_byte].decode(self.encoding)

        if child_nodes and not any(child_node.children or child_node.type in self.get_block_delimiter_types()
                                   for child_node in child_nodes):
            children.append(CodeBlock(
                type=CodeBlockType.CODE,
                pre_code=content_bytes[end_byte:child_nodes[0].start_byte].decode(self.encoding),
                content=content_bytes[child_nodes[0].start_byte:child_nodes[-1].end_byte].decode(self.encoding),
                start_line=child_nodes[0].start_point[0],
                end_line=child_nodes[-1].end_point[0],))
        else:
            for child in child_nodes:
                if child.type in self.get_child_node_block_types():
                    child_blocks = []
                    if child.children:
                        for child_child in child.children:
                            child_blocks.append(self.parse_code(content_bytes, child_child, start_byte=end_byte))
                            end_byte = child_child.end_byte
                    if self._is_error(child):
                        children.append(CodeBlock(
                            type=CodeBlockType.ERROR,
                            tree_sitter_type=node.type,
                            start_line=node.start_point[0],
                            end_line=end_line,
                            pre_code=pre_code,
                            content=code,
                            children=child_blocks
                        ))
                    else:
                        children.extend(child_blocks)
                else:
                    children.append(self.parse_code(content_bytes, child, start_byte=end_byte))
                    end_byte = child.end_byte

        if not node.parent and child_nodes and child_nodes[-1].end_byte < node.end_byte:
            children.append(CodeBlock(
                type=CodeBlockType.SPACE,
                pre_code=content_bytes[child_nodes[-1].end_byte:node.end_byte].decode(self.encoding),
                start_line=child_nodes[-1].start_point[0],
                end_line=child_nodes[-1].end_point[0],
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

    def parse(self, content: str) -> CodeBlock:
        tree = self.tree_parser.parse(bytes(content, self.encoding))
        codeblock = self.parse_code(content.encode(self.encoding), tree.walk().node)
        codeblock.language = self.language
        return codeblock
