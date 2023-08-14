from typing import Callable, Optional, List

import tree_sitter_languages
from tree_sitter import Node

from codeblocks.codeblocks import CodeBlock
from codeblocks.language.java import get_java_block_type, get_java_blocks


class CodeBlockParser:

    def __init__(self, language: str, encoding: str = "utf8"):
        try:
            self.tree_parser = tree_sitter_languages.get_parser(language)
        except Exception as e:
            print(f"Could not get parser for language {language}. Check ")
            raise e

        if language == "java":
            self.get_block_type = get_java_block_type
            self.get_child_blocks: Callable[[Node], List[Node]] = get_java_blocks

        self.encoding = encoding

    def parse_code(self, contents: str, node: Node) -> CodeBlock:
        if node.prev_sibling:
            start_byte = node.prev_sibling.end_byte
        elif node.parent:
            start_byte = node.parent.end_byte
        else:
            start_byte = 0

        pre_code = contents[start_byte:node.start_byte]

        block_type = self.get_block_type(node)

        children = []

        child_nodes = self.get_child_blocks(node)

        first_node = child_nodes[0] if child_nodes else None
        if first_node:
            if first_node.prev_sibling:
                end_byte = first_node.prev_sibling.end_byte
            else:
                end_byte = first_node.start_byte
        else:
            end_byte = node.end_byte

        code = contents[node.start_byte:end_byte]

        if child_nodes:
            for child in child_nodes:
                children.append(self.parse_code(contents, child))

        return CodeBlock(
            type=block_type,
            tree_sitter_type=node.type,
            pre_code=pre_code,
            content=code,
            children=children
        )

    def parse(self, content: str) -> CodeBlock:
        tree = self.tree_parser.parse(bytes(content, self.encoding))

        if not tree.root_node.children or tree.root_node.children[0].type == "ERROR":
            raise Exception("Code is invalid")

        return self.parse_code(content, tree.root_node)
