from typing import Optional, List

from tree_sitter import Node

from codeblocks import CodeBlockType


def find_block_node(node: Node):
    for child in node.children:
        if child.type.endswith("block") or child.type.endswith("body"):
            return child
    return None


class Language:

    def __init__(self, use_indentation_level: bool = False):
        self.use_indentation_level = use_indentation_level

    def get_block_type(self, node: Node) -> Optional[CodeBlockType]:
        pass

    def get_child_blocks(self, node: Node) -> List[Node]:
        pass

    def comment(self, comment: str) -> str:
        pass
