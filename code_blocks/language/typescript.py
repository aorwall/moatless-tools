from typing import Tuple, Optional

from code_blocks.language.utils import find_block_node
from tree_sitter import Node


def get_js_ts_block(node: Node) -> Tuple[int, Optional[Node]]:
    block_node_types = [
        "function_declaration", "function", "class_declaration", "if_statement",
        "for_statement", "while_statement", "do_statement", "try_statement", "switch_statement",
        "arrow_function"
    ]

    if node.type in block_node_types:
        block_node = find_block_node(node)
        return block_node.start_byte if block_node else None, block_node

    return -1, None
