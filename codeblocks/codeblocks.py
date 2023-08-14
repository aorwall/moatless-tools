from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional

from tree_sitter import Node


class CodeBlockType(str, Enum):
    PROGRAM = "program"
    DECLARATION = "declaration"
    CLASS = "class"
    FUNCTION = "function"
    STATEMENT = "statement"
    COMMENT = "comment"
    COMMENTED_OUT_CODE = "commented_out_code"
    NONE = "none"
    BLOCK_DELIMITER = "block_delimiter"


@dataclass
class CodeBlock:
    content: str
    type: CodeBlockType = None
    tree_sitter_type: Optional[str] = None
    pre_code: str = ""
    children: List["CodeBlock"] = field(default_factory=list)
    parent: Optional["CodeBlock"] = field(default=None)

    def __post_init__(self):
        for child in self.children:
            child.parent = self

    def __str__(self):
        child_code = "".join([str(child) for child in self.children])
        return f"{self.pre_code}{self.content}{child_code}"

    def __eq__(self, other):
        if not isinstance(other, CodeBlock):
            return False
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        return {
            "code": self.content,
            "type": self.type,
            "tree_sitter_type": self.tree_sitter_type,
            "pre_code": self.pre_code,
            "children": [child.to_dict() for child in self.children]
        }

    def root(self):
        if self.parent:
            return self.parent.root()
        return self


def find_node(node: Node, node_type: str):
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def get_python_block(node: Node) -> Tuple[int, Optional[Node]]:
    block_node_types = [
        "function_definition", "class_definition", "if_statement",
        "for_statement", "while_statement", "try_statement", "with_statement"
    ]

    if node.type in block_node_types:
        block_node = find_node(node, "block")
        return block_node.start_byte if block_node else None, block_node

    return -1, None


def get_js_ts_block(node: Node) -> Tuple[int, Optional[Node]]:
    block_node_types = [
        "function_declaration", "function", "class_declaration", "if_statement",
        "for_statement", "while_statement", "do_statement", "try_statement", "switch_statement",
        "arrow_function"
    ]

    if node.type in block_node_types:
        block_node = find_node(node, "block")
        return block_node.start_byte if block_node else None, block_node

    return -1, None
