from typing import Optional, Tuple

from tree_sitter import Node

from codeblocks import CodeBlockType
from codeblocks.parser.parser import CodeParser, commented_out_keywords, find_type

statement_node_types = [
    "if_statement", "for_statement", "while_statement", "try_statement", "with_statement",
    "expression_statement", "elif_clause", "else_clause", "except_clause", "finally_clause"
]

child_block_types = ["ERROR", "block"]

block_delimiters = [
    ":"
]


class PythonParser(CodeParser):

    def __init__(self):
        super().__init__("python")

    def get_first_child(self, node: Node):
        if node.children:
            return node.children[0]
        return None

    def find_block_child(self, node: Node):
        delimiter = find_type(node, [":"])
        if not delimiter:
            return self.get_first_child(node)
        if delimiter.next_sibling.type == "block":
            return self.get_first_child(delimiter.next_sibling)
        else:
            return delimiter.next_sibling

    def get_block_definition(self, node: Node) -> Tuple[CodeBlockType, Optional[Node]]:
        if node.type == "ERROR" and len(node.children) == 1:
            node = node.children[0]

        if node.type == "decorated_definition" and len(node.children) > 1:
            node = node.children[-1]

        if node.type == "block" and node.children:
            node = node.children[0]

        if node.type == "module":
            return CodeBlockType.MODULE, self.get_first_child(node)

        if node.type == "function_definition":
            return CodeBlockType.FUNCTION, self.find_block_child(node)

        if node.type == "class_definition":
            return CodeBlockType.CLASS, self.find_block_child(node)

        if node.type in ["import_statement", "import_from_statement", "future_import_statement"]:
            return CodeBlockType.IMPORT, None

        if node.type in block_delimiters:
            return CodeBlockType.BLOCK_DELIMITER, None

        if "comment" in node.type:
            comment = node.text.decode("utf8").strip()
            if comment.startswith("# ...") or any(keyword in comment.lower() for keyword in commented_out_keywords):
                return CodeBlockType.COMMENTED_OUT_CODE, None
            else:
                return CodeBlockType.COMMENT, None

        if node.type in statement_node_types:
            return CodeBlockType.STATEMENT, self.find_block_child(node)

        if node.type in ["return_statement"]:
            if len(node.children) > 1:
                return CodeBlockType.CODE, node.children[1]
            else:
                return CodeBlockType.CODE, None

        if node.type == "assignment":
            return CodeBlockType.CODE, find_type(node, ["="])

        if node.type == "pair":
            return CodeBlockType.CODE, find_type(node, [":"])

        if node.type == "dictionary":
            return CodeBlockType.CODE, find_type(node, ["{"])

        if node.type == "ERROR":
            return CodeBlockType.ERROR, None

        return CodeBlockType.CODE, None
