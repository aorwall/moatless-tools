from typing import Optional, List

from tree_sitter import Node

from codeblocks import CodeBlockType, CodeBlock
from codeblocks.parser.parser import CodeParser, commented_out_keywords, _find_type

compound_node_types = [
    "function_definition", "class_definition", "if_statement",
    "for_statement", "while_statement", "try_statement", "with_statement",
    "expression_statement", "dictionary"
]

child_block_types = ["ERROR", "block"]

block_delimiters = [
    ":"
]


def _find_delimiter_index(node: Node):
    for i, child in enumerate(node.children):
        if child.type == ":":
            return i
    return -1


class PythonParser(CodeParser):

    def __init__(self):
        super().__init__("python")

    def get_child_node_block_types(self):
        return child_block_types

    def get_compound_node_types(self):
        return compound_node_types

    def get_block_delimiter_types(self):
        return block_delimiters

    def get_block_type(self, node: Node) -> Optional[CodeBlockType]:
        if node.type == "decorated_definition" and len(node.children) > 1:
            node = node.children[-1]
        if node.type == "module":
            return CodeBlockType.MODULE
        elif node.type == "function_definition":
            return CodeBlockType.FUNCTION
        elif node.type == "class_definition":
            return CodeBlockType.CLASS
        elif node.type in ["import_statement", "import_from_statement", "future_import_statement"]:
            return CodeBlockType.IMPORT
        elif node.type in block_delimiters:
            return CodeBlockType.BLOCK_DELIMITER
        elif "comment" in node.type:
            comment = node.text.decode("utf8").strip()
            if comment.startswith("# ...") or any(keyword in comment.lower() for keyword in commented_out_keywords):
                return CodeBlockType.COMMENTED_OUT_CODE
            else:
                return CodeBlockType.COMMENT
        else:
            return CodeBlockType.CODE

    def get_child_nodes(self, node: Node) -> List[Node]:
        if node.type == "module":
            return node.children

        if node.type in ["decorated_definition", "expression_statement"] and node.children \
                and any(child.children for child in node.children):
            node = node.children[-1]

        if node.type == "assignment":
            delimiter, _ = _find_type(node, "=")
            if delimiter:
                return node.children[delimiter + 1:]

        if node.type == "dictionary":
            delimiter, _ = _find_type(node, "{")
            if delimiter is not None:
                return node.children[delimiter:]

        delimiter_index = _find_delimiter_index(node)
        if delimiter_index != -1:
            return node.children[delimiter_index:]
        else:
            return []

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
