from typing import List, Tuple, Optional

import tree_sitter_languages
from tree_sitter import Node

from ghostcoder.codeblocks.codeblocks import CodeBlock, CodeBlockType
from ghostcoder.codeblocks.parser.comment import get_comment_symbol

commented_out_keywords = ["rest of the code", "existing code", "other code"]
child_block_types = ["ERROR", "block"]
module_types = ["program", "module"]


def _find_type(node: Node, type: str):
    for i, child in enumerate(node.children):
        if child.type == type:
            return i, child
    return None, None


def find_type(node: Node, types: List[str]):
    for child in node.children:
        if child.type in types:
            return child
    return None


def find_nested_type(node: Node, type: str, levels: int = -1):
    if levels == 0:
        return None
    if node.type == type:
        return node
    for child in node.children:
        found_node = find_nested_type(child, type, levels-1)
        if found_node:
            return found_node
    return None


class CodeParser:

    def __init__(self, language: str, encoding: str = "utf8"):
        try:
            self.tree_parser = tree_sitter_languages.get_parser(language)
            self.tree_language = tree_sitter_languages.get_language(language)

        except Exception as e:
            print(f"Could not get parser for language {language}.")
            raise e
        self.encoding = encoding
        self.language = language

    def is_commented_out_code(self, node: Node):
        comment = node.text.decode("utf8").strip()
        return (comment.startswith(f"{get_comment_symbol(self.language)} ...") or
                any(keyword in comment.lower() for keyword in commented_out_keywords))

    def get_block_definition(self, node: Node):
        if node.children:
            return CodeBlockType.CODE, node.children[0], node.children[-1]
        return None, None, None

    def get_block_definition_2(self, node: Node, content_bytes: bytes, start_byte: int = 0) -> Tuple[Optional[CodeBlock], Optional[Node], Optional[Node]]:
        return None, None, None

    def parse_code(self, content_bytes: bytes, node: Node, start_byte: int = 0, level: int = 0) -> Tuple[CodeBlock, Node]:
        pre_code = content_bytes[start_byte:node.start_byte].decode(self.encoding)
        end_line = node.end_point[0]

        code_block, first_child, last_child = self.get_block_definition_2(node, content_bytes, start_byte)
        if not code_block:
            block_type, first_child, last_child = self.get_block_definition(node)

        if first_child:
            end_byte = self.get_previous(first_child, node)
        else:
            end_byte = node.end_byte

        if not code_block:
            code = content_bytes[node.start_byte:end_byte].decode(self.encoding)
            code_block = CodeBlock(
                type=block_type,
                identifier=code.split("\n")[0].strip(),
                tree_sitter_type=node.type,
                start_line=node.start_point[0],
                end_line=end_line,
                pre_code=pre_code,
                content=code,
                language=self.language
            )


        #print("block_type: ", block_type, "node_type: ", node.type, "first_child: ", first_child, "last_child: ", last_child, "start_byte", start_byte, "node.start_byte", node.start_byte,  "node.end_byte", node.end_byte)

        l = last_child.type if last_child else "none"
        #print(f"start [{level}]: {code} (last child {l}, end byte {end_byte})")

        next_node = first_child
        while next_node:
            if next_node.children and next_node.type == "block":  # TODO: This should be handled in get_block_definition
                next_node = next_node.children[0]

            #print(f"next  [{level}]: -> {next_node.type} - {next_node.start_byte}")

            child_block, child_last_node = self.parse_code(content_bytes, next_node, start_byte=end_byte, level=level+1)
            if not child_block.content:
                if child_block.children:
                    child_block.children[0].pre_code = child_block.pre_code + child_block.children[0].pre_code
                    child_block.children[0].__post_init__()  # FIXME
                    code_block.append_children(child_block.children)
            else:
                code_block.append_child(child_block)

            if child_last_node:
                #print(f"next  [{level}]: child_last_node -> {child_last_node.start_byte}")

                next_node = child_last_node

            end_byte = next_node.end_byte

            if next_node == last_child:
                break

            if next_node.next_sibling:
                next_node = next_node.next_sibling
            else:
                next_node = self.get_parent_next(next_node, last_child or node)

        #print(f"end   [{level}]: {code}")

        if not node.parent and node.end_byte > end_byte:
            code_block.append_child(CodeBlock(
                type=CodeBlockType.SPACE,
                pre_code=content_bytes[end_byte:node.end_byte].decode(self.encoding),
                start_line=end_line,
                end_line=node.end_point[0],
                content="",
            ))

        return code_block, next_node

    def get_previous(self, node: Node, origin_node: Node):
        if node == origin_node:
            return node.start_byte
        if node.prev_sibling:
            return node.prev_sibling.end_byte
        elif node.parent:
            return self.get_previous(node.parent, origin_node)
        else:
            return node.start_byte

    def get_parent_next(self, node: Node, orig_node: Node):
        if node != orig_node:
            if node.next_sibling:
                return node.next_sibling
            else:
                return self.get_parent_next(node.parent, orig_node)
        return None

    def has_error(self, node: Node):
        if node.type == "ERROR":
            return True
        if node.children:
            return any(self.has_error(child) for child in node.children)
        return False

    def parse(self, content: str) -> CodeBlock:
        content_in_bytes = bytes(content, self.encoding)
        tree = self.tree_parser.parse(content_in_bytes)
        codeblock, _ = self.parse_code(content_in_bytes, tree.walk().node)
        codeblock.language = self.language
        return codeblock
