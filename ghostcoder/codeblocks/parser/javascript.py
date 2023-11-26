import logging
from typing import Optional, Tuple

from tree_sitter import Node

from ghostcoder.codeblocks.codeblocks import CodeBlockType, CodeBlock
from ghostcoder.codeblocks.parser.parser import CodeParser

block_delimiters = [
    "{",
    "}",
    ";",
    "(",
    ")"
]

class JavaScriptParser(CodeParser):

    def __init__(self, language: str = "javascript", **kwargs):
        super().__init__(language, **kwargs)

        query_contents = self._read_query("javascript.scm")
        if language in ["tsx", "typescript"]:
            query_contents += "\n\n" + self._read_query("typescript.scm")
        else:
            query_contents += "\n\n" + self._read_query("javascript_only.scm")
        if language in ["tsx", "javascript"]:
            query_contents += "\n\n" + self._read_query("jsx.scm")
        self.query = self.tree_language.query(query_contents)

        if self.apply_gpt_tweaks:
            self.gpt_query = self._build_query("javascript_gpt.scm")

    def find_in_tree(self, node: Node) -> Tuple[Optional[CodeBlockType], Optional[Node], Optional[Node], Optional[Node]]:
        if node.type in block_delimiters:  # TODO: Move to query
            return CodeBlockType.BLOCK_DELIMITER, None, None, None

        if self.apply_gpt_tweaks:
            gpt_match = self.find_match(node, True)
            if all(gpt_match):
                self.debug_log(f"GPT match: {node.type}")
                return gpt_match

        return self.find_match(node)

    def find_match(self, node: Node, gpt_tweaks: bool = False) -> Tuple[CodeBlockType, Node, Node, Node]:
        if gpt_tweaks:
            query = self.gpt_query
        else:
            query = self.query

        captures = query.captures(node)

        identifier_node = None
        first_child = None
        block_type = None
        last_child = None

        for node_match, tag in captures:
            self.debug_log(f"Found tag {tag} on node {node_match}")

            if tag == "root" and node != node_match:
                self.debug_log(f"Skipping root match on {node.type}, return type {block_type}")
                return block_type, first_child, identifier_node, last_child

            if tag == "check_child":
                return self.find_match(node_match)

            if tag == "identifier":
                identifier_node = node_match
            elif tag == "child.first":
                first_child = node_match
            elif tag == "child.last":
                last_child = node_match

            if not block_type:
                block_type = self._get_block_type(tag)

        return block_type, first_child, identifier_node, last_child

    def _get_block_type(self, tag: str):
        if tag == "definition.code":
            return CodeBlockType.CODE
        elif tag == "definition.comment":
            return CodeBlockType.COMMENT
        elif tag == "definition.import":
            return CodeBlockType.IMPORT
        elif tag == "definition.class":
            return CodeBlockType.CLASS
        elif tag == "definition.function":
            return CodeBlockType.FUNCTION
        elif tag == "definition.statement":
            return CodeBlockType.STATEMENT
        elif tag == "definition.block":
            return CodeBlockType.BLOCK
        elif tag == "definition.module":
            return CodeBlockType.MODULE
        elif tag == "definition.block_delimiter":
            return CodeBlockType.BLOCK_DELIMITER
        return None

    def get_block_definition_2(self, node: Node, content_bytes: bytes, start_byte: int = 0) -> Tuple[Optional[CodeBlock], Optional[Node], Optional[Node]]:
        block_type, first_child, identifier_node, last_child = self.find_in_tree(node)
        if not block_type:
            return None, None, None

        self.debug_log(f"Found match on node type {node.type} with block type {block_type}")

        #if not last_child:
            #if node.next_sibling and node.next_sibling.type == ";":
            #    last_child = node.next_sibling
        #    if node.children:
        #        last_child = node.children[-1]

        #    if not first_child and last_child: # and last_child.type == ";":
        #        first_child = last_child

        #    if first_child and last_child and first_child.end_byte > last_child.end_byte:
        #        last_child = first_child

        pre_code = content_bytes[start_byte:node.start_byte].decode(self.encoding)
        end_line = node.end_point[0]

        if first_child:
            end_byte = self.get_previous(first_child, node)
        else:
            end_byte = node.end_byte

        code = content_bytes[node.start_byte:end_byte].decode(self.encoding)

        if identifier_node:
            identifier = content_bytes[identifier_node.start_byte:identifier_node.end_byte].decode(self.encoding)
        else:
            identifier = None

        if block_type == CodeBlockType.FUNCTION and identifier == "constructor":
            block_type = CodeBlockType.CONSTRUCTOR

        if block_type == CodeBlockType.COMMENT and "..." in node.text.decode("utf8"):
            block_type = CodeBlockType.COMMENTED_OUT_CODE

        # Support Jest tests
        if block_type == CodeBlockType.FUNCTION and code.startswith("describe("):
            block_type = CodeBlockType.TEST_SUITE
        if block_type == CodeBlockType.FUNCTION and code.startswith("it("):
            block_type = CodeBlockType.TEST_CASE

        # Expect class on components with an identifier starting with upper case
        if block_type == CodeBlockType.FUNCTION and identifier and identifier[0].isupper():
            block_type = CodeBlockType.CLASS

        code_block = CodeBlock(
                type=block_type,
                identifier=identifier,
                tree_sitter_type=node.type,
                start_line=node.start_point[0],
                end_line=end_line,
                pre_code=pre_code,
                content=code,
                language=self.language,
                children=[]
            )

        return code_block, first_child, last_child

    def get_block_definition(self, node: Node):
        return CodeBlockType.CODE, None, None
