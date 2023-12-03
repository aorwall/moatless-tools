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
        self.queries = self._build_queries(query_contents)

        if self.apply_gpt_tweaks:
            self.gpt_queries = self._build_queries(self._read_query("javascript_gpt.scm"))

    def get_block_definition(self, node: Node, content_bytes: bytes, start_byte: int = 0) -> Tuple[Optional[CodeBlock], Optional[Node], Optional[Node]]:
        first_child, identifier_node, last_child = None, None, None

        if node.type in block_delimiters:  # TODO: Move to query
            block_type = CodeBlockType.BLOCK_DELIMITER
        else:
            block_type, first_child, identifier_node, last_child = self.find_in_tree(node)
            if block_type:
                self.debug_log(f"Found match on node type {node.type} with block type {block_type}")
            else:
                self.debug_log(f"Found no match on node type {node.type} set block type {CodeBlockType.CODE}")
                block_type = CodeBlockType.CODE

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
            identifier = code  # TODO: Extract test description from node
        if block_type == CodeBlockType.FUNCTION and code.startswith("it("):
            block_type = CodeBlockType.TEST_CASE
            identifier = code  # TODO: Extract test description from node

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

