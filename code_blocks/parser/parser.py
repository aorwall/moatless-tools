from typing import Optional, List

import tree_sitter_languages
from tree_sitter import Node

from code_blocks.codeblocks import CodeBlock, CodeBlockType



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

    def get_block_type(self, node: Node) -> Optional[CodeBlockType]:
        pass

    def get_child_blocks(self, node: Node) -> List[Node]:
        pass

    def get_block_node_types(self) -> List[str]:
        return []

    def _is_error(self, node: Node) -> bool:
        if node.type != "ERROR":
            return False
        if len(node.children) == 1 and node.children[0].type in self.get_block_node_types():
            return False
        return True


    def parse_code(self, contents: str, node: Node, start_byte: int = 0) -> List[CodeBlock]:
        pre_code = contents[start_byte:node.start_byte]

        block_type = self.get_block_type(node)
        child_nodes = self.get_child_blocks(node)

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

        code = contents[node.start_byte:end_byte]

        for child in child_nodes:
            if child.type in ["ERROR", "block"]:
                child_children = []
                if child.children:
                    for child_child in child.children:
                        child_children.extend(self.parse_code(contents, child_child, start_byte=end_byte))
                        end_byte = child_child.end_byte
                if self._is_error(child):
                    children.append(CodeBlock(
                        type=CodeBlockType.ERROR,
                        tree_sitter_type=node.type,
                        start_line=node.start_point[0],
                        end_line=end_line,
                        pre_code=pre_code,
                        content=code,
                        children=child_children
                    ))
                else:
                    children.extend(child_children)
            else:
                children.extend(self.parse_code(contents, child, start_byte=end_byte))
                end_byte = child.end_byte


        if not node.parent and child_nodes and child_nodes[-1].end_byte < node.end_byte:
            children.append(CodeBlock(
                type=CodeBlockType.SPACE,
                pre_code=contents[child_nodes[-1].end_byte:node.end_byte],
                start_line=child_nodes[-1].start_point[0],
                end_line=child_nodes[-1].end_point[0],
                content="",
        ))

        return [CodeBlock(
            type=block_type,
            tree_sitter_type=node.type,
            start_line=node.start_point[0],
            end_line=end_line,
            pre_code=pre_code,
            content=code,
            children=children
        )]

    def parse(self, content: str) -> CodeBlock:
        tree = self.tree_parser.parse(bytes(content, self.encoding))
        blocks = self.parse_code(content, tree.root_node)
        if len(blocks) > 1:
            block_string = "\n".join([f"- {b.type.value} {b.content}" for b in blocks])
            print("Expect only one root block, but got more. Will return the first one. "
                  f"Blocks:\n{block_string}")
        return blocks[0]
