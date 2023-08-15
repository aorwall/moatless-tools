import tree_sitter_languages
from tree_sitter import Node

from code_blocks.codeblocks import CodeBlock, CodeBlockType
from code_blocks.language import get_language


class CodeParser:

    def __init__(self, language: str, encoding: str = "utf8"):
        self.language = get_language(language)
        if not self.language:
            print(f"Could not get parser for language {language}.")
            raise Exception(f"Could not get parser for language {language}.")

        try:
            self.tree_parser = tree_sitter_languages.get_parser(language)
        except Exception as e:
            print(f"Could not get parser for language {language}.")
            raise e
        self.encoding = encoding

    def parse_code(self, contents: str, node: Node, start_byte: int = 0) -> CodeBlock:
        pre_code = contents[start_byte:node.start_byte]

        block_type = self.language.get_block_type(node)
        child_nodes = self.language.get_child_blocks(node)

        children = []

        first_node = child_nodes[0] if child_nodes else None
        if first_node:
            if first_node.prev_sibling:
                end_byte = first_node.prev_sibling.end_byte
            else:
                end_byte = first_node.start_byte
        else:
            end_byte = node.end_byte

        code = contents[node.start_byte:end_byte]

        for child in child_nodes:
            children.append(self.parse_code(contents, child, start_byte=end_byte))
            end_byte = child.end_byte

        if not node.parent and child_nodes and child_nodes[-1].end_byte < node.end_byte:
            children.append(CodeBlock(
                type=CodeBlockType.SPACE,
                pre_code=contents[child_nodes[-1].end_byte:node.end_byte],
                content="",
        ))

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
