from typing import List

from code_blocks.codeblocks import CodeBlock, CodeBlockType
from code_blocks.parser import CodeParser


class CodeMerger:

    def __init__(self, language: str):
        self.language = language
        try:
            self.parser = CodeParser(language)
        except Exception as e:
            print(f"Could not get parser for language {language}.")
            raise e

    @staticmethod
    def find_next_equal_block(original_blocks: List[CodeBlock],
                              updated_blocks: List[CodeBlock],
                              start_original: int,
                              start_updated: int):
        i = start_original

        while i < len(original_blocks):
            j = start_updated
            while j < len(updated_blocks):
                if original_blocks[i].content == updated_blocks[j].content:
                    return i, j
                j += 1
            i += 1

        return -1, -1

    def merge_blocks(self, original_block: CodeBlock, updated_block: CodeBlock) -> CodeBlock:
        merged_block = CodeBlock(
            type=original_block.type,
            tree_sitter_type=original_block.tree_sitter_type,
            pre_code=original_block.pre_code,
            content=original_block.content,
            children=[]
        )

        i = 0
        j = 0
        while i < len(original_block.children) and j < len(updated_block.children):
            original_block_child = original_block.children[i]
            updated_block_child = updated_block.children[j]

            if original_block_child == updated_block_child:
                merged_block.children.append(original_block_child)
                i += 1
                j += 1
            elif updated_block_child.type == CodeBlockType.COMMENTED_OUT_CODE:
                j += 1
                orig_next, update_next = self.find_next_equal_block(original_block.children, updated_block.children, i,
                                                                    j)

                if orig_next == -1 or update_next == -1:
                    orig_next = len(original_block.children)
                    update_next = len(updated_block.children)

                merged_block.children.extend(original_block.children[i:orig_next])

                i = orig_next
                if update_next > j:
                    merged_block.children.extend(updated_block.children[j:update_next])

                j = update_next
            elif (original_block_child.content == updated_block_child.content and
                  original_block_child.children and updated_block_child.children):
                merged_block.children.append(self.merge_blocks(original_block_child, updated_block_child))
                i += 1
                j += 1
            else:
                merged_block.children.append(updated_block_child)
                j += 1

        return merged_block

    def merge(self, original_content: str, updated_content: str):
        original_block = self.parser.parse(original_content)
        updated_block = self.parser.parse(updated_content)
        merged_block = self.merge_blocks(original_block, updated_block)
        return str(merged_block)
