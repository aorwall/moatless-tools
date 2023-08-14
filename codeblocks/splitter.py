from typing import Optional, List

from codeblocks.codeblocks import CodeBlock, CodeBlockType
from codeblocks.parser import CodeBlockParser


class CodeSplitter:

    def __init__(
            self,
            language: str,
            chunk_lines: int = 40,
            max_chars: int = 1500
    ):
        self.language = language
        self.chunk_lines = chunk_lines
        self.max_chars = max_chars
        self.outcommented_code = "// ..." # TODO: Support all languages

        try:
            self.parser = CodeBlockParser(language)
        except Exception as e:
            print(f"Could not get parser for language {language}.")
            raise e

    def _outcommented_block(self, block: CodeBlock):
        return CodeBlock(
            type=CodeBlockType.COMMENTED_OUT_CODE,
            pre_code=block.pre_code,
            content=self.outcommented_code)

    def _count_code_blocks(self, codeblocks: List[CodeBlock]) -> int:
        """Count blocks that aren't delimiters or commented out code"""
        count = 0
        for child in codeblocks:
            if child.type != CodeBlockType.BLOCK_DELIMITER and child.type != CodeBlockType.COMMENTED_OUT_CODE:
                count += 1
        return count

    def _finish_chunk_block(self, chunk_block: CodeBlock, child_blocks: [CodeBlock], i: int) -> Optional[CodeBlock]:
        if self._count_code_blocks(chunk_block.children) > 0:
            if i < len(child_blocks):
                if self._count_code_blocks(child_blocks[i:]) > 0:
                    chunk_block.children.append(self._outcommented_block(child_blocks[i]))
                for c in child_blocks[i:]:
                    if c.type == CodeBlockType.BLOCK_DELIMITER:
                        chunk_block.children.append(c)

            if chunk_block.parent:
                chunk_block.parent = self.trim_code_block(chunk_block.parent, chunk_block)

            return chunk_block

        return None

    def _new_chunk_block(self, codeblock: CodeBlock, i: int, parent: Optional[CodeBlock] = None) -> CodeBlock:
        new_children = []
        for child in codeblock.children[:i]:
            if child.type == CodeBlockType.BLOCK_DELIMITER:
                new_children.append(child)

        new_children.append(self._outcommented_block(codeblock.children[i-1]))
        new_children.append(codeblock.children[i])

        return self._new_chunk_bock(codeblock, children=new_children, parent=parent)

    def _new_chunk_bock(self, codeblock: CodeBlock, children: List[CodeBlock], parent: Optional[CodeBlock] = None):
        new_chunk = CodeBlock(
            type=codeblock.type,
            content=codeblock.content,
            pre_code=codeblock.pre_code,
            tree_sitter_type=codeblock.tree_sitter_type,
            children=children,
            parent=parent
        )

        return new_chunk

    def _chunk_block(self, codeblock: CodeBlock, parent: Optional[CodeBlock] = None) -> List[CodeBlock]:
        chunk_blocks = []
        current_chunk = self._new_chunk_bock(codeblock=codeblock, children=[], parent=parent)

        for i, child_block in enumerate(codeblock.children):
            # TODO: Check level?
            if child_block.type in [CodeBlockType.PROGRAM, CodeBlockType.CLASS, CodeBlockType.FUNCTION] and child_block.children:
                if (len(current_chunk.children) > 0
                        and current_chunk.children[-1].type != CodeBlockType.COMMENTED_OUT_CODE):
                    current_chunk.children.append(self._outcommented_block(child_block))
                chunk_blocks.extend(self._chunk_block(child_block, codeblock))
            else:
                if len(str(child_block)) > self.max_chars:
                    finished_chunk = self._finish_chunk_block(current_chunk, codeblock.children, i)
                    if finished_chunk:
                        chunk_blocks.append(finished_chunk)

                    current_chunk = self._new_chunk_block(codeblock, i, parent)
                    chunk_blocks.extend(self._chunk_block(child_block, codeblock))
                elif len(str(current_chunk)) + len(str(child_block)) > self.max_chars:
                    finished_chunk = self._finish_chunk_block(current_chunk, codeblock.children, i)
                    if finished_chunk:
                        chunk_blocks.append(finished_chunk)

                    current_chunk = self._new_chunk_block(codeblock, i, parent)
                else:
                    current_chunk.children.append(child_block)

        finished_chunk = self._finish_chunk_block(current_chunk, codeblock.children, len(codeblock.children))
        if finished_chunk:
            chunk_blocks.append(finished_chunk)

        return chunk_blocks

    def trim_code_block(self, codeblock: CodeBlock, keep_child: CodeBlock):
        children = []
        for child in codeblock.children:
            if child.type == CodeBlockType.BLOCK_DELIMITER:
                children.append(child)
            elif child.content != keep_child.content:
                if not children or children[-1].type != CodeBlockType.COMMENTED_OUT_CODE:
                    children.append(self._outcommented_block(child))
            else:
                children.append(keep_child)

        trimmed_block = CodeBlock(
            content=codeblock.content,
            pre_code=codeblock.pre_code,
            type=codeblock.type,
            parent=codeblock.parent,
            children=children
        )

        if codeblock.parent:
            codeblock.parent = self.trim_code_block(trimmed_block.parent, trimmed_block)

        return trimmed_block

    def split_text(self, text: str) -> List[str]:
        codeblock = self.parser.parse(text)
        chunk_blocks = self._chunk_block(codeblock)
        return [str(block.root()) for block in chunk_blocks]
