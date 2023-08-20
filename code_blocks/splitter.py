from typing import Optional, List

from code_blocks.codeblocks import CodeBlock, CodeBlockType
from code_blocks.parser import create_parser

non_code_blocks = [CodeBlockType.BLOCK_DELIMITER, CodeBlockType.COMMENTED_OUT_CODE, CodeBlockType.SPACE]


class CodeSplitter:

    def __init__(
            self,
            language: str,
            chunk_lines: int = 40,
            max_chars: int = 1500
    ):
        self.chunk_lines = chunk_lines
        self.max_chars = max_chars

        self.language = language
        try:
            self.parser = create_parser(language)
        except Exception as e:
            print(f"Could not get parser for language {language}.")
            raise e

    def comment(self, comment: str) -> str:
        if self.language == "python":
            return f"# {comment}"
        else:
            return f"// {comment}"


    def _comment_out_block(self, block: CodeBlock):
        return CodeBlock(
            type=CodeBlockType.COMMENTED_OUT_CODE,
            pre_code=block.pre_code,
            content=self.comment("..."))

    @staticmethod
    def _count_code_blocks(codeblocks: List[CodeBlock]) -> int:
        count = 0
        for child in codeblocks:
            if child.type not in non_code_blocks and child.content:
                count += 1
        return count

    def _finish_chunk_block(self, chunk_block: CodeBlock, child_blocks: [CodeBlock], i: int) -> Optional[CodeBlock]:
        if self._count_code_blocks(chunk_block.children) > 0:
            if i < len(child_blocks):
                if self._count_code_blocks(child_blocks[i:]) > 0:
                    chunk_block.children.append(self._comment_out_block(child_blocks[i]))
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

        new_children.append(self._comment_out_block(codeblock.children[i - 1]))
        new_children.append(codeblock.children[i])

        return self._new_chunk_bock(codeblock, children=new_children, parent=parent)

    @staticmethod
    def _new_chunk_bock(codeblock: CodeBlock, children: List[CodeBlock], parent: Optional[CodeBlock] = None):
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
            if self.separate_block(child_block, codeblock):
                if (len(current_chunk.children) > 0
                        and current_chunk.children[-1].type != CodeBlockType.COMMENTED_OUT_CODE):
                    current_chunk.children.append(self._comment_out_block(child_block))
                chunk_blocks.extend(self._chunk_block(child_block, codeblock))
            else:
                if len(child_block.to_string()) > self.max_chars:
                    finished_chunk = self._finish_chunk_block(current_chunk, codeblock.children, i)
                    if finished_chunk:
                        chunk_blocks.append(finished_chunk)

                    current_chunk = self._new_chunk_block(codeblock, i, parent)
                    chunk_blocks.extend(self._chunk_block(child_block, codeblock))
                elif len(child_block.to_string()) + len(child_block.to_string()) > self.max_chars:
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

    @staticmethod
    def separate_block(child_block, codeblock):
        separate_block = child_block.children and (
                child_block.type in [CodeBlockType.MODULE, CodeBlockType.CLASS] or
                (child_block.type == CodeBlockType.FUNCTION and
                 codeblock.type in [CodeBlockType.MODULE, CodeBlockType.CLASS]))
        return separate_block

    def trim_code_block(self, codeblock: CodeBlock, keep_child: CodeBlock):
        children = []
        for child in codeblock.children:
            if child.type == CodeBlockType.BLOCK_DELIMITER:
                children.append(child)
            elif child.content != keep_child.content:
                if (child.type not in non_code_blocks and
                        (not children or children[-1].type != CodeBlockType.COMMENTED_OUT_CODE)):
                    children.append(self._comment_out_block(child))
            else:
                children.append(keep_child)

        trimmed_block = CodeBlock(
            content=codeblock.content,
            pre_code=codeblock.pre_code,
            type=codeblock.type,
            parent=codeblock.parent,
            children=children
        )

        if trimmed_block.parent:
            trimmed_block.parent = self.trim_code_block(trimmed_block.parent, trimmed_block)

        return trimmed_block

    def split_text(self, text: str) -> List[str]:
        codeblock = self.parser.parse(text)
        chunk_blocks = self._chunk_block(codeblock)
        return [block.root().to_string() for block in chunk_blocks]
