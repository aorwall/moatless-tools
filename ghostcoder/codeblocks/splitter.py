import logging
from typing import Optional, List

from llama_index.callbacks import CallbackManager
from llama_index.text_splitter import TextSplitter
from llama_index.text_splitter.code_splitter import DEFAULT_CHUNK_LINES, DEFAULT_LINES_OVERLAP, DEFAULT_MAX_CHARS
from pydantic import Field

from ghostcoder.codeblocks.codeblocks import CodeBlock, CodeBlockType
from ghostcoder.codeblocks.parser.create import create_parser

non_code_blocks = [CodeBlockType.BLOCK_DELIMITER, CodeBlockType.COMMENTED_OUT_CODE, CodeBlockType.SPACE]


class CodeSplitter(TextSplitter):

    language: str = Field(
        description="The programming language of the code being split."
    )
    chunk_lines: int = Field(
        default=DEFAULT_CHUNK_LINES,
        description="The number of lines to include in each chunk.",
    )
    chunk_lines_overlap: int = Field(
        default=DEFAULT_LINES_OVERLAP,
        description="How many lines of code each chunk overlaps with.",
    )
    max_chars: int = Field(
        default=DEFAULT_MAX_CHARS, description="Maximum number of characters per chunk."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    def __init__(
        self,
        language: str,
        chunk_lines: int = 40,
        chunk_lines_overlap: int = 15,
        max_chars: int = 1500,
        callback_manager: Optional[CallbackManager] = None,
    ):
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls):
        return "CodeSplitter"

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
                if (self._count_code_blocks(child_blocks[i:]) > 0 and
                        chunk_block.children[-1].type != CodeBlockType.COMMENTED_OUT_CODE):
                    chunk_block.children.append(child_blocks[i].create_commented_out_block())
                for c in child_blocks[i:]:
                    if c.type == CodeBlockType.BLOCK_DELIMITER:
                        chunk_block.children.append(c)

            return chunk_block

        return None

    def _new_chunk_block(self, codeblock: CodeBlock, i: int, parent: Optional[CodeBlock] = None) -> CodeBlock:
        new_children = []
        for child in codeblock.children[:i]:
            if child.type == CodeBlockType.BLOCK_DELIMITER:
                new_children.append(child)

        if len(new_children) < i:
            new_children.append(codeblock.children[i - 1].create_commented_out_block())

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

        if new_chunk.parent:
            new_chunk.parent = new_chunk.parent.trim_code_block(new_chunk)

        return new_chunk

    def _chunk_block(self, codeblock: CodeBlock, parent: Optional[CodeBlock] = None) -> List[CodeBlock]:
        chunk_blocks = []
        child_chunks = []
        current_chunk = self._new_chunk_bock(codeblock=codeblock, children=[], parent=parent)

        for i, child_block in enumerate(codeblock.children):
            if self.separate_block(child_block, codeblock):
                if (len(current_chunk.children) > 0
                        and current_chunk.children[-1].type != CodeBlockType.COMMENTED_OUT_CODE):
                    current_chunk.children.append(child_block.create_commented_out_block())
                child_chunks.extend(self._chunk_block(child_block, codeblock))
            else:
                if child_block.length_without_whitespace() > self.max_chars:
                    finished_chunk = self._finish_chunk_block(current_chunk, codeblock.children, i)
                    if finished_chunk:
                        chunk_blocks.append(finished_chunk)
                        current_chunk = self._new_chunk_block(codeblock, i, parent)
                    child_chunks.extend(self._chunk_block(child_block, codeblock))
                elif current_chunk.root().length_without_whitespace() + child_block.length_without_whitespace() > self.max_chars:
                    finished_chunk = self._finish_chunk_block(current_chunk, codeblock.children, i)
                    if finished_chunk:
                        chunk_blocks.append(finished_chunk)
                        current_chunk = self._new_chunk_block(codeblock, i, parent)
                    current_chunk.children.append(child_block)
                else:
                    current_chunk.children.append(child_block)

        finished_chunk = self._finish_chunk_block(current_chunk, codeblock.children, len(codeblock.children))
        if finished_chunk:
            chunk_blocks.append(finished_chunk)

        chunk_blocks.extend(child_chunks)

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
                    children.append(child.create_commented_out_block())
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
        try:
            parser = create_parser(self.language)
        except Exception as e:
            logging.warning(f"Could not get parser for language {self.language}.")
            raise e

        codeblock = parser.parse(text)
        chunk_blocks = self._chunk_block(codeblock)
        return [block.root().to_string().strip() for block in chunk_blocks]

