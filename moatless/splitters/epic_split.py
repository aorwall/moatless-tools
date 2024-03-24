import re
from enum import Enum
from typing import Sequence, List, Optional, Any, Callable

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import NodeParser, TextSplitter
from llama_index.core.node_parser.node_utils import logger
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship
from llama_index.core.utils import get_tqdm_iterable, get_tokenizer

from moatless.codeblocks import create_parser, CodeParser
from moatless.codeblocks.codeblocks import NON_CODE_BLOCKS, PathTree, CodeBlock, CodeBlockType
from moatless.splitters.code_splitter_v2 import CodeSplitterV2


CodeBlockChunk = List[CodeBlock]


def count_chunk_tokens(chunks: CodeBlockChunk) -> int:
    return sum([chunk.tokens for chunk in chunks])


def count_parent_tokens(codeblock: CodeBlock) -> int:
    tokens = codeblock.tokens
    if codeblock.parent:
        tokens += codeblock.parent.tokens
    return tokens


SPLIT_BLOCK_TYPES = [
    CodeBlockType.FUNCTION,
    CodeBlockType.CLASS,
    CodeBlockType.TEST_SUITE,
    CodeBlockType.TEST_CASE,
    CodeBlockType.MODULE,
]


class CommentStrategy(Enum):

    # Keep comments
    INCLUDE = "include"

    # Always associate comments before a code block with the code block
    ASSOCIATE = "associate"

    # Exclude comments in parsed chunks
    EXCLUDE = "exclude"

    # ONLY = "only"  TODO?


class EpicSplitter(NodeParser):

    text_splitter: TextSplitter = Field(
        description="Text splitter to use for splitting non code documents into nodes."
    )

    include_non_code_files: bool = Field(
        default=True, description="Whether or not to include non code files."
    )

    non_code_file_extensions: List[str] = Field(
        default=["md", "txt"], description="File extensions to consider as non code files."
    )

    comment_strategy: CommentStrategy = Field(
        default=CommentStrategy.INCLUDE, description="Comment strategy to use."
    )

    chunk_size: int = Field(
        default=1500, description="Chunk size to use for splitting code documents."
    )

    min_chunk_size: int = Field(
        default=256, description="Min tokens to split code."
    )

    max_chunk_size: int = Field(
        default=2000, description="Max tokens in one chunk."
    )

    hard_token_limit: int = Field(
        default=6000, description="Hard token limit for a chunk."
    )

    _parser: CodeParser = PrivateAttr()
    #_fallback_code_splitter: Optional[TextSplitter] = PrivateAttr() TODO: Implement fallback when tree sitter fails

    def __init__(
        self,
        chunk_size: int = 1024,
        language: str = "python", # TODO: Shouldn't have to set this
        min_chunk_size: int = 256,
        max_chunk_size: int = 1500,
        hard_token_limit: int = 6000,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        text_splitter: Optional[TextSplitter] = None,
        comment_strategy: CommentStrategy = CommentStrategy.ASSOCIATE,
        #fallback_code_splitter: Optional[TextSplitter] = None,
        include_non_code_files: bool = True,
        tokenizer: Optional[Callable] = None,
        non_code_file_extensions: Optional[List[str]] = ["md", "txt"],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        callback_manager = callback_manager or CallbackManager([])

        tokenizer = tokenizer or get_tokenizer()

        try:
            self._parser = create_parser(language, tokenizer=tokenizer)
        except Exception as e:
            logger.warning(
                f"Could not get parser for language {language}. Error: {e}")
            raise e

        #self._fallback_code_splitter = fallback_code_splitter

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=0,
            text_splitter=text_splitter or CodeSplitterV2(chunk_size=chunk_size, language="python"),
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            hard_token_limit=hard_token_limit,
            comment_strategy=comment_strategy,
            include_non_code_files=include_non_code_files,
            non_code_file_extensions=non_code_file_extensions,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls):
        return "GhostcoderNodeParser"

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        all_nodes: List[BaseNode] = []

        for node in nodes_with_progress:
            file_path = node.metadata.get("file_path")
            content = node.get_content()

            try:
                # TODO: Derive language from file extension
                codeblock = self._parser.parse(content)
            except Exception as e:
                logger.warning(
                    f"Failed to use epic splitter to split {file_path}. Fallback to treesitter_split(). Error: {e}")
                # TODO: Fall back to treesitter or text split
                continue

            chunks = self._chunk_contents(codeblock=codeblock, file_path=file_path)

            for chunk in chunks:
                content = self._to_context_string(codeblock, chunk)
                chunk_node = self._create_node(content, node, chunk=chunk)
                if chunk_node:
                    all_nodes.append(chunk_node)

        return all_nodes

    def _chunk_contents(self, codeblock: CodeBlock = None, file_path: str = "") -> List[CodeBlockChunk]:
        tokens = codeblock.sum_tokens()
        if tokens == 0:
            logger.debug(f"Skipping file {file_path} because it has no tokens.")
            return []

        if codeblock.find_errors():
            logger.warning(
                f"Failed to use spic splitter to split {file_path}. {len(codeblock.find_errors())} codeblocks with type ERROR. Fallback to treesitter_split()")
            # TODO: Fall back to treesitter or text split
            return []

        if all(codeblock.type in NON_CODE_BLOCKS for block in codeblock.children):
            logger.info(f"Skipping file {file_path} because it has no code blocks.")
            return []

        if tokens < self.min_chunk_size:
            return [[codeblock]]

        return self._chunk_block(codeblock, file_path)

    def _chunk_block(self, codeblock: CodeBlock, file_path: str = None) -> list[CodeBlockChunk]:
        chunks: list[CodeBlockChunk] = []
        current_chunk = []
        comment_chunk = []

        parent_tokens = count_parent_tokens(codeblock)

        ignoring_comment = False

        for child in codeblock.children:
            if child.type == CodeBlockType.COMMENT:
                if self.comment_strategy == CommentStrategy.EXCLUDE:
                    continue
                elif self._ignore_comment(child) or ignoring_comment:
                    ignoring_comment = True
                    continue
                # not first
                elif self.comment_strategy == CommentStrategy.ASSOCIATE and current_chunk:
                    comment_chunk.append(child)
                    continue
            else:
                if child.tokens > self.max_chunk_size:
                    start_content = child.content[:100]
                    logger.warning(f"Skipping code block {child.path_string()} in {file_path} as it has {child.tokens} tokens which is"
                                   f" more than chunk size {self.chunk_size}. Content: {start_content}...")
                    continue

                ignoring_comment = False

            if ((child.type in SPLIT_BLOCK_TYPES and child.sum_tokens() > self.min_chunk_size)
                    or parent_tokens + child.sum_tokens() > self.max_chunk_size):
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []

                current_chunk.extend(comment_chunk)
                comment_chunk = []
                current_chunk.append(child)

                child_chunks = self._chunk_block(child)

                if child_chunks:
                    first_child_chunk = child_chunks[0]

                    if parent_tokens + child.tokens + count_chunk_tokens(first_child_chunk) < self.max_chunk_size:
                        current_chunk.extend(first_child_chunk)
                        chunks.append(current_chunk)
                        chunks.extend(child_chunks[1:])
                        current_chunk = []
                    else:
                        chunks.append(current_chunk)
                        chunks.extend(child_chunks)
                        current_chunk = []

                continue

            new_token_count = parent_tokens + count_chunk_tokens(current_chunk) + child.sum_tokens()
            if codeblock.type not in SPLIT_BLOCK_TYPES and new_token_count < self.max_chunk_size \
                    or new_token_count < self.chunk_size:

                current_chunk.extend(comment_chunk)
                current_chunk.append(child)
            else:
                if current_chunk:
                    current_chunk.extend(comment_chunk)
                    chunks.append(current_chunk)
                current_chunk = [child]

            comment_chunk = []
            child_blocks = child.get_all_child_blocks()
            current_chunk.extend(child_blocks)

        if chunks and count_chunk_tokens(current_chunk) < self.min_chunk_size:
            chunks[-1].extend(current_chunk)
        else:
            chunks.append(current_chunk)

        return chunks

    def _ignore_comment(self, codeblock: CodeBlock) -> bool:
        return re.search(r"(?i)copyright|license|author", codeblock.content) or not codeblock.content

    def _to_context_string(self, codeblock: CodeBlock, show_blocks: List["CodeBlock"]) -> str:
        contents = ""

        if codeblock.pre_lines:
            contents += "\n" * (codeblock.pre_lines - 1)
            for line in codeblock.content_lines:
                if line:
                    contents += "\n" + codeblock.indentation + line
                else:
                    contents += "\n"
        else:
            contents += codeblock.pre_code + codeblock.content

        has_outcommented_code = False
        for i, child in enumerate(codeblock.children):
            show_code = child in show_blocks
            if show_code:
                if has_outcommented_code and child.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
                    if codeblock.type not in [CodeBlockType.CLASS, CodeBlockType.MODULE, CodeBlockType.TEST_SUITE]:
                        contents += child.create_commented_out_block("... other code").to_string()
                contents += self._to_context_string(
                    codeblock=child,
                    show_blocks=show_blocks)
                has_outcommented_code = False
            elif child.has_any_block(show_blocks):
                contents += self._to_context_string(
                    codeblock=child,
                    show_blocks=show_blocks)
                has_outcommented_code = False
            elif child.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
                has_outcommented_code = True

        if has_outcommented_code and codeblock.type not in [CodeBlockType.CLASS, CodeBlockType.MODULE, CodeBlockType.TEST_SUITE]:
            contents += child.create_commented_out_block("... other code").to_string()

        return contents

    def _create_node(self, content: str, node: BaseNode, chunk: CodeBlockChunk = None) -> Optional[TextNode]:
        metadata = {
            "file_path": node.metadata.get("file_path"),
            "file_name": node.metadata.get("file_name"),
            "file_type": node.metadata.get("file_type"),
        }

        if chunk:
            metadata["start_line"] = chunk[0].start_line
            metadata["end_line"] = chunk[-1].end_line

        content = content.strip()

        tokens = get_tokenizer()(content)
        if len(tokens) > self.hard_token_limit:
            logger.warning(f"Chunk in {node.metadata.get('file_path')} has {len(tokens)} tokens, will cut of at {self.hard_token_limit} tokens.")
            tokens = tokens[:self.hard_token_limit]

            try:
                # TODO: Should be generic like get_tokenizer()
                import tiktoken
                enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                content = enc.decode(tokens)
            except Exception as e:
                logger.warning(f"Failed to decode tokens to text. Error: {e}")
                return None

        metadata["tokens"] = len(tokens)

        excluded_embed_metadata_keys = node.excluded_embed_metadata_keys.copy()
        excluded_embed_metadata_keys.extend(["start_line", "end_line", "tokens"])

        node_id = node.id_

        if chunk:
            node_id += f"_{chunk[0].path_string()}_{chunk[-1].path_string()}"

        return TextNode(
            id_=node_id,
            text=content,
            metadata=metadata,
            excluded_embed_metadata_keys=excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
            metadata_seperator=node.metadata_seperator,
            metadata_template=node.metadata_template,
            text_template=node.text_template,
            relationships={NodeRelationship.SOURCE: node.as_related_node_info()},
        )

    def _count_tokens(self, text: str):
        tokenizer = get_tokenizer()
        return len(tokenizer(text))


if "__main__" == __name__:
    with open("/tmp/repos/sqlfluff/src/sqlfluff/dialects/dialect_postgres_keywords.py", "r") as file:
        content = file.read()

    parser = create_parser("python", tokenizer=get_tokenizer())

    splitter = EpicSplitter(
        chunk_size=750,
        min_chunk_size=100,
        comment_strategy=CommentStrategy.ASSOCIATE,
    )

    codeblock = parser.parse(content)

    print(codeblock.to_tree())

    tokenizer = get_tokenizer()

    chunks = splitter._chunk_block(codeblock)

    for i, chunk in enumerate(chunks):
        content = splitter._to_context_string(codeblock, chunk)
        print(f"\n========== Chunk {i} ==========\n\n")

        tokens = len(get_tokenizer()(content))
        print(f"Tokens: {tokens}")

        for block in chunk:
            print(f"- {block.path_string()}")

        print(content)

