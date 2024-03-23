import logging
import traceback
from typing import Sequence, List, Optional, Dict, Any, Callable

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import NodeParser, TextSplitter
from llama_index.core.node_parser.node_utils import logger
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship
from llama_index.core.utils import get_tqdm_iterable, get_tokenizer

from epicsplit.codeblocks import create_parser, CodeParser
from epicsplit.codeblocks.codeblocks import NON_CODE_BLOCKS, PathTree, CodeBlock
from epicsplit.splitters.code_splitter_v2 import CodeSplitterV2


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

    chunk_size: int = Field(
        default=1500, description="Chunk size to use for splitting code documents."
    )

    min_chunk_size: int = Field(
        default=256, description="Min tokens to split code."
    )

    max_chunk_size: int = Field(
        default=4000, description="Max tokens in oen chunk."
    )
    _parser: CodeParser = PrivateAttr()
    #_fallback_code_splitter: Optional[TextSplitter] = PrivateAttr() TODO: Implement fallback when tree sitter fails

    def __init__(
        self,
        chunk_size: int = 1024,
        language: str = "python",
        min_chunk_size: int = 256,
        max_chunk_size: int = 4000,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        text_splitter: Optional[TextSplitter] = None,
        #fallback_code_splitter: Optional[TextSplitter] = None,
        include_non_code_files: bool = True,
        tokenizer: Optional[Callable] = None,
        non_code_file_extensions: Optional[List[str]] = ["md", "txt"],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        callback_manager = callback_manager or CallbackManager([])

        try:
            self._parser = create_parser(language)
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
            tokens = self._count_tokens(content)
            if tokens == 0:
                logger.debug(f"Skipping file {file_path} because it has no tokens.")
                continue

            # TODO: Derive language from file extension

            try:
                codeblock = self._parser.parse(node.get_content())
            except Exception as e:
                logger.warning(f"Failed to use ghostcoder to split {file_path}. Fallback to treesitter_split(). Error: {e}")
                # TODO: Fall back to treesitter or text split
                continue

            if codeblock.find_errors():
                logger.warning(f"Failed to use ghostcoder to split {file_path}. {len(codeblock.find_errors())} codeblocks with type ERROR. Fallback to treesitter_split()")
                # TODO: Fall back to treesitter or text split
                continue

            if all(codeblock.type in NON_CODE_BLOCKS for block in codeblock.children):
                logger.info(f"Skipping file {file_path} because it has no code blocks.")
                continue

            if tokens < self.min_chunk_size:
                all_nodes.append(self._create_node(content, node))
                continue

            all_nodes.extend(self.chunk_block(codeblock, node))

        return all_nodes


    def chunk_block(self, codeblock: CodeBlock, root_node: BaseNode = None) -> List[BaseNode]:
        nodes: List[BaseNode] = []

        current_path_tree = None
        start_line = 0

        splitted_block = None
        splitted_blocks = codeblock.get_indexable_blocks()
        for splitted_block in splitted_blocks:
            if current_path_tree:
                new_path_tree = current_path_tree.model_copy(deep=True)
                new_path_tree.add_to_tree(splitted_block.full_path())

                content = codeblock._to_context_string(new_path_tree, show_commented_out_code_comment=False).strip()
                if self._count_tokens(content) < self.min_chunk_size:
                    current_path_tree = new_path_tree
                else:
                    content = codeblock._to_context_string(current_path_tree,
                                                           show_commented_out_code_comment=False).strip()
                    node = self._create_node(content, root_node, tree=current_path_tree, start_line=start_line,
                                             end_line=splitted_block.start_line - 1)
                    nodes.append(node)
                    current_path_tree = None

            if not current_path_tree:
                current_path_tree = PathTree()
                current_path_tree.add_to_tree(splitted_block.full_path())
                start_line = splitted_block.start_line

            content = codeblock._to_context_string(current_path_tree, show_commented_out_code_comment=False).strip()
            if self._count_tokens(content) < self.min_chunk_size:
                continue

            tokens = count_tokens(content)
            if tokens > self.chunk_size:
                trimmed_block = codeblock.trim_code_block(splitted_block)

                try:
                    text_splits = self.text_splitter.split_text(trimmed_block.to_string())
                    for text_split in text_splits:
                        end_line = start_line + text_split.count("\n")
                        node = self._create_node(text_split, root_node, start_line=start_line, end_line=end_line)
                        nodes.append(node)
                        start_line = end_line
                except Exception as e:
                    logger.error(f"Failed to split text in code block {trimmed_block.path_string()} in {root_node.id_}. Error: {e}")
                    # TODO: Fallback split?

                current_path_tree = None
                continue

        if current_path_tree and splitted_block:
            content = codeblock._to_context_string(current_path_tree, show_commented_out_code_comment=False).strip()

            node = self._create_node(content, root_node, tree=current_path_tree, start_line=start_line, end_line=splitted_block.end_line)
            nodes.append(node)

        return nodes

    def chunk_block_to_tree(self, rootblock: CodeBlock, codeblock: CodeBlock, node: BaseNode = None) -> List[PathTree]:
        trees: List[PathTree] = []

        current_path_tree = None

        splitted_block = None
        for splitted_block in codeblock.children:
            if current_path_tree:
                new_path_tree = current_path_tree.model_copy(deep=True)
                new_path_tree.add_to_tree(splitted_block.full_path())

                content = rootblock._to_context_string(new_path_tree, show_commented_out_code_comment=False).strip()
                if self._count_tokens(content) < self.min_chunk_size:
                    current_path_tree = new_path_tree
                else:
                    trees.append(current_path_tree)
                    current_path_tree = None

            if not current_path_tree:
                current_path_tree = PathTree()
                current_path_tree.add_to_tree(splitted_block.full_path())

            content = rootblock._to_context_string(current_path_tree, show_commented_out_code_comment=False).strip()
            if self._count_tokens(content) < self.min_chunk_size:
                continue

            tokens = count_tokens(content)
            if tokens > self.chunk_size:
                for child in codeblock.children:  # TODO: Do recursive
                    trees.extend(self.chunk_block_to_tree(rootblock, child, node))

                current_path_tree = None
                continue

        if current_path_tree and splitted_block:
            trees.append(current_path_tree)

        trees = self._merge_trees(rootblock, trees)

        return trees

    def _merge_trees(self, codeblock: CodeBlock, trees: List[PathTree]) -> List[PathTree]:
        combined_trees = []
        current_tree = PathTree()
        for tree in trees:
            if current_tree:
                current_chunk_content = codeblock._to_context_string(current_tree, show_commented_out_code_comment=False).strip()
                current_chunk_tokens = self._count_tokens(current_chunk_content)
                new_path_tree = current_tree.model_copy(deep=True)
            else:
                current_chunk_tokens = 0
                new_path_tree = PathTree()

            new_path_tree.merge(tree)
            chunk_content = codeblock._to_context_string(current_tree,
                                                         show_commented_out_code_comment=False).strip()
            chunk_tokens = self._count_tokens(chunk_content)

            if chunk_tokens < self.chunk_size:
                current_tree = new_path_tree
                continue

            if chunk_tokens > self.chunk_size > current_chunk_tokens:
                current_tree = PathTree()
                current_tree.merge(tree)

                combined_trees.append(current_tree)
                current_tree = None
                continue

            combined_trees.append(current_tree)

        if current_tree:
            combined_trees.append(current_tree)

        return combined_trees

    def _create_node(self, content: str, node: BaseNode, tree: PathTree = None, start_line: int = None, end_line: int = None) -> TextNode:
        metadata = {
            "file_path": node.metadata.get("file_path"),
            "file_name": node.metadata.get("file_name"),
            "file_type": node.metadata.get("file_type"),
        }

        if start_line is not None:
            metadata["start_line"] = start_line
            metadata["end_line"] = end_line

        metadata["tokens"] = self._count_tokens(content)

        excluded_embed_metadata_keys = node.excluded_embed_metadata_keys.copy()
        excluded_embed_metadata_keys.extend(["start_line", "end_line", "tokens"])

        node_id = node.id_

        if tree:
            node_id += f"_{tree.hash()}"
        elif start_line:
            node_id += f"_{start_line}"

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
