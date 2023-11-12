import logging
import traceback
from typing import Sequence, List, Optional, Dict

from llama_index import Document
from llama_index.callbacks import CBEventType, CallbackManager
from llama_index.callbacks.schema import EventPayload
from llama_index.node_parser import NodeParser, SimpleNodeParser
from llama_index.node_parser.extractors import MetadataExtractor
from llama_index.schema import BaseNode, MetadataMode, TextNode, NodeRelationship
from llama_index.text_splitter import TokenTextSplitter, SplitterType, get_default_text_splitter
from llama_index.utils import get_tqdm_iterable
from pydantic import Field

from ghostcoder.codeblocks import create_parser, CodeBlock, CodeBlockType
from ghostcoder.utils import count_tokens


class CodeNodeParser(NodeParser):
    """Route to the right node parser depending on language set in document metadata"""

    text_splitter: SplitterType = Field(
        description="The text splitter to use when splitting documents."
    )
    include_metadata: bool = Field(
        default=True, description="Whether or not to consider metadata when splitting."
    )
    include_prev_next_rel: bool = Field(
        default=True, description="Include prev/next node relationships."
    )
    metadata_extractor: Optional[MetadataExtractor] = Field(
        default=None, description="Metadata extraction pipeline to apply to nodes."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    @classmethod
    def from_defaults(
        cls,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        text_splitter: Optional[SplitterType] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> "CodeNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        text_splitter = text_splitter or get_default_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            callback_manager=callback_manager,
        )
        return cls(
            text_splitter=text_splitter,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            metadata_extractor=metadata_extractor,
            _node_parser_map={}
        )

    @classmethod
    def class_name(cls):
        return "CodeNodeParser"

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:

        with self.callback_manager.event(
                CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )

            all_nodes: List[BaseNode] = []
            for document in documents_with_progress:
                language = document.metadata.get("language", None)
                if language:
                    try:
                        parser = create_parser(language)
                    except Exception as e:
                        logging.warning(f"Could not get parser for language {language}. Will not parse document {document.id_}")
                        continue

                    content = document.get_content(metadata_mode=MetadataMode.NONE)
                    if not content:
                        logging.warning(f"Could not get content for document {document.id_}")
                        continue

                    codeblock = parser.parse(content)
                    logging.debug(codeblock.to_tree(include_tree_sitter_type=False,
                                                    show_tokens=True,
                                                    include_types=[CodeBlockType.FUNCTION, CodeBlockType.CLASS]))

                    splitted_blocks = codeblock.split_blocks()

                    for splitted_block in splitted_blocks:
                        definitions, parent = self.get_parent_and_definitions(splitted_block)

                        node_metadata = document.metadata
                        node_metadata["definition"] = splitted_block.content
                        node_metadata["block_type"] = str(splitted_block.type)

                        if splitted_block.identifier:
                            node_metadata["identifier"] = splitted_block.identifier
                        else:
                            node_metadata["identifier"] = splitted_block.content[:80].replace("\n", "\\n")

                        node_metadata["start_line"] = splitted_block.start_line

                        tokens = count_tokens(parent.to_string())
                        if tokens > 4000:
                            logging.info(f"Skip node [{node_metadata['identifier']}] in {document.id_} with {tokens} tokens")
                            continue

                        if tokens > 1000:
                            logging.info(f"Big node [{node_metadata['identifier']}] in {document.id_} with {tokens} tokens")

                        # TODO: Add relationships between code blocks
                        node = TextNode(
                            text=parent.to_string(),
                            embedding=document.embedding,
                            metadata=node_metadata,
                            excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                            excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                            metadata_seperator=document.metadata_seperator,
                            metadata_template=document.metadata_template,
                            text_template=document.text_template,
                            relationships={NodeRelationship.SOURCE: document.as_related_node_info()},
                        )

                        all_nodes.append(node)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes

    def get_parent_and_definitions(self, codeblock: CodeBlock) -> (List[str], CodeBlock):
        definitions = [codeblock.content]
        if codeblock.parent:
            parent_defs, parent = self.get_parent_and_definitions(codeblock.parent)
            definitions.extend(parent_defs)
            return definitions, parent
        else:
            return definitions, codeblock
