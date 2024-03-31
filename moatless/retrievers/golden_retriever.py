import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, QueryBundle, get_tokenizer
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.data_structs.data_structs import EmptyIndexStruct, IndexDict
from llama_index.core.node_parser import NodeParser
from llama_index.core.storage.docstore import DocumentStore, SimpleDocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore, VectorStoreQuery
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from moatless.retriever import CodeSnippetRetriever, CodeSnippet
from moatless.splitters.epic_split import EpicSplitter
from moatless.store.simple_faiss import SimpleFaissVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionPipelineSetup:
    name: str
    splitter: NodeParser
    embed_model: BaseEmbedding
    dimensions: int = None


class RetrieverScope(Enum):
    CODE_SNIPPETS = "code_snippets"
    FILES = "files"


class GoldenRetriever(CodeSnippetRetriever):

    def __init__(self,
                 vector_store: BasePydanticVectorStore,
                 docstore: DocumentStore,
                 embed_model: BaseEmbedding,
                 select_file_model: str = "gpt-3.5-turbo-0125",
                 selection_token_interval: int = 10000,
                 tokenizer=None,
                 retriever_strategy: RetrieverScope = RetrieverScope.CODE_SNIPPETS):
        self._tokenizer = tokenizer or get_tokenizer()
        self._vector_store = vector_store
        self._docstore = docstore
        self._embed_model = embed_model
        self._select_file_model = select_file_model
        self._selection_token_interval = selection_token_interval
        self._retriever_strategy = retriever_strategy

    def retrieve(self, query: str) -> List[CodeSnippet]:
        query_embedding = self._embed_model.get_query_embedding(query)

        query_bundle = VectorStoreQuery(
            query_str=query,
            query_embedding=query_embedding,
            similarity_top_k=250)

        result = self._vector_store.query(query_bundle)
        code_snippets = []

        ignored_removed_docs = 0

        for node_id, distance in zip(result.ids, result.similarities):
            node_doc = self._docstore.get_document(node_id, raise_error=False)
            if not node_doc:
                ignored_removed_docs += 1
                continue

            if self._retriever_strategy == RetrieverScope.FILES:
                with open(node_doc.metadata['file_path'], "r") as f:
                    content = f.read()
                code_snippet = CodeSnippet(
                    id=node_doc.metadata['file_path'],
                    file_path=node_doc.metadata['file_path'],
                    distance=distance,
                    content=content,
                    start_line=0,
                    end_line=len(content.split("\n")),
                )
            else:
                code_snippet = CodeSnippet(
                    id=node_doc.id_,
                    file_path=node_doc.metadata['file_path'],
                    distance=distance,
                    content=node_doc.get_content(),
                    start_line=node_doc.metadata.get('start_line', None),
                    end_line=node_doc.metadata.get('end_line', None),
                )

            # TODO: Run select files when sum tokens > self._selection_token_interval

            code_snippets.append(code_snippet)
        logger.info(f"Found {len(code_snippets)} code snippets. Ignored {ignored_removed_docs} removed code snippets.")

        return code_snippets
