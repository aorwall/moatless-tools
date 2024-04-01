import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from llama_index.core import get_tokenizer
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.storage.docstore import DocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore, VectorStoreQuery

logger = logging.getLogger(__name__)


class RetrieverScope(Enum):
    CODE_SNIPPETS = "code_snippets"
    FILES = "files"


@dataclass
class CodeSnippet:
    id: str
    file_path: str
    content: str
    distance: float = 0.0
    tokens: int = None
    language: str = "python"
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    start_block: Optional[str] = None
    end_block: Optional[str] = None


class RetrieverStrategy(Enum):
    CODE_SNIPPETS = "code_snippets"
    FILES = "files"


class CodeSnippetRetriever:

    def __init__(self,
                 vector_store: BasePydanticVectorStore,
                 docstore: DocumentStore,
                 embed_model: BaseEmbedding,
                 tokenizer=None,
                 max_context_tokens: int = 200000,
                 retriever_strategy: RetrieverScope = RetrieverScope.CODE_SNIPPETS):
        self._tokenizer = tokenizer or get_tokenizer()
        self._vector_store = vector_store
        self._docstore = docstore
        self._embed_model = embed_model
        self._max_context_tokens = max_context_tokens
        self._retriever_strategy = retriever_strategy

    def retrieve(self, query: str) -> List[CodeSnippet]:
        query_embedding = self._embed_model.get_query_embedding(query)

        query_bundle = VectorStoreQuery(
            query_str=query,
            query_embedding=query_embedding,
            similarity_top_k=1000)

        result = self._vector_store.query(query_bundle)
        code_snippets = []

        ignored_removed_docs = 0

        accumulated_tokens = 0

        for node_id, distance in zip(result.ids, result.similarities):
            node_doc = self._docstore.get_document(node_id, raise_error=False)
            if not node_doc:
                ignored_removed_docs += 1
                continue

            tokens = None
            if 'tokens' in node_doc.metadata:
                tokens = node_doc.metadata['tokens']

                if tokens + accumulated_tokens > self._max_context_tokens:
                    logger.debug(f"Reached max context tokens {self._max_context_tokens}). Currently {accumulated_tokens} tokens.")
                    break

            accumulated_tokens += tokens

            if self._retriever_strategy == RetrieverScope.FILES:
                with open(node_doc.metadata['file_path'], "r") as f:
                    content = f.read()
                code_snippet = CodeSnippet(
                    id=node_doc.metadata['file_path'],
                    file_path=node_doc.metadata['file_path'],
                    distance=distance,
                    content=content,
                    tokens=tokens,
                    start_line=0,
                    end_line=len(content.split("\n"))
                )
            else:
                code_snippet = CodeSnippet(
                    id=node_doc.id_,
                    file_path=node_doc.metadata['file_path'],
                    distance=distance,
                    content=node_doc.get_content(),
                    tokens=tokens,
                    start_line=node_doc.metadata.get('start_line', None),
                    end_line=node_doc.metadata.get('end_line', None),
                    start_block=node_doc.metadata.get('start_block', None),
                    end_block=node_doc.metadata.get('end_block', None)
                )

            code_snippets.append(code_snippet)

        logger.info(f"Found {len(code_snippets)} code snippets. Ignored {ignored_removed_docs} removed code snippets.")
        return code_snippets
