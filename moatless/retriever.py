import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from llama_index.core import get_tokenizer
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.storage.docstore import DocumentStore
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)

logger = logging.getLogger(__name__)


@dataclass
class CodeSnippet:
    id: str
    file_path: str
    content: str = None
    distance: float = 0.0
    tokens: int = None
    language: str = "python"
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    start_block: Optional[str] = None
    end_block: Optional[str] = None


class CodeSnippetRetriever:

    def __init__(
        self,
        vector_store: BasePydanticVectorStore,
        docstore: DocumentStore,
        embed_model: BaseEmbedding,
        tokenizer=None,
        max_context_tokens: int = 200000,
    ):
        self._tokenizer = tokenizer or get_tokenizer()
        self._vector_store = vector_store
        self._docstore = docstore
        self._embed_model = embed_model
        self._max_context_tokens = max_context_tokens

    def retrieve(
        self,
        query: str,
        file_names: List[str] = None,
        keyword_filters: List[str] = None,
        top_k: int = 250,
    ) -> List[CodeSnippet]:
        query_embedding = self._embed_model.get_query_embedding(query)

        query_bundle = VectorStoreQuery(
            query_str=query, query_embedding=query_embedding, similarity_top_k=top_k
        )

        result = self._vector_store.query(query_bundle)
        code_snippets = []

        filtered_out_snippets = 0
        ignored_removed_snippets = 0

        accumulated_tokens = 0

        for node_id, distance in zip(result.ids, result.similarities):
            node_doc = self._docstore.get_document(node_id, raise_error=False)
            if not node_doc:
                ignored_removed_snippets += 1
                # TODO: Retry to get top_k results
                continue

            if file_names and not any(
                file_name in node_doc.metadata["file_path"] for file_name in file_names
            ):
                filtered_out_snippets += 1
                continue

            if keyword_filters:
                # TODO: Should be filtered on real metadata...
                if (
                    not any(
                        keyword.lower() in node_doc.get_content().lower()
                        for keyword in keyword_filters
                    )
                    and not any(
                        keyword in node_doc.metadata.get("start_block", "").lower()
                        for keyword in keyword_filters
                    )
                    and not any(
                        keyword in node_doc.metadata.get("end_block", "").lower()
                        for keyword in keyword_filters
                    )
                ):
                    filtered_out_snippets += 1
                    continue

            if "tokens" in node_doc.metadata:
                tokens = node_doc.metadata["tokens"]

                if tokens + accumulated_tokens > self._max_context_tokens:
                    logger.debug(
                        f"Reached max context tokens {self._max_context_tokens}). Currently {accumulated_tokens} tokens."
                    )
                    break

                accumulated_tokens += tokens
            else:
                tokens = None

            code_snippet = CodeSnippet(
                id=node_doc.id_,
                file_path=node_doc.metadata["file_path"],
                distance=distance,
                content=node_doc.get_content(),
                tokens=tokens,
                start_line=node_doc.metadata.get("start_line", None),
                end_line=node_doc.metadata.get("end_line", None),
                start_block=node_doc.metadata.get("start_block", None),
                end_block=node_doc.metadata.get("end_block", None),
            )

            code_snippets.append(code_snippet)

        logger.info(
            f"Found {len(code_snippets)} code snippets. Ignored {ignored_removed_snippets} removed code snippets. Filtered out {filtered_out_snippets} code snippets."
        )
        return code_snippets
