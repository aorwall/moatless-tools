import logging
import os

import faiss
from llama_index.core import SimpleDirectoryReader, get_tokenizer
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import NodeRelationship
from llama_index.core.storage import docstore
from llama_index.core.storage.docstore import SimpleDocumentStore, DocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from moatless.code_graph import CodeGraph
from moatless.code_index import CodeIndex
from moatless.codeblocks import CodeBlock
from moatless.codeblocks.parser.python import PythonParser
from moatless.retriever import CodeSnippetRetriever
from moatless.splitters.epic_split import EpicSplitter, CommentStrategy
from moatless.store.simple_faiss import SimpleFaissVectorStore

logger = logging.getLogger(__name__)


class CodeBaseIngestionPipeline:

    def __init__(
        self,
        vector_store: BasePydanticVectorStore,
        docstore: DocumentStore,
        path: str,
        min_chunk_size=100,
        chunk_size=1500,
        hard_token_limit=2000,
        max_chunks=200,
        embed_model: BaseEmbedding = None,
        num_workers: int = None,
    ):
        self.path = path
        self.vector_store = vector_store
        self.docstore = docstore
        self._embed_model = embed_model

        self._code_graph = CodeGraph()

        self._splitter = EpicSplitter(
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
            code_graph=self._code_graph,  # TODO: Just to add all nodes to the code graph. This should be run after node parsing probably.
            hard_token_limit=hard_token_limit,
            max_chunks=max_chunks,
            language="python",
            comment_strategy=CommentStrategy.ASSOCIATE,
        )
        self.num_workers = num_workers

    @classmethod
    def from_path(cls, path: str, persist_dir: str = ".storage", **kwargs):
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

        # TODO: Try to download store

        try:
            vector_store = SimpleFaissVectorStore.from_persist_dir(persist_dir)
        except:
            faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(1536))
            vector_store = SimpleFaissVectorStore(faiss_index)

        try:
            docstore = SimpleDocumentStore.from_persist_dir(persist_dir)
        except:
            docstore = SimpleDocumentStore()

        return cls(vector_store=vector_store, docstore=docstore, path=path, **kwargs)

    def run(self, input_files: list[str] = None):
        reader = SimpleDirectoryReader(
            input_dir=self.path,
            exclude=[  # TODO: Shouldn't be hardcoded and filtered
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            input_files=input_files,
            filename_as_id=True,
            required_exts=[".py"],  # TODO: Shouldn't be hardcoded and filtered
            recursive=True,
        )

        embed_pipeline = IngestionPipeline(
            transformations=[self._embed_model],
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
            docstore=self.docstore,
            vector_store=self.vector_store,
        )

        docs = reader.load_data()
        logger.info(f"Read {len(docs)} documents")

        prepared_nodes = self._splitter.get_nodes_from_documents(
            docs, show_progress=True
        )

        tokenizer = get_tokenizer()
        sum_tokens = 0
        for node in prepared_nodes:
            tokens = len(tokenizer(node.get_content()))
            sum_tokens += tokens

            node.metadata["file_path"] = node.metadata["file_path"].replace(
                self.path, ""
            )
            if node.metadata["file_path"].startswith("/"):
                node.metadata["file_path"] = node.metadata["file_path"][1:]

            # To not use ref_id in pipeline and not change hash because of updated metadata
            node.relationships.pop(NodeRelationship.SOURCE, None)
            node.metadata.pop("file_size", None)
            node.metadata.pop("creation_date", None)
            node.metadata.pop("last_modified_date", None)

        logger.info(f"Prepared {len(prepared_nodes)} nodes with {sum_tokens} tokens")

        embedded_nodes = embed_pipeline.run(
            nodes=list(prepared_nodes), show_progress=True, num_workers=self.num_workers
        )

        tokens = sum([len(tokenizer(node.get_content())) for node in embedded_nodes])
        logger.info(f"Embedded {len(embedded_nodes)} vectors with {tokens} tokens")

        return len(embedded_nodes), tokens

    def index(self):
        return CodeIndex(retriever=self.retriever(), graph=self._code_graph)

    def retriever(self, max_context_tokens: int = 200000):
        return CodeSnippetRetriever(
            vector_store=self.vector_store,
            docstore=self.docstore,
            embed_model=self._embed_model,
            max_context_tokens=max_context_tokens,
        )

    def persist(self, persist_dir: str):
        self.vector_store.persist(persist_dir)
        self.docstore.persist(
            os.path.join(persist_dir, docstore.types.DEFAULT_PERSIST_FNAME)
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)

    import dotenv

    dotenv.load_dotenv("../../.env")

    repo_path = "/tmp/repos/pytest-dev_pytest"
    persist_dir = "/tmp/repos/pytest-dev_pytest-storage"

    ingestion = CodeBaseIngestionPipeline.from_path(
        path=repo_path,
        splitter=EpicSplitter(chunk_size=750, min_chunk_size=100, language="python"),
        embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
    )

    ingestion.run()
