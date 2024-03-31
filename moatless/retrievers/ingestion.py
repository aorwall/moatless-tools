import logging
import os
from dataclasses import dataclass
from enum import Enum

import chromadb
import faiss
from llama_index.core import SimpleDirectoryReader, get_tokenizer
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import NodeRelationship
from llama_index.core.storage.docstore import SimpleDocumentStore, DocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_PERSIST_FNAME
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from moatless.retrievers.golden_retriever import GoldenRetriever
from moatless.splitters.epic_split import EpicSplitter
from moatless.store.simple_faiss import SimpleFaissVectorStore


logger = logging.getLogger(__name__)

@dataclass
class IngestionPipelineSetup:
    name: str
    splitter: NodeParser
    embed_model: BaseEmbedding
    dimensions: int = None


class RetrieverStrategy(Enum):
    CODE_SNIPPETS = "code_snippets"
    FILES = "files"


class Ingestion:

    def __init__(self,
                 vector_store: BasePydanticVectorStore,
                 docstore: DocumentStore,
                 pipeline_setup: IngestionPipelineSetup,
                 path: str,
                 num_workers: int = None,
                 perist_dir: str = None):
        self.vector_store = vector_store
        self.docstore = docstore
        self.pipeline_setup = pipeline_setup
        self.path = path
        self.perist_dir = perist_dir
        self.num_workers = num_workers

    def run(self):
        reader = SimpleDirectoryReader(
            input_dir=self.path,
            exclude=[  # TODO: Shouldn't be hardcoded and filtered
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],  # TODO: Shouldn't be hardcoded and filtered
            recursive=True,
        )

        prepare_pipeline = IngestionPipeline(
            transformations=[self.pipeline_setup.splitter]
        )

        embed_pipeline = IngestionPipeline(
            transformations=[self.pipeline_setup.embed_model],
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
            docstore=self.docstore,
            vector_store=self.vector_store
        )

        docs = reader.load_data()
        logger.info(f"Read {len(docs)} documents")

        prepared_nodes = prepare_pipeline.run(documents=docs, show_progress=True, num_workers=self.num_workers)

        tokenizer = get_tokenizer()
        sum_tokens = 0
        for node in prepared_nodes:
            tokens = len(tokenizer(node.get_content()))
            sum_tokens += tokens

            # To not use ref_id in pipeline and not change hash because of updated metadata
            node.relationships.pop(NodeRelationship.SOURCE, None)
            node.metadata.pop('file_size', None)
            node.metadata.pop('creation_date', None)
            node.metadata.pop('last_modified_date', None)

        logger.info(f"Prepared {len(prepared_nodes)} nodes with {sum_tokens} tokens")

        embedded_nodes = embed_pipeline.run(nodes=list(prepared_nodes), show_progress=True, num_workers=self.num_workers)

        tokens = sum([len(tokenizer(node.get_content())) for node in embedded_nodes])
        logger.info(f"Embedded {len(embedded_nodes)} vectors with {tokens} tokens")

        return len(embedded_nodes), tokens


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)

    import dotenv
    dotenv.load_dotenv("../../.env")

    repo_path = "/tmp/repos/astropy"
    persist_dir = "/tmp/repos/astropy-storage"

    #db = chromadb.PersistentClient(path="/tmp/testing/chroma.db")
    #chroma_collection = db.get_or_create_collection("files")
    #vector_store = ChromaVectorStore(chroma_collection)

    try:
        vector_store = SimpleFaissVectorStore.from_persist_dir(persist_dir)
    except:
        faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(1536))
        vector_store = SimpleFaissVectorStore(faiss_index)

    #try:
    #simple_store = SimpleFaissVectorStore.from_persist_dir("/tmp/testing")
    #except:
    #    simple_store = SimpleFaissVectorStore()

    try:
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir)
    except:
        docstore = SimpleDocumentStore()

    try:
        index_store = SimpleIndexStore.from_persist_dir(persist_dir)
    except:
        index_store = SimpleIndexStore()

    pipeline_setup = IngestionPipelineSetup(
        name="text-embedding-3-small--epic-splitter-v4-100-750",
        splitter=EpicSplitter(chunk_size=750, min_chunk_size=100, language="python"),
        embed_model=OpenAIEmbedding(model="text-embedding-3-small")
    )

    ingestion = Ingestion(
        vector_store=vector_store,
        docstore=docstore,
        pipeline_setup=pipeline_setup,
        path=repo_path,
        perist_dir=persist_dir,
    )

    ingestion.run()

    docstore.persist(persist_path=os.path.join(persist_dir, DEFAULT_PERSIST_FNAME))
    logger.info(f"Docstore persisted to {persist_dir}")
    vector_store.persist(persist_dir=persist_dir)
    logger.info(f"Vector store persisted to {persist_dir}")

    #simple_store.persist("/tmp/testing/vector_store.json")

    retriever = GoldenRetriever(
        docstore=ingestion.docstore,
        vector_store=vector_store,
        embed_model=pipeline_setup.embed_model)

    snippets = retriever.retrieve("_test_build_index")

    for snippet in snippets[:10]:
        logger.info(f"{snippet.file_path} : {snippet.id}")