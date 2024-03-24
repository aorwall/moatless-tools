import logging
import os
from dataclasses import dataclass
from typing import List

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, get_tokenizer
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.node_parser import NodeParser
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import TransformComponent, Document, BaseNode, NodeRelationship
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from moatless.retriever import CodeSnippetRetriever, CodeSnippet
from moatless.splitters.epic_split import EpicSplitter


@dataclass
class IngestionPipelineSetup:
    name: str
    splitter: NodeParser
    embed_model: BaseEmbedding
    dimensions: int = None


class LlamaIndexCodeSnippetRetriever(CodeSnippetRetriever):

    def __init__(self,
                 retriever: BaseRetriever,
                 reader: BaseReader = None,
                 prepare_pipeline: IngestionPipeline = None,
                 perist_dir: str = None,
                 embed_pipeline: IngestionPipeline = None):
        self.reader = reader
        self.prepare_pipeline = prepare_pipeline
        self.perist_dir = perist_dir
        self.embed_pipeline = embed_pipeline
        self.retriever = retriever


    @classmethod
    def from_vector_store(cls,
                          vector_store: BasePydanticVectorStore,
                          embed_model: BaseEmbedding):
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        retriever = vector_index.as_retriever(similarity_top_k=250)

        return cls(
            retriever=retriever,
        )

    @classmethod
    def from_pipeline_setup(cls,
                            vector_store: BasePydanticVectorStore,
                            pipeline_setup: IngestionPipelineSetup,
                            path: str,
                            perist_dir: str = None):
        reader = SimpleDirectoryReader(
            input_dir=path,
            exclude=[ # TODO: Shouldn't be hardcoded and filtered
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
            transformations=[pipeline_setup.splitter]
        )

        embed_pipeline = IngestionPipeline(
            transformations=[pipeline_setup.embed_model],
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
            docstore=SimpleDocumentStore(),
            vector_store=vector_store
        )

        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=pipeline_setup.embed_model,
        )

        retriever = vector_index.as_retriever(similarity_top_k=250)

        return cls(
            reader=reader,
            prepare_pipeline=prepare_pipeline,
            perist_dir=perist_dir,
            embed_pipeline=embed_pipeline,
            retriever=retriever,
        )

    def read_docs(self):
        docs = self.reader.load_data()
        print(f"Found {len(docs)} documents")
        return docs

    def prepare_nodes(self, docs: List[Document]):
        prepare_perist_dir = f"{self.perist_dir}/prepare_pipeline"

        #if os.path.exists(prepare_perist_dir):
        #    try:
                #self.prepare_pipeline.load(prepare_perist_dir)
        #    except Exception as e:
        #        logging.error(f"Failed to load pipeline from {prepare_perist_dir}: {e}")

        prepared_nodes = self.prepare_pipeline.run(documents=docs, show_progress=True)  # TODO: Set num_workers to run in parallell
        #self.prepare_pipeline.persist(prepare_perist_dir)

        tokenizer = get_tokenizer()

        return prepared_nodes

    def embed_nodes(self, nodes: List[BaseNode]):
        embed_perist_dir = f"{self.perist_dir}/embed_pipeline"

        if os.path.exists(embed_perist_dir):
            try:
                self.embed_pipeline.load(embed_perist_dir)
            except Exception as e:
                logging.error(f"Failed to load pipeline from {embed_perist_dir}: {e}")

        embedded_nodes = self.embed_pipeline.run(nodes=nodes, show_progress=True)  # TODO: Set num_workers to run in parallell
        self.embed_pipeline.persist(embed_perist_dir)

        return embedded_nodes

    def run_index(self) -> (int, int):
        docs = self.read_docs()
        prepared_nodes = self.prepare_nodes(docs)

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

        print(f"Prepared {len(prepared_nodes)} nodes with {sum_tokens} tokens")

        embedded_nodes = self.embed_nodes(list(prepared_nodes))
        tokens = sum([len(tokenizer(node.get_content())) for node in embedded_nodes])
        print(f"Embedded {len(embedded_nodes)} vectors with {tokens} tokens")

        return len(embedded_nodes), tokens

    def retrieve(self, query: str) -> List[CodeSnippet]:
        result = self.retriever.retrieve(query)
        return [CodeSnippet(
            path=node.node.metadata['file_path'],
            content=node.node.get_content(),
            start_line=node.node.metadata.get('start_line', None),
            end_line=node.node.metadata.get('end_line', None),
        ) for node in result]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    pipeline_setup = IngestionPipelineSetup(
        name="text-embedding-3-small--epic-splitter-v3-100-750",
        splitter=EpicSplitter(chunk_size=750, min_chunk_size=100, language="python"),
        embed_model=OpenAIEmbedding(model="text-embedding-3-small")
    )

    retriever = LlamaIndexCodeSnippetRetriever.from_pipeline_setup(
        pipeline_setup=pipeline_setup,
        path="/tmp/repos/sqlfluff",
        perist_dir="/tmp/repos/sqlfluff/text-embedding-3-small--epic-splitter-v3-100-750",
    )

    retriever.run_index()
