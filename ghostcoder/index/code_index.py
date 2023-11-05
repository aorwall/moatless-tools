import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from llama_index import ServiceContext, StorageContext, load_index_from_storage, VectorStoreIndex, Document, \
    KnowledgeGraphIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.types import VectorStore
from pydantic import BaseModel, Field

from ghostcoder.codeblocks import create_parser, CodeBlockType
from ghostcoder.filerepository import FileRepository
from ghostcoder.index.node_parser import CodeNodeParser


class BlockSearchHit(BaseModel):
    score: float = Field(default=0, description ="The similarity score of the block.")
    type: str = Field(description="The type of the block.")
    content: str = Field(description="The content of the block.")


class FileSearchHit(BaseModel):
    path: str = Field(description="The path of the file.")
    blocks: List[BlockSearchHit] = Field(description="The blocks of the file.")


class CodeIndex:

    def __init__(self,
                 repository: FileRepository,
                 index_dir: str,
                 reload: bool = False,
                 vector_store: Optional[VectorStore] = None):
        self.repository = repository
        self.vector_store = vector_store
        self.index_dir = index_dir

        node_parser = CodeNodeParser.from_defaults(include_metadata=False)
        self.service_context = ServiceContext.from_defaults(node_parser=node_parser)

        docs = self._get_documents()

        if reload:
            self.initiate_index(docs)
            return

        try:
            storage_context = StorageContext.from_defaults(persist_dir=index_dir, vector_store=self.vector_store)
            self._index = load_index_from_storage(storage_context=storage_context, service_context=self.service_context, show_progress=True)
            logging.info("Index loaded from storage.")
            if index_dir:
                self.refresh(docs)
        except FileNotFoundError:
            logging.info("Index not found. Creating a new one...")
            self.initiate_index(docs)

    def initiate_index(self, docs):
        logging.info(f"Creating new index with {len(docs)} documents...")
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self._index = VectorStoreIndex.from_documents(documents=docs, service_context=self.service_context,
                                                      storage_context=storage_context, show_progress=True)
        if self.index_dir:
            self._index.storage_context.persist(persist_dir=self.index_dir)
            logging.info("New index created and persisted to storage.")

    def refresh(self, documents: List[Document]):
        docs_to_refresh = []
        for document in documents:
            existing_doc_hash = self._index._docstore.get_document_hash(document.get_doc_id())
            if existing_doc_hash != document.hash or existing_doc_hash is None:
                logging.debug(f"Found document to refresh: {document.get_doc_id()}. Existing hash: {existing_doc_hash}, new hash: {document.hash}")
                docs_to_refresh.append((existing_doc_hash, document))

        logging.info(f"Found {len(docs_to_refresh)} documents to refresh.")
        for i, (existing_doc_hash, document) in enumerate(docs_to_refresh):
            if existing_doc_hash != document.hash:
                logging.info(f"Refresh {document.get_doc_id()} ({i + 1}/{len(docs_to_refresh)})")
                self._index.update_ref_doc(document)
            elif existing_doc_hash is None:
                print(f"Insert {document.get_doc_id()} ({i + 1}/{len(docs_to_refresh)})")
                self._index.insert(document)

    def _get_documents(self):
        documents = []
        for file in self.repository.file_tree().traverse():
            if file.language in ["python", "java", "javascript", "typescript", "tsx"]:  # TODO: only supported
                data = self.repository.get_file_content(file.path)
                metadata = {
                    "path": file.path,
                    "is_test": file.test,
                    "language": file.language,
                }

                doc = Document(text=data, metadata=metadata)
                doc.id_ = str(file.path)

                documents.append(doc)
        return documents

    def search(self, query):
        retriever = self._index.as_retriever(similarity_top_k=20)

        nodes = retriever.retrieve(query)
        logging.info(f"Got {len(nodes)} hits")

        hits = {}
        for node in nodes:
            path = node.node.metadata.get("path")
            if path not in hits:
                hits[path] = FileSearchHit(path=path, blocks=[])
            hits[path].blocks.append(BlockSearchHit(
                similarity_score=node.score,
                type=node.node.metadata.get("type"),
                content=node.node.get_content()
            ))

        return hits.values()


if __name__ == "__main__":
    repo_dir = Path("")
    index_dir = ".index"
    exclude_dirs = [index_dir, ".index", ".gcoder"]

    logging_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=logging_format)

    repository = FileRepository(repo_path=repo_dir, exclude_dirs=exclude_dirs)

    db = chromadb.PersistentClient(path=index_dir + "/.chroma_db")
    chroma_collection = db.get_or_create_collection("collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    code_index = CodeIndex(repository=repository, index_dir=index_dir, vector_store=vector_store, reload=True)

    query = """
"""

    hits = code_index.search(query)
    for hit in hits:
        print(hit.path)
        for codeblock in hit.blocks:
            print(codeblock.score)
            print(codeblock.type)
            print(codeblock.content)
