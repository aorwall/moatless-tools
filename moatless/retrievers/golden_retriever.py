import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, QueryBundle
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


class RetrieverStrategy(Enum):
    CODE_SNIPPETS = "code_snippets"
    FILES = "files"


class GoldenRetriever(CodeSnippetRetriever):

    def __init__(self,
                 vector_store: BasePydanticVectorStore,
                 docstore: DocumentStore,
                 embed_model: BaseEmbedding,
                 retriever_strategy: RetrieverStrategy = RetrieverStrategy.CODE_SNIPPETS):
        self._vector_store = vector_store
        self._docstore = docstore
        self._embed_model = embed_model
        self._retriever_strategy = retriever_strategy

    def retrieve(self, query: str) -> List[CodeSnippet]:

        query_embedding = self._embed_model.get_query_embedding(query)

        query_bundle = VectorStoreQuery(
            query_str=query,
            query_embedding=query_embedding,
            similarity_top_k=250)

        result = self._vector_store.query(query_bundle)
        files = []

        ignored_removed_docs = 0

        for node_id, distance in zip(result.ids, result.similarities):
            node_doc = self._docstore.get_document(node_id, raise_error=False)
            if not node_doc:
                ignored_removed_docs += 1
                continue

            if self._retriever_strategy == RetrieverStrategy.FILES:
                with open(node_doc.metadata['file_path'], "r") as f:
                    content = f.read()
                files.append(CodeSnippet(
                    id=node_doc.metadata['file_path'],
                    file_path=node_doc.metadata['file_path'],
                    distance=distance,
                    content=content,
                    start_line=0,
                    end_line=len(content.split("\n")),
                ))
            else:
                files.append(CodeSnippet(
                    id=node_doc.id_,
                    file_path=node_doc.metadata['file_path'],
                    distance=distance,
                    content=node_doc.get_content(),
                    start_line=node_doc.metadata.get('start_line', None),
                    end_line=node_doc.metadata.get('end_line', None),
                ))

        logger.info(f"Found {len(files)} code snippets. Ignored {ignored_removed_docs} removed code snippets.")

        return files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    repo_path = "/tmp/repos/astropy"
    persist_dir = "/tmp/repos/astropy-storage"

    docstore = SimpleDocumentStore.from_persist_dir(persist_dir)
    vector_store = SimpleFaissVectorStore.from_persist_dir(persist_dir)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    retriever = GoldenRetriever(
        docstore=docstore,
        vector_store=vector_store,
        embed_model=embed_model)

    #retriever.run_index()

    #simple_store.persist("/tmp/testing/vector_store.json")

    query = """Issue when passing empty lists/arrays to WCS transformations
    The following should not fail but instead should return empty lists/arrays:

    \`\`\`
    In [1]: from astropy.wcs import WCS

    In [2]: wcs = WCS('2MASS_h.fits')

    In [3]: wcs.wcs_pix2world([], [], 0)
    ---------------------------------------------------------------------------
    InconsistentAxisTypesError                Traceback (most recent call last)
    <ipython-input-3-e2cc0e97941a> in <module>()
    ----> 1 wcs.wcs_pix2world([], [], 0)

    ~/Dropbox/Code/Astropy/astropy/astropy/wcs/wcs.py in wcs_pix2world(self, *args, **kwargs)
       1352         return self._array_converter(
       1353             lambda xy, o: self.wcs.p2s(xy, o)['world'],
    -> 1354             'output', *args, **kwargs)
       1355     wcs_pix2world.__doc__ = \"""
       1356         Transforms pixel coordinates to world coordinates by doing

    ~/Dropbox/Code/Astropy/astropy/astropy/wcs/wcs.py in _array_converter(self, func, sky, ra_dec_order, *args)
       1267                     "a 1-D array for each axis, followed by an origin.")
       1268 
    -> 1269             return _return_list_of_arrays(axes, origin)
       1270 
       1271         raise TypeError(

    ~/Dropbox/Code/Astropy/astropy/astropy/wcs/wcs.py in _return_list_of_arrays(axes, origin)
       1223             if ra_dec_order and sky == 'input':
       1224                 xy = self._denormalize_sky(xy)
    -> 1225             output = func(xy, origin)
       1226             if ra_dec_order and sky == 'output':
       1227                 output = self._normalize_sky(output)

    ~/Dropbox/Code/Astropy/astropy/astropy/wcs/wcs.py in <lambda>(xy, o)
       1351             raise ValueError("No basic WCS settings were created.")
       1352         return self._array_converter(
    -> 1353             lambda xy, o: self.wcs.p2s(xy, o)['world'],
       1354             'output', *args, **kwargs)
       1355     wcs_pix2world.__doc__ = \"""

    InconsistentAxisTypesError: ERROR 4 in wcsp2s() at line 2646 of file cextern/wcslib/C/wcs.c:
    ncoord and/or nelem inconsistent with the wcsprm.
    \`\`\`"""
    snippets = retriever.retrieve(query)

    for snippet in snippets[:10]:
        print(f"{snippet.file_path} : {snippet.start_line} - {snippet.end_line} : {snippet.id}")