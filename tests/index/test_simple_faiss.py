import os

from llama_index.core.vector_stores import VectorStoreQuery

from moatless.index import SimpleFaissVectorStore


def test_search_index():
    index_store_dir = os.getenv("INDEX_STORE_DIR")
    vector_store = SimpleFaissVectorStore.from_persist_dir(f"{index_store_dir}/django__django-12419")

    query_bundle = VectorStoreQuery(
        query_str="SECURE_REFERRER_POLICY setting",
        similarity_top_k=100,
    )

    result = vector_store.query(query_bundle)
    for res in result.ids:
        print(res)