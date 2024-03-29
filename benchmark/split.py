import logging

import faiss
from dotenv import load_dotenv
from llama_index.core import get_tokenizer
from llama_index.core.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.core.node_parser import TokenTextSplitter, CodeSplitter
from llama_index.core.node_parser.interface import TextSplitter, NodeParser
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.openai import OpenAIEmbedding

from moatless.splitters.code_splitter_v2 import CodeSplitterV2
from moatless.splitters.epic_split import EpicSplitter, CommentStrategy
from moatless.store.simple_faiss import SimpleFaissVectorStore


def generate_splitter_report(splitter: NodeParser, text: str):
    tokenize = get_tokenizer()
    node = TextNode(text=text, metadata={"file_path": "test.py"})

    chunks = splitter._parse_nodes([node])

    md = "\n\n## " + splitter.__class__.__name__ + "\n\n"
    md += f"{len(chunks)} chunks"
    for i, chunk in enumerate(chunks):
        md += f"\n\n#### Split {i + 1}\n"

        tokens = len(tokenize(chunk.get_content()))
        md += "" + str(tokens) + " tokens"

        start_line = chunk.metadata.get("start_line", None)
        if start_line is not None:
            end_line = chunk.metadata.get("end_line", None)
            md += f", line: {start_line} - {end_line}"

        md += f"\n\n```python\n{chunk.get_content()}\n```\n\n"

    write_file(splitter.__class__.__name__ + ".md", md)


def read_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


def write_file(file_path: str, content: str):
    with open(file_path, "w") as file:
        file.write(content)


def generate_splits(file, query):
    content = read_file(file)

    splitter = EpicSplitter(
        chunk_size=750,
        min_chunk_size=100,
        comment_strategy=CommentStrategy.ASSOCIATE,
    )

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    node = TextNode(text=content, metadata={"file_path": file})

    chunks = {}
    contents = []
    ids = []
    for chunk_node in splitter._parse_nodes([node]):
        contents.append(chunk_node.get_content())
        ids.append(chunk_node.id_)
        chunks[chunk_node.id_] = chunk_node

    embeddings = embed_model._get_text_embeddings(contents)

    query_embedding = embed_model._get_text_embedding(query)

    similarities, ids = get_top_k_embeddings(query_embedding=query_embedding, embeddings=embeddings, embedding_ids=ids, similarity_top_k=25)

    print("===== get_top_k_embeddings ====")
    for id in ids:
        print(f"{chunks[id].metadata.get('start_line')} - {chunks[id].metadata.get('end_line')}")
        print(chunks[id].get_content())



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.WARNING)

    load_dotenv("../.env")

    query = """Possible bug in io.fits related to D exponents
I came across the following code in ``fitsrec.py``:

\`\`\`python
        # Replace exponent separator in floating point numbers
        if 'D' in format:
            output_field.replace(encode_ascii('E'), encode_ascii('D'))
\`\`\`

I think this may be incorrect because as far as I can tell ``replace`` is not an in-place operation for ``chararray`` (it returns a copy). Commenting out this code doesn't cause any tests to fail so I think this code isn't being tested anyway."""

    generate_splits("/tmp/repos/astropy/astropy/io/fits/fitsrec.py", query)
