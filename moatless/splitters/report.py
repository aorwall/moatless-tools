import logging

from llama_index.core import get_tokenizer
from llama_index.core.node_parser import TokenTextSplitter, CodeSplitter
from llama_index.core.node_parser.interface import TextSplitter, NodeParser
from llama_index.core.schema import TextNode

from moatless.splitters.code_splitter_v2 import CodeSplitterV2
from moatless.splitters.epic_split import EpicSplitter, CommentStrategy


def generate_markdown(splitter: NodeParser, file_path: str) -> str:
    with open(file_path, "r") as file:
        content = file.read()

    tokenize = get_tokenizer()
    node = TextNode(text=content, metadata={"file_path": file_path})

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

    return md


def read_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()

def write_file(file_path: str, content: str):
    with open(file_path, "w") as file:
        file.write(content)


def generate_splits(file):
    content = read_file(file)

    splitter = EpicSplitter(
        chunk_size=750,
        min_chunk_size=100,
        comment_strategy=CommentStrategy.ASSOCIATE,
    )

    generate_markdown(splitter, content)





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)
    generate_splits("/tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py")
