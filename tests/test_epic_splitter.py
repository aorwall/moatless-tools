from llama_index.core import get_tokenizer

from moatless.codeblocks import create_parser
from moatless.splitters.epic_split import EpicSplitter, CommentStrategy


def test_parse_case_1():
    with open("data/python/split_case_1.py", "r") as file:
        content = file.read()

    parser = create_parser("python", tokenizer=get_tokenizer())

    splitter = EpicSplitter(
        chunk_size=750,
        min_chunk_size=100,
        comment_strategy=CommentStrategy.ASSOCIATE,
    )

    codeblock = parser.parse(content)

    print(codeblock.to_tree())


def test_split_case_1():
    with open(
            "/tmp/repos/sympy/sympy/integrals/rubi/rules/sine.py", "r") as file:
        content = file.read()

    parser = create_parser("python", tokenizer=get_tokenizer())

    splitter = EpicSplitter(
        chunk_size=750,
        min_chunk_size=100,
        comment_strategy=CommentStrategy.ASSOCIATE,
    )

    codeblock = parser.parse(content)

    print(codeblock.to_tree(only_identifiers=False))

    chunks = splitter._chunk_block(codeblock)

    for i, chunk in enumerate(chunks):
        content = splitter._to_context_string(codeblock, chunk)
        print(f"\n========== Chunk {i} ==========\n\n")

        tokens = len(get_tokenizer()(content))
        print(f"Tokens: {tokens}")

        paths = ",".join([block.path_string() for block in chunk])
        print()
        print(paths)

        print(content)



def test_split_case_2():
    """
    Split case with class with functions and long comments.
    """
    with open("data/python/split_case_2.py", "r") as file:
        content = file.read()

    parser = create_parser("python", tokenizer=get_tokenizer())

    splitter = EpicSplitter(
        chunk_size=750,
        min_chunk_size=100,
        comment_strategy=CommentStrategy.ASSOCIATE,
    )

    codeblock = parser.parse(content)

    print(codeblock.to_tree(only_identifiers=False))

    chunks = splitter._chunk_block(codeblock)

    for i, chunk in enumerate(chunks):
        content = splitter._to_context_string(codeblock, chunk)
        print(f"\n========== Chunk {i} ==========\n\n")

        tokens = len(get_tokenizer()(content))
        print(f"Tokens: {tokens}")

        for block in chunk:
            print(f"- {block.path_string()}")

        print(content)
