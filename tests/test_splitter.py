from code_blocks.codeblocks import CodeBlock, CodeBlockType
from code_blocks.splitter import CodeSplitter

java_code = CodeBlock(
    content="public class TreeSitterExample ",
    type=CodeBlockType.CLASS,
    pre_code="",
    children=[
        CodeBlock(content="{",
                  type=CodeBlockType.BLOCK_DELIMITER,
                  pre_code=""),
        CodeBlock(
            content="int myVariable = 10;",
            pre_code="\n\n    "),
        CodeBlock(
            content="public void myMethod(int parameter) ",
            type=CodeBlockType.FUNCTION,
            pre_code="\n\n    ",
            children=[
                CodeBlock(
                    content="{",
                    type=CodeBlockType.BLOCK_DELIMITER,
                    pre_code=""),
                CodeBlock(
                    content="myVariable = parameter;",
                    type=CodeBlockType.CODE,
                    pre_code="\n        "),
                CodeBlock(
                    content="}",
                    pre_code="\n    ",
                    type=CodeBlockType.BLOCK_DELIMITER)]),
        CodeBlock(content="}",
                  type=CodeBlockType.BLOCK_DELIMITER,
                  pre_code="\n")
    ])


def test_chunk_block():
    splitter = CodeSplitter("java")

    chunks = splitter._chunk_block(java_code)

    i = 1
    for chunk in chunks:
        print("--- Chunk", i, "---")
        print(str(chunk.root()))
        i += 1


def test_trim_blocks():
    splitter = CodeSplitter("java")
    splitter.trim_code_block(java_code, java_code.children[2])
    assert str(java_code.children[3].root()) == """public class TreeSitterExample {

    // ...

    public void myMethod(int parameter) {
        myVariable = parameter;
    }
}"""


def test_java_class():
    with open("java/treesitterexample.java", "r") as f:
        content = f.read()

    with open("java/treesitterexample_splitted.md", "r") as f:
        expected = f.read()


    splitter = CodeSplitter("java", max_chars=400)

    chunks = splitter.split_text(content)

    mkdown = ""

    i = 1
    for chunk in chunks:
        mkdown += f"# Chunk {i}\n```java\n{chunk}\n```\n\n"
        i += 1

    print(mkdown)

    assert mkdown == expected


def test_split_python_class():
    with open("python/treesitterexample.py", "r") as f:
        content = f.read()

    with open("python/treesitterexample_splitted.md", "r") as f:
        expected = f.read()

    splitter = CodeSplitter("python", max_chars=400)

    chunks = splitter.split_text(content)

    mkdown = ""

    i = 1
    for chunk in chunks:
        mkdown += f"# Chunk {i}\n```python\n{chunk}\n```\n\n"
        i += 1

    print(mkdown)

    assert mkdown == expected