import json

from codeblocks import CodeBlockType
from codeblocks.parser.java import JavaParser


def test_java_class():
    with open("java/Example.java", "r") as f:
        content = f.read()

    parser = JavaParser()

    code_blocks = parser.parse(content)

    print(code_blocks.trim_code_block(include_types=[CodeBlockType.MODULE, CodeBlockType.CLASS, CodeBlockType.FUNCTION]).to_string())
    print(code_blocks.to_tree())
    assert code_blocks.to_string() == content


def test_java_invalid():
    with open("java/Book_invalid_update.java", "r") as f:
        content = f.read()

    parser = JavaParser()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())
    assert len(code_blocks.find_errors()) == 3
