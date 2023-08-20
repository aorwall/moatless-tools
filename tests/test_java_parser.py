import json

from code_blocks.parser.java import JavaParser


def test_java_class():
    with open("java/example.java", "r") as f:
        content = f.read()

    parser = JavaParser()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())
    assert code_blocks.to_string() == content


def test_java_invalid():
    with open("java/Book_invalid_update.java", "r") as f:
        content = f.read()

    parser = JavaParser()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())
    assert len(code_blocks.find_errors()) == 3
