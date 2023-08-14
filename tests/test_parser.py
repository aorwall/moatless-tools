import json
from dataclasses import asdict

from codeblocks.parser import CodeBlockParser


def test_java_class():
    with open("java/treesitterexample.java", "r") as f:
        content = f.read()

    parser = CodeBlockParser("java")

    code_blocks = parser.parse(content)

    #code = str(code_blocks)
    assert str(code_blocks) == content
