import json

from codeblocks import CodeBlockType
from codeblocks.parser.java import JavaParser
from codeblocks.parser.parser import CodeParser


def test_java_class():
    with open("java/Example.java", "r") as f:
        content = f.read()
    parser = CodeParser("java")
    code_blocks = parser.parse(content)
    print(code_blocks.to_tree())
    assert code_blocks.to_string() == content
