import json

from code_blocks.parser import CodeParser


def test_java_class():
    with open("java/treesitterexample.java", "r") as f:
        content = f.read()

    parser = CodeParser("java")

    code_blocks = parser.parse(content)

    print(json.dumps(code_blocks.to_dict(), indent=2))
    assert str(code_blocks) == content


def test_python_class():
    with open("python/treesitterexample.py", "r") as f:
        content = f.read()
    with open("python/treesitterexample.json", "r") as f:
        expected_json = f.read()
    parser = CodeParser("python")

    code_blocks = parser.parse(content)

    assert str(code_blocks) == content
    assert json.dumps(code_blocks.to_dict(), indent=2) == expected_json


def test_python_with_comment():
    with open("python/calculator_insert1.py", "r") as f:
        content = f.read()
    parser = CodeParser("python")

    code_blocks = parser.parse(content)

    print(json.dumps(code_blocks.to_dict(), indent=2))

    assert str(code_blocks) == content
