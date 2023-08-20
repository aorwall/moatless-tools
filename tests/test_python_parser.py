import json

from code_blocks.parser.python import PythonParser


def test_python_with_comment():
    with open("python/calculator_insert1.py", "r") as f:
        content = f.read()
    parser = PythonParser()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content



def test_python_has_error_blocks():
    with open("python/dsl_dsl_find_update.py", "r") as f:
        content = f.read()
    parser = PythonParser()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert len(code_blocks.find_errors()) == 1
    assert code_blocks.find_errors()[0].to_string() == """
        for item in data
            # ...
            elif item[0] == EDGE:"""


def test_python_if_and_indentation_parsing_2():
    with open("python/if_clause.py", "r") as f:
        content = f.read()
    parser = PythonParser()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content



def test_python_two_classes():
    with open("python/binary_search.py", "r") as f:
        content = f.read()
    parser = PythonParser()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content


def test_python_sublist():
    with open("python/sublist.py", "r") as f:
        content = f.read()
    parser = PythonParser()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content

