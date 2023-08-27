import json

from codeblocks import CodeBlockType
from codeblocks.parser.python import PythonParser

parser = PythonParser()

def test_python_calculator():
    with open("python/calculator.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    print(code_blocks.to_string())

    assert code_blocks.to_string() == content

def test_python_with_comment():
    with open("python/calculator_insert1.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content

def test_python_with_comment_2():
    with open("python/calculator_insert2.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content



def test_python_example():
    with open("python/example.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))

    assert code_blocks.to_string() == content

def test_python_all_treesitter_types():
    with open("python/treesitter_types.py", "r") as f:
        content = f.read()
    with open("python/treesitter_types_expected.txt", "r") as f:
        expected_tree = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=False))

    assert code_blocks.to_tree() == expected_tree
    assert code_blocks.to_string() == content


def test_expression_list():
    code_blocks = parser.parse("1, 2, 3")
    assert code_blocks.to_tree() == """ 0 module ``
  1 code `1`
  1 code `,`
  1 code `2`
  1 code `,`
  1 code `3`
"""


def test_outer_inner_def():
    codeblocks = parser.parse("""def outer():
    x = 10
    def inner():
        nonlocal x
        x = 20
    inner()
    print(x)
""")
    print(codeblocks.to_tree(include_tree_sitter_type=True))


def test_python_has_error_blocks():
    with open("python/dsl_dsl_find_update.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree(include_tree_sitter_type=True))
    print(code_blocks.find_errors()[0].to_string())

    assert len(code_blocks.find_errors()) == 1
    assert code_blocks.find_errors()[0].to_string() == """
            # ...
            elif item[0] == EDGE:"""


def test_python_if_and_indentation_parsing_2():
    with open("python/if_clause.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content



def test_python_two_classes():
    with open("python/binary_search.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content


def test_python_sublist():
    with open("python/sublist.py", "r") as f:
        content = f.read()

    code_blocks = parser.parse(content)

    print(code_blocks.to_tree())

    assert code_blocks.to_string() == content
