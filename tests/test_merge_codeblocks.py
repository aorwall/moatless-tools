from typing import List

from moatless.codeblocks.parser.python import PythonParser


def _verify_merge(original_content, updated_content, assertions):
    parser = PythonParser(apply_gpt_tweaks=True)

    original_block = parser.parse(original_content)
    updated_block = parser.parse(updated_content)

    print(f"Original block:\n{original_block.to_tree()}")
    print(f"Updated block:\n{updated_block.to_tree()}")

    original_block.merge(updated_block)
    print(f"Merged block:\n{original_block.to_tree()}")

    print(original_block.merge_history)

    assertions(original_block)


def test_merge_function_with_updated_line():
    original_content = """def foo():
    bar = 1"""

    updated_content = """def foo():
    bar = 2"""

    def assertion(original_block):
        foo_func = original_block.children[0]
        assert foo_func.identifier == "foo"
        assert foo_func.to_string() == "def foo():\n    bar = 2"

    _verify_merge(original_content, updated_content, assertion)


def test_merge_function_with_new_lines():
    original_content = """def foo():
    bar = 1"""

    updated_content = """def foo():
    bar = 1
    bar += 2"""

    def assertion(original_block):
        foo_func = original_block.children[0]
        assert foo_func.identifier == "foo"
        assert foo_func.to_string() == "def foo():\n    bar = 1\n    bar += 2"

    _verify_merge(original_content, updated_content, assertion)


def test_merge_function_with_removed_lines():
    original_content = """def foo():
    bar = 1
    bar += 2"""

    updated_content = """def foo():
    bar = 1"""

    def assertion(original_block):
        foo_func = original_block.children[0]
        assert foo_func.identifier == "foo"
        assert foo_func.to_string() == "def foo():\n    bar = 1"

    _verify_merge(original_content, updated_content, assertion)


def test_merge_with_outcommented_function():
    original_content = """class Foo:
    
    def foo():
        print('hello world')
        
    def bar():
        x = 1
        y = x + 1
"""

    updated_content = """class Foo:

    def foo():
        # ... existing code

    def bar():
        x = 1
        y = x + 2
    """

    def assertion(original_block):
        foo_class = original_block.children[0]
        assert foo_class.identifier == "Foo"
        assert len(foo_class.children) == 2
        assert foo_class.children[0].identifier == "foo"
        assert (
            foo_class.children[0].to_string()
            == "\n\n    def foo():\n        print('hello world')"
        )
        assert foo_class.children[1].identifier == "bar"
        assert (
            foo_class.children[1].to_string()
            == "\n\n    def bar():\n        x = 1\n        y = x + 2"
        )

    _verify_merge(original_content, updated_content, assertion)


def test_pytest_dev__pytest_5808():
    with open("data/python/pytest-dev__pytest-5808/original_pastebin.py", "r") as f:
        original_content = f.read()

    with open("data/python/pytest-dev__pytest-5808/updated_pastebin.py", "r") as f:
        updated_content = f.read()

    with open("data/python/pytest-dev__pytest-5808/expected_pastebin.py", "r") as f:
        expected_content = f.read()

    def assertion(original_block):
        assert original_block.to_string() == expected_content

    _verify_merge(original_content, updated_content, assertion)


def test_pytest_dev__pytest_5808_ignore_comments():
    with open("data/python/pytest-dev__pytest-5808/original_pastebin.py", "r") as f:
        original_content = f.read()

    with open("data/python/pytest-dev__pytest-5808/updated_pastebin_2.py", "r") as f:
        updated_content = f.read()

    with open("data/python/pytest-dev__pytest-5808/expected_pastebin.py", "r") as f:
        expected_content = f.read()

    def assertion(original_block):
        assert original_block.to_string() == expected_content

    _verify_merge(original_content, updated_content, assertion)


def test_pytest_dev__pytest_5808_only_function():
    with open("data/python/pytest-dev__pytest-5808/original_pastebin.py", "r") as f:
        original_content = f.read()

    with open(
        "data/python/pytest-dev__pytest-5808/update_pastebin_create_new_paste.py", "r"
    ) as f:
        updated_content = f.read()

    with open("data/python/pytest-dev__pytest-5808/expected_pastebin.py", "r") as f:
        expected_content = f.read()

    parser = PythonParser(apply_gpt_tweaks=True)

    original_block = parser.parse(original_content)
    updated_block = parser.parse(updated_content)

    print(f"Original block:\n{original_block.to_tree()}")
    print(f"Updated block:\n{updated_block.to_tree()}")

    original_block.replace_by_path(["create_new_paste"], updated_block.children[0])
    print(f"Merged block:\n{original_block.to_tree()}")

    assert original_block.to_string() == expected_content


def test_replace_function():
    with open("data/python/replace_function/original.py", "r") as f:
        original_content = f.read()

    with open("data/python/replace_function/update.py", "r") as f:
        updated_content = f.read()

    with open("data/python/replace_function/expected.py", "r") as f:
        expected_content = f.read()

    parser = PythonParser(apply_gpt_tweaks=True)

    original_block = parser.parse(original_content)
    updated_block = parser.parse(updated_content)

    print(f"Original block:\n{original_block.to_tree()}")
    print(f"Updated block:\n{updated_block.to_tree()}")

    block_path = ["MatrixShaping", "_eval_col_insert"]
    changed_block = find_by_path_recursive(updated_block, block_path)

    original_block.replace_by_path(block_path, changed_block)
    print(f"Merged block:\n{original_block.to_tree()}")

    assert original_block.to_string() == expected_content


def find_by_path_recursive(codeblock, block_path: List[str]):
    found = codeblock.find_by_path(block_path)
    if not found and len(block_path) > 1:
        return find_by_path_recursive(codeblock, block_path[1:])
    return found


def test_replace_function_with_context():
    with open("data/python/replace_function/original.py", "r") as f:
        original_content = f.read()

    with open("data/python/replace_function/update_with_context.py", "r") as f:
        updated_content = f.read()

    with open("data/python/replace_function/expected.py", "r") as f:
        expected_content = f.read()

    parser = PythonParser(apply_gpt_tweaks=True)

    original_block = parser.parse(original_content)
    updated_block = parser.parse(updated_content)

    print(f"Original block:\n{original_block.to_tree()}")
    print(f"Updated block:\n{updated_block.to_tree()}")

    change_block = updated_block.find_by_path(["MatrixShaping", "_eval_col_insert"])

    original_block.replace_by_path(["MatrixShaping", "_eval_col_insert"], change_block)
    print(f"Merged block:\n{original_block.to_tree()}")

    assert original_block.to_string() == expected_content


def test_replace_function_with_indentation():
    with open("data/python/replace_function/original.py", "r") as f:
        original_content = f.read()

    with open("data/python/replace_function/update_with_indentation.py", "r") as f:
        updated_content = f.read()

    with open("data/python/replace_function/expected.py", "r") as f:
        expected_content = f.read()

    parser = PythonParser(apply_gpt_tweaks=True)

    original_block = parser.parse(original_content)
    updated_block = parser.parse(updated_content)

    print(f"Original block:\n{original_block.to_tree()}")
    print(f"Updated block:\n{updated_block.to_tree()}")

    change_block = updated_block.find_by_path(["_eval_col_insert"])

    original_block.replace_by_path(["MatrixShaping", "_eval_col_insert"], change_block)
    print(f"Merged block:\n{original_block.to_tree()}")

    assert original_block.to_string() == expected_content
