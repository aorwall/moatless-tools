from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.parser.python import PythonParser


def _verify_parsing(content, assertion):
    parser = PythonParser(apply_gpt_tweaks=True)

    codeblock = parser.parse(content)

    print(codeblock.to_tree())

    assertion(codeblock)


def test_function():
    content = """def foo():
    # ... existing code
    print('hello world')"""

    def assertion(codeblock):
        assert len(codeblock.children) == 1
        assert codeblock.children[0].identifier == "foo"
        assert len(codeblock.children[0].children) == 2

    _verify_parsing(content, assertion)


def test_outcommented_function():
    content = """def foo():
    # ... existing code"""

    def assertion(codeblock):
        assert len(codeblock.children) == 1
        assert codeblock.children[0].identifier == "foo"
        assert codeblock.children[0].type == CodeBlockType.COMMENTED_OUT_CODE

    _verify_parsing(content, assertion)


def test_function_followed_by_comment():
    content = """def foo():
    print('hello world')
    
# comment
"""

    def assertion(codeblock):
        assert len(codeblock.children) == 3

    _verify_parsing(content, assertion)


def test_outcommented_function_with_decorator():
    content = """@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # ... existing code"""

    def assertion(codeblock):
        assert len(codeblock.children) == 1
        assert codeblock.children[0].identifier == "pytest_configure"
        assert codeblock.children[0].type == CodeBlockType.COMMENTED_OUT_CODE

    _verify_parsing(content, assertion)


def test_outcommented_functions():
    content = """
@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # ... existing code

def pytest_unconfigure(config):
    # ... existing code

def create_new_paste(contents):
    import re

def pytest_terminal_summary(terminalreporter):
    # ... existing code"""

    def assertion(codeblock):
        assert len(codeblock.children) == 4
        assert ([child.identifier for child in codeblock.children] ==
                ["pytest_configure", "pytest_unconfigure", "create_new_paste", "pytest_terminal_summary"])
        assert ([child.type for child in codeblock.children] ==
                [CodeBlockType.COMMENTED_OUT_CODE, CodeBlockType.COMMENTED_OUT_CODE, CodeBlockType.FUNCTION, CodeBlockType.COMMENTED_OUT_CODE])

    _verify_parsing(content, assertion)


def test_realworld_example():
    with open("data/python/pytest-dev__pytest-5808/updated_pastebin.py", "r") as f:
        content = f.read()

    def assertion(codeblock):
        pass

    _verify_parsing(content, assertion)