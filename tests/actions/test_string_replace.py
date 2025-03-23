import pytest
from unittest.mock import Mock

from moatless.actions.string_replace import StringReplace, StringReplaceArgs, find_potential_matches, find_exact_matches
from moatless.file_context import FileContext
from moatless.repository.repository import InMemRepository


@pytest.fixture
def repository():
    repo = InMemRepository()
    repo.save_file("test.py", """def hello():
    message = "Hello World"
    print(message)
""")
    return repo

@pytest.fixture
def file_context(repository):
    context = FileContext(repo=repository)
    # Add file to context to ensure it's available
    context.add_file("test.py", show_all_spans=True)
    return context


@pytest.mark.asyncio
async def test_string_replace_basic(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test.py",
        old_str='    message = "Hello World"',
        new_str='    message = "Hello Universe"',
        scratch_pad="Updating greeting message"
    )
    
    observation = await action.execute(args, file_context)
    print(observation.message)
    content = file_context.get_file("test.py").content
    assert 'message = "Hello Universe"' in content
    assert "def hello():" in content  # Verify the rest of the file is intact
    assert "print(message)" in content

@pytest.mark.asyncio
async def test_string_replace_not_found(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test.py",
        old_str='not_existing_string',
        new_str='new_string',
        scratch_pad="Trying to replace non-existent string"
    )
    
    observation = await action.execute(args, file_context)
    
    assert observation.properties.get("fail_reason") == "string_not_found"

@pytest.mark.asyncio
async def test_string_replace_multiple_occurrences(repository, file_context):
    # Create file with multiple occurrences
    repository.save_file("test2.py", """def hello():
    message = "test"
    print(message)
    message = "test"
""")
    file_context.add_file("test2.py", show_all_spans=True)
    
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test2.py",
        old_str='    message = "test"',  # Include proper indentation in search string
        new_str='    message = "updated"',
        scratch_pad="Updating test messages"
    )
    
    observation = await action.execute(args, file_context)
    
    assert 'multiple_occurrences' in observation.properties.get('flags', [])

@pytest.mark.asyncio
async def test_string_replace_file_not_found(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="nonexistent.py",
        old_str='old_string',
        new_str='new_string',
        scratch_pad="Trying to modify non-existent file"
    )
    
    observation = await action.execute(args, file_context)
    
    assert observation.properties.get("fail_reason") == "file_not_found"

@pytest.mark.asyncio
async def test_string_replace_same_string(repository, file_context):
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test.py",
        old_str='    message = "Hello World"',
        new_str='    message = "Hello World"',
        scratch_pad="Trying to replace with same string"
    )
    
    observation = await action.execute(args, file_context)
    print(observation.message)
    
    assert observation.properties.get("fail_reason") == "no_changes"

@pytest.mark.asyncio
async def test_string_replace_with_indentation(repository, file_context):
    # Create file with indented content - note the proper indentation
    repository.save_file("test3.py", """class Test:
    def method(self):
        value = "old"
        return value
""")
    file_context.add_file("test3.py", show_all_spans=True)
    
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test3.py",
        old_str='        value = "old"',  # Include proper indentation in search string
        new_str='        value = "new"',
        scratch_pad="Updating indented value"
    )
    
    observation = await action.execute(args, file_context)
    
    content = file_context.get_file("test3.py").content
    assert '        value = "new"' in content
    assert "class Test:" in content  # Verify the rest of the file is intact
    assert "def method(self):" in content

@pytest.mark.asyncio
async def test_string_replace_with_newlines(repository, file_context):
    # Create file with multiline content - note the proper indentation
    repository.save_file("test4.py", """def old_function():
    print("line1")
    print("line2")
""")
    file_context.add_file("test4.py", show_all_spans=True)
    
    action = StringReplace(repository=repository)
    args = StringReplaceArgs(
        path="test4.py",
        old_str='''def old_function():
    print("line1")
    print("line2")''',
        new_str='''def new_function():
    print("new_line1")
    print("new_line2")''',
        scratch_pad="Replacing entire function"
    )
    
    observation = await action.execute(args, file_context)
    
    content = file_context.get_file("test4.py").content
    assert 'def new_function():' in content
    assert '    print("new_line1")' in content  # Check indented content
    assert '    print("new_line2")' in content



def test_find_potential_matches_indentation_differs():
    content = """def hello():
        message = "Hello World"
        print(message)"""

    # Search with different indentation
    old_str = 'message = "Hello World"'

    matches = find_potential_matches(old_str, content)

    assert len(matches) == 1
    assert matches[0]['diff_reason'] == 'line_breaks_differ'
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 2
    assert matches[0]['content'] == '        message = "Hello World"'


def test_find_potential_matches_multiple_matches():
    content = """def process():
    data = transform().filter()

    result = (
        transform()
        .filter()
    )
    
    data = transform().filter()

    final = transform().filter()"""

    old_str = 'data = transform().filter()'
    matches = find_potential_matches(old_str, content)
    assert len(matches) == 2
    assert all(m['diff_reason'] == 'line_breaks_differ' for m in matches)

    old_str = 'transform().filter()'
    matches = find_potential_matches(old_str, content)
    assert len(matches) == 1
    assert all(m['diff_reason'] == 'line_breaks_differ' for m in matches)


def test_find_potential_matches_no_match():
    content = """def hello():
    message = "Hello World"
    print(message)"""

    old_str = 'not_existing_content'

    matches = find_potential_matches(old_str, content)

    assert len(matches) == 0


def test_find_potential_matches_multiline_with_different_indents():
    content = """def example():
    if condition:
        with open('file.txt') as f:
            process(f)"""

    # Search with different indentation levels
    old_str = """if condition:
with open('file.txt') as f:
    process(f)"""

    matches = find_potential_matches(old_str, content)

    assert len(matches) == 1
    assert matches[0]['diff_reason'] == 'line_breaks_differ'
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 4
    assert '    if condition:\n        with open(\'file.txt\') as f:\n            process(f)' in matches[0]['content']


def test_find_exact_matches_basic():
    # Matches test_string_replace_basic setup
    content = """def hello():
    message = "Hello World"
    print(message)
"""
    
    matches = find_exact_matches('    message = "Hello World"', content)
    assert len(matches) == 1
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 2
    assert matches[0]['content'] == '    message = "Hello World"'
    assert matches[0]['diff_reason'] == 'exact_match'

def test_find_exact_matches_multiline():
    content = """def process():
    if True:
        print("a")
        print("b")
    return None"""
    
    old_str = '''    if True:
        print("a")
        print("b")'''
    
    matches = find_exact_matches(old_str, content)
    assert len(matches) == 1
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 4
    assert matches[0]['content'] == old_str
    assert matches[0]['diff_reason'] == 'exact_match'

def test_find_exact_matches_multiple_occurrences():
    content = """def test():
    print("hello")
    print("world")
    print("hello")"""
    
    matches = find_exact_matches('    print("hello")', content)
    assert len(matches) == 2
    assert matches[0]['start_line'] == 2
    assert matches[1]['start_line'] == 4
    assert matches[0]['end_line'] == 2
    assert matches[1]['end_line'] == 4
    assert all(m['diff_reason'] == 'exact_match' for m in matches)
    assert all(m['content'] == '    print("hello")' for m in matches)

    # Test no matches
    matches = find_exact_matches('not_found', content)
    assert len(matches) == 0

def test_find_potential_matches_latex_case():
    # Real case from sympy/printing/latex.py
    content = """def test():
        return (r"\\left\\["
              + r", ".join(self._print(el) for el in printset)
              + r"\\right\\]")"""
    
    old_str = """return (r"\\left\\[" + r", ".join(self._print(el) for el in printset) + r"\\right\\]")"""
    
    matches = find_potential_matches(old_str, content)
    
    assert len(matches) == 1
    assert matches[0]['diff_reason'] == 'line_breaks_differ'
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 4
    assert matches[0]['content'] == '''        return (r"\\left\\["
              + r", ".join(self._print(el) for el in printset)
              + r"\\right\\]")'''

def test_find_potential_matches_multiline_string_concatenation():
    content = """    default_error_messages = {
        'invalid': _("'%(value)s' value has an invalid format. It must be in "
                     "[DD] [HH:[MM:]]ss[.uuuuuu] format.")
    }"""
    
    # Test with single-line version
    old_str = "'invalid': _(\"'%(value)s' value has an invalid format. It must be in [DD] [HH:[MM:]]ss[.uuuuuu] format.\")"
    
    matches = find_potential_matches(old_str, content)
    
    assert len(matches) == 1
    assert matches[0]['diff_reason'] == 'line_breaks_differ'
    assert matches[0]['start_line'] == 2
    assert matches[0]['end_line'] == 3
    assert matches[0]['content'] == """        'invalid': _("'%(value)s' value has an invalid format. It must be in "
                     "[DD] [HH:[MM:]]ss[.uuuuuu] format.")"""