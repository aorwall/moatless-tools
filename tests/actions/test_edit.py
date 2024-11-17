import pytest
from pathlib import Path

from moatless.actions.edit import ClaudeEditTool, EditActionArguments
from moatless.actions.model import Observation
from moatless.file_context import FileContext
from moatless.repository import FileRepository
from moatless.repository.repository import Repository

@pytest.fixture
def repo(tmp_path):
    return FileRepository(repo_path=str(tmp_path))

@pytest.fixture
def file_context(repo):
    return FileContext(repo=repo)


@pytest.fixture
def test_file(repo):
    file_path = Path(repo.repo_path) / "src" / "test.py"
    content = """def hello():
    print("Hello, World!")
    return True

def add(a, b):
    return a + b
"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)

    print(file_path)
    return "/src/test.py"

@pytest.fixture
def edit_action():
    return ClaudeEditTool()

def test_view_command(edit_action, file_context, test_file):
    args = EditActionArguments(
        scratch_pad="View the file content",
        command="view",
        path=str(test_file),
        view_range=None
    )
    
    result = edit_action.execute(args, file_context)
    assert isinstance(result, Observation)
    assert "def hello():" in result.message
    assert "def add(a, b):" in result.message

def test_view_with_range(edit_action, file_context, test_file):
    args = EditActionArguments(
        scratch_pad="View specific lines",
        command="view",
        path=str(test_file),
        view_range=[1, 3]
    )
    
    result = edit_action.execute(args, file_context)
    print(result.message)
    assert isinstance(result, Observation)
    assert "def hello():" in result.message
    assert "print(" in result.message
    assert "def add" not in result.message

def test_str_replace_command(edit_action, file_context, test_file):
    args = EditActionArguments(
        scratch_pad="Replace string in file",
        command="str_replace",
        path=str(test_file),
        old_str='    print("Hello, World!")',
        new_str='    print("Hi, World!")'
    )
    
    result = edit_action.execute(args, file_context)
    assert isinstance(result, Observation)
    assert result.properties.get("diff")
    assert 'print("Hi, World!")' in file_context.get_file(str(test_file)).content

def test_insert_command(edit_action, file_context, test_file):
    args = EditActionArguments(
        scratch_pad="Insert new line",
        command="insert",
        path=str(test_file),
        insert_line=1,
        new_str="# This is a test function"
    )
    
    result = edit_action.execute(args, file_context)
    assert isinstance(result, Observation)
    assert result.properties.get("diff")
    assert "# This is a test function" in file_context.get_file(str(test_file)).content



def test_str_replace_multiple_occurrences(edit_action, file_context, test_file):
    # Modify file to have multiple occurrences

    file_context.get_file(str(test_file)).apply_changes("print('test')\nprint('test')")

    args = EditActionArguments(
        scratch_pad="Replace string with multiple occurrences",
        command="str_replace",
        path=str(test_file),
        old_str="print('test')",
        new_str="print('updated')"
    )
    
    result = edit_action.execute(args, file_context)
    assert isinstance(result, Observation)
    assert result.expect_correction
    assert "Multiple occurrences" in result.message

def test_invalid_insert_line(edit_action, file_context, test_file):
    args = EditActionArguments(
        scratch_pad="Insert at invalid line",
        command="insert",
        path=str(test_file),
        insert_line=1000,
        new_str="This should fail"
    )
    
    result = edit_action.execute(args, file_context)
    assert isinstance(result, Observation)
    assert result.expect_correction 