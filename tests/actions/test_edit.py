from pathlib import Path

import pytest

from moatless.actions.edit import ClaudeEditTool, EditActionArguments
from moatless.actions.model import Observation
from moatless.file_context import FileContext
from moatless.repository import FileRepository
from moatless.repository.repository import InMemRepository


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

    return "/src/test.py"

@pytest.fixture
def edit_action(repository):
    return ClaudeEditTool(repository=repository)

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

def test_create_command(edit_action, file_context, repo):
    new_file = Path(repo.repo_path) / "new.py"
    content = "print('New file')\n"
    
    args = EditActionArguments(
        scratch_pad="Create a new file",
        command="create",
        path=str(new_file),
        file_text=content
    )
    
    result = edit_action.execute(args, file_context)
    assert isinstance(result, Observation)
    assert "File created successfully" in result.message

def test_str_replace_command(edit_action, file_context, test_file):
    args = EditActionArguments(
        scratch_pad="Replace string in file",
        command="str_replace",
        path=str(test_file),
        old_str='print("Hello, World!")',
        new_str='print("Hi, World!")'
    )

    file_context.add_file(test_file, show_all_spans=True)
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


def test_file_not_found(edit_action, file_context, repo):
    non_existent = Path(repo.repo_path) / "nonexistent.py"
    args = EditActionArguments(
        scratch_pad="Try to view non-existent file",
        command="view",
        path=str(non_existent)
    )
    
    result = edit_action.execute(args, file_context)
    assert isinstance(result, Observation)
    assert result.expect_correction
    assert "does not exist. Please provide a valid path" in result.message


def test_str_replace_multiple_occurrences(edit_action, file_context, repo):
    # Modify file to have multiple occurrences
    file_path = Path(repo.repo_path) / "src" / "test_multiple.py"
    content = """print('test')
print('test')
"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)

    file_context.add_file("src/test_multiple.py", show_all_spans=True)

    args = EditActionArguments(
        scratch_pad="Replace string with multiple occurrences",
        command="str_replace",
        path="src/test_multiple.py",
        old_str="print('test')",
        new_str="print('updated')"
    )
    
    result = edit_action.execute(args, file_context)
    assert isinstance(result, Observation)
    assert "Multiple occurrences" in result.message
    assert result.properties.get("flags") == ["multiple_occurrences"]

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


