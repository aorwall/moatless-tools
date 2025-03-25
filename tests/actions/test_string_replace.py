import pytest
import pytest_asyncio
import textwrap
from unittest.mock import Mock
from pathlib import Path

from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.actions.schema import Observation
from moatless.file_context import FileContext, ContextFile
from moatless.repository.repository import InMemRepository
from moatless.workspace import Workspace


def dedent(text):
    """Remove common leading whitespace from a multi-line string."""
    return textwrap.dedent(text).strip()


@pytest_asyncio.fixture
async def repository():
    repo = InMemRepository()
    # Add a file with specific content to test string replacement
    repo.save_file("existing.py", """def hello_world():
    message = "Hello World"
    return message

if __name__ == "__main__":
    print(hello_world())
""")
    return repo


@pytest_asyncio.fixture
async def workspace(repository):
    workspace = Mock(spec=Workspace)
    workspace.repository = repository
    return workspace


@pytest_asyncio.fixture
async def file_context(repository):
    context = FileContext(repo=repository, shadow_mode=False)
    return context


@pytest.mark.asyncio
async def test_string_replace_basic(file_context, repository, workspace):
    """Test basic string replacement with StringReplace."""
    # Set up string replace action
    string_replace_action = StringReplace()
    string_replace_action.workspace = workspace
    
    # Double-check the file content to use the exact string
    content = repository.get_file_content("existing.py")
    print(f"Original content:\n{content}")
    
    # Make sure the file is in context with show_all_spans=True and mark as viewed
    file_context.add_file("existing.py", show_all_spans=True)
    context_file = file_context.get_file("existing.py")
    context_file.was_viewed = True
    
    # Find the exact string we want to replace
    lines = content.splitlines()
    message_line = None
    for line in lines:
        if 'message = "Hello World"' in line:
            message_line = line
            break
    
    # Replace the message
    replace_args = StringReplaceArgs(
        path="existing.py",
        old_str=message_line,
        new_str=message_line.replace('Hello World', 'Hello Universe')
    )
    
    # Execute the string replace action
    replace_observation = await string_replace_action.execute(replace_args, file_context=file_context)
    
    # Verify the observation
    assert isinstance(replace_observation, Observation)
    assert "The file existing.py has been edited" in replace_observation.message
    assert "Review the changes" in replace_observation.summary
    assert "cat -n" in replace_observation.message
    
    # Verify content change
    modified_content = repository.get_file_content("existing.py")
    assert 'message = "Hello Universe"' in modified_content
    assert "def hello_world():" in modified_content  # Ensure other parts are preserved


@pytest.mark.asyncio
async def test_string_replace_multiline(file_context, repository, workspace):
    """Test replacing multiple lines at once."""
    # Set up string replace action
    string_replace_action = StringReplace()
    string_replace_action.workspace = workspace
    
    # Make sure the file is in context with show_all_spans=True and mark as viewed
    file_context.add_file("existing.py", show_all_spans=True)
    context_file = file_context.get_file("existing.py")
    context_file.was_viewed = True
    
    # Verify initial content
    content = repository.get_file_content("existing.py")
    print(f"Original content:\n{content}")
    
    # Replace the entire function
    original_func = """def hello_world():
    message = "Hello World"
    return message"""
    
    new_func = """def hello_universe():
    message = "Hello Universe"
    greeting = f"{message}!"
    return greeting"""
    
    replace_args = StringReplaceArgs(
        path="existing.py",
        old_str=original_func,
        new_str=new_func
    )
    
    # Execute string replace action
    replace_observation = await string_replace_action.execute(replace_args, file_context=file_context)
    
    # Verify the observation
    assert isinstance(replace_observation, Observation)
    assert "The file existing.py has been edited" in replace_observation.message
    
    # Verify content change
    modified_content = repository.get_file_content("existing.py")
    assert "def hello_universe():" in modified_content
    assert 'greeting = f"{message}!"' in modified_content


@pytest.mark.asyncio
async def test_string_replace_non_existent(file_context, repository, workspace):
    """Test replacement with non-existent string."""
    # Set up string replace action
    string_replace_action = StringReplace()
    string_replace_action.workspace = workspace
    
    # Make sure the file is in context with show_all_spans=True and mark as viewed
    file_context.add_file("existing.py", show_all_spans=True)
    context_file = file_context.get_file("existing.py")
    context_file.was_viewed = True
    
    replace_args = StringReplaceArgs(
        path="existing.py",
        old_str="This string does not exist in the file",
        new_str="New content"
    )
    
    # Execute string replace action
    replace_observation = await string_replace_action.execute(replace_args, file_context=file_context)
    
    # Verify error observation
    assert "String 'This string does not exist in the file' not found" in replace_observation.message
    assert replace_observation.properties.get("fail_reason") == "string_not_found"
    
    # Content should remain unchanged
    content = repository.get_file_content("existing.py")
    assert "New content" not in content


@pytest.mark.asyncio
async def test_string_replace_non_existent_file(file_context, repository, workspace):
    """Test replacement in a non-existent file."""
    # Set up string replace action
    string_replace_action = StringReplace()
    string_replace_action.workspace = workspace
    
    # Don't add the file to context - it doesn't exist
    
    replace_args = StringReplaceArgs(
        path="nonexistent.py",
        old_str="Any string",
        new_str="New content"
    )
    
    # Execute string replace action
    replace_observation = await string_replace_action.execute(replace_args, file_context=file_context)
    
    # Verify error observation indicates file not found
    assert replace_observation.properties.get("fail_reason") == "file_not_found"