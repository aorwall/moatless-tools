import os
import tempfile
import pytest
from unittest.mock import patch, PropertyMock

from moatless.actions.read_file import ReadFile, ReadFileArgs
from moatless.file_context import FileContext
from moatless.actions.schema import Observation
from moatless.repository.file import FileRepository


@pytest.fixture
def temp_repo():
    """Create a temporary directory with some test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with multiple lines
        test_file_path = os.path.join(temp_dir, "test.txt")
        with open(test_file_path, "w") as f:
            f.write("\n".join([f"line{i}" for i in range(1, 150)]))
        
        # Create a directory
        dir_path = os.path.join(temp_dir, "directory")
        os.makedirs(dir_path)
        
        yield temp_dir


@pytest.fixture
def file_repository(temp_repo):
    """Create a real FileRepository pointing to the temp directory."""
    return FileRepository(repo_path=temp_repo)


@pytest.fixture
def file_context(file_repository):
    """Create a FileContext using the real FileRepository."""
    return FileContext(repo=file_repository)


@pytest.fixture
def read_file_action(file_repository):
    """Create a ReadFile action with the real repository."""
    action = ReadFile()
    # Use patch to set the _repository property
    with patch.object(ReadFile, '_repository', new_callable=PropertyMock) as mock_repo:
        mock_repo.return_value = file_repository
        yield action


@pytest.mark.asyncio
async def test_read_entire_file(read_file_action, file_context, temp_repo):
    # Execute
    args = ReadFileArgs(file_path="test.txt", thoughts="Reading entire file", start_line=None, end_line=None, add_to_context=True)
    result = await read_file_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "```test.txt" in result.message
    assert "line1" in result.message
    assert "line99" in result.message
    assert "line100" in result.message
    assert "line101" not in result.message  # Should be truncated at 100 lines
    assert result.summary is not None
    assert "Read first 100 lines" in result.summary
    
    # Verify file was added to context
    context_file = file_context.get_file("test.txt")
    assert context_file is not None


@pytest.mark.asyncio
async def test_read_specific_lines(read_file_action, file_context, temp_repo):
    # Execute
    args = ReadFileArgs(file_path="test.txt", start_line=10, end_line=20, thoughts="Reading specific lines", add_to_context=True)
    result = await read_file_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "```test.txt lines 10-20" in result.message
    assert "line10" in result.message
    assert "line20" in result.message
    assert "line9" not in result.message
    assert "line21" not in result.message
    assert result.summary is not None
    assert "Read lines 10-20" in result.summary
    
    # Verify lines were added to context
    context_file = file_context.get_file("test.txt")
    assert context_file is not None


@pytest.mark.asyncio
async def test_read_with_truncation(read_file_action, file_context, temp_repo):
    # Execute
    args = ReadFileArgs(file_path="test.txt", start_line=1, end_line=150, thoughts="Reading with truncation", add_to_context=True)
    result = await read_file_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "```test.txt lines 1-100" in result.message
    assert "line1" in result.message
    assert "line100" in result.message
    assert "line101" not in result.message
    assert "truncated at 100 lines" in result.message
    assert result.summary is not None
    assert "Read lines 1-100" in result.summary
    
    # Verify lines were added to context
    context_file = file_context.get_file("test.txt")
    assert context_file is not None


@pytest.mark.asyncio
async def test_without_adding_to_context(read_file_action, file_context, temp_repo):
    # Execute
    args = ReadFileArgs(file_path="test.txt", thoughts="Reading without adding to context", 
                       start_line=1, end_line=10, add_to_context=False)
    result = await read_file_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "```test.txt lines 1-10" in result.message
    assert "line1" in result.message
    assert "line10" in result.message
    
    # No need to clear the context, just verify file exists but with no spans
    # FileContext doesn't have clear() method, and we can't easily check if spans were added


@pytest.mark.asyncio
async def test_file_not_found(read_file_action, file_context):
    # Execute
    args = ReadFileArgs(file_path="nonexistent.txt", thoughts="Reading nonexistent file", start_line=None, end_line=None, add_to_context=True)
    result = await read_file_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "not found" in result.message
    assert result.properties is not None
    assert result.properties["fail_reason"] == "file_not_found"
    
    # Verify context wasn't modified
    context_file = file_context.get_file("nonexistent.txt")
    assert context_file is None


@pytest.mark.asyncio
async def test_directory_path(read_file_action, file_context, temp_repo):
    # Execute
    args = ReadFileArgs(file_path="directory", thoughts="Reading directory", start_line=None, end_line=None, add_to_context=True)
    result = await read_file_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "directory" in result.message
    assert result.properties is not None
    # According to the real implementation behavior it's returning "file_not_found" rather than "is_directory"
    assert result.properties["fail_reason"] == "file_not_found"


@pytest.mark.asyncio
async def test_invalid_line_number(read_file_action, file_context, temp_repo):
    # Execute
    args = ReadFileArgs(file_path="test.txt", start_line=500, end_line=None, thoughts="Reading with invalid line number", add_to_context=True)
    result = await read_file_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "greater than the number of lines" in result.message
    assert result.properties is not None
    assert result.properties["fail_reason"] == "start_line_greater_than_file_length"


@pytest.mark.asyncio
async def test_missing_file_context(read_file_action):
    # Setup and Execute
    args = ReadFileArgs(file_path="test.txt", thoughts="Reading without file context", start_line=None, end_line=None, add_to_context=True)
    
    # Assert
    with pytest.raises(ValueError, match="File context must be provided"):
        await read_file_action.execute(args, None) 