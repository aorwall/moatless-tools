import pytest
import pytest_asyncio
from unittest.mock import Mock

from moatless.actions.cleanup_context import CleanupContext, CleanupContextArgs
from moatless.file_context import FileContext
from moatless.repository.repository import InMemRepository
from moatless.workspace import Workspace


@pytest_asyncio.fixture
async def repository():
    repo = InMemRepository()
    # Add test files with content
    repo.save_file("file1.py", "# Test file 1\nprint('file1')")
    repo.save_file("file2.py", "# Test file 2\nprint('file2')")
    repo.save_file("file3.py", "# Test file 3\nprint('file3')")
    repo.save_file("target.py", "# Target file\nprint('target')")
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
async def test_cleanup_context_success(workspace, file_context):
    """Test successful removal of files from context."""
    # Setup - add some files to context using proper method
    file_context.add_file("file1.py")
    file_context.add_file("file2.py") 
    file_context.add_file("file3.py")
    
    action = CleanupContext()
    action._workspace = workspace
    
    args = CleanupContextArgs(file_paths=["file1.py", "file3.py"])
    result = await action.execute(args, file_context)
    
    # Verify files were removed
    assert "file1.py" not in file_context._files
    assert "file2.py" in file_context._files  # Should remain
    assert "file3.py" not in file_context._files
    
    # Verify result
    assert result.message is not None
    assert "Successfully removed 2 files from file context" in result.message
    assert result.summary == "Removed 2 files from context"
    assert result.properties["removed_count"] == 2
    assert result.properties["not_found_count"] == 0
    assert result.properties["removed_files"] == ["file1.py", "file3.py"]
    assert result.properties["not_found_files"] == []


@pytest.mark.asyncio
async def test_cleanup_context_file_not_found(workspace, file_context):
    """Test handling of files not found in context."""
    # Setup - add only one file to context
    file_context.add_file("file1.py")
    
    action = CleanupContext()
    action._workspace = workspace
    
    args = CleanupContextArgs(file_paths=["file1.py", "nonexistent.py"])
    result = await action.execute(args, file_context)
    
    # Verify only existing file was removed
    assert "file1.py" not in file_context._files
    
    # Verify result
    assert "Successfully removed file1.py from file context" in result.message
    assert "File nonexistent.py was not found in the current context" in result.message
    assert result.summary == "Removed 1 files, 1 not found"
    assert result.properties["removed_count"] == 1
    assert result.properties["not_found_count"] == 1


@pytest.mark.asyncio
async def test_cleanup_context_no_files_found(workspace, file_context):
    """Test handling when no requested files are found in context."""
    # Setup - empty context (don't add any files)
    
    action = CleanupContext()
    action._workspace = workspace
    
    args = CleanupContextArgs(file_paths=["nonexistent1.py", "nonexistent2.py"])
    result = await action.execute(args, file_context)
    
    # Verify no files were removed
    assert len(file_context._files) == 0
    
    # Verify result indicates failure
    assert result.summary == "No files removed - 2 not found in context"
    assert result.properties["removed_count"] == 0
    assert result.properties["not_found_count"] == 2
    assert result.properties["fail_reason"] == "no_files_removed"


@pytest.mark.asyncio
async def test_cleanup_context_single_file(workspace, file_context):
    """Test removal of a single file."""
    # Setup
    file_context.add_file("target.py")
    
    action = CleanupContext()
    action._workspace = workspace
    
    args = CleanupContextArgs(file_paths=["target.py"])
    result = await action.execute(args, file_context)
    
    # Verify file was removed
    assert "target.py" not in file_context._files
    
    # Verify result for single file
    assert "Successfully removed target.py from file context" in result.message
    assert result.summary == "Removed 1 files from context"


@pytest.mark.asyncio
async def test_cleanup_context_no_file_context():
    """Test that action fails when file context is not provided."""
    action = CleanupContext()
    action._workspace = Mock()
    
    args = CleanupContextArgs(file_paths=["file.py"])
    
    with pytest.raises(ValueError, match="File context must be provided"):
        await action.execute(args, None)


@pytest.mark.asyncio
async def test_cleanup_context_wrong_args_type(workspace, file_context):
    """Test that action fails when wrong argument type is provided."""
    action = CleanupContext()
    action._workspace = workspace
    
    # Use a different ActionArguments type
    from moatless.actions.schema import ActionArguments
    wrong_args = ActionArguments(thoughts="test")
    
    with pytest.raises(ValueError, match="Expected CleanupContextArgs"):
        await action.execute(wrong_args, file_context)


def test_cleanup_context_args_validation():
    """Test validation of CleanupContextArgs."""
    # Valid args
    args = CleanupContextArgs(file_paths=["file1.py", "file2.py"])
    assert args.file_paths == ["file1.py", "file2.py"]
    
    # Test log_name for single file
    single_args = CleanupContextArgs(file_paths=["single.py"])
    assert single_args.log_name == "CleanupContext(single.py)"
    
    # Test log_name for multiple files
    multi_args = CleanupContextArgs(file_paths=["file1.py", "file2.py"])
    assert multi_args.log_name == "CleanupContext(2 files)"
    
    # Test to_prompt for single file
    assert "Remove single.py from file context" in single_args.to_prompt()
    
    # Test to_prompt for multiple files
    assert "Remove 2 files from file context" in multi_args.to_prompt()


def test_cleanup_context_args_empty_list():
    """Test that empty file list is rejected by pydantic validation."""
    # Pydantic rejects empty list due to min_length=1 constraint
    with pytest.raises(ValueError, match="List should have at least 1 item"):
        CleanupContextArgs(file_paths=[])


@pytest.mark.asyncio
async def test_cleanup_context_integration(workspace, file_context, repository):
    """Integration test that verifies cleanup action with real file operations."""
    # Set up cleanup action and initialize with workspace
    cleanup_action = CleanupContext()
    await cleanup_action.initialize(workspace)
    
    # Verify initial repository state
    assert repository.file_exists("file1.py")
    assert repository.file_exists("file2.py")
    assert repository.file_exists("file3.py")
    
    # Add files to context with show_all_spans=True (like real usage)
    file_context.add_file("file1.py", show_all_spans=True)
    file_context.add_file("file2.py", show_all_spans=True) 
    file_context.add_file("file3.py", show_all_spans=True)
    
    # Verify files are properly in context
    assert file_context.has_file("file1.py")
    assert file_context.has_file("file2.py")
    assert file_context.has_file("file3.py")
    assert len(file_context.file_paths) == 3
    
    # Execute cleanup action to remove 2 files
    cleanup_args = CleanupContextArgs(file_paths=["file1.py", "file3.py"])
    result = await cleanup_action.execute(cleanup_args, file_context)
    
    # Verify the action result
    assert "Successfully removed 2 files from file context" in result.message
    assert "file1.py" in result.message
    assert "file3.py" in result.message
    assert result.summary == "Removed 2 files from context"
    assert result.properties["removed_count"] == 2
    assert result.properties["not_found_count"] == 0
    
    # Verify context state after cleanup
    assert not file_context.has_file("file1.py")
    assert file_context.has_file("file2.py")  # Should remain
    assert not file_context.has_file("file3.py")
    assert len(file_context.file_paths) == 1
    assert file_context.file_paths == ["file2.py"]
    
    # Verify repository files are untouched (cleanup only affects context)
    assert repository.file_exists("file1.py")
    assert repository.file_exists("file2.py")
    assert repository.file_exists("file3.py")
    
    # Verify content of remaining file is accessible
    remaining_file = file_context.get_file("file2.py")
    assert remaining_file is not None
    assert "Test file 2" in remaining_file.content