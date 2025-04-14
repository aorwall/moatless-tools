import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from moatless.actions.list_files import ListFiles, ListFilesArgs
from moatless.file_context import FileContext
from moatless.actions.schema import Observation
from moatless.repository.file import FileRepository
from moatless.workspace import Workspace
from moatless.environment.local import LocalBashEnvironment, EnvironmentExecutionError


@pytest.fixture
def temp_repo():
    """Create a temporary directory with some test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directories and files with different extensions
        os.makedirs(os.path.join(temp_dir, "src", "components"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "src", "utils"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "tests"), exist_ok=True)

        # Create JS files
        with open(os.path.join(temp_dir, "src", "components", "Button.js"), "w") as f:
            f.write("// Button component\n")

        with open(os.path.join(temp_dir, "src", "components", "Card.js"), "w") as f:
            f.write("// Card component\n")

        with open(os.path.join(temp_dir, "src", "utils", "helpers.js"), "w") as f:
            f.write("// Helper functions\n")

        # Create TS files
        with open(os.path.join(temp_dir, "src", "index.ts"), "w") as f:
            f.write("// Main entry point\n")

        # Create test files
        with open(os.path.join(temp_dir, "tests", "Button.test.js"), "w") as f:
            f.write("// Button tests\n")

        yield temp_dir


@pytest.fixture
def file_repository(temp_repo):
    """Create a FileRepository pointing to the temp directory."""
    return FileRepository(repo_path=temp_repo)


@pytest.fixture
def file_context(file_repository):
    """Create a FileContext using the repository."""
    return FileContext(repo=file_repository)


@pytest.fixture
def workspace(file_repository):
    """Create a workspace with the repository."""
    workspace = MagicMock(spec=Workspace)
    workspace.repository = file_repository
    workspace.environment = None
    return workspace


@pytest.fixture
def list_files_action(workspace):
    """Create a ListFiles action with the repository."""
    action = ListFiles()
    action._workspace = workspace
    return action


@pytest.mark.asyncio
async def test_list_files_root_directory(list_files_action, file_context, temp_repo):
    """Test listing files in the root directory."""
    # Execute
    args = ListFilesArgs(directory="", recursive=False, thoughts="Listing root directory contents")
    result = await list_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Contents of directory '(root)'" in result.message
    assert "📁 src" in result.message
    assert "📁 tests" in result.message
    assert "Button.js" not in result.message  # Should not include files from subdirectories


@pytest.mark.asyncio
async def test_list_files_subdirectory(list_files_action, file_context, temp_repo):
    """Test listing files in a specific subdirectory."""
    # Execute
    args = ListFilesArgs(directory="src/components", recursive=False, thoughts="Listing components directory contents")
    result = await list_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Contents of directory 'src/components'" in result.message
    assert "📄 Button.js" in result.message
    assert "📄 Card.js" in result.message
    assert "index.ts" not in result.message  # File not in this directory
    assert "utils" not in result.message  # Not showing directories from parent


@pytest.mark.asyncio
async def test_list_files_recursive(list_files_action, file_context, temp_repo):
    """Test listing files recursively from a directory."""
    # Execute
    args = ListFilesArgs(directory="src", recursive=True, thoughts="Listing src directory contents recursively")
    result = await list_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Contents of directory 'src' (recursive)" in result.message
    assert "📁 src/components" in result.message
    assert "📁 src/utils" in result.message
    assert "📄 src/components/Button.js" in result.message
    assert "📄 src/components/Card.js" in result.message
    assert "📄 src/utils/helpers.js" in result.message
    assert "📄 src/index.ts" in result.message


@pytest.mark.asyncio
async def test_list_files_empty_directory(list_files_action, file_context, temp_repo):
    """Test listing files in an empty directory."""
    # Create an empty directory
    empty_dir = os.path.join(temp_repo, "empty")
    os.makedirs(empty_dir, exist_ok=True)

    # Execute
    args = ListFilesArgs(directory="empty", recursive=False, thoughts="Listing empty directory contents")
    result = await list_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Directory is empty" in result.message


@pytest.mark.asyncio
async def test_list_files_nonexistent_directory(list_files_action, file_context, temp_repo):
    """Test listing files in a non-existent directory."""
    # Mock the LocalBashEnvironment.execute to return error text
    original_execute = LocalBashEnvironment.execute

    async def mock_execute_with_error(self, command):
        if "nonexistent" in command:
            return "find: ./nonexistent: No such file or directory"
        return await original_execute(self, command)

    # Apply the mock
    with patch.object(LocalBashEnvironment, "execute", mock_execute_with_error):
        # Execute
        args = ListFilesArgs(directory="nonexistent", recursive=False, thoughts="Listing non-existent directory")
        result = await list_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Error" in result.message
    assert "No such directory" in result.message


@pytest.mark.asyncio
async def test_list_files_with_environment(list_files_action, file_context, temp_repo):
    """Test listing files using the environment."""
    # Create a mock environment and set it on the workspace
    env = LocalBashEnvironment(cwd=temp_repo)
    list_files_action.workspace.environment = env

    # Execute
    args = ListFilesArgs(directory="src", recursive=False, thoughts="Testing with workspace environment")
    result = await list_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "components" in result.message
    assert "utils" in result.message
    assert "index.ts" in result.message


@pytest.mark.asyncio
async def test_list_files_environment_failure_fallback(list_files_action, file_context, temp_repo):
    """Test fallback to repository's list_directory when environment commands fail."""
    # Mock the LocalBashEnvironment.execute to fail
    original_execute = LocalBashEnvironment.execute

    async def mock_execute_error(self, command):
        if "find" in command:
            raise EnvironmentExecutionError("Command failed", -1, "Error")
        return await original_execute(self, command)

    # Mock the repository's list_directory method
    mock_repo = MagicMock(spec=FileRepository)
    mock_repo.list_directory.return_value = {"directories": ["fallback_dir"], "files": ["fallback_file.txt"]}

    # Keep a reference to the original repository
    original_repo = list_files_action._workspace.repository

    try:
        # Replace the repository with our mock by patching the workspace
        list_files_action._workspace.repository = mock_repo
        file_context._repo = mock_repo

        # Apply the execute mock
        with patch.object(LocalBashEnvironment, "execute", mock_execute_error):
            # Execute
            args = ListFilesArgs(directory="src", recursive=False, thoughts="Testing fallback mechanism")
            result = await list_files_action.execute(args, file_context)

        # Assert fallback was used
        assert isinstance(result, Observation)
        assert result.message is not None
        assert "fallback_dir" in result.message
        assert "fallback_file.txt" in result.message
        mock_repo.list_directory.assert_called_once_with("src")

    finally:
        # Restore the original repository
        list_files_action._workspace.repository = original_repo


@pytest.mark.asyncio
async def test_list_files_after_deletion(list_files_action, file_context, temp_repo):
    """Test that deleted files are not shown in listing."""
    # First create a file
    test_file_path = os.path.join(temp_repo, "temp_test_file.txt")
    with open(test_file_path, "w") as f:
        f.write("This is a test file")

    # Verify file is listed
    args = ListFilesArgs(directory="", recursive=False, thoughts="Checking file exists")
    result = await list_files_action.execute(args, file_context)
    assert "📄 temp_test_file.txt" in result.message

    # Delete the file
    os.remove(test_file_path)

    # Verify file is no longer listed
    args = ListFilesArgs(directory="", recursive=False, thoughts="Checking file was deleted")
    result = await list_files_action.execute(args, file_context)
    assert "📄 temp_test_file.txt" not in result.message


@pytest.mark.asyncio
async def test_list_files_symlinks(list_files_action, file_context, temp_repo):
    """Test that symlinks don't lead to duplicate listings when using -xdev."""
    # Create a directory with a file
    external_dir = os.path.join(temp_repo, "external")
    os.makedirs(external_dir, exist_ok=True)
    external_file = os.path.join(external_dir, "external_file.txt")
    with open(external_file, "w") as f:
        f.write("This is an external file")

    # Create a symlink in the main directory
    symlink_path = os.path.join(temp_repo, "symlink_to_external")
    try:
        os.symlink(external_dir, symlink_path, target_is_directory=True)
    except (OSError, AttributeError):
        pytest.skip("Symlink creation not supported or requires elevated permissions")

    # Verify the symlink is shown when listing recursively
    args = ListFilesArgs(directory="", recursive=True, thoughts="Testing symlink handling")
    result = await list_files_action.execute(args, file_context)

    # The symlink should be treated as a directory and listed
    assert "external" in result.message

    # Without -xdev, find would follow symlinks and potentially show duplicate content
    # or infinite recursion. With -xdev, we avoid this issue.
    assert "external_file.txt" in result.message
