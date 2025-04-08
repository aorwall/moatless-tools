import os
import tempfile
import pytest
from unittest.mock import patch, PropertyMock, AsyncMock, MagicMock

from moatless.actions.glob import GlobTool, GlobArgs
from moatless.file_context import FileContext
from moatless.actions.schema import Observation
from moatless.repository.file import FileRepository
from moatless.workspace import Workspace


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
            f.write("// Button component")
        
        with open(os.path.join(temp_dir, "src", "components", "Card.js"), "w") as f:
            f.write("// Card component")
            
        with open(os.path.join(temp_dir, "src", "utils", "helpers.js"), "w") as f:
            f.write("// Helper functions")
            
        # Create TS files
        with open(os.path.join(temp_dir, "src", "index.ts"), "w") as f:
            f.write("// Main entry point")
            
        # Create test files
        with open(os.path.join(temp_dir, "tests", "Button.test.js"), "w") as f:
            f.write("// Button tests")
            
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
    return workspace


@pytest.fixture
def glob_tool(workspace):
    """Create a GlobTool action with the repository."""
    action = GlobTool()
    action._workspace = workspace
    return action


@pytest.mark.asyncio
async def test_find_by_extension(glob_tool, file_context, temp_repo):
    """Test finding files by extension."""
    # Execute
    args = GlobArgs(pattern="**/*.js", max_results=100, thoughts="Finding all JavaScript files")
    result = await glob_tool.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found" in result.message
    assert "Button.js" in result.message
    assert "Card.js" in result.message
    assert "helpers.js" in result.message
    assert "Button.test.js" in result.message
    assert "index.ts" not in result.message


@pytest.mark.asyncio
async def test_find_in_specific_directory(glob_tool, file_context, temp_repo):
    """Test finding files in a specific directory."""
    # Execute
    args = GlobArgs(pattern="src/components/*.js", max_results=100, thoughts="Finding files in components directory")
    result = await glob_tool.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Button.js" in result.message
    assert "Card.js" in result.message
    assert "helpers.js" not in result.message
    assert "Button.test.js" not in result.message


@pytest.mark.asyncio
async def test_max_results_limit(glob_tool, file_context, temp_repo):
    """Test that max_results parameter limits the results."""
    # Execute
    args = GlobArgs(pattern="**/*.js", max_results=2, thoughts="Testing max results limit")
    result = await glob_tool.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found 2 files" in result.message
    # Count the number of file paths in the result
    file_paths = [line for line in result.message.split('\n') if line.strip().startswith('📄')]
    assert len(file_paths) == 2


@pytest.mark.asyncio
async def test_no_matching_files(glob_tool, file_context, temp_repo):
    """Test behavior when no files match the pattern."""
    # Execute
    args = GlobArgs(pattern="**/*.cpp", max_results=100, thoughts="Finding files that don't exist")
    result = await glob_tool.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "No files found" in result.message


@pytest.mark.asyncio
async def test_invalid_pattern(glob_tool, file_context):
    """Test behavior with an invalid pattern."""
    # Mock the matching_files method of FileRepository
    original_matching_files = glob_tool.workspace.repository.matching_files
    
    async def mock_error(file_pattern):
        raise ValueError("Invalid pattern")
    
    # Apply the mock
    with patch.object(glob_tool.workspace.repository.__class__, 'matching_files', 
                     side_effect=mock_error):
        # Execute
        args = GlobArgs(pattern="[invalid", max_results=100, thoughts="Testing invalid pattern")
        result = await glob_tool.execute(args, file_context)
    
    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Error executing glob search" in result.message


@pytest.mark.asyncio
async def test_missing_workspace(file_context):
    """Test behavior when workspace is missing."""
    # Setup a GlobTool without a workspace
    glob_tool = GlobTool()
    
    # Execute
    args = GlobArgs(pattern="**/*.js", max_results=100, thoughts="Testing missing workspace")
    with pytest.raises(RuntimeError, match="No workspace set"):
        await glob_tool.execute(args, file_context)


@pytest.mark.asyncio
async def test_missing_repository(file_context):
    """Test behavior when repository is missing."""
    # Setup a GlobTool with a workspace but no repository
    glob_tool = GlobTool()
    workspace = MagicMock(spec=Workspace)
    workspace.repository = None
    glob_tool._workspace = workspace
    
    # Execute
    args = GlobArgs(pattern="**/*.js", max_results=100, thoughts="Testing missing repository")
    
    # The GlobTool._execute implementation raises ValueError, but the base Action.execute
    # raises RuntimeError first when checking workspace.repository, so we need to check for that
    with pytest.raises((ValueError, RuntimeError)):
        await glob_tool.execute(args, file_context)


@pytest.mark.asyncio
async def test_repository_without_matching_files(file_context):
    """Test behavior when repository doesn't support matching_files."""
    # Setup a GlobTool with a repository that doesn't have matching_files
    glob_tool = GlobTool()
    workspace = MagicMock(spec=Workspace)
    
    # Create a real repo-like object but without the matching_files method
    class RepoWithoutMatching:
        def __init__(self):
            pass
            
    # Create a mock repository without matching_files method
    mock_repo = RepoWithoutMatching()
    
    workspace.repository = mock_repo
    glob_tool._workspace = workspace
    
    # Execute
    args = GlobArgs(pattern="**/*.js", max_results=100, thoughts="Testing repository without matching_files")
    
    with pytest.raises(ValueError, match="Repository does not support glob matching"):
        await glob_tool.execute(args, file_context) 