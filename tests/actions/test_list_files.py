import os
import tempfile
import pytest
import subprocess
from unittest.mock import patch, MagicMock

from moatless.actions.list_files import ListFiles, ListFilesArgs
from moatless.file_context import FileContext
from moatless.actions.schema import Observation
from moatless.repository.file import FileRepository
from moatless.workspace import Workspace
from moatless.environment.local import LocalBashEnvironment, EnvironmentExecutionError


def init_git_repo(repo_path):
    """Initialize a git repository in the given path."""
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)

    # Create .gitignore file
    with open(os.path.join(repo_path, ".gitignore"), "w") as f:
        f.write("# Files to ignore\n")
        f.write("*.log\n")
        f.write("ignored_dir/\n")
        f.write("*_ignored.js\n")

    # Configure git user for the test
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True, capture_output=True)

    # Add .gitignore to git
    subprocess.run(["git", "add", ".gitignore"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit with .gitignore"], cwd=repo_path, check=True, capture_output=True
    )


@pytest.fixture
def temp_repo():
    """Create a temporary directory with some test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directories and files with different extensions
        os.makedirs(os.path.join(temp_dir, "src", "components"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "src", "utils"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "tests"), exist_ok=True)

        # Create a directory that should be ignored
        os.makedirs(os.path.join(temp_dir, "ignored_dir"), exist_ok=True)

        # Create JS files
        with open(os.path.join(temp_dir, "src", "components", "Button.js"), "w") as f:
            f.write("// Button component\n")

        with open(os.path.join(temp_dir, "src", "components", "Card.js"), "w") as f:
            f.write("// Card component\n")

        with open(os.path.join(temp_dir, "src", "components", "Card_ignored.js"), "w") as f:
            f.write("// Ignored card component\n")

        with open(os.path.join(temp_dir, "src", "utils", "helpers.js"), "w") as f:
            f.write("// Helper functions\n")

        # Create TS files
        with open(os.path.join(temp_dir, "src", "index.ts"), "w") as f:
            f.write("// Main entry point\n")

        # Create test files
        with open(os.path.join(temp_dir, "tests", "Button.test.js"), "w") as f:
            f.write("// Button tests\n")

        # Create log file that should be ignored
        with open(os.path.join(temp_dir, "debug.log"), "w") as f:
            f.write("Debug log\n")

        # Create a file in the ignored directory
        with open(os.path.join(temp_dir, "ignored_dir", "ignored_file.txt"), "w") as f:
            f.write("This file should be ignored\n")

        # Initialize git repository
        try:
            init_git_repo(temp_dir)

            # Add files to git
            subprocess.run(["git", "add", "src", "tests"], cwd=temp_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Add source files"], cwd=temp_dir, check=True, capture_output=True)
        except Exception as e:
            print(f"Warning: Could not initialize git repository: {e}")
            # Tests will fall back to non-git behavior

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
    # Set up environment for all tests
    workspace.environment = LocalBashEnvironment(cwd=file_repository.repo_path)
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
    # If git is available, this should be ignored due to .gitignore pattern
    # If git is not available, this will be shown
    if "respecting .gitignore" in result.message:
        assert "📄 Card_ignored.js" not in result.message
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
    assert "📁 src/components" in result.message or "📁 components" in result.message
    assert "📁 src/utils" in result.message or "📁 utils" in result.message
    assert "Button.js" in result.message
    assert "Card.js" in result.message
    assert "helpers.js" in result.message
    assert "index.ts" in result.message


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
    assert "Directory nonexistent does not exist" in result.message


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
    mock_repo.shadow_mode = False

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

    # Add file to git to make it visible
    try:
        subprocess.run(["git", "add", "temp_test_file.txt"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pytest.skip("Could not add test file to git")

    # Verify file is listed
    args = ListFilesArgs(directory="", recursive=False, thoughts="Checking file exists")
    result = await list_files_action.execute(args, file_context)
    if "respecting .gitignore" in result.message:
        # Only check this if git is available
        assert "temp_test_file.txt" in result.message

    # Delete the file
    os.remove(test_file_path)

    # Remove from git to update git's view
    try:
        subprocess.run(["git", "rm", "--cached", "temp_test_file.txt"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pass

    # Verify file is no longer listed
    args = ListFilesArgs(directory="", recursive=False, thoughts="Checking file was deleted")
    result = await list_files_action.execute(args, file_context)
    assert "temp_test_file.txt" not in result.message


@pytest.mark.asyncio
async def test_list_files_symlinks(list_files_action, file_context, temp_repo):
    """Test that symlinks don't lead to duplicate listings when using -xdev."""
    # Create a directory with a file
    external_dir = os.path.join(temp_repo, "external")
    os.makedirs(external_dir, exist_ok=True)
    external_file = os.path.join(external_dir, "external_file.txt")
    with open(external_file, "w") as f:
        f.write("This is an external file")

    # Add to git to make it visible when respecting .gitignore
    try:
        subprocess.run(["git", "add", "external"], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add external dir"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pass

    # Create a symlink in the main directory
    symlink_path = os.path.join(temp_repo, "symlink_to_external")
    try:
        os.symlink(external_dir, symlink_path, target_is_directory=True)
    except (OSError, AttributeError):
        pytest.skip("Symlink creation not supported or requires elevated permissions")

    # Verify the symlink is shown when listing recursively
    args = ListFilesArgs(directory="", recursive=True, thoughts="Testing symlink handling")
    result = await list_files_action.execute(args, file_context)

    # The external directory should be shown
    assert "external" in result.message

    # When git is available, the file will be shown if it was committed
    git_available = "respecting .gitignore" in result.message
    if git_available:
        # Check if file was successfully committed to git
        git_ls_files = subprocess.run(
            ["git", "ls-files", "external"], cwd=temp_repo, capture_output=True, text=True
        ).stdout.strip()

        if git_ls_files:
            assert "external_file.txt" in result.message


@pytest.mark.asyncio
async def test_git_respects_gitignore(list_files_action, file_context, temp_repo):
    """Test that git integration respects .gitignore patterns."""
    # Set up environment
    env = LocalBashEnvironment(cwd=temp_repo)
    list_files_action.workspace.environment = env

    # Execute
    args = ListFilesArgs(directory="", recursive=True, thoughts="Testing .gitignore respect")
    result = await list_files_action.execute(args, file_context)

    # Check if git is available (look for the indicator in the message)
    if "respecting .gitignore" in result.message:
        # Git is available, so .gitignore should be respected
        assert "📁 ignored_dir" not in result.message
        assert "ignored_file.txt" not in result.message
        assert "debug.log" not in result.message
        assert "Card_ignored.js" not in result.message
    else:
        # Git is not available, so these might be shown
        # We skip assertions in this case
        pytest.skip("Git not available in test environment")


@pytest.mark.asyncio
async def test_max_results_limit(list_files_action, file_context, temp_repo):
    """Test that max_results parameter limits the output."""
    # Create multiple files to test limit
    for i in range(20):
        with open(os.path.join(temp_repo, f"file_{i}.txt"), "w") as f:
            f.write(f"Content {i}")

    # Add files to git to make them visible
    try:
        subprocess.run(["git", "add", "file_*.txt"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        # If we can't add to git, test will still run but may need to use non-git mode
        pass

    # Execute with a limit of 5
    args = ListFilesArgs(directory="", recursive=False, max_results=5, thoughts="Testing max_results")
    result = await list_files_action.execute(args, file_context)

    # Count total files and directories in the result
    total_items = 0
    for line in result.message.split("\n"):
        if line.startswith("📁 ") or line.startswith("📄 "):
            total_items += 1

    # There should be max 5 items total
    assert total_items <= 5

    # If git is available, files and directories in result may be limited to what's in git
    is_limited = len(result.properties.get("directories", [])) + len(result.properties.get("files", [])) >= 5

    if is_limited:
        assert "Results limited to 5" in result.message


@pytest.mark.asyncio
async def test_directories_prioritized_over_files(list_files_action, file_context, temp_repo):
    """Test that directories are shown before files until max_results is reached."""
    # Create multiple directories and files
    for i in range(10):
        os.makedirs(os.path.join(temp_repo, f"dir_{i}"), exist_ok=True)
        with open(os.path.join(temp_repo, f"file_{i}.txt"), "w") as f:
            f.write(f"Content {i}")

    # Add to git to make them visible
    try:
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add test files"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pass

    # Execute with a limit that's less than total items
    args = ListFilesArgs(directory="", recursive=False, max_results=8, thoughts="Testing directory prioritization")
    result = await list_files_action.execute(args, file_context)

    # Split result into lines and get items
    lines = result.message.split("\n")
    dir_lines = [line for line in lines if line.startswith("📁 ")]
    file_lines = [line for line in lines if line.startswith("📄 ")]

    # If we have more than 8 directories, only directories should be shown
    if len(result.properties.get("directories", [])) >= 8:
        assert len(file_lines) == 0, "When enough directories exist, no files should be shown"
    else:
        # Otherwise, all directories should be shown first, then files
        total_dirs = len(result.properties.get("directories", []))
        total_files = len(result.properties.get("files", []))
        assert total_dirs + total_files <= 8, "Total items should not exceed max_results"
        
        # All directories should appear before any files in the message
        if dir_lines and file_lines:
            last_dir_index = max(i for i, line in enumerate(lines) if line.startswith("📁 "))
            first_file_index = min(i for i, line in enumerate(lines) if line.startswith("📄 "))
            assert last_dir_index < first_file_index, "All directories should appear before files"


@pytest.mark.asyncio
async def test_hidden_directories_excluded(list_files_action, file_context, temp_repo):
    """Test that directories starting with '.' are hidden by default."""
    # Create directories starting with '.' and regular directories
    os.makedirs(os.path.join(temp_repo, ".hidden_dir"), exist_ok=True)
    os.makedirs(os.path.join(temp_repo, ".another_hidden"), exist_ok=True)
    os.makedirs(os.path.join(temp_repo, "visible_dir"), exist_ok=True)
    
    # Create some files in these directories
    with open(os.path.join(temp_repo, ".hidden_dir", "hidden_file.txt"), "w") as f:
        f.write("Hidden content")
    with open(os.path.join(temp_repo, "visible_dir", "visible_file.txt"), "w") as f:
        f.write("Visible content")

    # Add to git (though they might be ignored by .gitignore)
    try:
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add test dirs"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pass

    # Execute listing
    args = ListFilesArgs(directory="", recursive=False, thoughts="Testing hidden directory exclusion")
    result = await list_files_action.execute(args, file_context)

    # Check that hidden directories are not shown
    assert ".hidden_dir" not in result.message, "Hidden directories starting with '.' should not be shown"
    assert ".another_hidden" not in result.message, "Hidden directories starting with '.' should not be shown"
    assert "visible_dir" in result.message, "Regular directories should be shown"

    # Also check properties
    directories = result.properties.get("directories", [])
    assert ".hidden_dir" not in directories, "Hidden directories should not be in properties"
    assert ".another_hidden" not in directories, "Hidden directories should not be in properties"
    assert "visible_dir" in directories, "Regular directories should be in properties"


@pytest.mark.asyncio
async def test_recursive_hidden_directories_excluded(list_files_action, file_context, temp_repo):
    """Test that hidden directories are excluded in recursive mode too."""
    # Create nested structure with hidden directories
    os.makedirs(os.path.join(temp_repo, "parent", ".hidden_subdir"), exist_ok=True)
    os.makedirs(os.path.join(temp_repo, "parent", "visible_subdir"), exist_ok=True)
    
    # Create some files
    with open(os.path.join(temp_repo, "parent", ".hidden_subdir", "file.txt"), "w") as f:
        f.write("Hidden content")
    with open(os.path.join(temp_repo, "parent", "visible_subdir", "file.txt"), "w") as f:
        f.write("Visible content")

    # Add to git
    try:
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add nested test dirs"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pass

    # Execute recursive listing
    args = ListFilesArgs(directory="parent", recursive=True, thoughts="Testing recursive hidden directory exclusion")
    result = await list_files_action.execute(args, file_context)

    # Check that hidden subdirectories are not shown
    assert ".hidden_subdir" not in result.message, "Hidden subdirectories should not be shown in recursive mode"
    assert "visible_subdir" in result.message, "Regular subdirectories should be shown in recursive mode"

    # Also check properties
    directories = result.properties.get("directories", [])
    hidden_dirs = [d for d in directories if ".hidden_subdir" in d]
    assert len(hidden_dirs) == 0, "No hidden directories should be in properties for recursive listing"


@pytest.mark.asyncio
async def test_breadth_first_directory_ordering(list_files_action, file_context, temp_repo):
    """Test that directories are listed breadth-first (root level first, then by depth)."""
    # Create a complex nested directory structure
    nested_dirs = [
        "level1_a",
        "level1_b", 
        "level1_a/level2_a",
        "level1_a/level2_b",
        "level1_b/level2_c",
        "level1_a/level2_a/level3_a",
        "level1_a/level2_a/level3_b",
        "level1_b/level2_c/level3_c"
    ]
    
    for dir_path in nested_dirs:
        os.makedirs(os.path.join(temp_repo, dir_path), exist_ok=True)
        # Create a file in each directory to ensure it's tracked
        with open(os.path.join(temp_repo, dir_path, "file.txt"), "w") as f:
            f.write("content")

    # Add to git
    try:
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add nested directories"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pass

    # Execute recursive listing
    args = ListFilesArgs(directory="", recursive=True, thoughts="Testing breadth-first ordering")
    result = await list_files_action.execute(args, file_context)

    # Get the directory order from properties 
    directories = result.properties.get("directories", [])
    
    # Filter to only our test directories
    test_dirs = [d for d in directories if d.startswith("level1_")]
    
    # Expected order: level 1 dirs first, then level 2, then level 3
    # Within each level, alphabetical order
    expected_order = [
        "level1_a",           # depth 0
        "level1_b",           # depth 0  
        "level1_a/level2_a",  # depth 1
        "level1_a/level2_b",  # depth 1
        "level1_b/level2_c",  # depth 1
        "level1_a/level2_a/level3_a",  # depth 2
        "level1_a/level2_a/level3_b",  # depth 2
        "level1_b/level2_c/level3_c"   # depth 2
    ]
    
    # Check that our test directories appear in breadth-first order
    actual_order = [d for d in directories if d in expected_order]
    assert actual_order == expected_order, f"Expected breadth-first order {expected_order}, got {actual_order}"


@pytest.mark.asyncio 
async def test_performance_improvement_no_xargs(list_files_action, file_context, temp_repo):
    """Test that the new commands don't use slow xargs approach."""
    import time
    
    # Create many directories to test performance
    for i in range(50):
        os.makedirs(os.path.join(temp_repo, f"perf_test_dir_{i:02d}"), exist_ok=True)
        for j in range(5):
            subdir = os.path.join(temp_repo, f"perf_test_dir_{i:02d}", f"subdir_{j}")
            os.makedirs(subdir, exist_ok=True)
            with open(os.path.join(subdir, "file.txt"), "w") as f:
                f.write("test content")

    # Add to git
    try:
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add performance test dirs"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pass

    # Measure time for recursive listing
    start_time = time.time()
    args = ListFilesArgs(directory="", recursive=True, thoughts="Testing performance")
    result = await list_files_action.execute(args, file_context)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Should complete much faster than the old 3+ minute xargs approach
    # Even with 250+ directories, should complete in under 5 seconds
    assert execution_time < 5.0, f"Directory listing took {execution_time:.2f}s, expected < 5.0s"
    
    # Verify we got results
    assert len(result.properties.get("directories", [])) > 50, "Should have found many test directories"


@pytest.mark.asyncio
async def test_sort_breadth_first_function():
    """Test the sort_breadth_first utility function directly."""
    from moatless.actions.list_files import sort_breadth_first
    
    # Test various directory paths
    unsorted_paths = [
        "deep/nested/path/dir",
        "root_dir",
        "another_root",
        "level1/level2",
        "level1/another_level2", 
        "different_level1/level2/level3",
        "level1/level2/level3"
    ]
    
    result = sort_breadth_first(unsorted_paths)
    
    # Expected: depth 0 first, then 1, then 2, then 3, alphabetical within each depth
    expected = [
        "another_root",          # depth 0
        "root_dir",              # depth 0  
        "level1/another_level2", # depth 1
        "level1/level2",         # depth 1
        "different_level1/level2/level3", # depth 2
        "level1/level2/level3",  # depth 2
        "deep/nested/path/dir"   # depth 3
    ]
    
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.asyncio
async def test_directories_with_spaces_and_special_chars(list_files_action, file_context, temp_repo):
    """Test that directories with spaces and special characters are handled correctly."""
    # Create directories with problematic names
    problematic_dirs = [
        "dir with spaces",
        "dir-with-dashes", 
        "dir_with_underscores",
        "dir.with.dots",
        "dir(with)parens",
        "dir[with]brackets",
        "dir{with}braces"
    ]
    
    for dir_name in problematic_dirs:
        dir_path = os.path.join(temp_repo, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        # Create a file in each directory
        with open(os.path.join(dir_path, "test.txt"), "w") as f:
            f.write("test content")
    
    # Add to git  
    try:
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add special directories"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pass
        
    # Set up environment
    env = LocalBashEnvironment(cwd=temp_repo)
    list_files_action.workspace.environment = env

    # Execute 
    args = ListFilesArgs(directory="", recursive=False, thoughts="Testing special characters")
    result = await list_files_action.execute(args, file_context)
    
    # Should not contain bash error messages
    assert "unary operator expected" not in result.message
    assert "bash: line" not in result.message
    
    # Should contain the directories
    for dir_name in problematic_dirs:
        assert dir_name in result.message or dir_name.replace(" ", r"\ ") in result.message


@pytest.mark.asyncio
async def test_empty_and_malformed_directory_names(list_files_action, file_context, temp_repo):
    """Test handling of edge cases that might cause bash parsing issues."""
    # Create some normal directories
    normal_dirs = ["normal1", "normal2"]
    for dir_name in normal_dirs:
        os.makedirs(os.path.join(temp_repo, dir_name), exist_ok=True)
        with open(os.path.join(temp_repo, dir_name, "file.txt"), "w") as f:
            f.write("content")
    
    # Add to git
    try:
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add normal directories"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pass
    
    # Set up environment  
    env = LocalBashEnvironment(cwd=temp_repo)
    list_files_action.workspace.environment = env

    # Execute
    args = ListFilesArgs(directory="", recursive=True, thoughts="Testing bash robustness")
    result = await list_files_action.execute(args, file_context)
    
    # The key test: should not contain bash error messages
    assert "unary operator expected" not in result.message
    assert "bash: line" not in result.message
    assert "command not found" not in result.message
    
    # Should successfully list directories
    assert "normal1" in result.message
    assert "normal2" in result.message


@pytest.mark.asyncio
async def test_very_deep_directory_structure(list_files_action, file_context, temp_repo):
    """Test with deeply nested directories that might stress bash command parsing."""
    # Create a very deep directory structure
    deep_path = temp_repo
    for i in range(10):  # Create 10 levels deep
        deep_path = os.path.join(deep_path, f"level_{i}")
        os.makedirs(deep_path, exist_ok=True)
        
    # Add a file at the deepest level
    with open(os.path.join(deep_path, "deep_file.txt"), "w") as f:
        f.write("very deep content")
    
    # Add to git
    try:
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add deep structure"], cwd=temp_repo, check=True, capture_output=True)
    except Exception:
        pass
    
    # Set up environment
    env = LocalBashEnvironment(cwd=temp_repo)
    list_files_action.workspace.environment = env

    # Execute recursive listing
    args = ListFilesArgs(directory="", recursive=True, thoughts="Testing deep structure")
    result = await list_files_action.execute(args, file_context)
    
    # Should not contain bash errors
    assert "unary operator expected" not in result.message
    assert "bash: line" not in result.message
    
    # Should contain some of the deep directories
    assert "level_0" in result.message
    assert "level_1" in result.message
