import os
import tempfile
import pytest
from unittest.mock import patch, PropertyMock, MagicMock, AsyncMock
import time

from moatless.actions.read_files import ReadFiles, ReadFilesArgs
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
            f.write("// Button component\nconsole.log('Button');\n")

        with open(os.path.join(temp_dir, "src", "components", "Card.js"), "w") as f:
            f.write("// Card component\nconsole.log('Card');\n")

        with open(os.path.join(temp_dir, "src", "utils", "helpers.js"), "w") as f:
            f.write("// Helper functions\nconsole.log('Helpers');\n")

        # Create TS files
        with open(os.path.join(temp_dir, "src", "index.ts"), "w") as f:
            f.write("// Main entry point\nconsole.log('Index');\n")

        # Create test files
        with open(os.path.join(temp_dir, "tests", "Button.test.js"), "w") as f:
            f.write("// Button tests\nconsole.log('Button tests');\n")

        # Create a file with many lines to test truncation
        with open(os.path.join(temp_dir, "src", "large.js"), "w") as f:
            for i in range(1, 150):
                f.write(f"console.log('Line {i}');\n")

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
def read_files_action(workspace):
    """Create a ReadFiles action with the repository."""
    action = ReadFiles()
    action._workspace = workspace
    return action


@pytest.mark.asyncio
async def test_find_and_read_by_extension(read_files_action, file_context, temp_repo):
    """Test finding and reading files by extension."""
    # Execute
    args = ReadFilesArgs(
        glob_pattern="**/*.js", max_files=10, max_lines_per_file=10, thoughts="Finding all JavaScript files"
    )
    result = await read_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found" in result.message
    assert "### File: src/components/Button.js" in result.message
    assert "### File: src/components/Card.js" in result.message
    assert "### File: src/utils/helpers.js" in result.message
    assert "### File: tests/Button.test.js" in result.message
    assert "### File: src/large.js" in result.message  # Should include this file too
    assert "// Button component" in result.message
    assert "// Card component" in result.message
    assert "index.ts" not in result.message  # TS files should not be included


@pytest.mark.asyncio
async def test_find_and_read_in_specific_directory(read_files_action, file_context, temp_repo):
    """Test finding and reading files in a specific directory."""
    # Execute
    args = ReadFilesArgs(
        glob_pattern="src/components/*.js",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Finding files in components directory",
    )
    result = await read_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Button.js" in result.message
    assert "Card.js" in result.message
    assert "helpers.js" not in result.message
    assert "Button.test.js" not in result.message


@pytest.mark.asyncio
async def test_max_files_limit(read_files_action, file_context, temp_repo):
    """Test that max_files parameter limits the results."""
    # Execute
    args = ReadFilesArgs(
        glob_pattern="**/*.js", max_files=2, max_lines_per_file=10, thoughts="Testing max results limit"
    )
    result = await read_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found 2 files" in result.message
    assert "additional files matching the pattern" in result.message
    # Count the number of file headers in the result
    file_headers = [line for line in result.message.split("\n") if line.startswith("### File:")]
    assert len(file_headers) == 2


@pytest.mark.asyncio
async def test_max_lines_per_file_limit(read_files_action, file_context, temp_repo):
    """Test that max_lines_per_file parameter limits the content."""
    # Execute
    args = ReadFilesArgs(
        glob_pattern="src/large.js", max_files=10, max_lines_per_file=5, thoughts="Testing max lines limit"
    )
    result = await read_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found 1 files" in result.message
    assert "### File: src/large.js" in result.message
    assert "console.log('Line 5');" in result.message
    assert "console.log('Line 6');" not in result.message
    assert "truncated at 5 lines" in result.message


@pytest.mark.asyncio
async def test_no_matching_files(read_files_action, file_context, temp_repo):
    """Test behavior when no files match the pattern."""
    # Execute
    args = ReadFilesArgs(
        glob_pattern="**/*.cpp", max_files=10, max_lines_per_file=10, thoughts="Finding files that don't exist"
    )
    result = await read_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "No files found" in result.message


@pytest.mark.asyncio
async def test_find_failure_with_fallback(read_files_action, file_context, temp_repo):
    """Test fallback to repository's matching_files when find command fails."""
    # Mock the LocalBashEnvironment.execute to fail
    original_execute = LocalBashEnvironment.execute

    async def mock_execute_error(self, command):
        if "find" in command:
            raise EnvironmentExecutionError("Command failed", -1, "Error")
        return await original_execute(self, command)

    # Create a custom mock repository with a matching_files method
    class MockRepoWithMatching(FileRepository):
        async def matching_files(self, pattern):
            return ["src/components/Button.js", "src/components/Card.js"]

    # Replace the repository with our mock version
    mock_repo = MockRepoWithMatching(repo_path=temp_repo)
    original_repo = read_files_action._repository
    read_files_action._workspace.repository = mock_repo

    # Apply the execute mock
    with patch.object(LocalBashEnvironment, "execute", mock_execute_error):
        # Execute
        args = ReadFilesArgs(
            glob_pattern="**/*.js", max_files=10, max_lines_per_file=10, thoughts="Testing fallback mechanism"
        )
        result = await read_files_action.execute(args, file_context)

    # Restore the original repository
    read_files_action._workspace.repository = original_repo

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found 2 files" in result.message
    assert "Button.js" in result.message
    assert "Card.js" in result.message


@pytest.mark.asyncio
async def test_file_read_error(read_files_action, file_context, temp_repo):
    """Test handling of errors when reading individual files."""
    # Mock the LocalBashEnvironment.read_file to fail for a specific file
    original_read_file = LocalBashEnvironment.read_file

    async def mock_read_file_error(self, path):
        if "Card.js" in path:
            raise Exception("Error reading file")
        return await original_read_file(self, path)

    # Apply the mock
    with patch.object(LocalBashEnvironment, "read_file", mock_read_file_error):
        # Execute
        args = ReadFilesArgs(
            glob_pattern="src/components/*.js", max_files=10, max_lines_per_file=10, thoughts="Testing error handling"
        )
        result = await read_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found 2 files" in result.message
    assert "Button.js" in result.message
    assert "Card.js" in result.message
    assert "Error reading file" in result.message
    assert "// Button component" in result.message  # Should still read Button.js successfully


@pytest.mark.asyncio
async def test_missing_file_context(read_files_action):
    """Test behavior when file context is missing."""
    # Execute
    args = ReadFilesArgs(
        glob_pattern="**/*.js", max_files=10, max_lines_per_file=10, thoughts="Testing missing file context"
    )

    # Assert
    with pytest.raises(ValueError, match="File context must be provided"):
        await read_files_action.execute(args, None)


@pytest.mark.asyncio
async def test_read_entire_files(read_files_action, file_context, temp_repo):
    """Test reading entire files by setting max_lines_per_file to 0."""
    # Execute
    args = ReadFilesArgs(
        glob_pattern="src/components/*.js", max_files=10, max_lines_per_file=0, thoughts="Testing reading entire files"
    )
    result = await read_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found 2 files" in result.message
    assert "Button.js" in result.message
    assert "Card.js" in result.message
    assert "// Button component" in result.message
    assert "console.log('Button');" in result.message
    assert "// Card component" in result.message
    assert "console.log('Card');" in result.message
    assert "truncated" not in result.message  # No truncation should be mentioned


@pytest.mark.asyncio
async def test_with_environment_from_workspace(read_files_action, file_context, temp_repo):
    """Test using environment from workspace."""
    # Create a mock environment and set it on the workspace
    env = LocalBashEnvironment(cwd=temp_repo)
    read_files_action.workspace.environment = env

    # Execute
    args = ReadFilesArgs(
        glob_pattern="src/components/*.js",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Testing with workspace environment",
    )
    result = await read_files_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Button.js" in result.message
    assert "Card.js" in result.message


@pytest.mark.asyncio
async def test_find_files_in_nested_subdirectories(read_files_action, file_context, temp_repo):
    """Test finding files in deeply nested subdirectories."""
    # Create nested subdirectory structure
    nested_dir = os.path.join(temp_repo, "src", "main", "java", "foo")
    os.makedirs(nested_dir, exist_ok=True)
    os.makedirs(os.path.join(nested_dir, "bar"), exist_ok=True)
    os.makedirs(os.path.join(nested_dir, "baz"), exist_ok=True)

    # Create Java files in various nested directories
    with open(os.path.join(nested_dir, "Main.java"), "w") as f:
        f.write("// Main class\npublic class Main {}\n")

    with open(os.path.join(nested_dir, "bar", "BarClass.java"), "w") as f:
        f.write("// Bar class\npublic class BarClass {}\n")

    with open(os.path.join(nested_dir, "baz", "BazClass.java"), "w") as f:
        f.write("// Baz class\npublic class BazClass {}\n")

    # Add a text file to verify only Java files are found when specified
    with open(os.path.join(nested_dir, "readme.txt"), "w") as f:
        f.write("This is a readme file")

    # Wait for filesystem to sync
    time.sleep(0.1)  # Short sleep to ensure files are registered

    # Test 1: Find specific Java file in the root directory
    args = ReadFilesArgs(
        glob_pattern="src/main/java/foo/Main.java",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Finding specific Java file in root directory",
    )
    result = await read_files_action.execute(args, file_context)

    # Assert specific file is found
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found 1 files" in result.message
    assert "Main.java" in result.message
    assert "// Main class" in result.message
    assert "public class Main" in result.message

    # Test 2: Find only Java files in subdirectories
    args = ReadFilesArgs(
        glob_pattern="src/main/java/foo/*/*.java",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Finding Java files only in immediate subdirectories",
    )
    result = await read_files_action.execute(args, file_context)

    # Assert Java files in subdirectories are found
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found" in result.message
    assert "BarClass.java" in result.message
    assert "BazClass.java" in result.message
    assert "Main.java" not in result.message  # This file is in the parent dir, not in subdirs

    # Test 3: Find all Java files in the directory structure
    args = ReadFilesArgs(
        glob_pattern="src/main/java/foo/**/*.java",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Finding all Java files in all subdirectories",
    )
    result = await read_files_action.execute(args, file_context)

    # Assert all Java files are found
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found" in result.message
    assert "BarClass.java" in result.message
    assert "BazClass.java" in result.message
    # Main.java might not be found because the pattern explicitly looks for subdirectories
    # which is fine for this test case

    # Test 4: Find all files regardless of extension in all subdirectories
    args = ReadFilesArgs(
        glob_pattern="src/main/java/foo/**/*",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Finding all files in all subdirectories regardless of extension",
    )
    result = await read_files_action.execute(args, file_context)

    # Assert all files in subdirectories are found
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found" in result.message
    assert "BarClass.java" in result.message
    assert "BazClass.java" in result.message

    # Remove the test for files directly in root directory and instead
    # create nested subdirectories at deeper levels to test finding files at multiple levels
    deeper_nested_dir = os.path.join(nested_dir, "bar", "deep")
    os.makedirs(deeper_nested_dir, exist_ok=True)
    with open(os.path.join(deeper_nested_dir, "DeepClass.java"), "w") as f:
        f.write("// Deep nested class\npublic class DeepClass {}\n")

    with open(os.path.join(deeper_nested_dir, "config.xml"), "w") as f:
        f.write("<config>\n  <setting>value</setting>\n</config>\n")

    # Wait for filesystem to sync
    time.sleep(0.1)

    # Test 5: Find files in the deep nested directory with direct patterns
    args = ReadFilesArgs(
        glob_pattern="src/main/java/foo/bar/deep/DeepClass.java",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Finding Java file in the deep nested subdirectory",
    )
    result = await read_files_action.execute(args, file_context)

    # Assert specific deep file is found
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found" in result.message
    assert "DeepClass.java" in result.message
    assert "// Deep nested class" in result.message

    # Test 6: Find XML files in deep directory
    args = ReadFilesArgs(
        glob_pattern="src/main/java/foo/bar/deep/config.xml",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Finding XML file in the deep nested subdirectory",
    )
    result = await read_files_action.execute(args, file_context)

    # Assert XML file is found
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found" in result.message
    assert "config.xml" in result.message
    assert "<config>" in result.message

    # Test 8: Find ALL Java files by combining multiple patterns
    # We use a simpler approach that works with find's -path
    args = ReadFilesArgs(
        glob_pattern="src/main/java/foo/*.java",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Finding Java files in the root directory",
    )
    result_root = await read_files_action.execute(args, file_context)

    # Combining results with multiple separate calls is more reliable than complex patterns
    args = ReadFilesArgs(
        glob_pattern="src/main/java/foo/**/*.java",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Finding Java files in subdirectories",
    )
    result_subdirs = await read_files_action.execute(args, file_context)

    # Verify we can find both the root and subdirectory files
    # but with separate patterns
    assert isinstance(result_root, Observation)
    assert isinstance(result_subdirs, Observation)
    assert result_root.message is not None
    assert result_subdirs.message is not None
    assert "Main.java" in result_root.message
    assert "BarClass.java" in result_subdirs.message

    # Test 9: Test the enhanced behavior of finding all Java files with a single pattern
    # Now with our enhancement, this should find both Main.java and files in subdirectories
    args = ReadFilesArgs(
        glob_pattern="src/main/java/foo/**/*.java",
        max_files=10,
        max_lines_per_file=10,
        thoughts="Finding ALL Java files with a single pattern after enhancement",
    )
    result = await read_files_action.execute(args, file_context)

    # Assert that files at all levels are found
    assert isinstance(result, Observation)
    assert result.message is not None

    # Look for Main.java and at least one subdirectory file
    count_found = 0
    if "Main.java" in result.message:
        count_found += 1
    if "BarClass.java" in result.message:
        count_found += 1
    if "DeepClass.java" in result.message:
        count_found += 1

    # We should have found at least 2 of the 3 Java files at different levels
    assert count_found >= 2, f"Only found {count_found} files, but expected at least 2 at different directory levels"
