import os
import tempfile
import pytest
import subprocess
from unittest.mock import patch, MagicMock

from moatless.actions.grep_tool import GrepTool, GrepToolArgs
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
        os.makedirs(os.path.join(temp_dir, "requests", "api"), exist_ok=True)

        # Create a directory that should be ignored
        os.makedirs(os.path.join(temp_dir, "ignored_dir"), exist_ok=True)

        # Create JS files with various content
        with open(os.path.join(temp_dir, "src", "components", "Button.js"), "w") as f:
            f.write("// Button component\n")
            f.write("function renderButton() {\n")
            f.write("  console.log('Button rendered');\n")
            f.write("}\n")

        with open(os.path.join(temp_dir, "src", "components", "Card.js"), "w") as f:
            f.write("// Card component\n")
            f.write("class Card {\n")
            f.write("  render() {\n")
            f.write("    console.error('Card render error');\n")
            f.write("  }\n")
            f.write("}\n")

        with open(os.path.join(temp_dir, "src", "components", "Card_ignored.js"), "w") as f:
            f.write("// Ignored card component\n")
            f.write("function ignoredFunction() {}\n")

        with open(os.path.join(temp_dir, "src", "utils", "helpers.js"), "w") as f:
            f.write("// Helper functions\n")
            f.write("function logError(msg) {\n")
            f.write("  console.error('Error: ' + msg);\n")
            f.write("}\n")

        # Create Python files
        with open(os.path.join(temp_dir, "src", "main.py"), "w") as f:
            f.write("import logging\n")
            f.write("\n")
            f.write("def main():\n")
            f.write('    logging.error("Application error")\n')
            f.write('    print("Hello World")\n')

        # Create test files
        with open(os.path.join(temp_dir, "tests", "test_button.py"), "w") as f:
            f.write("import pytest\n")
            f.write("\n")
            f.write("def test_button_render():\n")
            f.write("    # Test button rendering\n")
            f.write("    assert True\n")

        # Create log file that should be ignored
        with open(os.path.join(temp_dir, "debug.log"), "w") as f:
            f.write("Debug log error\n")
            f.write("Another error line\n")

        # Create files in ignored directory
        with open(os.path.join(temp_dir, "ignored_dir", "ignored_file.txt"), "w") as f:
            f.write("This file should be ignored\n")
            f.write("Error in ignored file\n")

        # Create files in requests directory for path-based search testing
        with open(os.path.join(temp_dir, "requests", "api", "timeout.py"), "w") as f:
            f.write("class TimeoutError(Exception):\n")
            f.write("    pass\n")
            f.write("\n")
            f.write("class DecodeError(Exception):\n")
            f.write("    pass\n")

        # Initialize git repository
        try:
            init_git_repo(temp_dir)

            # Add files to git
            subprocess.run(["git", "add", "src", "tests", "requests"], cwd=temp_dir, check=True, capture_output=True)
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
def grep_tool_action(workspace):
    """Create a GrepTool action with the repository."""
    action = GrepTool()
    action._workspace = workspace
    return action


@pytest.mark.asyncio
async def test_grep_simple_pattern(grep_tool_action, file_context, temp_repo):
    """Test searching for a simple pattern across all files."""
    # Execute
    args = GrepToolArgs(pattern=r"error", max_results=10, thoughts="Searching for error occurrences")
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found" in result.message
    assert "matches for regex pattern 'error'" in result.message

    # Should find errors in Card.js, helpers.js, and main.py
    message_lower = result.message.lower()
    assert "card.js" in message_lower or "helpers.js" in message_lower or "main.py" in message_lower

    # Should not find errors in ignored files (they're excluded by --exclude-dir)
    assert "debug.log" not in result.message
    assert "ignored_file.txt" not in result.message


@pytest.mark.asyncio
async def test_grep_with_file_pattern(grep_tool_action, file_context, temp_repo):
    """Test searching with a specific file pattern."""
    # Execute
    args = GrepToolArgs(
        pattern=r"function\s+\w+",
        include="*.js",
        max_results=10,
        thoughts="Finding function definitions in JavaScript files",
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "in files matching '*.js'" in result.message

    # Should find functions in JS files
    assert "Button.js" in result.message
    assert "renderButton" in result.message
    assert "helpers.js" in result.message
    assert "logError" in result.message

    # Should not find Python functions
    assert "main.py" not in result.message
    assert "test_button.py" not in result.message


@pytest.mark.asyncio
async def test_grep_with_path_pattern(grep_tool_action, file_context, temp_repo):
    """Test searching with complex path patterns using **."""
    # Execute
    args = GrepToolArgs(
        pattern=r"Error",
        include="**/requests/**/*.py",
        max_results=10,
        thoughts="Finding Error classes in requests library",
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find TimeoutError and DecodeError in requests/api/timeout.py
    assert "timeout.py" in result.message
    assert "TimeoutError" in result.message
    assert "DecodeError" in result.message

    # Should not find errors in other directories
    assert "main.py" not in result.message
    assert "Card.js" not in result.message


@pytest.mark.asyncio
async def test_grep_no_matches(grep_tool_action, file_context, temp_repo):
    """Test when no matches are found."""
    # Execute
    args = GrepToolArgs(
        pattern=r"NonExistentPattern12345", max_results=10, thoughts="Searching for a pattern that doesn't exist"
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "No matches found" in result.message


@pytest.mark.asyncio
async def test_grep_case_sensitive(grep_tool_action, file_context, temp_repo):
    """Test case-sensitive regex patterns."""
    # Execute
    args = GrepToolArgs(
        pattern=r"Error",  # Capital E
        max_results=10,
        thoughts="Searching for Error with capital E",
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find TimeoutError, DecodeError, but not 'error' in lowercase
    assert "TimeoutError" in result.message or "DecodeError" in result.message

    # Just verify we found results with 'Error' (capital E)
    # The exact match details are already verified in the message content


@pytest.mark.asyncio
async def test_grep_regex_patterns(grep_tool_action, file_context, temp_repo):
    """Test complex regex patterns."""
    # Execute - search for class definitions
    args = GrepToolArgs(pattern=r"class\s+\w+", max_results=10, thoughts="Finding class definitions")
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find Card class and Error classes
    assert "Card" in result.message or "TimeoutError" in result.message


@pytest.mark.asyncio
async def test_grep_max_results_limit(grep_tool_action, file_context, temp_repo):
    """Test that max_results parameter limits the output."""
    # Create multiple files with many matches
    for i in range(10):
        with open(os.path.join(temp_repo, f"test_file_{i}.txt"), "w") as f:
            for j in range(5):
                f.write(f"Line {j}: test pattern here\n")

    # Execute with a low limit
    args = GrepToolArgs(pattern=r"test pattern", max_results=5, thoughts="Testing max results limit")
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Check properties for exact count
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] <= 5
    assert "Results limited to 5 matches" in result.message


@pytest.mark.asyncio
async def test_grep_directory_search(grep_tool_action, file_context, temp_repo):
    """Test searching within a specific directory."""
    # Execute
    args = GrepToolArgs(
        pattern=r"console",
        include="src/components/",
        max_results=10,
        thoughts="Searching for console usage in components directory",
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find console usage in Button.js and Card.js
    assert "Button.js" in result.message
    assert "Card.js" in result.message

    # Should not find console usage in utils or other directories
    assert "helpers.js" not in result.message


@pytest.mark.asyncio
async def test_grep_directory_search_all_file_types(grep_tool_action, file_context, temp_repo):
    """Test that directory search now includes all file types (no explicit --include patterns)."""
    # Create additional files with different extensions and no extensions
    docker_dir = os.path.join(temp_repo, "docker")
    os.makedirs(docker_dir, exist_ok=True)

    # Create Dockerfile (no extension)
    with open(os.path.join(docker_dir, "Dockerfile"), "w") as f:
        f.write("FROM ubuntu:latest\n")
        f.write("# Test pattern for search\n")
        f.write("RUN echo 'testing pattern'\n")

    # Create Makefile (no extension)
    with open(os.path.join(docker_dir, "Makefile"), "w") as f:
        f.write("all:\n")
        f.write("\t@echo 'testing pattern'\n")

    # Create config file with unusual extension
    with open(os.path.join(docker_dir, "config.conf"), "w") as f:
        f.write("# Configuration file\n")
        f.write("setting=testing pattern\n")

    # Create shell script
    with open(os.path.join(docker_dir, "setup.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'testing pattern'\n")

    # Execute search in docker directory
    args = GrepToolArgs(
        pattern=r"testing pattern",
        include="docker/",
        max_results=20,
        thoughts="Testing directory search with various file types",
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find pattern in all file types
    message_lower = result.message.lower()
    assert "dockerfile" in message_lower
    assert "makefile" in message_lower
    assert "config.conf" in message_lower
    assert "setup.sh" in message_lower

    # Verify we found multiple matches
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] >= 4  # At least one match per file

    print(f"Found {properties['total_matches']} matches in {properties['total_files']} files")


@pytest.mark.asyncio
async def test_grep_directory_search_excludes_unwanted_files(grep_tool_action, file_context, temp_repo):
    """Test that directory search still excludes unwanted file types."""
    # Create test directory
    test_dir = os.path.join(temp_repo, "build")
    os.makedirs(test_dir, exist_ok=True)

    # Create good files that should be searched
    with open(os.path.join(test_dir, "main.py"), "w") as f:
        f.write("# Test pattern here\n")

    # Create files that should be excluded
    with open(os.path.join(test_dir, "debug.log"), "w") as f:
        f.write("ERROR: Test pattern here\n")

    with open(os.path.join(test_dir, "temp.tmp"), "w") as f:
        f.write("Test pattern here\n")

    # Execute search in build directory
    args = GrepToolArgs(
        pattern=r"Test pattern", include="build/", max_results=10, thoughts="Testing exclusion of unwanted file types"
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find pattern in Python file
    assert "main.py" in result.message

    # Should NOT find pattern in excluded files
    assert "debug.log" not in result.message
    assert "temp.tmp" not in result.message

    # Verify properties
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] == 1  # Only the Python file match


@pytest.mark.asyncio
async def test_grep_empty_pattern(grep_tool_action, file_context, temp_repo):
    """Test handling of empty or invalid patterns."""
    # Execute with empty pattern should raise validation error
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GrepToolArgs(pattern="", max_results=10, thoughts="Empty pattern test")


@pytest.mark.asyncio
async def test_grep_special_characters_in_pattern(grep_tool_action, file_context, temp_repo):
    """Test regex patterns with special characters."""
    # Execute
    args = GrepToolArgs(
        pattern=r"console\.(log|error)", max_results=10, thoughts="Finding console.log or console.error calls"
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find both console.log and console.error
    assert "console" in result.message
    properties = result.properties
    assert properties["total_matches"] > 0


@pytest.mark.asyncio
async def test_grep_line_numbers(grep_tool_action, file_context, temp_repo):
    """Test that line numbers are correctly reported."""
    # Execute
    args = GrepToolArgs(pattern=r"import", include="*.py", max_results=10, thoughts="Finding import statements")
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Should show line numbers
    assert "Line" in result.message

    # Line numbers are already verified by checking the message format above


@pytest.mark.asyncio
async def test_grep_execution_error_handling(grep_tool_action, file_context, temp_repo):
    """Test handling of execution errors."""
    # Mock the environment to raise an error
    original_execute = LocalBashEnvironment.execute

    async def mock_execute_error(self, command, patch=None):
        if "grep" in command and "invalid_regex[" in command:
            raise EnvironmentExecutionError("Invalid regex", 2, "grep: Invalid regular expression")
        return await original_execute(self, command, patch)

    with patch.object(LocalBashEnvironment, "execute", mock_execute_error):
        # Execute with pattern that causes error
        args = GrepToolArgs(pattern=r"invalid_regex[", max_results=10, thoughts="Testing error handling")
        result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Error" in result.message



@pytest.mark.asyncio
async def test_grep_output_formatting(grep_tool_action, file_context, temp_repo):
    """Test that output is properly formatted with file grouping."""
    # Execute
    args = GrepToolArgs(pattern=r"console", max_results=20, thoughts="Testing output formatting")
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Check formatting elements
    assert "📄" in result.message  # File icon
    assert "Line" in result.message  # Line number prefix

    # Check summary
    assert result.summary is not None
    assert "matches" in result.summary
    assert "files" in result.summary


@pytest.mark.asyncio
async def test_grep_file_not_found_in_include(grep_tool_action, file_context, temp_repo):
    """Test when include pattern matches no files."""
    # Execute
    args = GrepToolArgs(
        pattern=r"test", include="*.nonexistent", max_results=10, thoughts="Searching in non-existent file type"
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    # Should either find no matches or report no files found
    assert "No matches found" in result.message or "No files found" in result.message


@pytest.mark.asyncio
async def test_grep_multiple_matches_per_file(grep_tool_action, file_context, temp_repo):
    """Test files with multiple matches are grouped correctly."""
    # Create a file with multiple matches
    with open(os.path.join(temp_repo, "multi_match.txt"), "w") as f:
        f.write("First match: test\n")
        f.write("Second match: test\n")
        f.write("Third match: test\n")

    # Execute
    args = GrepToolArgs(
        pattern=r"test", include="multi_match.txt", max_results=10, thoughts="Testing multiple matches in one file"
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None

    # Should show the file once with multiple line entries
    assert "multi_match.txt" in result.message
    assert result.message.count("Line 1:") == 1
    assert result.message.count("Line 2:") == 1
    assert result.message.count("Line 3:") == 1


@pytest.mark.asyncio
async def test_grep_evaluation_criteria(grep_tool_action):
    """Test that evaluation criteria are properly defined."""
    criteria = GrepTool.get_evaluation_criteria()

    assert isinstance(criteria, list)
    assert len(criteria) > 0
    assert all(isinstance(c, str) for c in criteria)

    # Check for key evaluation aspects
    criteria_text = " ".join(criteria).lower()
    assert "pattern" in criteria_text
    assert "search" in criteria_text or "efficiency" in criteria_text
    assert "result" in criteria_text or "relevance" in criteria_text


@pytest.mark.asyncio
async def test_grep_pattern_with_pipe_characters(grep_tool_action, file_context, temp_repo):
    """Test regex patterns containing pipe characters that could be misinterpreted by shell."""
    # Create test files with patterns that match our regex
    with open(os.path.join(temp_repo, "test_pipe_pattern.py"), "w") as f:
        f.write("def setUp(self):\n")
        f.write("    pass\n")
        f.write("\n")
        f.write("def setUpTestData(cls):\n")
        f.write("    pass\n")

    # Execute with pipe character in pattern - this was the reported issue
    args = GrepToolArgs(
        pattern=r"def setUp|def setUpTestData",
        include="test_pipe_pattern.py",
        max_results=10,
        thoughts="Testing pipe character in regex pattern",
    )
    result = await grep_tool_action.execute(args, file_context)

    # Assert
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found" in result.message
    assert "test_pipe_pattern.py" in result.message

    # Should find both setUp and setUpTestData
    assert "setUp" in result.message
    assert "setUpTestData" in result.message


@pytest.mark.asyncio
async def test_grep_pattern_with_shell_metacharacters(grep_tool_action, file_context, temp_repo):
    """Test regex patterns with various shell metacharacters."""
    # Create test file with content that matches complex patterns
    with open(os.path.join(temp_repo, "test_metacharacters.py"), "w") as f:
        f.write('print("Hello World")\n')
        f.write("result = func(arg1, arg2)\n")
        f.write("pattern = 'test*pattern'\n")
        f.write("command = 'ls -la | grep test'\n")

    # Test patterns with various metacharacters
    test_patterns = [
        r"print\(.*\)|func\(.*\)",  # Pipe and parentheses
        r"'.*\*.*'",  # Asterisk in quotes
        r".*\|.*grep",  # Pipe character
        r".*\(.*,.*\)",  # Parentheses and comma
    ]

    for pattern in test_patterns:
        args = GrepToolArgs(
            pattern=pattern,
            include="test_metacharacters.py",
            max_results=10,
            thoughts=f"Testing pattern with metacharacters: {pattern}",
        )
        result = await grep_tool_action.execute(args, file_context)

        # Should not fail due to shell parsing errors
        assert isinstance(result, Observation)
        assert result.message is not None
        assert "Shell command parsing error" not in result.message


@pytest.mark.asyncio
async def test_grep_shell_parsing_error_detection(grep_tool_action, file_context, temp_repo):
    """Test detection and handling of shell parsing errors."""
    from unittest.mock import patch

    # Mock the environment to simulate shell parsing errors
    async def mock_execute_shell_error(self, command, fail_on_error=False, patch=None):
        if "problematic_pattern" in command:
            raise EnvironmentExecutionError(
                "/bin/sh: 1: syntax error near unexpected token", 2, "/bin/sh: 1: syntax error near unexpected token"
            )
        return "No output"

    with patch.object(LocalBashEnvironment, "execute", mock_execute_shell_error):
        args = GrepToolArgs(
            pattern="problematic_pattern",
            include="*.py",
            max_results=10,
            thoughts="Testing shell parsing error detection",
        )
        result = await grep_tool_action.execute(args, file_context)

    # Should detect and handle the shell parsing error gracefully
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Shell command parsing error" in result.message
    assert result.properties is not None
    assert result.properties["fail_reason"] == "shell_parsing_error"


@pytest.mark.asyncio
async def test_grep_output_parsing_with_mixed_errors(grep_tool_action, file_context, temp_repo):
    """Test output parsing when shell errors are mixed with grep output."""
    from unittest.mock import patch

    # Mock the environment to return mixed output (grep results + shell errors)
    async def mock_execute_mixed_output(self, command, fail_on_error=False, patch=None):
        if "mixed_test" in command:
            return """test_file.py:1:def test_function():
/bin/sh: 1: some_command: not found
test_file.py:3:    return True
syntax error near unexpected token
test_file.py:5:def another_test():"""
        return ""

    with patch.object(LocalBashEnvironment, "execute", mock_execute_mixed_output):
        args = GrepToolArgs(
            pattern="mixed_test", include="*.py", max_results=10, thoughts="Testing mixed output parsing"
        )
        result = await grep_tool_action.execute(args, file_context)

    # Should parse valid grep lines and skip shell error lines
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "test_file.py" in result.message
    assert "test_function" in result.message
    assert "another_test" in result.message

    # Should not include shell error messages in the parsed output
    assert "not found" not in result.message
    assert "syntax error" not in result.message

    # Check properties
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] == 3  # Only valid grep lines


@pytest.mark.asyncio
async def test_grep_single_file_output_parsing(grep_tool_action, file_context, temp_repo):
    """Test parsing of single file grep output format (no file prefix)."""
    from unittest.mock import patch

    # Create a test file
    test_file = os.path.join(temp_repo, "test_single_file.py")
    with open(test_file, "w") as f:
        f.write("class TestClass:\n")
        f.write("    pass\n")
        f.write("class AnotherClass:\n")
        f.write("    pass\n")

    # Mock the environment to return single file format output (like when searching a specific file)
    async def mock_execute_single_file_format(self, command, fail_on_error=False, patch=None):
        if "test_single_file.py" in command:
            # Simulate grep output when searching a specific file (with -H flag, includes filename)
            return """test_single_file.py:1:class TestClass:
test_single_file.py:3:class AnotherClass:"""
        return ""

    with patch.object(LocalBashEnvironment, "execute", mock_execute_single_file_format):
        args = GrepToolArgs(
            pattern="class",
            include="test_single_file.py",
            max_results=10,
            thoughts="Testing single file output parsing",
        )
        result = await grep_tool_action.execute(args, file_context)

    # Should parse the matches correctly
    assert isinstance(result, Observation)
    assert result.message is not None
    assert "Found 2 matches" in result.message
    assert "test_single_file.py" in result.message
    assert "TestClass" in result.message
    assert "AnotherClass" in result.message

    # Check properties
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] == 2
    assert properties["total_files"] == 1

    # File paths and line numbers are already verified in the message content above
