import asyncio
import os
import pytest
import tempfile

from moatless.environment.local import LocalBashEnvironment
from moatless.environment.base import EnvironmentExecutionError


@pytest.mark.asyncio
async def test_execute():
    """Test the execute method of LocalBashEnvironment."""
    env = LocalBashEnvironment()

    # Test successful command execution
    result = await env.execute("echo 'Hello World'")
    assert result.strip() == "Hello World"

    # Test with cwd
    with tempfile.TemporaryDirectory() as temp_dir:
        env_with_cwd = LocalBashEnvironment(cwd=temp_dir)
        result = await env_with_cwd.execute("pwd")
        assert temp_dir in result

    # Test command that fails
    with pytest.raises(EnvironmentExecutionError):
        await env.execute("command_that_does_not_exist")

    # Test with environment variables
    env_with_vars = LocalBashEnvironment(env={"TEST_VAR": "test_value"})
    result = await env_with_vars.execute("echo $TEST_VAR")
    assert "test_value" in result


@pytest.mark.asyncio
async def test_read_file():
    """Test the read_file method of LocalBashEnvironment."""
    env = LocalBashEnvironment()

    # Create a temporary file with known content
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write("Test content")
        temp_path = temp_file.name

    try:
        # Read the file
        content = await env.read_file(temp_path)
        assert content == "Test content"

        # Test with a directory as cwd
        dir_path = os.path.dirname(temp_path)
        file_name = os.path.basename(temp_path)
        env_with_cwd = LocalBashEnvironment(cwd=dir_path)
        content = await env_with_cwd.read_file(file_name)
        assert content == "Test content"

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            await env.read_file("non_existent_file.txt")
    finally:
        # Clean up
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_write_file():
    """Test the write_file method of LocalBashEnvironment."""
    env = LocalBashEnvironment()

    # Create a temporary path for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test writing to a file
        test_file = os.path.join(temp_dir, "test_file.txt")
        await env.write_file(test_file, "Test content")

        # Verify the content was written correctly
        with open(test_file, "r") as f:
            content = f.read()
            assert content == "Test content"

        # Test writing to a file in a subdirectory that doesn't exist yet
        test_subdir_file = os.path.join(temp_dir, "subdir", "test_file.txt")
        await env.write_file(test_subdir_file, "Subdir test content")

        # Verify the content was written correctly and the directory was created
        assert os.path.exists(os.path.join(temp_dir, "subdir"))
        with open(test_subdir_file, "r") as f:
            content = f.read()
            assert content == "Subdir test content"

        # Test with a directory as cwd
        env_with_cwd = LocalBashEnvironment(cwd=temp_dir)
        await env_with_cwd.write_file("cwd_test_file.txt", "CWD test content")

        # Verify the content was written correctly
        with open(os.path.join(temp_dir, "cwd_test_file.txt"), "r") as f:
            content = f.read()
            assert content == "CWD test content"


@pytest.mark.asyncio
async def test_read_write_integration():
    """Test the integration of read_file and write_file methods."""
    env = LocalBashEnvironment()

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write content to a file
        test_file = os.path.join(temp_dir, "test_file.txt")
        original_content = "Original content"
        await env.write_file(test_file, original_content)

        # Read the content back
        read_content = await env.read_file(test_file)
        assert read_content == original_content

        # Modify the content
        modified_content = "Modified content"
        await env.write_file(test_file, modified_content)

        # Read the modified content
        read_modified_content = await env.read_file(test_file)
        assert read_modified_content == modified_content
