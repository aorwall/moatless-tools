import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from moatless.environment.base import BaseEnvironment
from moatless.environment.local import LocalBashEnvironment


class TestEnvironmentExecutePythonCode:
    @pytest.mark.asyncio
    async def test_execute_python_code_success(self):
        """Test successful execution of Python code through environment."""
        # Create a mock environment
        env = LocalBashEnvironment()
        
        # Mock the underlying methods
        env.write_file = AsyncMock()
        env.execute = AsyncMock(return_value="Hello from Python!")
        
        # Execute Python code
        code = "print('Hello from Python!')"
        result = await env.execute_python_code(code)
        
        # Verify write_file was called with temp file in current directory
        assert env.write_file.called
        temp_file_path = env.write_file.call_args[0][0]
        assert "moatless_temp_" in temp_file_path
        assert temp_file_path.endswith(".py")
        assert "/" not in temp_file_path  # Should be in current directory, not a full path
        assert env.write_file.call_args[0][1] == code
        
        # Verify execute was called twice (once for Python, once for cleanup)
        assert env.execute.call_count == 2
        python_cmd = env.execute.call_args_list[0][0][0]
        assert python_cmd.startswith("python ")
        assert temp_file_path in python_cmd
        
        cleanup_cmd = env.execute.call_args_list[1][0][0]
        assert cleanup_cmd.startswith("rm -f ")
        assert temp_file_path in cleanup_cmd
        
        # Verify result
        assert result == "Hello from Python!"

    @pytest.mark.asyncio
    async def test_execute_python_code_no_cleanup(self):
        """Test execution without cleanup."""
        # Create a mock environment
        env = LocalBashEnvironment()
        
        # Mock the underlying methods
        env.write_file = AsyncMock()
        env.execute = AsyncMock(return_value="Test output")
        
        # Execute Python code without cleanup
        code = "print('Test')"
        result = await env.execute_python_code(code, cleanup=False)
        
        # Verify execute was called only once (no cleanup)
        assert env.execute.call_count == 1
        python_cmd = env.execute.call_args[0][0]
        assert python_cmd.startswith("python ")
        
        # Verify result
        assert result == "Test output"

    @pytest.mark.asyncio
    async def test_execute_python_code_with_error(self):
        """Test handling of execution errors."""
        # Create a mock environment
        env = LocalBashEnvironment()
        
        # Mock the underlying methods
        env.write_file = AsyncMock()
        env.execute = AsyncMock()
        env.execute.side_effect = [Exception("Syntax error"), None]  # Error on Python execution, success on cleanup
        
        # Execute Python code
        code = "print('Invalid syntax"
        
        with pytest.raises(Exception, match="Syntax error"):
            await env.execute_python_code(code)
        
        # Verify cleanup was still attempted
        assert env.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_python_code_cleanup_error_ignored(self):
        """Test that cleanup errors are ignored."""
        # Create a mock environment
        env = LocalBashEnvironment()
        
        # Mock the underlying methods
        env.write_file = AsyncMock()
        env.execute = AsyncMock()
        env.execute.side_effect = ["Success", Exception("Cleanup failed")]
        
        # Execute Python code - should not raise even if cleanup fails
        code = "print('Test')"
        result = await env.execute_python_code(code)
        
        # Verify result is still returned
        assert result == "Success"
        
    @pytest.mark.asyncio
    async def test_unique_temp_filenames(self):
        """Test that unique temporary filenames are generated."""
        # Create a mock environment
        env = LocalBashEnvironment()
        
        # Mock the underlying methods
        env.write_file = AsyncMock()
        env.execute = AsyncMock(return_value="")
        
        # Execute different code snippets
        code1 = "print('Code 1')"
        code2 = "print('Code 2')"
        
        await env.execute_python_code(code1)
        await env.execute_python_code(code2)
        
        # Get the temp file paths used
        temp_file1 = env.write_file.call_args_list[0][0][0]
        temp_file2 = env.write_file.call_args_list[1][0][0]
        
        # Verify they are different
        assert temp_file1 != temp_file2
        assert "moatless_temp_" in temp_file1
        assert "moatless_temp_" in temp_file2