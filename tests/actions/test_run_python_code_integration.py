import asyncio
import tempfile
import os

import pytest

from moatless.actions.run_python_code import RunPythonCode, RunPythonCodeArgs
from moatless.environment.local import LocalBashEnvironment
from moatless.file_context import FileContext
from moatless.workspace import Workspace


class TestRunPythonCodeIntegration:
    """Integration tests using real LocalBashEnvironment."""
    
    @pytest.fixture
    def workspace_and_temp_dir(self):
        """Create a real workspace with LocalBashEnvironment."""
        # Create a temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        try:
            environment = LocalBashEnvironment(cwd=temp_dir)
            workspace = Workspace(environment=environment)
            yield workspace, temp_dir
        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_execute_simple_print(self, workspace_and_temp_dir):
        """Test executing a simple print statement."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        code = "print('Hello from integration test!')"
        args = RunPythonCodeArgs(code=code)
        
        result = await action.execute(args, None)
        
        assert result.message.startswith("Python output:")
        assert "Hello from integration test!" in result.message
        assert not result.properties  # No error properties

    @pytest.mark.asyncio
    async def test_execute_multiline_code_with_imports(self, workspace_and_temp_dir):
        """Test executing multiline code with imports."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        code = """
import json
import sys

data = {"test": "value", "number": 42}
print(f"Python version: {sys.version.split()[0]}")
print(f"JSON output: {json.dumps(data)}")
"""
        args = RunPythonCodeArgs(code=code)
        
        result = await action.execute(args, None)
        
        assert "Python output:" in result.message
        assert "Python version:" in result.message
        assert 'JSON output: {"test": "value", "number": 42}' in result.message

    @pytest.mark.asyncio
    async def test_execute_with_file_operations(self, workspace_and_temp_dir):
        """Test code that performs file operations in the workspace."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        code = """
# Write a test file
with open('test_output.txt', 'w') as f:
    f.write('This is a test file created by RunPythonCode')
    
# Read it back
with open('test_output.txt', 'r') as f:
    content = f.read()
    
print(f"File content: {content}")
print("File operation successful!")
"""
        args = RunPythonCodeArgs(code=code)
        
        result = await action.execute(args, None)
        
        assert "Python output:" in result.message
        assert "File content: This is a test file created by RunPythonCode" in result.message
        assert "File operation successful!" in result.message
        
        # Verify the file was actually created in the workspace
        test_file_path = os.path.join(temp_dir, 'test_output.txt')
        assert os.path.exists(test_file_path)
        with open(test_file_path, 'r') as f:
            assert f.read() == 'This is a test file created by RunPythonCode'

    @pytest.mark.asyncio
    async def test_execute_with_error_handling(self, workspace_and_temp_dir):
        """Test handling of Python errors."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        code = """
# This will cause a NameError
print(undefined_variable)
"""
        args = RunPythonCodeArgs(code=code)
        
        result = await action.execute(args, None)
        
        # The error output should be captured and returned as successful execution
        # This allows the LLM to see and understand the error
        assert "Python output:" in result.message
        assert "NameError: name 'undefined_variable' is not defined" in result.message
        assert "Traceback" in result.message

    @pytest.mark.asyncio
    async def test_temp_file_cleanup(self, workspace_and_temp_dir):
        """Test that temporary files are cleaned up."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        # Get current working directory (where temp files are created) before execution
        files_before = set(os.listdir(temp_dir))
        
        code = "print('Testing temp file cleanup')"
        args = RunPythonCodeArgs(code=code)
        
        result = await action.execute(args, None)
        
        assert "Python output:" in result.message
        
        # Give a moment for cleanup to complete
        await asyncio.sleep(0.1)
        
        # Check that no new moatless_temp files remain in the working directory
        files_after = set(os.listdir(temp_dir))
        new_files = files_after - files_before
        moatless_temp_files = [f for f in new_files if f.startswith('moatless_temp_')]
        
        assert len(moatless_temp_files) == 0, f"Temporary files not cleaned up: {moatless_temp_files}"

    @pytest.mark.asyncio
    async def test_execute_with_data_processing(self, workspace_and_temp_dir):
        """Test more complex data processing code."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        code = """
# Simulate data processing
data = [1, 2, 3, 4, 5]
squared = [x**2 for x in data]
total = sum(squared)

print(f"Original data: {data}")
print(f"Squared values: {squared}")
print(f"Sum of squares: {total}")

# Dictionary operations
info = {
    "count": len(data),
    "sum": sum(data),
    "average": sum(data) / len(data)
}

for key, value in info.items():
    print(f"{key}: {value}")
"""
        args = RunPythonCodeArgs(code=code)
        
        result = await action.execute(args, None)
        
        assert "Python output:" in result.message
        assert "Original data: [1, 2, 3, 4, 5]" in result.message
        assert "Squared values: [1, 4, 9, 16, 25]" in result.message
        assert "Sum of squares: 55" in result.message
        assert "count: 5" in result.message
        assert "average: 3.0" in result.message