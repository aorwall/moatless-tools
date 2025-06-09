import asyncio
import tempfile

import pytest

from moatless.actions.run_python_code import RunPythonCode, RunPythonCodeArgs
from moatless.environment.local import LocalBashEnvironment
from moatless.workspace import Workspace

import logging

logger = logging.getLogger(__name__)

class TestRunPythonCodeErrorOutput:
    """Test that full error output is returned, not just generic error messages."""
    
    @pytest.fixture
    def workspace_and_temp_dir(self):
        """Create a real workspace with LocalBashEnvironment."""
        temp_dir = tempfile.mkdtemp()
        try:
            environment = LocalBashEnvironment(cwd=temp_dir)
            workspace = Workspace(environment=environment)
            yield workspace, temp_dir
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_import_error_returns_full_traceback(self, workspace_and_temp_dir):
        """Test that import errors return the full traceback, not just 'Error executing code:'."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        # Code that will cause an ImportError
        code_with_import_error = """
import nonexistent_module_that_definitely_does_not_exist
print("This line should never be reached")
"""
        
        args = RunPythonCodeArgs(code=code_with_import_error)
        result = await action.execute(args, None)

        logger.info(result.message)
        
        # The result should NOT just say "Error executing code:"
        # It should contain the full Python traceback
        assert "Python output:" in result.message, "Should show Python output with error details"
        assert "ModuleNotFoundError" in result.message or "ImportError" in result.message, "Should contain the actual error type"
        assert "nonexistent_module_that_definitely_does_not_exist" in result.message, "Should mention the problematic module"
        assert "Traceback" in result.message, "Should include the full traceback"
        
        # Ensure we're NOT getting a generic error message
        assert not result.message.startswith("Error executing code:"), "Should not start with generic error message"
        assert result.properties is None or result.properties.get("fail_reason") != "execution_error", "Should not set fail_reason for Python errors"

    @pytest.mark.asyncio
    async def test_syntax_error_returns_full_output(self, workspace_and_temp_dir):
        """Test that syntax errors return the full Python error output."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        # Code with syntax error
        code_with_syntax_error = """
print("Missing closing quote
print("This line won't be reached")
"""
        
        args = RunPythonCodeArgs(code=code_with_syntax_error)
        result = await action.execute(args, None)
        
        # Should return the full Python syntax error output
        assert "Python output:" in result.message, "Should show Python output"
        assert "SyntaxError" in result.message, "Should contain SyntaxError"
        assert "unterminated string literal" in result.message or "EOL while scanning string literal" in result.message, "Should contain specific syntax error details"
        
        # Should NOT be a generic error
        assert not result.message.startswith("Error executing code:"), "Should not be generic error"

    @pytest.mark.asyncio
    async def test_runtime_error_returns_full_output(self, workspace_and_temp_dir):
        """Test that runtime errors return the full Python error output."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        # Code that will cause a runtime error
        code_with_runtime_error = """
def divide_by_zero():
    return 10 / 0

print("About to call divide_by_zero...")
result = divide_by_zero()
print(f"Result: {result}")
"""
        
        args = RunPythonCodeArgs(code=code_with_runtime_error)
        result = await action.execute(args, None)
        
        # Should return the full Python error output including the function call stack
        assert "Python output:" in result.message, "Should show Python output"
        assert "ZeroDivisionError" in result.message, "Should contain ZeroDivisionError"
        assert "division by zero" in result.message, "Should contain specific error message"
        assert "About to call divide_by_zero..." in result.message, "Should include output before the error"
        assert "divide_by_zero" in result.message, "Should show the function name in traceback"
        assert "Traceback" in result.message, "Should include full traceback"

    @pytest.mark.asyncio
    async def test_mixed_output_and_error(self, workspace_and_temp_dir):
        """Test that both stdout and stderr are captured when code produces both."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        # Code that produces both normal output and an error
        code_with_mixed_output = """
import sys

print("Starting script...")
print("Normal output to stdout")
print("Writing to stderr", file=sys.stderr)

for i in range(3):
    print(f"Loop iteration {i}")

# Now cause an error
undefined_variable
"""
        
        args = RunPythonCodeArgs(code=code_with_mixed_output)
        result = await action.execute(args, None)
        
        # Should capture both the normal output AND the error
        assert "Python output:" in result.message, "Should show Python output"
        assert "Starting script..." in result.message, "Should include initial output"
        assert "Loop iteration" in result.message, "Should include loop output"
        assert "Writing to stderr" in result.message, "Should capture stderr output"
        assert "NameError" in result.message, "Should include the error"
        assert "undefined_variable" in result.message, "Should mention the problematic variable"

    @pytest.mark.asyncio
    async def test_no_generic_error_messages(self, workspace_and_temp_dir):
        """Test that we never get generic 'Error executing code:' messages for Python errors."""
        workspace, temp_dir = workspace_and_temp_dir
        action = RunPythonCode()
        await action.initialize(workspace)
        
        # Test multiple types of errors
        error_codes = [
            "import this_module_does_not_exist",  # ImportError
            "undefined_var",  # NameError  
            "1 / 0",  # ZeroDivisionError
            "print('unclosed string",  # SyntaxError
            "[].pop()",  # IndexError
        ]
        
        for code in error_codes:
            args = RunPythonCodeArgs(code=code)
            result = await action.execute(args, None)
            
            # For all Python errors, we should get the actual Python output, not generic error messages
            assert not result.message.startswith("Error executing code:"), f"Code '{code}' returned generic error instead of Python output"
            assert "Python output:" in result.message, f"Code '{code}' should show Python output"
            assert result.properties is None or result.properties.get("fail_reason") != "execution_error", f"Code '{code}' should not set execution_error flag"