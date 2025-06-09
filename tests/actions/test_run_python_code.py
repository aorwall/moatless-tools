import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from moatless.actions.run_python_code import RunPythonCode, RunPythonCodeArgs
from moatless.file_context import FileContext
from moatless.workspace import Workspace


class TestRunPythonCodeAction:
    @pytest.fixture
    def file_context(self):
        mock_file_context = MagicMock(spec=FileContext)
        return mock_file_context

    @pytest.fixture
    def workspace(self):
        mock_workspace = MagicMock(spec=Workspace)
        mock_workspace.environment = AsyncMock()
        return mock_workspace

    @pytest.fixture
    def run_python_code_action(self, workspace):
        action = RunPythonCode()
        asyncio.run(action.initialize(workspace))
        return action

    @pytest.mark.asyncio
    async def test_execute_python_code_success(self, file_context, workspace):
        """Test successful execution of Python code."""
        # Mock successful code execution
        expected_output = "Hello, World!"
        workspace.environment.execute_python_code.return_value = expected_output

        # Create and initialize the action
        action = RunPythonCode()
        await action.initialize(workspace)

        # Execute the action
        code = "print('Hello, World!')"
        args = RunPythonCodeArgs(code=code)
        result = await action.execute(args, file_context)

        # Verify the method was called correctly
        workspace.environment.execute_python_code.assert_called_once_with(code)
        
        # Verify the output
        assert "Python output:" in result.message
        assert expected_output in result.message
        assert "print('Hello, World!')" in result.summary

    @pytest.mark.asyncio
    async def test_execute_multiline_code(self, file_context, workspace):
        """Test execution of multiline Python code."""
        # Mock successful code execution
        expected_output = "Result: 15"
        workspace.environment.execute_python_code.return_value = expected_output

        # Create and initialize the action
        action = RunPythonCode()
        await action.initialize(workspace)

        # Execute the action with multiline code
        code = """def add(a, b):
    return a + b

result = add(5, 10)
print(f"Result: {result}")"""
        
        args = RunPythonCodeArgs(code=code)
        result = await action.execute(args, file_context)

        # Verify the method was called correctly
        workspace.environment.execute_python_code.assert_called_once_with(code)
        
        # Verify the output and summary
        assert "Python output:" in result.message
        assert expected_output in result.message
        assert "def add(a, b):..." in result.summary

    @pytest.mark.asyncio
    async def test_execute_code_with_error(self, file_context, workspace):
        """Test handling of code execution errors."""
        # Mock code execution error
        error_message = "NameError: name 'undefined_variable' is not defined"
        workspace.environment.execute_python_code.side_effect = Exception(error_message)

        # Create and initialize the action
        action = RunPythonCode()
        await action.initialize(workspace)

        # Execute the action
        code = "print(undefined_variable)"
        args = RunPythonCodeArgs(code=code)
        result = await action.execute(args, file_context)

        # Verify error handling
        assert "Error executing code:" in result.message
        assert error_message in result.message
        assert result.properties.get("fail_reason") == "execution_error"

    @pytest.mark.asyncio
    async def test_execute_without_environment(self, file_context):
        """Test error handling when no environment is available."""
        # Create workspace without environment
        workspace = MagicMock(spec=Workspace)
        workspace.environment = None

        # Create and initialize the action
        action = RunPythonCode()
        await action.initialize(workspace)

        # Execute should raise ValueError
        args = RunPythonCodeArgs(code="print('test')")
        
        with pytest.raises(ValueError, match="Environment is required to run Python code"):
            await action.execute(args, file_context)

    @pytest.mark.asyncio
    async def test_execute_with_timeout_parameter(self, file_context, workspace):
        """Test that timeout parameter is properly handled in args schema."""
        # Mock successful code execution
        expected_output = "Long running task completed"
        workspace.environment.execute_python_code.return_value = expected_output

        # Create and initialize the action
        action = RunPythonCode()
        await action.initialize(workspace)

        # Execute with custom timeout
        code = """import time
time.sleep(1)
print("Long running task completed")"""
        
        args = RunPythonCodeArgs(code=code, timeout=60)
        result = await action.execute(args, file_context)

        # Verify execution occurred
        workspace.environment.execute_python_code.assert_called_once_with(code)
        assert "Python output:" in result.message

    def test_args_schema_validation(self):
        """Test that the RunPythonCodeArgs schema validates correctly."""
        # Test valid args
        args = RunPythonCodeArgs(code="print('test')")
        assert args.code == "print('test')"
        assert args.timeout == 30

        # Test args with custom timeout
        args = RunPythonCodeArgs(
            code="import time; time.sleep(2)",
            timeout=120
        )
        assert args.code == "import time; time.sleep(2)"
        assert args.timeout == 120

    def test_action_name(self):
        """Test that the action has the correct name."""
        action = RunPythonCode()
        assert action.name == "RunPythonCode"
        assert RunPythonCode.get_name() == "RunPythonCode"
        
    @pytest.mark.asyncio
    async def test_empty_code_summary(self, file_context, workspace):
        """Test summary generation for empty code."""
        # Mock successful execution
        workspace.environment.execute_python_code.return_value = ""

        # Create and initialize the action
        action = RunPythonCode()
        await action.initialize(workspace)

        # Execute with empty code
        args = RunPythonCodeArgs(code="")
        result = await action.execute(args, file_context)

        # Verify summary handles empty code gracefully
        assert "Executed Python code: <empty code>" in result.summary