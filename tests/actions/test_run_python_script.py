import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from moatless.actions.run_python_script import RunPythonScript, RunPythonScriptArgs
from moatless.environment import EnvironmentExecutionError
from moatless.file_context import FileContext
from moatless.workspace import Workspace


class TestRunPythonScriptAction:
    @pytest.fixture
    def file_context(self):
        mock_file_context = MagicMock(spec=FileContext)
        mock_file_context.shadow_mode = False
        mock_file_context.generate_git_patch.return_value = None
        return mock_file_context

    @pytest.fixture
    def workspace(self):
        mock_workspace = MagicMock(spec=Workspace)
        mock_workspace.environment = AsyncMock()
        return mock_workspace

    @pytest.fixture
    def run_python_script_action(self, workspace):
        action = RunPythonScript()
        asyncio.run(action.initialize(workspace))
        return action

    @pytest.mark.asyncio
    async def test_execute_python_script_success(self, file_context, workspace):
        """Test successful execution of a Python script."""
        # Mock successful script execution
        expected_output = "Hello, World!\nScript completed successfully."
        workspace.environment.execute.return_value = expected_output

        # Create and initialize the action
        action = RunPythonScript()
        await action.initialize(workspace)

        # Execute the action
        args = RunPythonScriptArgs(script_path="test_script.py")
        result = await action.execute(args, file_context)

        # Verify the command was called correctly
        workspace.environment.execute.assert_called_once_with("python test_script.py", patch=None, fail_on_error=True)

        # Verify the output
        assert "Python output:" in result.message
        assert expected_output in result.message

    @pytest.mark.asyncio
    async def test_execute_python_script_with_args(self, file_context, workspace):
        """Test execution of a Python script with command line arguments."""
        # Mock successful script execution
        expected_output = "Arguments processed: arg1 arg2"
        workspace.environment.execute.return_value = expected_output

        # Create and initialize the action
        action = RunPythonScript()
        await action.initialize(workspace)

        # Execute the action with arguments
        args = RunPythonScriptArgs(script_path="test_script.py", args=["arg1", "arg2"])
        result = await action.execute(args, file_context)

        # Verify the command was called correctly with arguments
        workspace.environment.execute.assert_called_once_with(
            "python test_script.py arg1 arg2", patch=None, fail_on_error=True
        )

        # Verify the output
        assert "Python output:" in result.message
        assert expected_output in result.message

    @pytest.mark.asyncio
    async def test_execute_python_script_with_error(self, file_context, workspace):
        """Test handling of script execution errors."""
        # Mock script execution error
        error_message = "ModuleNotFoundError: No module named 'nonexistent'"
        workspace.environment.execute.side_effect = EnvironmentExecutionError(
            "Script execution failed", return_code=1, stderr=error_message
        )

        # Create and initialize the action
        action = RunPythonScript()
        await action.initialize(workspace)

        # Execute the action
        args = RunPythonScriptArgs(script_path="failing_script.py")
        result = await action.execute(args, file_context)

        # Verify error handling
        assert "Python output:" in result.message
        assert error_message in result.message
        assert result.properties.get("fail_reason") == "execution_error"

    @pytest.mark.asyncio
    async def test_execute_without_environment(self, file_context):
        """Test error handling when no environment is available."""
        # Create workspace without environment
        workspace = MagicMock(spec=Workspace)
        workspace.environment = None

        # Create and initialize the action
        action = RunPythonScript()
        await action.initialize(workspace)

        # Execute should raise ValueError
        args = RunPythonScriptArgs(script_path="test_script.py")

        with pytest.raises(ValueError, match="Environment is required to run Python scripts"):
            await action.execute(args, file_context)

    @pytest.mark.asyncio
    async def test_execute_with_timeout_parameter(self, file_context, workspace):
        """Test that timeout parameter is properly handled in args schema."""
        # Mock successful script execution
        expected_output = "Long running script output"
        workspace.environment.execute.return_value = expected_output

        # Create and initialize the action
        action = RunPythonScript()
        await action.initialize(workspace)

        # Execute with custom timeout
        args = RunPythonScriptArgs(script_path="long_script.py", timeout=60)
        result = await action.execute(args, file_context)

        # Verify execution occurred (timeout handling would be in environment layer)
        workspace.environment.execute.assert_called_once_with("python long_script.py", patch=None, fail_on_error=True)
        assert "Python output:" in result.message

    def test_args_schema_validation(self):
        """Test that the RunPythonScriptArgs schema validates correctly."""
        # Test valid args
        args = RunPythonScriptArgs(script_path="test.py")
        assert args.script_path == "test.py"
        assert args.args == []
        assert args.timeout == 30

        # Test args with parameters
        args = RunPythonScriptArgs(script_path="script.py", args=["--verbose", "input.txt"], timeout=120)
        assert args.script_path == "script.py"
        assert args.args == ["--verbose", "input.txt"]
        assert args.timeout == 120

    def test_action_name(self):
        """Test that the action has the correct name."""
        action = RunPythonScript()
        assert action.name == "RunPythonScript"
        assert RunPythonScript.get_name() == "RunPythonScript"
