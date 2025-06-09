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
        # Test valid args with defaults
        args = RunPythonScriptArgs(script_path="test.py")
        assert args.script_path == "test.py"
        assert args.args == []
        assert args.timeout == 30
        assert args.max_output_tokens == 2000

        # Test args with all parameters
        args = RunPythonScriptArgs(
            script_path="script.py", 
            args=["--verbose", "input.txt"], 
            timeout=120,
            max_output_tokens=5000
        )
        assert args.script_path == "script.py"
        assert args.args == ["--verbose", "input.txt"]
        assert args.timeout == 120
        assert args.max_output_tokens == 5000

    def test_action_name(self):
        """Test that the action has the correct name."""
        action = RunPythonScript()
        assert action.name == "RunPythonScript"
        assert RunPythonScript.get_name() == "RunPythonScript"

    @pytest.mark.asyncio
    async def test_output_truncation_large_output(self, file_context, workspace):
        """Test that large outputs are truncated based on token count."""
        # Create a large output that exceeds token limit
        large_output = "This is a test line.\n" * 1000  # Should exceed 2000 tokens
        workspace.environment.execute.return_value = large_output

        # Create and initialize the action
        action = RunPythonScript()
        await action.initialize(workspace)

        # Execute with default max_output_tokens (2000)
        args = RunPythonScriptArgs(script_path="large_output_script.py")
        result = await action.execute(args, file_context)

        # Verify output was truncated
        assert "Python output:" in result.message
        assert "[Output truncated at 2000 tokens" in result.message
        assert "Please revise the script to show less output" in result.message
        assert result.properties.get("fail_reason") == "truncated"
        # The result should be shorter than the original
        assert len(result.message) < len(f"Python output:\n{large_output}")

    @pytest.mark.asyncio
    async def test_output_truncation_small_output(self, file_context, workspace):
        """Test that small outputs are not truncated."""
        # Create a small output that should not be truncated
        small_output = "Hello, World!\nThis is a small output."
        workspace.environment.execute.return_value = small_output

        # Create and initialize the action
        action = RunPythonScript()
        await action.initialize(workspace)

        # Execute with default max_output_tokens
        args = RunPythonScriptArgs(script_path="small_output_script.py")
        result = await action.execute(args, file_context)

        # Verify output was not truncated
        assert "Python output:" in result.message
        assert small_output in result.message
        assert "Output truncated" not in result.message
        assert result.properties.get("fail_reason") is None

    @pytest.mark.asyncio
    async def test_output_truncation_custom_limit(self, file_context, workspace):
        """Test that custom token limits work correctly."""
        # Create medium-sized output
        medium_output = "Test line\n" * 100
        workspace.environment.execute.return_value = medium_output

        # Create and initialize the action
        action = RunPythonScript()
        await action.initialize(workspace)

        # Execute with very low max_output_tokens
        args = RunPythonScriptArgs(script_path="medium_script.py", max_output_tokens=50)
        result = await action.execute(args, file_context)

        # Verify output was truncated according to custom limit
        assert "Python output:" in result.message
        assert "[Output truncated at 50 tokens" in result.message
        assert result.properties.get("fail_reason") == "truncated"

    @pytest.mark.asyncio
    async def test_error_output_truncation(self, file_context, workspace):
        """Test that error outputs are also truncated."""
        # Create large error output
        large_error = "Error: " + "Long error message. " * 500
        workspace.environment.execute.side_effect = EnvironmentExecutionError(
            "Script failed", return_code=1, stderr=large_error
        )

        # Create and initialize the action
        action = RunPythonScript()
        await action.initialize(workspace)

        # Execute with default max_output_tokens
        args = RunPythonScriptArgs(script_path="error_script.py")
        result = await action.execute(args, file_context)

        # Verify error output was truncated
        assert "Python output:" in result.message
        assert "[Error output truncated at 2000 tokens" in result.message
        assert result.properties.get("fail_reason") == "execution_error_truncated"

    def test_truncate_output_by_tokens_method(self):
        """Test the _truncate_output_by_tokens method directly."""
        action = RunPythonScript()
        
        # Test with empty output
        result, was_truncated = action._truncate_output_by_tokens("", 100)
        assert result == ""
        assert not was_truncated

        # Test with small output (should not be truncated)
        small_text = "Hello World"
        result, was_truncated = action._truncate_output_by_tokens(small_text, 100)
        assert result == small_text
        assert not was_truncated

        # Test with large single line (should be truncated)
        large_single_line = "word " * 1000
        result, was_truncated = action._truncate_output_by_tokens(large_single_line, 10)
        assert was_truncated
        assert len(result) < len(large_single_line)

        # Test with multiple lines (should be truncated by lines)
        large_multiline = "Test line\n" * 100
        result, was_truncated = action._truncate_output_by_tokens(large_multiline, 50)
        assert was_truncated
        assert result.count('\n') < large_multiline.count('\n')

    def test_improved_truncation_efficiency(self):
        """Test that the improved truncation preserves more content."""
        action = RunPythonScript()
        
        # Create test content with known token characteristics
        test_content = "Line 1: This is a test line with some content.\n" * 50
        original_tokens = action._truncate_output_by_tokens(test_content, 10000)[0]  # Get original without truncation
        
        # Test with moderate limit - should preserve significant portion
        result, was_truncated = action._truncate_output_by_tokens(test_content, 1000)
        if was_truncated:
            # Should preserve a significant portion, not just 400 tokens when limit is 2000
            result_tokens_estimate = len(result.split()) * 1.3  # Rough token estimate
            # Should get at least 70% of target tokens to avoid being too conservative
            assert result_tokens_estimate >= 700, f"Only got ~{result_tokens_estimate} tokens from 1000 limit"
