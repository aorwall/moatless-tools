import pytest
from unittest.mock import MagicMock, patch

from moatless.actions.view_diff import ViewDiff, ViewDiffArgs
from moatless.actions.schema import Observation
from moatless.file_context import FileContext
from moatless.environment.base import BaseEnvironment
from moatless.workspace import Workspace


@pytest.mark.asyncio
async def test_view_diff_shadow_workspace():
    """Test ViewDiff action with shadow workspace."""
    # Mock FileContext with shadow_workspace=True
    file_context = MagicMock(spec=FileContext)
    file_context.generate_git_patch.return_value = "mock diff content"

    # Create workspace with shadow_mode=True
    mock_workspace = MagicMock(spec=Workspace)
    mock_workspace.shadow_mode = True

    # Create action and set workspace
    action = ViewDiff()
    action.workspace = mock_workspace

    args = ViewDiffArgs(thoughts="Viewing diff in shadow workspace")

    observation = await action.execute(args, file_context)

    # Verify the result
    assert isinstance(observation, Observation)
    assert "Current changes in workspace:\nmock diff content" == observation.message

    # Verify the shadow workspace method was called
    file_context.generate_git_patch.assert_called_once()


@pytest.mark.asyncio
async def test_view_diff_no_changes():
    """Test ViewDiff action when there are no changes."""
    # Mock FileContext
    file_context = MagicMock(spec=FileContext)

    # Mock the environment
    mock_env = MagicMock(spec=BaseEnvironment)
    mock_env.execute.return_value = ""  # No changes

    # Create workspace with shadow_mode=False
    mock_workspace = MagicMock(spec=Workspace)
    mock_workspace.shadow_mode = False
    mock_workspace.environment = mock_env

    # Create action and set workspace
    action = ViewDiff()
    action.workspace = mock_workspace

    args = ViewDiffArgs(thoughts="Viewing diff with real git")

    observation = await action.execute(args, file_context)

    # Verify the result
    assert isinstance(observation, Observation)
    assert observation.message == "No changes detected in the workspace."


@pytest.mark.asyncio
async def test_view_diff_with_main_branch():
    """Test ViewDiff action that correctly gets diff with main branch."""
    # Mock FileContext
    file_context = MagicMock(spec=FileContext)

    # Mock the environment to return a valid diff for main branch
    mock_env = MagicMock(spec=BaseEnvironment)
    mock_env.execute.side_effect = [
        "diff from main branch"  # First call succeeds
    ]

    # Create workspace with shadow_mode=False
    mock_workspace = MagicMock(spec=Workspace)
    mock_workspace.shadow_mode = False
    mock_workspace.environment = mock_env

    # Create action and set workspace
    action = ViewDiff()
    action.workspace = mock_workspace

    args = ViewDiffArgs(thoughts="Viewing diff with main branch")

    observation = await action.execute(args, file_context)

    # Verify the result
    assert isinstance(observation, Observation)
    assert "Current changes in workspace:\ndiff from main branch" == observation.message

    # Verify git diff main was called
    mock_env.execute.assert_called_once_with("git diff main", fail_on_error=False)


@pytest.mark.asyncio
async def test_view_diff_fallback_to_master():
    """Test ViewDiff action falls back to master branch if main fails."""
    # Mock FileContext
    file_context = MagicMock(spec=FileContext)

    # Mock the environment to fail on main but succeed on master
    mock_env = MagicMock(spec=BaseEnvironment)
    mock_env.execute.side_effect = [
        "fatal: ambiguous argument 'main'",  # First call fails
        "diff from master branch",  # Second call succeeds
    ]

    # Create workspace with shadow_mode=False
    mock_workspace = MagicMock(spec=Workspace)
    mock_workspace.shadow_mode = False
    mock_workspace.environment = mock_env

    # Create action and set workspace
    action = ViewDiff()
    action.workspace = mock_workspace

    args = ViewDiffArgs(thoughts="Viewing diff with master fallback")

    observation = await action.execute(args, file_context)

    # Verify the result
    assert isinstance(observation, Observation)
    assert "Current changes in workspace:\ndiff from master branch" == observation.message

    # Verify both git diff commands were called
    assert mock_env.execute.call_count == 2
    mock_env.execute.assert_any_call("git diff main", fail_on_error=False)
    mock_env.execute.assert_any_call("git diff master", fail_on_error=False)


@pytest.mark.asyncio
async def test_view_diff_fallback_to_uncommitted():
    """Test ViewDiff action falls back to uncommitted changes if both branches fail."""
    # Mock FileContext
    file_context = MagicMock(spec=FileContext)

    # Mock the environment to fail on both branches but succeed on uncommitted
    mock_env = MagicMock(spec=BaseEnvironment)
    mock_env.execute.side_effect = [
        "fatal: ambiguous argument 'main'",  # First call fails
        "fatal: ambiguous argument 'master'",  # Second call fails
        "uncommitted changes",  # Third call succeeds
    ]

    # Create workspace with shadow_mode=False
    mock_workspace = MagicMock(spec=Workspace)
    mock_workspace.shadow_mode = False
    mock_workspace.environment = mock_env

    # Create action and set workspace
    action = ViewDiff()
    action.workspace = mock_workspace

    args = ViewDiffArgs(thoughts="Viewing uncommitted changes fallback")

    observation = await action.execute(args, file_context)

    # Verify the result
    assert isinstance(observation, Observation)
    assert "Current changes in workspace:\nuncommitted changes" == observation.message

    # Verify all git diff commands were called
    assert mock_env.execute.call_count == 3
    mock_env.execute.assert_any_call("git diff main", fail_on_error=False)
    mock_env.execute.assert_any_call("git diff master", fail_on_error=False)
    mock_env.execute.assert_any_call("git diff", fail_on_error=True)


@pytest.mark.asyncio
async def test_view_diff_error_handling():
    """Test ViewDiff action handles errors properly."""
    # Mock FileContext
    file_context = MagicMock(spec=FileContext)

    # Mock the environment to raise an exception
    mock_env = MagicMock(spec=BaseEnvironment)
    mock_env.execute.side_effect = Exception("Git command failed")

    # Create workspace with shadow_mode=False
    mock_workspace = MagicMock(spec=Workspace)
    mock_workspace.shadow_mode = False
    mock_workspace.environment = mock_env

    # Create action and set workspace
    action = ViewDiff()
    action.workspace = mock_workspace

    args = ViewDiffArgs(thoughts="Viewing diff with error")

    observation = await action.execute(args, file_context)

    # Verify the result shows error
    assert isinstance(observation, Observation)
    assert "Failed to get git diff" in observation.message
