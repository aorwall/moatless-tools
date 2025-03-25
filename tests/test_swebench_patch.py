import os
import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch as mock_patch

from moatless.runtime.local import SweBenchTestbedEnvironment
from moatless.storage.base import BaseStorage
from swebench.harness.constants import APPLY_PATCH_PASS, APPLY_PATCH_FAIL


@pytest.fixture
def mock_storage():
    storage = MagicMock(spec=BaseStorage)
    return storage


@pytest.fixture
def swebench_instance():
    return {
        "instance_id": "test-instance",
        "repo": "test/repo",
        "version": "1.0",
        "base_commit": "abc123",
        "test_patch": "dummy_patch"  # Added to avoid KeyError
    }


@pytest.fixture
def mock_test_spec():
    test_spec = MagicMock()
    test_spec.repo = "test/repo"
    return test_spec


@pytest.fixture
def temp_git_repo():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize a git repo in the temp directory
        os.system(f"cd {temp_dir} && git init")
        os.system(f"cd {temp_dir} && git config user.email 'test@example.com'")
        os.system(f"cd {temp_dir} && git config user.name 'Test User'")
        
        # Create a test file and commit it
        test_file = os.path.join(temp_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")
        
        os.system(f"cd {temp_dir} && git add test_file.txt")
        os.system(f"cd {temp_dir} && git commit -m 'Initial commit'")
        
        yield Path(temp_dir)


@pytest.mark.asyncio
async def test_apply_patch_successful(temp_git_repo, swebench_instance, mock_storage, mock_test_spec):
    # Create the environment with our test repo
    swebench_instance["base_commit"] = "HEAD"  # Use the most recent commit
    
    # Mock make_test_spec to avoid initialization errors
    with mock_patch('moatless.runtime.local.make_test_spec', return_value=mock_test_spec):
        env = SweBenchTestbedEnvironment(temp_git_repo, swebench_instance, mock_storage)
        
        # Create a simple patch that modifies the test file
        patch = """diff --git a/test_file.txt b/test_file.txt
--- a/test_file.txt
+++ b/test_file.txt
@@ -1,3 +1,3 @@
-Line 1
+Line 1 modified
 Line 2
 Line 3
"""
        
        # We'll directly test the _apply_patch method
        # Mock _reset_repository to avoid changing the actual repo
        env._reset_repository = AsyncMock(return_value=True)
        
        # Mock the _execute_command to return real git diff output
        original_execute_command = env._execute_command
        
        async def mock_execute_command(command, cwd=None):
            if "git diff" in command:
                return "diff --git a/test_file.txt b/test_file.txt\n--- a/test_file.txt\n+++ b/test_file.txt\n@@ -1,3 +1,3 @@\n-Line 1\n+Line 1 modified\nLine 2\nLine 3", 0
            elif "git ls-files --others" in command:
                return "", 0  # No untracked files
            return await original_execute_command(command, cwd)
        
        with mock_patch.object(env, '_execute_command', side_effect=mock_execute_command):
            # Apply the patch
            result = await env._apply_patch(patch)
            
            # Verify the result
            assert result is True


@pytest.mark.asyncio
async def test_apply_patch_with_warnings(temp_git_repo, swebench_instance, mock_storage, mock_test_spec):
    # Create the environment with our test repo
    swebench_instance["base_commit"] = "HEAD"  # Use the most recent commit
    
    # Mock make_test_spec to avoid initialization errors
    with mock_patch('moatless.runtime.local.make_test_spec', return_value=mock_test_spec):
        env = SweBenchTestbedEnvironment(temp_git_repo, swebench_instance, mock_storage)
        
        # Create a patch with trailing whitespace warnings
        patch = """diff --git a/test_file.txt b/test_file.txt
--- a/test_file.txt
+++ b/test_file.txt
@@ -1,3 +1,3 @@
-Line 1
+Line 1 modified    
 Line 2
 Line 3
"""
        
        # We'll directly test the _apply_patch method
        # Mock _reset_repository to avoid changing the actual repo
        env._reset_repository = AsyncMock(return_value=True)
        
        # Mock the _execute_command to simulate the git apply with warnings but successful diff
        async def mock_execute_command(command, cwd=None):
            if "git apply -v" in command:
                return "/testbed/temp_patch.diff:2: trailing whitespace.\n    \nChecking patch test_file.txt...", 0
            elif "git diff" in command:
                return "diff --git a/test_file.txt b/test_file.txt\n--- a/test_file.txt\n+++ b/test_file.txt\n@@ -1,3 +1,3 @@\n-Line 1\n+Line 1 modified    \nLine 2\nLine 3", 0
            elif "git ls-files --others" in command:
                return "", 0  # No untracked files
            return "", 0
        
        with mock_patch.object(env, '_execute_command', side_effect=mock_execute_command):
            # Apply the patch
            result = await env._apply_patch(patch)
            
            # Verify the result - should succeed despite warnings
            assert result is True


@pytest.mark.asyncio
async def test_apply_patch_file_not_found(temp_git_repo, swebench_instance, mock_storage, mock_test_spec):
    # Create the environment with our test repo
    swebench_instance["base_commit"] = "HEAD"  # Use the most recent commit
    
    # Mock make_test_spec to avoid initialization errors
    with mock_patch('moatless.runtime.local.make_test_spec', return_value=mock_test_spec):
        env = SweBenchTestbedEnvironment(temp_git_repo, swebench_instance, mock_storage)
        
        # Create a patch referencing a non-existent file
        patch = """diff --git a/non_existent.txt b/non_existent.txt
--- a/non_existent.txt
+++ b/non_existent.txt
@@ -1,3 +1,3 @@
-Line 1
+Line 1 modified
 Line 2
 Line 3
"""
        
        # We'll directly test the _apply_patch method
        # Mock _reset_repository to avoid changing the actual repo
        env._reset_repository = AsyncMock(return_value=True)
        
        # Mock the _execute_command to simulate the error
        async def mock_execute_command(command, cwd=None):
            if "git apply -v" in command:
                return "error: non_existent.txt: No such file or directory", 1
            elif "patch --batch" in command:
                return "can't find file to patch at input line 3", 1
            elif "git diff" in command:
                return "", 0  # No changes
            elif "git ls-files --others" in command:
                return "", 0  # No untracked files
            return "", 0
        
        with mock_patch.object(env, '_execute_command', side_effect=mock_execute_command):
            # Apply the patch
            result = await env._apply_patch(patch)
            
            # Verify the result - should fail
            assert result is False


@pytest.mark.asyncio
async def test_apply_patch_fallback_to_patch_command(temp_git_repo, swebench_instance, mock_storage, mock_test_spec):
    # Create the environment with our test repo
    swebench_instance["base_commit"] = "HEAD"  # Use the most recent commit
    
    # Mock make_test_spec to avoid initialization errors
    with mock_patch('moatless.runtime.local.make_test_spec', return_value=mock_test_spec):
        env = SweBenchTestbedEnvironment(temp_git_repo, swebench_instance, mock_storage)
        
        # Create a patch that git apply cannot handle but patch command can
        patch = """diff --git a/test_file.txt b/test_file.txt
--- a/test_file.txt
+++ b/test_file.txt
@@ -1,3 +1,3 @@
-Line 1
+Line 1 modified
 Line 2
 Line 3
"""
        
        # We'll directly test the _apply_patch method
        # Mock _reset_repository to avoid changing the actual repo
        env._reset_repository = AsyncMock(return_value=True)
        
        # Mock the _execute_command to simulate git apply failing but patch succeeding
        async def mock_execute_command(command, cwd=None):
            if "git apply -v" in command:
                return "error: patch failed: test_file.txt:1", 1
            elif "patch --batch" in command:
                return "patching file test_file.txt", 0
            elif "git diff" in command and not hasattr(mock_execute_command, "called_patch"):
                mock_execute_command.called_patch = True
                return "", 0  # No changes after git apply
            elif "git diff" in command:
                return "diff --git a/test_file.txt b/test_file.txt\n--- a/test_file.txt\n+++ b/test_file.txt\n@@ -1,3 +1,3 @@\n-Line 1\n+Line 1 modified\nLine 2\nLine 3", 0
            elif "git ls-files --others" in command:
                return "", 0  # No untracked files
            return "", 0
        
        with mock_patch.object(env, '_execute_command', side_effect=mock_execute_command):
            # Apply the patch
            result = await env._apply_patch(patch)
            
            # Verify the result - should succeed using patch command
            assert result is True


@pytest.mark.asyncio
async def test_apply_patch_adding_new_file(temp_git_repo, swebench_instance, mock_storage, mock_test_spec):
    # Create the environment with our test repo
    swebench_instance["base_commit"] = "HEAD"  # Use the most recent commit
    
    # Mock make_test_spec to avoid initialization errors
    with mock_patch('moatless.runtime.local.make_test_spec', return_value=mock_test_spec):
        env = SweBenchTestbedEnvironment(temp_git_repo, swebench_instance, mock_storage)
        
        # Create a patch that adds a new file
        patch = """diff --git a/new_file.txt b/new_file.txt
new file mode 100644
index 0000000..3be9c81
--- /dev/null
+++ b/new_file.txt
@@ -0,0 +1,3 @@
+Line 1
+Line 2
+Line 3
"""
        
        # We'll directly test the _apply_patch method
        # Mock _reset_repository to avoid changing the actual repo
        env._reset_repository = AsyncMock(return_value=True)
        
        # Mock the _execute_command to simulate adding a new file
        async def mock_execute_command(command, cwd=None):
            if "git apply -v" in command:
                return "Checking patch new_file.txt...", 0
            elif "git diff" in command:
                return "", 0  # No tracked changes
            elif "git ls-files --others" in command:
                return "new_file.txt", 0  # New untracked file
            return "", 0
        
        with mock_patch.object(env, '_execute_command', side_effect=mock_execute_command):
            # Apply the patch
            result = await env._apply_patch(patch)
            
            # Verify the result - should succeed due to new untracked file
            assert result is True 


@pytest.mark.asyncio
async def test_apply_patch_real_repo_operations(temp_git_repo, swebench_instance, mock_storage, mock_test_spec):
    """
    Test that actually performs git operations in a real repo without any mocking.
    This ensures we test the full flow including resetting the repo and verifying files exist.
    """
    # Create the environment with our test repo
    swebench_instance["base_commit"] = "HEAD"  # Use the most recent commit
    
    # We'll only mock make_test_spec, but use real git operations
    with mock_patch('moatless.runtime.local.make_test_spec', return_value=mock_test_spec):
        env = SweBenchTestbedEnvironment(temp_git_repo, swebench_instance, mock_storage)
        
        # Override skip_conda_activate to avoid conda errors in tests
        env._skip_conda_activate = True
        
        # 1. Create a patch that modifies an existing file AND adds a new file
        patch = """diff --git a/test_file.txt b/test_file.txt
--- a/test_file.txt
+++ b/test_file.txt
@@ -1,3 +1,3 @@
-Line 1
+Line 1 modified by real test
 Line 2
 Line 3
diff --git a/new_test_file.txt b/new_test_file.txt
new file mode 100644
index 0000000..3be9c81
--- /dev/null
+++ b/new_test_file.txt
@@ -0,0 +1,3 @@
+New file line 1
+New file line 2
+New file line 3
"""
        
        # Apply the patch - this will use real git operations
        result = await env._apply_patch(patch)
        
        # Verify the result
        assert result is True
        
        # Additional verification - check the actual file contents on disk
        with open(temp_git_repo / "test_file.txt", "r") as f:
            modified_content = f.read()
            assert "Line 1 modified by real test" in modified_content
            
        # Verify the new file was created
        new_file_path = temp_git_repo / "new_test_file.txt"
        assert new_file_path.exists()
        with open(new_file_path, "r") as f:
            new_file_content = f.read()
            assert "New file line 1" in new_file_content
            
        # Run a second test on the same repo to check the reset functionality
        second_patch = """diff --git a/test_file.txt b/test_file.txt
--- a/test_file.txt
+++ b/test_file.txt
@@ -1,3 +1,3 @@
-Line 1
+Second modification
 Line 2
 Line 3
diff --git a/another_new_file.txt b/another_new_file.txt
new file mode 100644
index 0000000..9f8906f
--- /dev/null
+++ b/another_new_file.txt
@@ -0,0 +1,1 @@
+Another new file
"""
        
        # Apply the second patch
        second_result = await env._apply_patch(second_patch)
        assert second_result is True
        
        # Verify that the first modification was lost (due to reset)
        with open(temp_git_repo / "test_file.txt", "r") as f:
            modified_content = f.read()
            assert "Second modification" in modified_content
            assert "Line 1 modified by real test" not in modified_content
            
        # Verify the first new file is gone and the second one exists
        assert not (temp_git_repo / "new_test_file.txt").exists()
        assert (temp_git_repo / "another_new_file.txt").exists() 