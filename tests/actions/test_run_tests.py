import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.file_context import FileContext
from moatless.testing.schema import TestFile, TestResult, TestStatus
from moatless.workspace import Workspace


class TestRunTestsAction:
    @pytest.fixture
    def file_context(self):
        mock_file_context = MagicMock(spec=FileContext)
        mock_file_context.file_exists = lambda path: True
        return mock_file_context

    @pytest.fixture
    def workspace(self):
        mock_workspace = MagicMock(spec=Workspace)
        mock_workspace.runtime = AsyncMock()
        return mock_workspace

    @pytest.fixture
    def run_tests_action(self, workspace):
        action = RunTests()
        asyncio.run(action.initialize(workspace))
        return action

    @pytest.mark.asyncio
    async def test_run_tests_filters_results_by_file(self, workspace):
        """Test that run_tests method correctly filters test results by file path."""
        # Setup mock test results
        test_results = [
            TestResult(
                status=TestStatus.PASSED,
                name="test_one",
                file_path="tests/file1_test.py",
                method="test_one"
            ),
            TestResult(
                status=TestStatus.FAILED,
                name="test_two",
                file_path="tests/file1_test.py",
                method="test_two",
                failure_output="Assertion failed"
            ),
            TestResult(
                status=TestStatus.PASSED,
                name="test_three",
                file_path="tests/file2_test.py",
                method="test_three"
            ),
        ]
        
        # Mock the runtime's run_tests method
        workspace.runtime.run_tests.return_value = test_results
        
        # Create the action and initialize it
        action = RunTests()
        await action.initialize(workspace)
        
        # Run tests directly using create_test_files method
        test_files = ["tests/file1_test.py", "tests/file2_test.py"]
        test_file_objects = action.create_test_files(test_file_paths=test_files, test_results=test_results)
        
        # Verify test results are assigned correctly
        assert len(test_file_objects) == 2
        
        # Find each file in the results
        file1_obj = None
        file2_obj = None
        for tf in test_file_objects:
            if tf.file_path == "tests/file1_test.py":
                file1_obj = tf
            elif tf.file_path == "tests/file2_test.py":
                file2_obj = tf
        
        assert file1_obj is not None
        assert file2_obj is not None
        
        # File 1 should have 2 test results
        assert len(file1_obj.test_results) == 2
        assert all(r.file_path == "tests/file1_test.py" for r in file1_obj.test_results)
        
        # File 2 should have 1 test result
        assert len(file2_obj.test_results) == 1
        assert all(r.file_path == "tests/file2_test.py" for r in file2_obj.test_results)
    
    @pytest.mark.asyncio
    async def test_execute_processes_results_correctly(self, file_context, workspace):
        """Test that the execute method correctly uses TestFile methods for generating output."""
        # Create test results data
        test_results = [
            TestResult(
                status=TestStatus.PASSED,
                name="test_one",
                file_path="tests/file1_test.py",
                method="test_one"
            ),
            TestResult(
                status=TestStatus.FAILED,
                name="test_two",
                file_path="tests/file1_test.py", 
                method="test_two",
                failure_output="Assertion failed"
            ),
        ]
        
        # Create test file objects
        test_file_objects = [
            TestFile(file_path="tests/file1_test.py", test_results=test_results)
        ]
        
        # Setup file_context and repository mocks
        file_context.file_exists.return_value = True
        workspace.runtime.run_tests.return_value = test_results
        
        # Mock repository checks
        mock_repository = MagicMock()
        mock_repository.file_exists.return_value = True
        mock_repository.is_directory.return_value = False
        
        # Create and initialize the action
        action = RunTests()
        await action.initialize(workspace)
        
        # Patch the repository property and execute the action
        with patch.object(RunTests, '_repository', new_callable=PropertyMock, return_value=mock_repository):
            args = RunTestsArgs(test_files=["tests/file1_test.py"])
            result = await action.execute(args, file_context)
            
            # Verify that the output contains test failure details and summary
            assert "Assertion failed" in result.message
            assert "tests/file1_test.py: 1 passed, 1 failed" in result.message
            assert "Total: 1 passed, 1 failed" in result.message
        
        # Verify artifact changes were created
        assert len(result.artifact_changes) == 1
        artifact = result.artifact_changes[0]
        assert artifact.artifact_id == "tests/file1_test.py"
        assert artifact.properties["passed"] == 1
        assert artifact.properties["failed"] == 1
        assert artifact.properties["total"] == 2 

class TestFileFilteringForTestResults:
    """Test that test results are correctly filtered by file path"""
    
    def test_file_result_filtering(self):
        """
        Directly test the file filtering logic without running the full action
        """
        # Create test data
        test_results = [
            TestResult(
                status=TestStatus.PASSED,
                name="test_one",
                file_path="tests/file1_test.py",
                method="test_one"
            ),
            TestResult(
                status=TestStatus.FAILED,
                name="test_two",
                file_path="tests/file1_test.py",
                method="test_two",
                failure_output="Assertion failed"
            ),
            TestResult(
                status=TestStatus.PASSED,
                name="test_three",
                file_path="tests/file2_test.py",
                method="test_three"
            ),
        ]
        
        # Create test file objects
        test_file1 = TestFile(file_path="tests/file1_test.py")
        test_file2 = TestFile(file_path="tests/file2_test.py")
        
        # Apply the filtering logic from the fix
        test_file1.test_results = [
            result for result in test_results 
            if result.file_path and result.file_path == test_file1.file_path
        ]
        
        test_file2.test_results = [
            result for result in test_results 
            if result.file_path and result.file_path == test_file2.file_path
        ]
        
        # Verify results
        assert len(test_file1.test_results) == 2
        assert all(r.file_path == "tests/file1_test.py" for r in test_file1.test_results)
        
        assert len(test_file2.test_results) == 1
        assert all(r.file_path == "tests/file2_test.py" for r in test_file2.test_results)
        
        # Check that test status and summary methods work as expected
        assert TestFile.get_test_counts([test_file1, test_file2]) == (2, 1, 0)  # 2 passed, 1 failed, 0 errors
        
        # Check summary output
        summary = TestFile.get_test_summary([test_file1, test_file2])
        assert "tests/file1_test.py: 1 passed, 1 failed" in summary
        assert "tests/file2_test.py: 1 passed, 0 failed" in summary
        assert "Total: 2 passed, 1 failed" in summary
        
        # Check failure details
        failure_details = TestFile.get_test_failure_details([test_file1, test_file2])
        assert "Assertion failed" in failure_details 