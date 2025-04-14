from unittest.mock import patch

import pytest
from moatless.testing.schema import TestFile, TestResult, TestStatus


def test_test_file_init():
    test_file = TestFile(file_path="tests/test_example.py")
    assert test_file.file_path == "tests/test_example.py"
    assert test_file.test_results == []


def test_get_test_summary_empty():
    summary = TestFile.get_test_summary([])
    assert summary == "No test results available."


def test_get_test_summary_with_results():
    # Create test files with results
    test_file1 = TestFile(file_path="tests/test_example1.py")
    test_file1.test_results = [
        TestResult(test_name="test1", status=TestStatus.PASSED),
        TestResult(test_name="test2", status=TestStatus.FAILED),
        TestResult(test_name="test3", status=TestStatus.ERROR),
    ]

    test_file2 = TestFile(file_path="tests/test_example2.py")
    test_file2.test_results = [
        TestResult(test_name="test4", status=TestStatus.PASSED),
        TestResult(test_name="test5", status=TestStatus.PASSED),
    ]

    summary = TestFile.get_test_summary([test_file1, test_file2])

    assert "tests/test_example1.py: 1 passed, 1 failed, 1 errors, 0 skipped" in summary
    assert "tests/test_example2.py: 2 passed, 0 failed, 0 errors, 0 skipped" in summary
    assert "Total: 3 passed, 1 failed, 1 errors, 0 skipped." in summary


def test_get_test_summary_passed():
    # Create test files with results
    test_file1 = TestFile(file_path="tests/test_example1.py")
    test_file1.test_results = [
        TestResult(test_name="test1", status=TestStatus.PASSED),
        TestResult(test_name="test2", status=TestStatus.PASSED),
        TestResult(test_name="test3", status=TestStatus.PASSED),
    ]

    test_file2 = TestFile(file_path="tests/test_example2.py")
    test_file2.test_results = [
        TestResult(test_name="test4", status=TestStatus.PASSED),
    ]

    summary = TestFile.get_test_summary([test_file1, test_file2])

    assert "tests/test_example1.py: 3 passed, 0 failed, 0 errors, 0 skipped" in summary
    assert "tests/test_example2.py: 1 passed, 0 failed, 0 errors, 0 skipped" in summary
    assert "Total: 4 passed, 0 failed, 0 errors, 0 skipped." in summary


def test_get_test_failure_details_empty():
    details = TestFile.get_test_failure_details([])
    assert details == ""


def test_get_test_failure_details_with_failures():
    # Create test file with failures
    test_file = TestFile(file_path="tests/test_example.py")
    test_file.test_results = [
        TestResult(
            test_name="test_fails",
            status=TestStatus.FAILED,
            failure_output="Expected 1 but got 2",
            file_path="tests/test_example.py",
        ),
        TestResult(
            test_name="test_errors",
            status=TestStatus.ERROR,
            failure_output="Division by zero",
            file_path="tests/test_example.py",
        ),
        TestResult(test_name="test_passes", status=TestStatus.PASSED),
    ]

    details = TestFile.get_test_failure_details([test_file])

    assert "tests/test_example.py" in details
    assert "Expected 1 but got 2" in details
    assert "Division by zero" in details
    assert "test_passes" not in details  # Passed tests should not be included


def test_get_test_counts():
    # Create test files with mixed results
    test_file1 = TestFile(file_path="tests/test_example1.py")
    test_file1.test_results = [
        TestResult(test_name="test1", status=TestStatus.PASSED),
        TestResult(test_name="test2", status=TestStatus.FAILED),
    ]

    test_file2 = TestFile(file_path="tests/test_example2.py")
    test_file2.test_results = [
        TestResult(test_name="test3", status=TestStatus.PASSED),
        TestResult(test_name="test4", status=TestStatus.ERROR),
    ]

    passed, failed, error = TestFile.get_test_counts([test_file1, test_file2])

    assert passed == 2
    assert failed == 1
    assert error == 1


def test_get_test_status_empty():
    status = TestFile.get_test_status([])
    assert status is None


def test_get_test_status_passed():
    test_file = TestFile(file_path="tests/test_example.py")
    test_file.test_results = [
        TestResult(test_name="test1", status=TestStatus.PASSED),
        TestResult(test_name="test2", status=TestStatus.PASSED),
    ]

    status = TestFile.get_test_status([test_file])
    assert status == TestStatus.PASSED


def test_get_test_status_failed():
    test_file = TestFile(file_path="tests/test_example.py")
    test_file.test_results = [
        TestResult(test_name="test1", status=TestStatus.PASSED),
        TestResult(test_name="test2", status=TestStatus.FAILED),
    ]

    status = TestFile.get_test_status([test_file])
    assert status == TestStatus.FAILED


def test_get_test_status_error():
    test_file = TestFile(file_path="tests/test_example.py")
    test_file.test_results = [
        TestResult(test_name="test1", status=TestStatus.PASSED),
        TestResult(test_name="test2", status=TestStatus.FAILED),
        TestResult(test_name="test3", status=TestStatus.ERROR),
    ]

    status = TestFile.get_test_status([test_file])
    assert status == TestStatus.ERROR


def test_get_test_failure_details_truncation():
    # Create a test file with a very long error message
    test_file = TestFile(file_path="tests/test_example.py")
    long_message = "x" * 3000  # 3000 character message that will be truncated

    test_file.test_results = [
        TestResult(
            test_name="test_long_error",
            status=TestStatus.ERROR,
            failure_output=long_message,
            file_path="tests/test_example.py",
        )
    ]

    details = TestFile.get_test_failure_details([test_file], max_chars_per_test=1000)

    # Check that truncation message is included
    assert "characters truncated" in details
    # The message should be shorter than the original
    assert len(details) < len(long_message) + 200  # Adding buffer for formatting


def test_get_test_failure_details_token_limit():
    # Create test file with results
    test_file = TestFile(file_path="tests/test_example.py")
    test_file.test_results = [
        TestResult(
            test_name="test1",
            status=TestStatus.FAILED,
            failure_output="First error message",
            file_path="tests/test_example.py",
        ),
        TestResult(
            test_name="test2",
            status=TestStatus.ERROR,
            failure_output="Second error message",
            file_path="tests/test_example.py",
        ),
    ]

    # Test with very small token limit
    with patch("moatless.utils.tokenizer.count_tokens", return_value=1000):
        details = TestFile.get_test_failure_details(
            [test_file],
            max_tokens=500,  # Less than one message worth of tokens
        )

        # With token limiting, only the first message should be included
        # since it would already exceed the token limit
        assert "First error message" in details
        assert "Second error message" not in details
