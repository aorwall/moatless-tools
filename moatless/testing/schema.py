from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, model_validator

from moatless.utils.tokenizer import count_tokens


class TestStatus(str, Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"

    def __str__(self):
        return self.value


class TraceItem(BaseModel):
    file_path: str
    method: Optional[str] = None
    line_number: Optional[int] = None
    output: str = ""


class TestResult(BaseModel):
    status: TestStatus = Field(..., description="Status of the test")
    name: Optional[str] = None
    file_path: Optional[str] = None
    method: Optional[str] = None
    failure_output: Optional[str] = None
    stacktrace: List[TraceItem] = Field(default_factory=list, description="List of stack trace items")

    @model_validator(mode="before")
    def convert_status_to_enum(cls, values):
        if isinstance(values.get("status"), str):
            values["status"] = TestStatus(values["status"])
        return values


class TestFile(BaseModel):
    file_path: str = Field(..., description="The path to the test file.")
    test_results: list[TestResult] = Field(default_factory=list, description="List of test results.")

    @staticmethod
    def get_test_summary(test_files: list["TestFile"], file_paths: Optional[list[str]] = None) -> str:
        """
        Returns a summary of test results, optionally filtered by specified test files.
        If file_paths is provided, only shows results for those files.
        Lists each file with its own results followed by an overall summary.

        Args:
            test_files: List of TestFile objects containing test results
            file_paths: Optional list of file paths to include in the summary

        Returns:
            str: Summary string of test results
        """
        if not test_files:
            return "No test results available."

        # Filter test files if specified
        included_test_files = []
        if file_paths:
            # Only include test files for the specified file paths
            for test_file in test_files:
                if test_file.file_path in file_paths:
                    included_test_files.append(test_file)
        else:
            included_test_files = test_files

        if not included_test_files:
            return "No test results available."

        # Collect results and generate per-file summaries
        all_results = []
        per_file_summary = []

        for test_file in included_test_files:
            file_results = test_file.test_results
            all_results.extend(file_results)

            # Calculate stats for this file
            file_failure_count = sum(1 for r in file_results if r.status == TestStatus.FAILED)
            file_error_count = sum(1 for r in file_results if r.status == TestStatus.ERROR)
            file_passed_count = len(file_results) - file_failure_count - file_error_count

            # Add to per-file summary
            per_file_summary.append(
                f"* {test_file.file_path}: {file_passed_count} passed, {file_failure_count} failed, {file_error_count} errors"
            )

        # Calculate overall stats
        failure_count = sum(1 for r in all_results if r.status == TestStatus.FAILED)
        error_count = sum(1 for r in all_results if r.status == TestStatus.ERROR)
        passed_count = len(all_results) - failure_count - error_count

        # Combine per-file summary with overall summary
        summary = "\n".join(per_file_summary)
        summary += f"\n\nTotal: {passed_count} passed, {failure_count} failed, {error_count} errors."

        return summary

    @staticmethod
    def get_test_failure_details(
        test_files: list["TestFile"],
        max_tokens: int = 8000,
        max_chars_per_test: int = 2000,
        file_paths: Optional[list[str]] = None,
    ) -> str:
        """
        Returns detailed output for each failed or errored test result.
        For long messages, shows the first and last portions with middle truncated.
        If file_paths is provided, only shows details for those files.

        Args:
            test_files: List of TestFile objects containing test results
            max_tokens: Maximum total tokens for all test details
            max_chars_per_test: Maximum characters per test message before truncating
            file_paths: Optional list of file paths to include in the details

        Returns:
            str: Formatted string containing details of failed tests
        """
        if not test_files:
            return ""

        # Filter test files if specified
        included_test_files = []
        if file_paths:
            # Only include test files for the specified file paths
            for test_file in test_files:
                if test_file.file_path in file_paths:
                    included_test_files.append(test_file)
        else:
            included_test_files = test_files

        sum_tokens = 0
        test_result_strings = []
        for test_file in included_test_files:
            for result in test_file.test_results:
                if result.status in [TestStatus.FAILED, TestStatus.ERROR] and result.failure_output:
                    attributes = ""
                    if result.file_path:
                        attributes += f"{result.file_path}"

                    if len(result.failure_output) > max_chars_per_test:
                        # Show first and last portions of the message
                        chars_per_section = max_chars_per_test // 2
                        start_section = result.failure_output[:chars_per_section]
                        end_section = result.failure_output[-chars_per_section:]
                        truncated_message = f"{start_section}\n\n... {len(result.failure_output) - max_chars_per_test} characters truncated ...\n\n{end_section}"
                    else:
                        truncated_message = result.failure_output

                    test_result_str = f"* {result.status.value} {attributes}>\n```\n{truncated_message}\n```\n"
                    test_result_tokens = count_tokens(test_result_str)
                    if sum_tokens + test_result_tokens > max_tokens:
                        break

                    sum_tokens += test_result_tokens
                    test_result_strings.append(test_result_str)

        return "\n".join(test_result_strings) if test_result_strings else ""

    @staticmethod
    def get_test_counts(test_files: list["TestFile"]) -> tuple[int, int, int]:
        """
        Returns counts of passed, failed, and errored tests.

        Args:
            test_files: List of TestFile objects containing test results

        Returns:
            Tuple[int, int, int]: A tuple containing (passed_count, failure_count, error_count)
        """
        all_results = []
        for test_file in test_files:
            all_results.extend(test_file.test_results)

        failure_count = sum(1 for r in all_results if r.status == TestStatus.FAILED)
        error_count = sum(1 for r in all_results if r.status == TestStatus.ERROR)
        passed_count = len(all_results) - failure_count - error_count

        return (passed_count, failure_count, error_count)

    @staticmethod
    def get_test_status(test_files: list["TestFile"]) -> Optional[TestStatus]:
        """
        Returns the overall test status based on all test results.
        Returns ERROR if any test has error status,
        FAILED if any test has failed status,
        PASSED if all tests passed,
        or None if no test results exist.

        Args:
            test_files: List of TestFile objects containing test results

        Returns:
            Optional[TestStatus]: The overall test status
        """
        all_results = []
        for test_file in test_files:
            all_results.extend(test_file.test_results)

        if not all_results:
            return None

        if any(r.status == TestStatus.ERROR for r in all_results):
            return TestStatus.ERROR
        elif any(r.status == TestStatus.FAILED for r in all_results):
            return TestStatus.FAILED
        else:
            return TestStatus.PASSED
