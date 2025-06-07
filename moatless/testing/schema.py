from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, model_validator


class TestStatus(str, Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

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
    timed_out: bool = Field(default=False, description="Whether the test timed out during execution")

    @model_validator(mode="before")
    def convert_status_to_enum(cls, values):
        if isinstance(values.get("status"), str):
            values["status"] = TestStatus(values["status"])
        return values


class TestFile(BaseModel):
    file_path: str = Field(..., description="The path to the test file.")
    test_results: list[TestResult] = Field(default_factory=list, description="List of test results.")

    @staticmethod
    def get_test_summary(test_files: list["TestFile"]) -> str:
        """
        Returns a summary of test results, optionally filtered by specified test files.
        If file_paths is provided, only shows results for those files.
        Lists each file with its own results followed by an overall summary.

        Args:
            test_files: List of TestFile objects containing test results

        Returns:
            str: Summary string of test results
        """
        if not test_files:
            return "No test results available."

        # Collect results and generate per-file summaries
        all_results = []
        per_file_summary = []

        for test_file in test_files:
            file_results = test_file.test_results
            all_results.extend(file_results)

            # Calculate stats for this file
            file_failure_count = sum(1 for r in file_results if r.status == TestStatus.FAILED)
            file_error_count = sum(1 for r in file_results if r.status == TestStatus.ERROR)
            file_skipped_count = sum(1 for r in file_results if r.status == TestStatus.SKIPPED)
            file_timeout_count = sum(1 for r in file_results if r.timed_out)
            file_passed_count = len(file_results) - file_failure_count - file_error_count - file_skipped_count

            # Add to per-file summary
            summary_parts = [
                f"{file_passed_count} passed",
                f"{file_failure_count} failed",
                f"{file_error_count} errors",
                f"{file_skipped_count} skipped",
            ]
            if file_timeout_count > 0:
                summary_parts.append(f"{file_timeout_count} timed out")

            per_file_summary.append(f"* {test_file.file_path}: {', '.join(summary_parts)}")

        # Calculate overall stats
        failure_count = sum(1 for r in all_results if r.status == TestStatus.FAILED)
        error_count = sum(1 for r in all_results if r.status == TestStatus.ERROR)
        skipped_count = sum(1 for r in all_results if r.status == TestStatus.SKIPPED)
        timeout_count = sum(1 for r in all_results if r.timed_out)
        passed_count = len(all_results) - failure_count - error_count - skipped_count

        # Combine per-file summary with overall summary
        if failure_count + error_count + skipped_count + passed_count > 0:
            summary = "\n".join(per_file_summary)
            total_parts = [
                f"{passed_count} passed",
                f"{failure_count} failed",
                f"{error_count} errors",
                f"{skipped_count} skipped",
            ]
            if timeout_count > 0:
                total_parts.append(f"{timeout_count} timed out")
            summary += f"\n\nTotal: {', '.join(total_parts)}."
        else:
            summary = ""

        return summary

    @staticmethod
    def get_test_failure_details(
        test_files: list["TestFile"], max_tokens: int = 8000, max_chars_per_test: int = 2000
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

        sum_tokens = 0
        test_result_strings = []

        # Process each test file
        for test_file in test_files:
            # Process each test result in the file
            for result in test_file.test_results:
                if result.status in [TestStatus.FAILED, TestStatus.ERROR, TestStatus.UNKNOWN] and result.failure_output:
                    attributes = ""
                    if result.file_path:
                        attributes += f"{result.file_path}"

                    # Add timeout indicator if the test timed out
                    if result.timed_out:
                        attributes += " (TIMED OUT)"

                    # Handle long failure output
                    if len(result.failure_output) > max_chars_per_test:
                        chars_per_section = max_chars_per_test // 2
                        start_section = result.failure_output[:chars_per_section]
                        end_section = result.failure_output[-chars_per_section:]
                        truncated_message = f"{start_section}\n\n... {len(result.failure_output) - max_chars_per_test} characters truncated ...\n\n{end_section}"
                    else:
                        truncated_message = result.failure_output

                    # Format the test result string
                    from moatless.utils.tokenizer import count_tokens

                    test_result_str = f"* {attributes}\n```\n{truncated_message}\n```\n"
                    test_result_tokens = count_tokens(test_result_str)

                    # Always include at least one result
                    if not test_result_strings or sum_tokens + test_result_tokens <= max_tokens:
                        test_result_strings.append(test_result_str)
                        sum_tokens += test_result_tokens

                        # If we've reached the token limit and have at least one result, break
                        if sum_tokens >= max_tokens and test_result_strings:
                            break

            # Break out of the outer loop if we've reached the token limit
            if sum_tokens >= max_tokens and test_result_strings:
                break

        # Join all the test result strings
        return "\n".join(test_result_strings)

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
