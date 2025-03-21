import re
from typing import List, Optional

from moatless.testing.schema import TestResult, TestStatus
from moatless.testing.test_output_parser import TestOutputParser
from moatless.testing.python.utils import create_generic_error_result


class SeabornParser(TestOutputParser):
    """Parser for test output from the Seaborn testing framework."""

    def parse_test_output(self, log: str, file_path: Optional[str] = None) -> List[TestResult]:
        """
        Parse the Seaborn test output and extract test results.

        Args:
            log: The test command output string
            file_path: Optional file path to filter results for

        Returns:
            List[TestResult]: List of test results
        """
        test_results = []
        for line in log.split("\n"):
            if line.startswith(TestStatus.FAILED.value):
                test_case = line.split()[1]
                test_results.append(TestResult(status=TestStatus.FAILED, name=test_case))
            elif f" {TestStatus.PASSED.value} " in line:
                parts = line.split()
                if parts[1] == TestStatus.PASSED.value:
                    test_case = parts[0]
                    test_results.append(TestResult(status=TestStatus.PASSED, name=test_case))
            elif line.startswith(TestStatus.PASSED.value):
                parts = line.split()
                test_case = parts[1]
                test_results.append(TestResult(status=TestStatus.PASSED, name=test_case))

        # If no test results were found but the log contains error indicators, create a generic error
        if not test_results and ("Traceback" in log or re.search(r"[A-Z][a-zA-Z]*Error", log) or "Exception:" in log):
            error_type = "Unknown Error"
            if "ImportError" in log:
                error_type = "Import Error"
            elif "ValueError" in log:
                error_type = "Value Error"
            elif "AttributeError" in log:
                error_type = "Attribute Error"

            test_results.append(create_generic_error_result(log, error_type))

        # If file_path is provided, filter results
        if file_path:
            filtered_results = []
            for result in test_results:
                if result.file_path and file_path in result.file_path:
                    filtered_results.append(result)
            return filtered_results

        return test_results
