import re
from typing import List, Optional

from moatless.testing.python.utils import create_generic_error_result
from moatless.testing.schema import TestResult, TestStatus
from moatless.testing.test_output_parser import TestOutputParser


class SympyParser(TestOutputParser):
    """Parser for test output from the Sympy testing framework."""

    def parse_test_output(self, log: str, file_path: Optional[str] = None) -> List[TestResult]:
        """
        Parse the Sympy test output and extract test results.

        Args:
            log: The test command output string
            file_path: Optional file path to filter results for

        Returns:
            List[TestResult]: List of test results
        """
        test_results = {}
        current_file = None

        for line in log.split("\n"):
            line = line.strip()

            # Check for file path
            if ".py[" in line:
                current_file = line.split("[")[0].strip()
                continue

            if line.startswith("test_"):
                split_line = line.split()
                if len(split_line) < 2:
                    continue

                test = split_line[0].strip()
                status = split_line[1]

                # Skip if file_path filter is provided and doesn't match
                if file_path and current_file and file_path not in current_file:
                    continue

                if status == "E":
                    test_results[test] = TestResult(
                        status=TestStatus.ERROR,
                        name=test,
                        method=test,
                        file_path=current_file,
                    )
                elif status == "F":
                    test_results[test] = TestResult(
                        status=TestStatus.FAILED,
                        name=test,
                        method=test,
                        file_path=current_file,
                    )
                elif status == "ok":
                    test_results[test] = TestResult(
                        status=TestStatus.PASSED,
                        name=test,
                        method=test,
                        file_path=current_file,
                    )
                elif status == "s":
                    test_results[test] = TestResult(
                        status=TestStatus.SKIPPED,
                        name=test,
                        method=test,
                        file_path=current_file,
                    )

        current_method = None
        current_file = None
        failure_output = []
        for line in log.split("\n"):
            pattern = re.compile(r"(_*) (.*)\.py:(.*) (_*)")
            match = pattern.match(line)
            if match:
                if current_method and current_method in test_results:
                    test_results[current_method].failure_output = "\n".join(failure_output)
                    test_results[current_method].file_path = current_file

                current_file = f"{match.group(2)}.py"
                current_method = match.group(3)
                failure_output = []
                continue

            if "tests finished" in line:
                if current_method and current_method in test_results:
                    test_results[current_method].failure_output = "\n".join(failure_output)
                    test_results[current_method].file_path = current_file
                break

            failure_output.append(line)

        if current_method and current_method in test_results:
            test_results[current_method].failure_output = "\n".join(failure_output)
            test_results[current_method].file_path = current_file

        # If no test results were found but the log contains error indicators, create a generic error
        if not test_results and ("Traceback" in log or re.search(r"[A-Z][a-zA-Z]*Error", log) or "Exception:" in log):
            error_type = "Unknown Error"
            if "ImportError" in log:
                error_type = "Import Error"
            elif "TypeError" in log:
                error_type = "Type Error"
            elif "SyntaxError" in log:
                error_type = "Syntax Error"

            # Extract file path from traceback if possible
            file_match = re.search(r'File "([^"]+)", line (\d+)', log)
            file_path_from_trace = None
            if file_match:
                file_path_from_trace = file_match.group(1)
                if file_path_from_trace.startswith("/testbed/"):
                    file_path_from_trace = file_path_from_trace[len("/testbed/") :]

            # Fall back to generic error
            results_list = list(test_results.values())
            results_list.append(create_generic_error_result(log, error_type))
            return results_list

        return list(test_results.values())
