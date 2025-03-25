import re
from typing import List, Optional

from moatless.testing.python.utils import clean_log, parse_traceback_line, parse_traceback, create_generic_error_result
from moatless.testing.schema import TestResult, TestStatus, TraceItem
from moatless.testing.test_output_parser import TestOutputParser


class DjangoParser(TestOutputParser):
    """Parser for test output from the Django testing framework."""

    def parse_test_output(self, log: str, file_path: Optional[str] = None) -> List[TestResult]:
        """
        Parse the Django test output and extract test results.

        Args:
            log: The test command output string
            file_path: Optional file path to filter results for

        Returns:
            List[TestResult]: List of test results
        """
        test_status_map = {}

        current_test = None
        current_method = None
        current_file_path = None
        current_traceback_item: Optional[TraceItem] = None
        current_output = []

        test_pattern = re.compile(r"^(\w+) \(([\w.]+)\)")

        log = clean_log(log)
        lines = log.split("\n")

        for line in lines:
            line = line.strip()

            match = test_pattern.match(line)
            if match:
                current_test = match.group(0)
                method_name = match.group(1)
                full_path = match.group(2).split(".")

                # Extract file path and class name
                file_path_parts = [part for part in full_path[:-1] if part[0].islower()]
                class_name = full_path[-1] if full_path[-1][0].isupper() else None

                current_file_path = "tests/" + "/".join(file_path_parts) + ".py"
                current_method = f"{class_name}.{method_name}" if class_name else method_name

            if current_test:
                if "..." in line:
                    swebench_name = line.split("...")[0].strip()
                else:
                    swebench_name = None

                if "... ok" in line or line == "ok":
                    if swebench_name:
                        current_test = swebench_name
                    test_status_map[current_method] = TestResult(
                        status=TestStatus.PASSED,
                        file_path=current_file_path,
                        name=current_test,
                        method=current_method,
                    )
                    current_test = None
                    current_method = None
                    current_file_path = None
                elif "FAIL" in line or "\nFAIL" in line:
                    if swebench_name:
                        current_test = swebench_name
                    test_status_map[current_method] = TestResult(
                        status=TestStatus.FAILED,
                        file_path=current_file_path,
                        name=current_test,
                        method=current_method,
                    )
                    current_test = None
                    current_method = None
                    current_file_path = None
                elif "ERROR" in line or "\nERROR" in line:
                    if swebench_name:
                        current_test = swebench_name
                    test_status_map[current_method] = TestResult(
                        status=TestStatus.ERROR,
                        file_path=current_file_path,
                        name=current_test,
                        method=current_method,
                    )
                    current_test = None
                    current_method = None
                    current_file_path = None
                elif " ... skipped" in line or "\nskipped" in line:
                    if swebench_name:
                        current_test = swebench_name
                    test_status_map[current_method] = TestResult(
                        status=TestStatus.SKIPPED,
                        file_path=current_file_path,
                        name=current_test,
                        method=current_method,
                    )
                    current_test = None
                    current_method = None
                    current_file_path = None
                continue

        for line in lines:
            if line.startswith("===================="):
                if current_method and current_output and current_method in test_status_map:
                    test_status_map[current_method].failure_output = "\n".join(current_output)
                current_method = None
                current_output = []
                current_traceback_item = None
            elif line.startswith("--------------------------") and current_traceback_item:
                if current_method and current_output and current_method in test_status_map:
                    test_status_map[current_method].failure_output = "\n".join(current_output)

                current_method = None
                current_output = []
                current_traceback_item = None
            elif line.startswith("ERROR: ") or line.startswith("FAIL: "):
                current_test = line.split(": ", 1)[1].strip()
                match = test_pattern.match(current_test)

                if match:
                    method_name = match.group(1)
                    full_path = match.group(2).split(".")
                    class_name = full_path[-1] if full_path[-1][0].isupper() else None
                    current_method = f"{class_name}.{method_name}" if class_name else method_name
                else:
                    current_method = current_test

            elif len(test_status_map) == 0 and "Traceback (most recent call last)" in line:
                # If traceback is logged but not tests we expect syntax error
                current_method = "traceback"
                test_status_map[current_method] = TestResult(
                    status=TestStatus.ERROR, name=current_method, method=current_method
                )

            elif current_method and not line.startswith("--------------------------"):
                current_output.append(line)
                file_path_result, line_number, method_name = parse_traceback_line(line)
                if file_path_result:
                    if current_traceback_item and current_method in test_status_map:
                        test_status_map[current_method].stacktrace.append(current_traceback_item)

                    current_traceback_item = TraceItem(
                        file_path=file_path_result, line_number=line_number, method=method_name
                    )
                elif current_traceback_item:
                    if current_traceback_item.output:
                        current_traceback_item.output += "\n"
                    current_traceback_item.output += line

        # Handle the last test case
        if current_method and current_output and current_method in test_status_map:
            test_status_map[current_method].failure_output = "\n".join(current_output)
            if current_traceback_item:
                test_status_map[current_method].stacktrace.append(current_traceback_item)

        # If no test results were found but the log contains error indicators, create generic error
        if not test_status_map and (
            "Traceback" in log or re.search(r"[A-Z][a-zA-Z]*Error", log) or "Exception:" in log
        ):
            error_type = "Unknown Error"
            if "ImportError" in log:
                error_type = "Import Error"
            elif "SyntaxError" in log:
                error_type = "Syntax Error"
            elif "AttributeError" in log:
                error_type = "Attribute Error"

            # Try to use parse_traceback for a more detailed result first
            traceback_result = parse_traceback(log)
            if traceback_result:
                test_status_map["traceback"] = traceback_result
            else:
                # Fall back to generic error
                test_status_map["error"] = create_generic_error_result(log, error_type)

        # Filter by file path if provided
        if file_path:
            filtered_results = {}
            for key, result in test_status_map.items():
                if result.file_path and file_path in result.file_path:
                    filtered_results[key] = result
            return list(filtered_results.values())

        return list(test_status_map.values())

    def parse_traceback(self, log: str) -> Optional[TestResult]:
        """
        Parse a traceback log into a TestResult object.

        Args:
            log: The traceback log to parse

        Returns:
            TestResult object or None if no traceback found
        """
        return parse_traceback(log)
