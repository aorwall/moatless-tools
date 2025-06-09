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

        # First pass: Parse test status lines
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

        # Second pass: Parse failure details and override statuses
        current_method = None
        subtest_counters = {}  # Track duplicate subtests
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
                
                # Check for subtests with conditions (e.g., "(condition=...)")
                subtest_suffix = ""
                if " (condition=" in current_test:
                    # Extract the base test name and the condition
                    base_test, condition = current_test.split(" (condition=", 1)
                    subtest_suffix = f" (condition={condition}"
                    match = test_pattern.match(base_test)
                else:
                    match = test_pattern.match(current_test)

                if match:
                    method_name = match.group(1)
                    full_path = match.group(2).split(".")
                    class_name = full_path[-1] if full_path[-1][0].isupper() else None
                    base_method = f"{class_name}.{method_name}" if class_name else method_name
                    
                    # For subtests, create a unique method identifier
                    if subtest_suffix:
                        # Check if we've seen this exact subtest before
                        subtest_key = base_method + subtest_suffix
                        if subtest_key in subtest_counters:
                            subtest_counters[subtest_key] += 1
                            current_method = f"{subtest_key} [{subtest_counters[subtest_key]}]"
                        else:
                            subtest_counters[subtest_key] = 1
                            current_method = subtest_key
                    else:
                        current_method = base_method
                    
                    # Check if this is a subtest of an existing test
                    if subtest_suffix:
                        # This is a subtest - create a new test result for it
                        file_path_parts = [part for part in full_path[:-1] if part[0].islower()]
                        test_file_path = "tests/" + "/".join(file_path_parts) + ".py"
                        test_status_map[current_method] = TestResult(
                            status=TestStatus.FAILED if line.startswith("FAIL: ") else TestStatus.ERROR,
                            file_path=test_file_path,
                            name=current_test,
                            method=current_method,
                        )
                    elif current_method in test_status_map:
                        # Override the status to FAILED or ERROR
                        if line.startswith("ERROR: "):
                            test_status_map[current_method].status = TestStatus.ERROR
                        else:
                            test_status_map[current_method].status = TestStatus.FAILED
                    else:
                        # Create new test result if not found
                        file_path_parts = [part for part in full_path[:-1] if part[0].islower()]
                        test_file_path = "tests/" + "/".join(file_path_parts) + ".py"
                        test_status_map[current_method] = TestResult(
                            status=TestStatus.FAILED if line.startswith("FAIL: ") else TestStatus.ERROR,
                            file_path=test_file_path,
                            name=current_test,
                            method=current_method,
                        )
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

        # Third pass: Ensure we didn't miss any tests from the failure output
        # Look for any FAIL/ERROR lines that we might have missed in first pass
        for line in lines:
            if line.startswith("FAIL: ") or line.startswith("ERROR: "):
                test_line = line.split(": ", 1)[1].strip()
                
                # Check for subtests with conditions
                subtest_suffix = ""
                if " (condition=" in test_line:
                    base_test, condition = test_line.split(" (condition=", 1)
                    subtest_suffix = f" (condition={condition}"
                    match = test_pattern.match(base_test)
                else:
                    match = test_pattern.match(test_line)

                if match:
                    method_name = match.group(1)
                    full_path = match.group(2).split(".")
                    class_name = full_path[-1] if full_path[-1][0].isupper() else None
                    base_method = f"{class_name}.{method_name}" if class_name else method_name
                    
                    # Create method identifier
                    if subtest_suffix:
                        method_identifier = base_method + subtest_suffix
                    else:
                        method_identifier = base_method
                    
                    # Check if we already have this exact test (including any with counters)
                    existing_keys = [k for k in test_status_map.keys() if k.startswith(method_identifier)]
                    if not existing_keys:
                        file_path_parts = [part for part in full_path[:-1] if part[0].islower()]
                        test_file_path = "tests/" + "/".join(file_path_parts) + ".py"
                        test_status_map[method_identifier] = TestResult(
                            status=TestStatus.FAILED if line.startswith("FAIL: ") else TestStatus.ERROR,
                            file_path=test_file_path,
                            name=test_line,
                            method=method_identifier,
                        )

        # Remove base tests that have subtests, but only if they actually have subtests
        base_tests_with_subtests = set()
        for method_key in test_status_map:
            if " (condition=" in method_key:
                # This is a subtest, extract the base test name
                base_test = method_key.split(" (condition=")[0]
                base_tests_with_subtests.add(base_test)
        
        # Remove base tests that have subtests from the results, 
        # but only if the base test actually exists and has subtests
        for base_test in base_tests_with_subtests:
            if base_test in test_status_map:
                # Check if this base test actually has subtests
                has_subtests = any(key.startswith(base_test + " (condition=") for key in test_status_map)
                if has_subtests:
                    del test_status_map[base_test]

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
