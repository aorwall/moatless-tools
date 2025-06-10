import re
from typing import List, Optional

from moatless.testing.python.utils import clean_log, create_generic_error_result
from moatless.testing.schema import TestResult, TestStatus, TraceItem
from moatless.testing.test_output_parser import TestOutputParser


class PyTestParser(TestOutputParser):
    """Parser for test output from the pytest framework."""

    def parse_test_output(self, log: str, file_path: Optional[str] = None) -> List[TestResult]:
        """
        Parse the pytest output and extract test results.

        Args:
            log: The test command output string
            file_path: Optional file path to filter results for

        Returns:
            List[TestResult]: List of test results
        """
        test_results = []
        test_errors = []

        failure_outputs = {}
        current_failure = None
        current_section = []
        option_pattern = re.compile(r"(.*?)\[(.*)\]")

        log = clean_log(log)

        test_summary_phase = False
        failures_phase = False
        errors_phase = False
        error_patterns = [
            (re.compile(r"ERROR collecting (.*) ___.*"), "collection"),
            (re.compile(r"ERROR at setup of (.*) ___.*"), "setup"),
            (re.compile(r"ERROR at teardown of (.*) ___.*"), "teardown"),
            (re.compile(r"ERROR (.*) ___.*"), "general"),
        ]

        for line in log.split("\n"):
            if "short test summary info" in line:
                test_summary_phase = True
                failures_phase = False
                errors_phase = False
                # Only include results for last test summary for now
                test_results = []
                continue

            if "=== FAILURES ===" in line:
                test_summary_phase = False
                failures_phase = True
                errors_phase = False
                continue

            if "=== ERRORS ===" in line:
                test_summary_phase = False
                failures_phase = True
                errors_phase = True
                continue

            # Remove ANSI codes and escape characters
            line = re.sub(r"\[(\d+)m", "", line)
            line = line.translate(str.maketrans("", "", "".join([chr(char) for char in range(1, 32)])))

            if (
                not failures_phase
                and any([line.startswith(x.value) for x in TestStatus])
                or any([line.endswith(x.value) for x in TestStatus])
            ):
                if line.startswith(TestStatus.FAILED.value):
                    line = line.replace(" - ", " ")

                test_case = line.split()
                if len(test_case) <= 1:
                    continue

                if any([line.startswith(x.value) for x in TestStatus]):
                    status_str = test_case[0]
                else:
                    status_str = test_case[-1]

                if status_str.endswith(":"):
                    status_str = status_str[:-1]

                if status_str != "SKIPPED" and "::" not in line:
                    continue

                try:
                    status = TestStatus(status_str)
                except ValueError:
                    status = TestStatus.ERROR

                # Handle SKIPPED cases with [number]
                if status == TestStatus.SKIPPED and test_case[1].startswith("[") and test_case[1].endswith("]"):
                    file_path_with_line = test_case[2]
                    test_file_path, line_number = file_path_with_line.split(":", 1)
                    method = None
                    full_name = " ".join(test_case[2:])
                else:
                    full_name = " ".join(test_case[1:])

                    has_option = option_pattern.search(full_name)
                    if has_option:
                        main, option = has_option.groups()
                        if option.startswith("/") and not option.startswith("//") and "*" not in option:
                            option = "/" + option.split("/")[-1]

                        # In the SWE-Bench dataset only the first word in an option is included for some reason...
                        if option and " " in option:
                            option = option.split()[0]
                            full_name = f"{main}[{option}"
                        else:
                            full_name = f"{main}[{option}]"

                    parts = full_name.split("::")
                    if len(parts) > 1:
                        test_file_path = parts[0]
                        method = ".".join(parts[1:])

                        if not has_option:
                            method = method.split()[0]
                    else:
                        test_file_path, method = None, None

                # Filter by file path if provided
                if file_path and test_file_path and file_path not in test_file_path:
                    continue

                test_results.append(TestResult(status=status, name=full_name, file_path=test_file_path, method=method))

            error_match = None
            error_type = None
            for pattern, err_type in error_patterns:
                match = pattern.search(line)
                if match:
                    error_match = match
                    error_type = err_type
                    break

            if error_match:
                if current_failure and current_section:
                    failure_outputs[current_failure].extend(current_section)

                if error_match.group(1).endswith(".py"):
                    current_failure = f"{error_type.capitalize()} error in {error_match.group(1)}"
                    test_errors.append(
                        TestResult(
                            status=TestStatus.ERROR,
                            name=current_failure,
                            file_path=error_match.group(1),
                        )
                    )
                    failure_outputs[current_failure] = []
                    current_section = []
                else:
                    current_failure = error_match.group(1)
                    failure_outputs[current_failure] = []
                    current_section = []
            elif line.startswith("_____"):
                if current_failure and current_section:
                    failure_outputs[current_failure].extend(current_section)
                current_failure = line.strip("_ ")
                failure_outputs[current_failure] = []
                current_section = []
            elif line.startswith("====="):
                if current_failure and current_section:
                    failure_outputs[current_failure].extend(current_section)
                current_failure = None
                current_section = []
            elif current_failure:
                current_section.append(line)

        # Add the last section if exists
        if current_failure and current_section:
            failure_outputs[current_failure].extend(current_section)

        test_results.extend(test_errors)

        # Add failure outputs to corresponding failed or error tests
        for test in test_results:
            if test.status in [TestStatus.PASSED, TestStatus.SKIPPED]:
                continue

            if test.method in failure_outputs:
                test.failure_output = "\n".join(failure_outputs[test.method])
            elif test.name in failure_outputs:
                test.failure_output = "\n".join(failure_outputs[test.name])

            # Truncate long outputs with teardown capture
            if test.failure_output and len(test.failure_output.splitlines()) > 25:
                teardown_idx = test.failure_output.find(
                    "--------------------------- Captured stdout teardown ---------------------------"
                )
                if teardown_idx != -1:
                    test.failure_output = test.failure_output[:teardown_idx].rstrip()

        # If we have no test results but the log indicates an error, create a generic error result
        if not test_results and (re.search(r"[A-Z][a-zA-Z]*Error", log) or "ImportError" in log or "Traceback" in log):
            error_type = "Unknown Error"
            if "ImportError" in log:
                error_type = "Import Error"
            elif "ValueError" in log:
                error_type = "Value Error"
            elif "TypeError" in log:
                error_type = "Type Error"
            elif "SyntaxError" in log:
                error_type = "Syntax Error"

            test_results.append(create_generic_error_result(log, error_type))
        
        # If we have no test results, try to parse the summary line
        if not test_results:
            # Look for pytest summary patterns like "X passed", "X failed", etc.
            summary_pattern = re.compile(r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|error)", re.IGNORECASE)
            summary_matches = summary_pattern.findall(log)
            
            if summary_matches:
                # Create a single test result representing the overall test run
                # Find the predominant status - if any failed/error, use that; otherwise use passed
                has_failures = False
                total_tests = 0
                status_counts = {}
                
                for count_str, status_str in summary_matches:
                    count = int(count_str)
                    total_tests += count
                    status_str_upper = status_str.upper()
                    
                    if status_str_upper in ["FAILED", "ERROR"]:
                        has_failures = True
                    
                    status_counts[status_str_upper] = count
                
                # Determine overall status
                if "ERROR" in status_counts and status_counts["ERROR"] > 0:
                    overall_status = TestStatus.ERROR
                elif "FAILED" in status_counts and status_counts["FAILED"] > 0:
                    overall_status = TestStatus.FAILED
                else:
                    overall_status = TestStatus.PASSED
                
                # Create a summary result
                test_results.append(TestResult(
                    status=overall_status,
                    name=f"Test Summary ({total_tests} tests)",
                    file_path=file_path,
                    method=None,
                    failure_output=None
                ))

        return test_results

    def detect_unhandled_pytest_error(self, log: str) -> bool:
        """
        Detects if the log contains unhandled pytest-style errors, excluding those covered by the regular parser.

        Args:
            log: The test command output string

        Returns:
            bool: True if an unhandled error is detected, False otherwise
        """
        unhandled_patterns = [
            r"ImportError while loading conftest",
            r"Error during collection",
            r"Error while loading fixture",
        ]

        for pattern in unhandled_patterns:
            if re.search(pattern, log):
                return True
        return False

    def parse_unhandled_pytest_error(self, log: str, test_name: str) -> TestResult:
        """
        Parses a pytest-style error log that isn't handled by the regular parser into a TestResult object.

        Args:
            log: The test command output string
            test_name: Name to use for the test result

        Returns:
            TestResult: A test result object with the error information
        """
        stacktrace = []
        failure_output = None

        # Extract each line of the stack trace
        trace_lines = log.split("\n")

        # Pattern to match: file_path:line_number: in method_name (method is the last item on the line)
        pattern = r"([^:]+):(\d+):\s+in\s+(.+)$"

        i = 0
        while i < len(trace_lines):
            trace = trace_lines[i]
            # Ensure trace is a string and search for matches
            if isinstance(trace, str):
                match = re.search(pattern, trace)

                if match:
                    file_path = match.group(1)
                    line_number = int(match.group(2))
                    method_name = match.group(3)

                    # Now look ahead to the next line for the output
                    if i + 1 < len(trace_lines):
                        output = trace_lines[i + 1].strip()  # Get the next line as output
                    else:
                        output = ""

                    trace_item = TraceItem(
                        file_path=file_path.strip(),
                        line_number=line_number,
                        method=method_name,
                        output=output,
                    )
                    stacktrace.append(trace_item)

                    i += 2
                else:
                    i += 1
            else:
                i += 1

        # Extract the final error type and message
        error_message_match = re.search(r"E\s+(\w+Error):\s+(.+)", log)
        if error_message_match:
            failure_output = f"{error_message_match.group(1)}: {error_message_match.group(2)}"
        else:
            failure_output = log

        test_result = TestResult(
            status=TestStatus.ERROR,
            name=test_name,
            failure_output=failure_output,
            stacktrace=stacktrace,
        )

        return test_result
