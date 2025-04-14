import re
from typing import List, Optional

from moatless.testing.schema import TestResult, TestStatus
from moatless.testing.test_output_parser import TestOutputParser


class MavenParser(TestOutputParser):
    """Parser for test output from Maven Surefire/Failsafe test framework."""

    # Max number of lines to keep in full for error messages
    MAX_ERROR_LINES = 20

    def parse_test_output(self, log: str, file_path: Optional[str] = None) -> List[TestResult]:
        """
        Parse the Maven test output and extract test results.

        Args:
            log: The test command output string
            file_path: Optional file path to filter results for

        Returns:
            List[TestResult]: List of test results
        """
        test_results = []

        # If the log starts with "Command failed", it's a command execution error
        if log.startswith("Command failed with return code"):
            # Extract the command from the error message if possible
            command_match = re.search(r"Command failed with return code \d+: (.+)", log)
            command = command_match.group(1) if command_match else "unknown command"

            return [
                TestResult(
                    status=TestStatus.ERROR,
                    name=f"Maven command failed: {command}",
                    file_path=file_path,
                    failure_output=log,
                )
            ]

        # Patterns to identify test results in Maven output
        test_pattern = re.compile(r"Running ([\w\.]+)")
        test_result_pattern = re.compile(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)")
        failure_pattern = re.compile(r"\[ERROR\] ([\w\.]+)(?:\([\w\.]+\))?\s*Time elapsed:")
        failure_detail_pattern = re.compile(r"^\[ERROR\]\s*at ([\w\.]+)\(([^:]+):(\d+)\)")

        # Patterns for compilation errors - support both [line,column] and [line] formats
        compilation_error_pattern = re.compile(r"\[ERROR\] (?:.*?/)?([\w/\.]+):\[(\d+),(\d+)\] (.+)")
        compilation_error_pattern2 = re.compile(r"\[ERROR\] (?:.*?/)?([\w/\.]+):\[(\d+)\] (.+)")

        # Capture test failure output
        failure_outputs = {}
        current_failure = None
        current_section = []

        # States for parsing
        current_test_class = None

        # Extract compilation errors with context
        compilation_errors = []

        # Find all compilation error sections
        compilation_sections = []
        current_comp_section = []
        in_compilation_error = False

        for line in log.splitlines():
            line = line.strip()

            # Detect start of compilation error section - match both error patterns
            if re.search(r"\[ERROR\] .*\.java:\[\d+", line):
                if in_compilation_error and current_comp_section:
                    compilation_sections.append(current_comp_section.copy())
                current_comp_section = [line]
                in_compilation_error = True
                continue

            # Continue collecting compilation error details
            if in_compilation_error:
                if line.startswith("[ERROR]") and not line.startswith("[ERROR] ->"):
                    current_comp_section.append(line)
                else:
                    # End of the section
                    if current_comp_section:
                        compilation_sections.append(current_comp_section.copy())
                    in_compilation_error = False
                    current_comp_section = []

        # Add the last section if exists
        if in_compilation_error and current_comp_section:
            compilation_sections.append(current_comp_section)

        # Process each compilation error section
        for section in compilation_sections:
            # Extract the first line which contains the file and position
            match = compilation_error_pattern.search(section[0])
            if match:
                source_file = match.group(1)
                line_number = match.group(2)
                column = match.group(3)

                # Combine all error lines as the message
                message = "\n".join(section)

                compilation_errors.append(
                    {"file": source_file, "line": line_number, "column": column, "message": message}
                )
            else:
                # Try the second pattern for [line] format
                match = compilation_error_pattern2.search(section[0])
                if match:
                    source_file = match.group(1)
                    line_number = match.group(2)

                    # Combine all error lines as the message
                    message = "\n".join(section)

                    compilation_errors.append(
                        {
                            "file": source_file,
                            "line": line_number,
                            "column": "0",  # Default column value
                            "message": message,
                        }
                    )

        # If we found compilation errors, report them
        if compilation_errors:
            for error in compilation_errors:
                test_results.append(
                    TestResult(
                        status=TestStatus.ERROR,
                        name=f"Compilation error in {error['file']} at line {error['line']}",
                        file_path=error["file"],
                        method=None,
                        failure_output=error["message"],
                    )
                )

            # If we have compilation errors, we might not have test results, but return anyway
            if test_results:
                return test_results

        # Check for common build problems
        build_problem_patterns = [
            (r"No tests were executed!", "No tests were executed"),
            (r"Failed to execute goal.*-Dtest=([\w\.]+)", "Failed to execute test"),
            (r"There are test failures", "Test failures occurred"),
            (r"RuntimeException: Failed to load ApplicationContext", "Failed to load Spring ApplicationContext"),
            (r"BuildTimeoutException", "Build timeout"),
            (r"OutOfMemoryError", "Out of memory error"),
            (
                r"Failed to execute goal org\.apache\.maven\.plugins:maven-compiler-plugin.*Compilation failure",
                "Maven compilation failure",
            ),
        ]

        for pattern, error_name in build_problem_patterns:
            match = re.search(pattern, log, re.MULTILINE)
            if match:
                # Extract more context for the error
                error_context = self._extract_error_context(log, match)

                # Create a specific error for build problems
                return [
                    TestResult(
                        status=TestStatus.ERROR,
                        name=error_name,
                        file_path=file_path,
                        failure_output=error_context if error_context else log,
                    )
                ]

        for line in log.splitlines():
            line = line.strip()

            # Find test class
            test_match = test_pattern.search(line)
            if test_match:
                current_test_class = test_match.group(1)
                continue

            # Capture test failures
            failure_match = failure_pattern.search(line)
            if failure_match:
                if current_failure and current_section:
                    failure_outputs[current_failure] = "\n".join(current_section)

                current_failure = failure_match.group(1)
                current_section = []
                continue

            # Collect failure details
            if current_failure and line and not line.startswith("[INFO]"):
                current_section.append(line)
                continue

            # Process test result summary
            summary_match = test_result_pattern.search(line)
            if summary_match and current_test_class:
                total = int(summary_match.group(1))
                failures = int(summary_match.group(2))
                errors = int(summary_match.group(3))
                skipped = int(summary_match.group(4))

                # Process passed tests
                passed = total - failures - errors - skipped
                if passed > 0:
                    test_results.append(
                        TestResult(
                            status=TestStatus.PASSED,
                            name=f"{passed} passed test(s) in {current_test_class}",
                            file_path=file_path or self._convert_class_to_file_path(current_test_class),
                            method=current_test_class,
                        )
                    )

                # Process skipped tests
                if skipped > 0:
                    test_results.append(
                        TestResult(
                            status=TestStatus.SKIPPED,
                            name=f"{skipped} skipped test(s) in {current_test_class}",
                            file_path=file_path or self._convert_class_to_file_path(current_test_class),
                            method=current_test_class,
                        )
                    )
                continue

        # Add the last section if exists
        if current_failure and current_section:
            failure_outputs[current_failure] = "\n".join(current_section)

        # Extract individual failed tests
        failed_test_pattern = re.compile(r"^\[ERROR\] ([\w\.]+)#(\w+)")
        for line in log.splitlines():
            line = line.strip()
            failed_test_match = failed_test_pattern.search(line)

            if failed_test_match:
                class_name = failed_test_match.group(1)
                method_name = failed_test_match.group(2)
                test_name = f"{class_name}#{method_name}"

                # Determine if it's an error or failure (errors usually have exceptions)
                status = TestStatus.ERROR if "Exception" in line else TestStatus.FAILED

                test_results.append(
                    TestResult(
                        status=status,
                        name=test_name,
                        file_path=file_path or self._convert_class_to_file_path(class_name),
                        method=method_name,
                        failure_output=failure_outputs.get(test_name, ""),
                    )
                )

        # If we got no results but have errors in the log, create a generic error result
        if not test_results and any(error in log for error in ["ERROR", "Exception", "Build failed", "BUILD FAILURE"]):
            # Try to extract more specific error information
            error_name = "Maven build error"

            # Look for specific error patterns in the output
            error_patterns = [
                (r"Failed to execute goal.*-Dtest=([\w\.]+)", lambda m: f"Failed to execute test for {m.group(1)}"),
                (r"Failed to execute goal.*?:\s+(.+?)\s*$", lambda m: f"Maven goal failed: {m.group(1)}"),
                (
                    r"java\.lang\.([\w]+Exception)(?::\s*(.+?))?$",
                    lambda m: f"{m.group(1)}: {m.group(2) if m.group(2) else ''}",
                ),
                (r"No tests were executed!", lambda m: "No tests were executed"),
                (r"There are test failures", lambda m: "Test failures occurred"),
            ]

            for pattern, name_func in error_patterns:
                match = re.search(pattern, log, re.MULTILINE)
                if match:
                    error_name = name_func(match)
                    # Extract the error context instead of using the entire log
                    error_output = self._extract_error_context(log, match)
                    break
            else:
                # If no specific patterns matched, use BUILD FAILURE section if present
                build_failure_match = re.search(r"(BUILD FAILURE.*?)(\[INFO\]|\Z)", log, re.MULTILINE | re.DOTALL)
                if build_failure_match:
                    error_output = build_failure_match.group(1)
                else:
                    error_output = log

            test_results.append(
                TestResult(status=TestStatus.ERROR, name=error_name, file_path=file_path, failure_output=error_output)
            )

        return test_results

    def _extract_error_context(self, log: str, match):
        """
        Extract error context around a match.

        Args:
            log: Full log text
            match: Regex match object for the error pattern

        Returns:
            String containing error context
        """
        if not match:
            return None

        # Get the matched line and position
        start_pos = match.start()
        end_pos = match.end()

        # Find the start of the line containing the match
        line_start = log.rfind("\n", 0, start_pos)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1  # Skip the newline

        # Find the section boundaries
        # Look for BUILD FAILURE
        build_failure_start = log.rfind("BUILD FAILURE", 0, start_pos)
        if build_failure_start != -1:
            # Look for the section start
            section_start = log.rfind("\n[INFO]", 0, build_failure_start)
            if section_start == -1:
                section_start = max(0, build_failure_start - 200)
        else:
            # If BUILD FAILURE not found, go back a reasonable amount
            section_start = max(0, line_start - 200)

        # Look for the end of the error section
        help_match = re.search(r"\[ERROR\] -> \[Help \d+\]", log[end_pos:])
        if help_match:
            section_end = end_pos + help_match.end()
        else:
            # If no Help marker, go forward a reasonable amount
            section_end = min(len(log), end_pos + 500)

        # Extract the relevant section
        error_section = log[section_start:section_end]

        # Count the number of lines
        line_count = error_section.count("\n")

        # If the section is small enough, return it as is
        if line_count <= self.MAX_ERROR_LINES:
            return error_section

        # Otherwise, focus on the error message and some context
        lines = error_section.split("\n")

        # Find the line with our error match
        match_line_idx = None
        for i, line in enumerate(lines):
            if match.group(0) in line:
                match_line_idx = i
                break

        if match_line_idx is not None:
            # Take some lines before and after the match
            start_idx = max(0, match_line_idx - 5)
            end_idx = min(len(lines), match_line_idx + 15)
            focused_lines = lines[start_idx:end_idx]

            # Add a note about truncation if needed
            if start_idx > 0:
                focused_lines.insert(0, "... (earlier lines omitted) ...")
            if end_idx < len(lines):
                focused_lines.append("... (additional lines omitted) ...")

            return "\n".join(focused_lines)

        # If we couldn't find the exact line, return the beginning and end of the section
        return "\n".join(lines[:10]) + "\n... (middle lines omitted) ...\n" + "\n".join(lines[-10:])

    def _convert_class_to_file_path(self, class_name: str) -> str:
        """
        Convert Java class name to file path.

        Args:
            class_name: Fully qualified Java class name

        Returns:
            File path corresponding to the class
        """
        if not class_name:
            return ""

        # Replace dots with slashes and add .java extension
        # e.g. com.example.MyTest -> src/main/java/com/example/MyTest.java
        # Since we don't know if it's in main or test, just provide the package path
        return class_name.replace(".", "/") + ".java"
