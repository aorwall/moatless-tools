import re
from typing import List, Optional, Tuple

from moatless.testing.schema import TestResult, TestStatus, TraceItem


def clean_log(log: str) -> str:
    """
    Remove ANSI color codes and escape sequences from log output

    Args:
        log: Log content to clean

    Returns:
        str: Cleaned log content
    """
    # Remove ANSI color codes like [0m[37m etc.
    log = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", log)

    # Remove specific problematic control characters while preserving important ones
    # \x0b: vertical tab
    # \x0c: form feed
    # \x1c-\x1f: file separators, group separators etc.
    control_chars = "\x0b\x0c\x1c\x1d\x1e\x1f"
    log = log.translate(str.maketrans("", "", control_chars))

    return log


def parse_traceback_line(line: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse a traceback line to extract file path, line number, and method name.

    Args:
        line: The traceback line to parse

    Returns:
        Tuple containing file path, line number, and method name, or None values if not found
    """
    pattern = r'File "([^"]+)", line (\d+), in (\S+)'

    match = re.search(pattern, line)

    if match:
        file_path = match.group(1)
        if file_path.startswith("/testbed/"):
            file_path = file_path.replace("/testbed/", "")
        line_number = int(match.group(2))
        method_name = match.group(3)

        return file_path, line_number, method_name
    else:
        return None, None, None


def parse_traceback(log: str) -> Optional[TestResult]:
    """
    Parse a traceback log into a TestResult object.

    Args:
        log: The traceback log to parse

    Returns:
        TestResult object or None if no traceback found
    """
    current_trace_item = None
    stacktrace = []

    for line in log.split("\n"):
        file_path, line_number, method_name = parse_traceback_line(line)
        if file_path:
            current_trace_item = TraceItem(file_path=file_path, line_number=line_number, method=method_name)
            stacktrace.append(current_trace_item)
        elif current_trace_item:
            if current_trace_item.output:
                current_trace_item.output += "\n"
            current_trace_item.output += line

    if not current_trace_item:
        return None

    return TestResult(
        status=TestStatus.ERROR,
        name="traceback",
        method=current_trace_item.method,
        file_path=current_trace_item.file_path,
        failure_output=log,
        stacktrace=stacktrace,
    )


def create_generic_error_result(log: str, error_type: str = "Unknown Error") -> TestResult:
    """
    Create a generic error TestResult when a more specific parser fails.

    This is used as a fallback mechanism when standard parsing fails but we still want to
    return an error result rather than an empty list.

    Args:
        log: The error log content
        error_type: Type of error to report in the name field

    Returns:
        TestResult: A TestResult object with ERROR status
    """
    # Try to extract file path and line number from any recognizable pattern
    file_path = None
    method = None
    line_number = None
    failure_output = log
    stacktrace = []

    # Look for pytest-style tracebacks
    pytest_pattern = r"([^:]+):(\d+): in ([^\n]+)\n\s+(.+)"
    for match in re.finditer(pytest_pattern, log):
        path, line, func, code = match.groups()
        if not file_path:
            file_path = path
            method = func
            line_number = int(line)
        stacktrace.append(TraceItem(file_path=path, line_number=int(line), method=func, output=code.strip()))

    # Look for ImportError specific patterns
    import_error_match = re.search(r'ImportError.+[\'"](.*)[\'"]', log)
    if import_error_match and not file_path:
        file_path = import_error_match.group(1)
        if file_path.startswith("/testbed/"):
            file_path = file_path.replace("/testbed/", "")

    # Extract actual error message
    error_message_match = re.search(r"([A-Z][a-zA-Z]*Error:[^\n]+)", log)
    if error_message_match:
        failure_output = error_message_match.group(1)

    # Create a TestResult with whatever we found
    return TestResult(
        status=TestStatus.ERROR,
        name=error_type,
        file_path=file_path,
        method=method,
        failure_output=failure_output,
        stacktrace=stacktrace,
    )
