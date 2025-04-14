import logging
from typing import Dict, List, Optional, Type

from moatless.testing.java.maven_parser import MavenParser
from moatless.testing.python.parser_registry import parse_log as parse_python_log
from moatless.testing.schema import TestResult, TestStatus
from moatless.testing.test_output_parser import TestOutputParser

logger = logging.getLogger(__name__)

# Map repository names to their respective parser classes
REPO_TO_PARSER_CLASS: Dict[str, Type[TestOutputParser]] = {
    # Java repositories with Maven
    "apache/maven": MavenParser,
    "spring-projects/spring-boot": MavenParser,
    "spring-projects/spring-framework": MavenParser,
    "elastic/elasticsearch": MavenParser,
    "hibernate/hibernate-orm": MavenParser,
    "apache/kafka": MavenParser,
    # Add more Java/Maven repositories as needed
}


def get_parser_for_repo(repo: str) -> TestOutputParser:
    """
    Get the appropriate parser for a repository.

    Args:
        repo: Repository name

    Returns:
        TestOutputParser: A parser instance for the repository
    """
    parser_class = REPO_TO_PARSER_CLASS.get(repo)

    # Default to Maven parser for Java projects that aren't explicitly listed
    if not parser_class and "/" in repo and repo.endswith("-java"):
        parser_class = MavenParser
    elif not parser_class:
        # Try to determine the appropriate parser based on other clues
        if repo.lower().endswith("java") or repo.lower().endswith("maven"):
            parser_class = MavenParser

    # If we still don't have a parser, use Maven as default
    if not parser_class:
        parser_class = MavenParser

    return parser_class()


def parse_log(log: str, repo: str, file_path: Optional[str] = None) -> List[TestResult]:
    """
    Parse test log using the appropriate parser for the repository.

    Args:
        log: Test output log content
        repo: Repository name
        file_path: Optional file path to filter results for

    Returns:
        List[TestResult]: List of parsed test results
    """
    # First try the python parsers
    try:
        # Check if it's a Python project by looking for Python-specific output
        python_indicators = ["pytest", "unittest", "python", 'File "', '.py"', "assert"]
        if any(indicator in log for indicator in python_indicators):
            return parse_python_log(log, repo, file_path)
    except Exception as e:
        logger.warning(f"Error using Python parser: {e}")

    # If not Python or Python parsing failed, try with Java
    parser = get_parser_for_repo(repo)
    logger.info(f"Parsing log for {repo} with {parser.__class__.__name__}")
    test_results = parser.parse_test_output(log, file_path)
    logger.info(f"Parsed {len(test_results)} test results")

    if not test_results:
        return [
            TestResult(
                file_path=file_path,
                failure_output=log,
                status=TestStatus.UNKNOWN,
                error_details=log,
            )
        ]

    # Skip testbed prefix in file paths
    for result in test_results:
        if result.file_path and result.file_path.startswith("/testbed/"):
            result.file_path = result.file_path[len("/testbed/") :]

        if result.failure_output:
            result.failure_output = result.failure_output.replace("/testbed/", "")

        if not result.file_path:
            result.file_path = file_path

    return test_results
