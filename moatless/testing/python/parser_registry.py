from typing import Dict, List, Optional, Type
import logging
from moatless.testing.schema import TestResult, TestStatus
from moatless.testing.test_output_parser import TestOutputParser
from moatless.testing.python.pytest_parser import PyTestParser
from moatless.testing.python.django_parser import DjangoParser
from moatless.testing.python.sympy_parser import SympyParser
from moatless.testing.python.seaborn_parser import SeabornParser

logger = logging.getLogger(__name__)

# Map repository names to their respective parser classes
REPO_TO_PARSER_CLASS: Dict[str, Type[TestOutputParser]] = {
    "astropy/astropy": PyTestParser,
    "django/django": DjangoParser,
    "marshmallow-code/marshmallow": PyTestParser,
    "matplotlib/matplotlib": PyTestParser,
    "mwaskom/seaborn": SeabornParser,
    "pallets/flask": PyTestParser,
    "psf/requests": PyTestParser,
    "pvlib/pvlib-python": PyTestParser,
    "pydata/xarray": PyTestParser,
    "pydicom/pydicom": PyTestParser,
    "pylint-dev/astroid": PyTestParser,
    "pylint-dev/pylint": PyTestParser,
    "pytest-dev/pytest": PyTestParser,
    "pyvista/pyvista": PyTestParser,
    "scikit-learn/scikit-learn": PyTestParser,
    "sqlfluff/sqlfluff": PyTestParser,
    "sphinx-doc/sphinx": PyTestParser,
    "sympy/sympy": SympyParser,
}


def get_parser_for_repo(repo: str) -> TestOutputParser:
    """
    Get the appropriate parser for a repository.

    Args:
        repo: Repository name

    Returns:
        TestOutputParser: A parser instance for the repository
    """
    parser_class = REPO_TO_PARSER_CLASS.get(repo, PyTestParser)
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
    parser = get_parser_for_repo(repo)
    logger.info(f"Parsing log for {repo} with {parser}")
    test_results = parser.parse_test_output(log, file_path)

    if not test_results:
        return [
            TestResult(
                file_path=file_path,
                failure_output=log,
                status=TestStatus.ERROR,
                error_details=log,
            )
        ]

    # Skip testbed prefix in file paths
    for result in test_results:
        if result.file_path and result.file_path.startswith("/testbed/"):
            result.file_path = result.file_path[len("/testbed/") :]

        if result.failure_output:
            result.failure_output = result.failure_output.replace("/testbed/", "")

    return test_results
