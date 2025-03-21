import re
from abc import ABC, abstractmethod
from typing import List, Optional

from moatless.testing.schema import TestResult, TestStatus


class TestOutputParser(ABC):
    """Base parser class for test output from different test frameworks."""

    @abstractmethod
    def parse_test_output(self, output: str, file_path: Optional[str] = None) -> List[TestResult]:
        """
        Parse the test command output and extract test results.

        Args:
            output: The test command output string
            file_path: Optional file path to filter results for

        Returns:
            List[TestResult]: List of test results
        """
        pass
