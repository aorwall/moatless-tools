import logging
from pathlib import Path
from typing import Optional, Tuple

from pydantic import PrivateAttr

from moatless.actions.model import Observation
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.utils.file import is_test

logger = logging.getLogger(__name__)


class CodeModificationMixin:
    """
    A mixin that provides common functionality for actions that modify code files.
    This includes path normalization, file validation, test running, and observation handling.
    """

    _runtime: RuntimeEnvironment | None = PrivateAttr(default=None)
    _code_index: CodeIndex | None = PrivateAttr(default=None)
    _repository: Repository | None = PrivateAttr(default=None)

    def normalize_path(self, file_path: str) -> str:
        """Normalize file path by removing /repo and leading /"""
        if file_path.startswith("/repo"):
            file_path = file_path[5:]
        if file_path.startswith("/"):
            file_path = file_path[1:]
        return file_path

    def validate_file_access(
        self, file_path: str, file_context: FileContext
    ) -> Tuple[Optional[Path], Optional[Observation]]:
        """
        Validate file access and return either a valid Path object or an error Observation.

        Args:
            file_path: The path to validate
            file_context: The file context
        Returns:
            Tuple of (Path object if valid, Error observation if invalid)
        """
        path = Path(file_path)

        if not file_context.file_exists(str(path)):
            return None, Observation(
                message=f"File {path} not found.",
                properties={"fail_reason": "file_not_found"},
            )

        if not file_context.has_file(str(path)):
            return None, Observation(
                message=f"You have not yet viewed the file {path}. Use ViewCode to view the parts of the file that you want to modify.",
                properties={"fail_reason": "file_not_in_context"},
            )

        return path, None

    def run_tests(
        self,
        file_path: str,
        file_context: FileContext,
    ) -> str:
        if not file_context.has_runtime:
            return ""

        if file_context.file_exists(file_path) and is_test(file_path):
            file_context.add_test_file(file_path)
        elif self._code_index:
            # If the file is not a test file, find test files that might be related to the file
            search_results = self._code_index.find_test_files(
                file_path, query=file_path, max_results=2, max_spans=2
            )

            for search_result in search_results:
                file_context.add_test_file(search_result.file_path)
        else:
            logger.warning(f"No code index cannot find test files for {file_path}")
            return ""

        file_context.run_tests()

        response_msg = f"Running tests for the following files:\n"
        for test_file in file_context.test_files:
            response_msg += f"* {test_file.file_path}\n"

        failure_details = file_context.get_test_failure_details()
        if failure_details:
            response_msg += f"\n{failure_details}"

        summary = f"\n{file_context.get_test_summary()}"
        response_msg += summary

        return response_msg

    def format_snippet_with_lines(self, snippet: str, start_line: int) -> str:
        """Format a code snippet with line numbers"""
        return "\n".join(
            f"{i + start_line:6}\t{line}" for i, line in enumerate(snippet.split("\n"))
        )
