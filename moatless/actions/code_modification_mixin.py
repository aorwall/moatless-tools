import logging
from pathlib import Path
from typing import Optional, Tuple

from opentelemetry import trace
from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.schema import Observation
from moatless.file_context import FileContext
from moatless.telemetry import instrument
from moatless.utils.file import is_test

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


class CodeModificationMixin:
    """
    A mixin that provides common functionality for actions that modify code files.
    This includes path normalization, file validation, test running, and observation handling.
    """

    persist_artifacts: bool = Field(False, description="Whether to persist artifacts after modifying code")
    auto_run_tests: bool = Field(True, description="Whether to automatically run tests after modifying code")

    def normalize_path(self, file_path: str) -> str:
        """Normalize file path by removing /repo and leading /"""
        if file_path.startswith("/repo"):
            file_path = file_path[5:]
        if file_path.startswith("/"):
            file_path = file_path[1:]
        return file_path

    def persist(self, file_context: FileContext):
        """Persist the modified files"""
        if not self.persist_artifacts:
            return

        file_context.persist()

    def validate_file_access(
        self, file_path: str, file_context: FileContext
    ) -> tuple[Optional[Path], Optional[Observation]]:
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
            )  # type: ignore

        if not file_context.has_file(str(path)):
            return None, Observation(
                message=f"You have not yet viewed the file {path}. Use ViewCode to view the parts of the file that you want to modify.",
                properties={"fail_reason": "file_not_in_context"},
            )  # type: ignore

        return path, None

    @tracer.start_as_current_span("run_tests")
    async def run_tests(
        self,
        file_path: str,
        file_context: FileContext,
    ) -> str:
        if not self.auto_run_tests:
            return ""

        if not file_context.has_runtime:
            logger.warning(f"No runtime, cannot run tests for {file_path}")
            return ""

        if file_context.file_exists(file_path) and is_test(file_path):
            file_context.add_test_file(file_path)
        elif self._workspace.code_index:
            # If the file is not a test file, find test files that might be related to the file
            search_results = await self._workspace.code_index.find_test_files(
                file_path, query=file_path, max_results=2, max_spans=2
            )

            for search_result in search_results:
                file_context.add_test_file(search_result.file_path)
        else:
            logger.warning(f"No code index, cannot find test files for {file_path}")
            return ""

        logger.info(f"Running tests for {file_path}")
        await file_context.run_tests()

        response_msg = ""
        if not file_context.test_files:
            response_msg = "No test files found. Consider adding tests to verify the changes.\n"
        elif file_context.has_test_patch():
            response_msg = "Running tests for the updated test files:\n"
        else:
            response_msg = "Running existing tests to verify no regressions."

        if file_context.test_files:
            for test_file in file_context.test_files:
                response_msg += f"* {test_file.file_path}\n"

        failure_details = file_context.get_test_failure_details()
        if failure_details:
            response_msg += f"\n{failure_details}"

        summary = f"\n{file_context.get_test_summary()}"
        response_msg += summary

        if not failure_details and not file_context.has_test_patch():
            response_msg += "\nConsider adding new test cases for the changes."

        return response_msg

    def format_snippet_with_lines(self, snippet: str, start_line: int) -> str:
        """Format a code snippet with line numbers"""
        return "\n".join(f"{i + start_line:6}\t{line}" for i, line in enumerate(snippet.split("\n")))
