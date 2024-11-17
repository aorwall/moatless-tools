import logging
from pathlib import Path
from typing import Optional, Tuple

from pydantic import PrivateAttr

from moatless.actions.model import Observation
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment

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
        self, file_path: str, file_context: FileContext, allow_missing: bool = False
    ) -> Tuple[Optional[Path], Optional[Observation]]:
        """
        Validate file access and return either a valid Path object or an error Observation.

        Args:
            file_path: The path to validate
            file_context: The file context
            allow_missing: Whether to allow missing files (for file creation)

        Returns:
            Tuple of (Path object if valid, Error observation if invalid)
        """
        path = Path(file_path)

        if not allow_missing and not file_context.file_exists(str(path)):
            return None, Observation(
                message=f"File {path} not found.",
                properties={"fail_reason": "file_not_found"},
            )

        if allow_missing and file_context.file_exists(str(path)):
            return None, Observation(
                message=f"File already exists at: {path}. Cannot overwrite existing file.",
                properties={"fail_reason": "file_exists"},
            )

        if not allow_missing:
            context_file = file_context.get_context_file(str(path))
            if not context_file:
                return None, Observation(
                    message=f"Could not get context for file: {path}",
                    properties={"fail_reason": "context_error"},
                )

        return path, None

    def run_tests_and_update_observation(
        self,
        observation: Observation,
        file_path: str,
        scratch_pad: str,
        file_context: FileContext,
    ) -> Observation:
        """Run tests and update the observation with test results"""
        if not observation.properties or not observation.properties.get("diff"):
            return observation

        if not self._runtime:
            return observation

        run_tests = RunTests(
            repository=self._repository,
            runtime=self._runtime,
            code_index=self._code_index,
        )

        test_observation = run_tests.execute(
            RunTestsArgs(
                scratch_pad=scratch_pad,
                test_files=[file_path],
            ),
            file_context,
        )

        observation.properties.update(test_observation.properties)
        observation.message += "\n\n" + test_observation.message

        return observation

    def format_snippet_with_lines(self, snippet: str, start_line: int) -> str:
        """Format a code snippet with line numbers"""
        return "\n".join(
            f"{i + start_line:6}\t{line}" for i, line in enumerate(snippet.split("\n"))
        )
