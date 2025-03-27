import logging
from pathlib import Path
from typing import Optional

from opentelemetry import trace

from moatless.actions.schema import Observation
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


class CodeModificationMixin:
    """
    A mixin that provides common functionality for actions that modify code files.
    This includes path normalization, file validation, test running, and observation handling.
    """

    def normalize_path(self, file_path: str) -> str:
        """Normalize file path by removing /repo and leading /"""
        if file_path.startswith("/repo"):
            file_path = file_path[5:]
        if file_path.startswith("/"):
            file_path = file_path[1:]
        return file_path

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

    def format_snippet_with_lines(self, snippet: str, start_line: int) -> str:
        """Format a code snippet with line numbers"""
        return "\n".join(f"{i + start_line:6}\t{line}" for i, line in enumerate(snippet.split("\n")))
