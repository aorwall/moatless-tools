import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import PrivateAttr

from moatless.artifacts.artifact import ArtifactReference
from moatless.artifacts.diagnostics.diagnostic import (
    DiagnosticArtifact,
    DiagnosticHandler,
    DiagnosticSeverity,
    Position,
    Range,
)
from moatless.environment.base import BaseEnvironment, EnvironmentExecutionError
from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)


class MyPyArtifactHandler(DiagnosticHandler):
    """
    A handler for generating and managing MyPy diagnostic artifacts.
    This handler runs MyPy on Python code and converts the output into DiagnosticArtifact objects.
    """

    _environment: BaseEnvironment = PrivateAttr(default=None)

    def __init__(self, storage: BaseStorage | None = None, environment: Optional[BaseEnvironment] = None, **kwargs):
        """
        Initialize the MyPy artifact handler.

        Args:
            storage: Storage backend to use for storing artifacts.
            environment: Optional environment to use for running MyPy commands.
                         If not provided, the default environment will be used.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(storage=storage, **kwargs)

        self._environment = environment or BaseEnvironment.get_default_environment()

    async def run_diagnostics(self, files: list[str] | None = None) -> list[DiagnosticArtifact]:
        """
        Run MyPy on the specified Python files and generate diagnostic artifacts.

        Args:
            files: List of Python files or directories to check

        Returns:
            List of DiagnosticArtifact objects generated from MyPy output
        """
        await self._ensure_mypy_installed()
        if files is None:
            files = ["."]

        cmd_parts = ["mypy", "--output", "json"]
        cmd_parts.extend(files)

        command = " ".join(cmd_parts)
        logger.info(f"Running MyPy command: {command}")

        try:
            stdout = await self._environment.execute(command)
            artifacts = self._parse_mypy_json_output(stdout)
            for artifact in artifacts:
                artifact = await self.create(artifact)
                await self.persist(artifact.id)
        except EnvironmentExecutionError as e:
            logger.warning(f"MyPy command failed with return code {e.return_code}: {e.stderr}")
            # If MyPy returned JSON output in stderr, try to parse it
            if e.stderr and e.stderr.strip().startswith("{"):
                try:
                    artifacts.extend(self._parse_mypy_json_output(e.stderr))
                except Exception as parse_error:
                    logger.error(f"Failed to parse MyPy error output: {parse_error}")

        return artifacts

    async def _ensure_mypy_installed(self) -> None:
        """
        Ensure that MyPy is installed in the current environment.
        Installs it if not present.
        """
        # Check if mypy is installed
        try:
            stdout = await self._environment.execute("pip list | grep mypy")
            mypy_installed = stdout and "mypy" in stdout
        except EnvironmentExecutionError:
            # grep returns non-zero exit code if pattern not found
            mypy_installed = False

        if not mypy_installed:
            logger.info("MyPy not found. Installing...")
            try:
                await self._environment.execute("pip install mypy")
                logger.info("MyPy installed successfully")
            except EnvironmentExecutionError as e:
                logger.error(f"Failed to install MyPy: {e.stderr}")
                raise RuntimeError(f"Failed to install MyPy: {e.stderr}") from e

    def _parse_mypy_json_output(self, output: str) -> list[DiagnosticArtifact]:
        """
        Parse MyPy JSON output and convert it to DiagnosticArtifact objects.

        MyPy JSON output format is:
        {"file": "path/to/file.py", "line": 123, "column": 45, "message": "...", "code": "...", "severity": "..."}

        Args:
            output: The JSON stdout from MyPy execution

        Returns:
            List of DiagnosticArtifact objects
        """
        artifacts = []

        # Parse each line of the output as a separate JSON object
        for line in output.splitlines():
            if not line.strip():
                continue

            try:
                report = json.loads(line)

                file_path = report.get("file", "")

                severity_map = {
                    "error": DiagnosticSeverity.ERROR,
                    "warning": DiagnosticSeverity.WARNING,
                    "note": DiagnosticSeverity.INFO,
                }
                severity = severity_map.get(report.get("severity", "error"), DiagnosticSeverity.ERROR)

                line_num = report.get("line", 1)
                column_num = report.get("column", 1)

                range_obj = Range(
                    start=Position(line=line_num, column=column_num),
                )

                file_reference = ArtifactReference(id=file_path, type="file")

                diagnostic = DiagnosticArtifact(
                    id=self.generate_id(),
                    severity=severity,
                    range=range_obj,
                    source="mypy",
                    message=report.get("message", ""),
                    code=report.get("code", "unknown"),
                    references=[file_reference],
                )

                artifacts.append(diagnostic)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse MyPy JSON output: {e}\nLine: {line}")
                continue

        return artifacts
