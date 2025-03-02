import json
import logging
from typing import List, Optional

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
    RewardScaleEntry,
)
from moatless.artifacts.artifact import ArtifactReference
from moatless.artifacts.diagnostics.diagnostic import (
    Diagnostic,
    DiagnosticArtifact,
    DiagnosticHandler,
    DiagnosticSeverity,
    Position,
    Range,
)
from moatless.environment.base import BaseEnvironment, EnvironmentExecutionError
from moatless.file_context import FileContext
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class RunMyPyArgs(ActionArguments):
    """
    Run MyPy type checking on the specified Python files.
    """

    thoughts: str = Field(..., description="Your reasoning on what files to check with MyPy.")
    files: list[str] = Field(..., description="The list of Python files or directories to check with MyPy")

    model_config = ConfigDict(title="RunMyPy")

    @property
    def log_name(self):
        return f"RunMyPy({', '.join(self.files)})"

    def to_prompt(self):
        return "Running MyPy type checking for the following files:\n" + "\n".join(f"* {file}" for file in self.files)


class RunMyPy(Action):
    """
    An action for running MyPy type checking on Python code and generating diagnostic artifacts.
    """

    ignore_notes: bool = Field(
        default=False,
        description="If true, ignore notes from MyPy.",
    )

    args_schema = RunMyPyArgs

    async def initialize(self, workspace: Workspace):
        await super().initialize(workspace)
        logger.info(f"Initialized RunMyPy action with workspace: {workspace}")
        await self._ensure_mypy_installed()

    async def _execute(self, args: RunMyPyArgs, file_context: FileContext | None = None) -> str | None:
        """
        Run MyPy on the specified Python files and generate diagnostic artifacts.
        """
        if not self.workspace or not self.workspace.environment:
            raise ValueError("Workspace with environment must be provided to execute the RunMyPy action.")

        if not args.files:
            return "No files specified for MyPy type checking."

        # Ensure MyPy is installed
        await self._ensure_mypy_installed()

        # Prepare the MyPy command
        cmd_parts = ["mypy", "--output", "json"]
        cmd_parts.extend(args.files)

        command = " ".join(cmd_parts)
        logger.info(f"Running MyPy command: {command}")

        artifacts = []
        try:
            stdout = await self.workspace.environment.execute(command)
            artifacts = self._parse_mypy_json_output(stdout, args.files)
        except EnvironmentExecutionError as e:
            logger.warning(f"MyPy command failed with return code {e.return_code}: {e.stderr}")
            # If MyPy returned JSON output in stderr, try to parse it
            if e.stderr and e.stderr.strip().startswith("{"):
                try:
                    artifacts = self._parse_mypy_json_output(e.stderr, args.files)
                except Exception as parse_error:
                    logger.error(f"Failed to parse MyPy error output: {parse_error}")
                    return f"Failed to run MyPy: {e.stderr}"
            else:
                return f"Failed to run MyPy: {e.stderr}"
        except Exception as e:
            logger.error(f"Unexpected error running MyPy: {str(e)}")
            return f"Unexpected error running MyPy: {str(e)}"

        # Save artifacts using the diagnostic handler from workspace
        diagnostic_handler = self.workspace.artifact_handlers.get("diagnostic")
        if not diagnostic_handler:
            logger.warning("No diagnostic handler found in workspace")
            return "No diagnostic handler found in workspace. MyPy results not saved."

        # Generate a summary message
        if not artifacts:
            return "MyPy found no type issues."

        # Create summary message using the to_prompt_message_content method
        summary_parts = [f"MyPy found {sum(len(artifact.diagnostics) for artifact in artifacts)} issues:"]

        for artifact in artifacts:
            content = artifact.to_prompt_message_content()
            if content["type"] == "text":
                summary_parts.append(content["text"])

        return "\n\n".join(summary_parts)

    def _parse_mypy_json_output(
        self, output: str, requested_files: Optional[list[str]] = None
    ) -> list[DiagnosticArtifact]:
        """
        Parse MyPy JSON output and convert it to DiagnosticArtifact objects.

        MyPy JSON output format is:
        {"file": "path/to/file.py", "line": 123, "column": 45, "message": "...", "code": "...", "severity": "..."}

        Args:
            output: The JSON stdout from MyPy execution
            requested_files: Optional list of files that were explicitly requested for checking

        Returns:
            List of DiagnosticArtifact objects, one per file
        """
        # Group diagnostics by file
        diagnostics_by_file = {}

        # Parse each line of the output as a separate JSON object
        for line in output.splitlines():
            if not line.strip():
                continue

            try:
                report = json.loads(line)

                file_path = report.get("file", "")

                # Skip files that weren't explicitly requested if requested_files is provided
                if requested_files and not any(file_path.startswith(req_file) for req_file in requested_files):
                    continue

                severity_map = {
                    "error": DiagnosticSeverity.ERROR,
                    "warning": DiagnosticSeverity.WARNING,
                    "note": DiagnosticSeverity.INFO,
                }
                severity = severity_map.get(report.get("severity", "error"), DiagnosticSeverity.ERROR)

                if self.ignore_notes and severity == DiagnosticSeverity.INFO:
                    continue

                line_num = report.get("line", 1)
                column_num = report.get("column", 1)

                range_obj = Range(
                    start=Position(line=line_num, column=column_num),
                )

                diagnostic = Diagnostic(
                    severity=severity,
                    range=range_obj,
                    source="mypy",
                    message=report.get("message", ""),
                    code=report.get("code", "unknown"),
                )

                # Group diagnostics by file
                if file_path not in diagnostics_by_file:
                    diagnostics_by_file[file_path] = []
                diagnostics_by_file[file_path].append(diagnostic)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse MyPy JSON output: {e}\nLine: {line}")
                continue

        # Create a DiagnosticArtifact for each file
        artifacts = []
        for file_path, diagnostics in diagnostics_by_file.items():
            artifact = DiagnosticArtifact(file_path=file_path, diagnostics=diagnostics)
            artifacts.append(artifact)

        return artifacts

    async def _ensure_mypy_installed(self) -> None:
        """
        Ensure that MyPy is installed in the current environment.
        Installs it if not present.
        """
        # Check if mypy is installed

        try:
            stdout = await self.workspace.environment.execute("pip list | grep mypy")
            mypy_installed = stdout and "mypy" in stdout
        except EnvironmentExecutionError:
            # grep returns non-zero exit code if pattern not found
            mypy_installed = False
        except Exception as e:
            logger.error(f"Error checking if MyPy is installed: {str(e)}")
            raise RuntimeError(f"Error checking if MyPy is installed: {str(e)}") from e

        if not mypy_installed:
            logger.info("MyPy not found. Installing...")
            try:
                await self.workspace.environment.execute("pip install mypy")
                logger.info("MyPy installed successfully")
            except Exception as e:
                logger.error(f"Failed to install MyPy: {str(e)}")
                raise RuntimeError(f"Failed to install MyPy: {str(e)}") from e
