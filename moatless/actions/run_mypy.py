import json
import logging
from typing import List, Optional

from moatless.workspace import Workspace
from pydantic import Field, ConfigDict

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
    RewardScaleEntry,
)
from moatless.artifacts.diagnostics.diagnostic import DiagnosticHandler, DiagnosticArtifact, DiagnosticSeverity, Range, Position
from moatless.artifacts.artifact import ArtifactReference
from moatless.file_context import FileContext
from moatless.environment.base import BaseEnvironment, EnvironmentExecutionError

logger = logging.getLogger(__name__)


class RunMyPyArgs(ActionArguments):
    """
    Run MyPy type checking on the specified Python files.
    """

    thoughts: str = Field(..., description="Your reasoning on what files to check with MyPy.")
    files: List[str] = Field(..., description="The list of Python files or directories to check with MyPy")

    model_config = ConfigDict(title="RunMyPy")

    @property
    def log_name(self):
        return f"RunMyPy({', '.join(self.files)})"

    def to_prompt(self):
        return f"Running MyPy type checking for the following files:\n" + "\n".join(f"* {file}" for file in self.files)


class RunMyPy(Action):
    """
    An action for running MyPy type checking on Python code and generating diagnostic artifacts.
    """
    args_schema = RunMyPyArgs

    async def initialize(self, workspace: Workspace):
        await super().initialize(workspace)        
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
            artifacts = self._parse_mypy_json_output(stdout)
        except EnvironmentExecutionError as e:
            logger.warning(f"MyPy command failed with return code {e.return_code}: {e.stderr}")
            # If MyPy returned JSON output in stderr, try to parse it
            if e.stderr and e.stderr.strip().startswith('{'):
                try:
                    artifacts = self._parse_mypy_json_output(e.stderr)
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
        
        # Save each artifact
        for artifact in artifacts:
            created_artifact = diagnostic_handler.create(artifact)
            await diagnostic_handler.persist(created_artifact.id)
        
        # Generate a summary message
        if not artifacts:
            return "MyPy found no type issues."
        
        # Group diagnostics by file
        diagnostics_by_file = {}
        for artifact in artifacts:
            file_path = artifact.references[0].id if artifact.references else "unknown"
            if file_path not in diagnostics_by_file:
                diagnostics_by_file[file_path] = []
            diagnostics_by_file[file_path].append(artifact)
        
        # Format the summary message
        summary = []
        summary.append(f"MyPy found {len(artifacts)} issues:")
        
        for file_path, file_artifacts in diagnostics_by_file.items():
            summary.append(f"\n## {file_path}")
            for artifact in file_artifacts:
                line = artifact.range.start.line
                col = artifact.range.start.column or 0
                severity = artifact.severity.value.upper()
                message = artifact.message
                summary.append(f"- {severity} at line {line}, col {col}: {message}")
        
        return "\n".join(summary)

    def _parse_mypy_json_output(self, output: str) -> List[DiagnosticArtifact]:
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
                    "note": DiagnosticSeverity.INFO
                }
                severity = severity_map.get(report.get("severity", "error"), DiagnosticSeverity.ERROR)
                
                line_num = report.get("line", 1)
                column_num = report.get("column", 1)
                
                range_obj = Range(
                    start=Position(line=line_num, column=column_num),
                )
                
                file_reference = ArtifactReference(
                    id=file_path,
                    type="file"
                )
                
                # Generate a unique ID for the diagnostic
                diagnostic_id = f"mypy-{file_path}-{line_num}-{column_num}"
                
                diagnostic = DiagnosticArtifact(
                    id=diagnostic_id,
                    severity=severity,
                    range=range_obj,
                    source="mypy",
                    message=report.get("message", ""),
                    code=report.get("code", "unknown"),
                    references=[file_reference]
                )
                
                artifacts.append(diagnostic)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse MyPy JSON output: {e}\nLine: {line}")
                continue
        
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
