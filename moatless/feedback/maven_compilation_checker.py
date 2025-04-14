import logging
import re
from typing import List, Optional

from moatless.feedback.base import BaseFeedbackGenerator
from moatless.node import FeedbackData, Node
from moatless.workspace import Workspace
from moatless.testing.java.maven_parser import MavenParser
from moatless.testing.schema import TestResult, TestStatus

logger = logging.getLogger(__name__)


class MavenCompilationChecker(BaseFeedbackGenerator):
    """
    Feedback generator that checks if a Maven project compiles successfully,
    and provides feedback about compilation issues if they exist.
    """

    maven_binary: str = "mvn"
    _workspace: Optional[Workspace] = None
    _parser: Optional[MavenParser] = None  # Fixed optional type

    async def initialize(self, workspace: Workspace):
        self._workspace = workspace
        self._parser = MavenParser()  # Initialize the Maven parser

        # Make sure environment is available
        if not hasattr(workspace, "environment") or workspace.environment is None:
            raise ValueError("Environment is required to check Maven compilation")

        # Check Maven availability
        try:
            maven_version = await self._workspace.environment.execute(
                f"{self.maven_binary} --version", fail_on_error=True
            )
            logger.info(f"Maven version: {maven_version.splitlines()[0] if maven_version else 'Unknown'}")
        except Exception as e:
            error_msg = f"Maven does not appear to be installed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def generate_feedback(self, node: Node) -> FeedbackData | None:
        """
        Generate feedback based on Maven compilation results.

        Args:
            node: The node to generate feedback for

        Returns:
            FeedbackData with compilation issues or None if compilation succeeds

        Raises:
            RuntimeError: If Maven is not installed or cannot be executed
        """
        if self._workspace is None:
            logger.warning("Workspace is not set for MavenCompilationChecker")
            return None

        if self._workspace.environment is None:
            logger.warning("Environment is not set in workspace")
            return None

        # Run 'mvn compile' to check if the project compiles
        logger.info("Running 'mvn compile' to check if the project compiles")
        try:
            compile_output = await self._workspace.environment.execute(f"{self.maven_binary} compile")
            compile_success = "BUILD SUCCESS" in compile_output

            if compile_success:
                logger.info("Maven compilation succeeded")
                return None  # No feedback needed for successful compilation

            # Parse compilation errors using the MavenParser
            feedback = self._parse_compilation_errors(compile_output)
            logger.warning("Maven compilation failed")

            return FeedbackData(feedback=feedback, analysis=None, completion=None, suggested_node_id=None)

        except Exception as e:
            error_msg = f"Failed to execute Maven compilation: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _parse_compilation_errors(self, compile_output: str) -> str:
        """
        Parse Maven compilation output and extract errors and warnings using MavenParser.

        Args:
            compile_output: The output from Maven compilation

        Returns:
            A formatted string with compilation errors and warnings
        """
        if not self._parser:
            self._parser = MavenParser()

        # Use MavenParser to parse the compile output
        test_results = self._parser.parse_test_output(compile_output)

        # Format the feedback
        feedback = "Maven compilation failed. Please fix the following issues:\n\n"

        # Group errors and warnings
        errors = [result for result in test_results if result.status == TestStatus.ERROR]

        # Also extract detailed error messages from the original output
        error_details = self._extract_error_details(compile_output)

        # Extract warnings from the output (MavenParser doesn't explicitly track warnings)
        warning_lines = []
        for line in compile_output.splitlines():
            if line.strip().startswith("[WARNING]") and "COMPILATION WARNING" not in line:
                warning_lines.append(line)

        if errors:
            feedback += "## Compilation Errors\n"
            for error in errors:
                # Extract line number from error name if available
                line_match = re.search(r"at line (\d+)", error.name) if error.name else None
                line_num = line_match.group(1) if line_match else "unknown"

                if error.file_path:
                    # Find matching detailed error message if available
                    detailed_message = ""
                    for detail in error_details:
                        if error.file_path.endswith(detail["file"]) and str(line_num) == str(detail["line"]):
                            detailed_message = f": {detail['message']}"
                            break

                    feedback += f"- **{error.file_path}** (line {line_num}){detailed_message}\n"
                else:
                    feedback += f"- {error.name}\n"
                    if error.failure_output:
                        feedback += f"  {error.failure_output.splitlines()[0]}\n"

            # Also include raw error messages to ensure we have the complete error details
            # This ensures we don't lose the specific error messaging from Maven
            feedback += "\n### Detailed Error Messages:\n"
            error_pattern = r"\[ERROR\] (.+\.java):\[\d+(?:,\d+)?\] (.+)"
            for line in compile_output.splitlines():
                if "[ERROR]" in line and ".java" in line and ":" in line:
                    match = re.search(error_pattern, line)
                    if match:
                        feedback += f"{line}\n"

            feedback += "\n"

        if warning_lines:
            feedback += "## Compilation Warnings\n"
            for warning in warning_lines:
                feedback += f"{warning}\n"
            feedback += "\n"

        feedback += "Please fix these issues before proceeding."

        return feedback

    def _extract_error_details(self, compile_output: str) -> List[dict]:
        """
        Extract detailed error information from the compilation output.
        This is now a compatibility wrapper around MavenParser.

        Args:
            compile_output: The output from Maven compilation

        Returns:
            A list of dictionaries with error details
        """
        error_details = []

        if not self._parser:
            self._parser = MavenParser()

        # Use MavenParser to parse compilation errors
        test_results = self._parser.parse_test_output(compile_output)

        # Convert TestResult objects to error_details format for compatibility
        for result in test_results:
            if result.status == TestStatus.ERROR and result.file_path:
                # Extract file name from path
                file_name = result.file_path.split("/")[-1]

                # Extract line number from name if available
                line_num = "0"
                if result.name:
                    match = re.search(r"at line (\d+)", result.name)
                    if match:
                        line_num = match.group(1)

                error_details.append(
                    {
                        "file": file_name,
                        "full_path": result.file_path,
                        "line": line_num,
                        "message": result.failure_output or result.name or "",
                    }
                )

        return error_details
