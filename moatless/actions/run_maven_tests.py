import logging
import os
import time
from pathlib import Path

from moatless.environment.base import EnvironmentExecutionError
from moatless.testing.java.maven_parser import MavenParser
from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
    RewardScaleEntry,
)
from moatless.artifacts.artifact import ArtifactChange
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from moatless.testing.schema import TestFile
from moatless.testing.schema import TestResult, TestStatus
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class RunMavenTestsArgs(ActionArguments):
    """
    Run the specified Maven tests in the codebase.

    Note: This action requires explicit paths to test files.
    - Only full test file paths are supported (e.g., "src/test/java/com/example/MyTest.java")
    - Directory paths, file patterns, and individual test methods are NOT supported
    - Non-existent test files will be reported as errors
    """

    test_files: list[str] = Field(
        ..., description="The list of explicit test file paths to run (must be file paths, not directories or patterns)"
    )

    model_config = ConfigDict(title="RunMavenTests")

    @property
    def log_name(self):
        return f"RunMavenTests({', '.join(self.test_files)})"

    def to_prompt(self):
        return "Running Maven tests for the following files:\n" + "\n".join(f"* {file}" for file in self.test_files)

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Run the Maven tests for our user authentication module",
                action=RunMavenTestsArgs(
                    thoughts="Running Maven tests to verify the authentication functionality works as expected",
                    test_files=[
                        "src/test/java/com/example/auth/AuthenticationTest.java",
                        "src/test/java/com/example/auth/LoginTest.java",
                    ],
                ),
            ),
            FewShotExample.create(
                user_input="After fixing the data validation bug, I need to make sure the validation tests pass",
                action=RunMavenTestsArgs(
                    thoughts="Running Maven tests to confirm that the bug fix resolved the validation issues",
                    test_files=["src/test/java/com/example/validation/InputValidatorTest.java"],
                ),
            ),
        ]


class RunMavenTests(Action):
    """
    Runs specified Maven test files in the codebase.

    Important requirements:
    - Only accepts explicit, complete file paths to test files
    - Does not support directories, file patterns, or individual test methods
    - Will validate that test files exist before attempting to run them
    - Will report errors for non-existent files or if directories are specified
    - Only runs when file_context.shadow_mode is False (working on actual files)
    """

    args_schema = RunMavenTestsArgs

    max_output_tokens: int = Field(
        2000,
        description="The maximum number of tokens in the test result output message",
    )

    # Maven binary to use
    maven_binary: str = "mvn"

    async def initialize(self, workspace: Workspace):
        await super().initialize(workspace)

        # Make sure environment is available
        if not hasattr(workspace, "environment") or workspace.environment is None:
            raise ValueError("Environment is required to run Maven tests")

        # Make sure maven is installed
        maven_version = await workspace.environment.execute(f"{self.maven_binary} --version", fail_on_error=True)
        if not maven_version:
            raise ValueError("Maven is not installed")
        
        

    async def execute(
        self,
        args: RunMavenTestsArgs,
        file_context: FileContext | None = None,
    ) -> Observation:
        """
        Run Maven tests for the specified files.
        """
        if file_context is None:
            raise ValueError("File context must be provided to execute the run tests action.")

        # Check if we're in shadow mode - if so, we can't run tests
        if hasattr(file_context, "shadow_mode") and file_context.shadow_mode:
            return Observation.create(
                message="Maven tests can only be run when file_context.shadow_mode is False (working on actual files, not in-memory).",
                properties={"test_results": [], "fail_reason": "shadow_mode_enabled"},
            )

        # Ensure workspace is set
        if self._workspace is None:
            raise ValueError("Workspace is not set for RunMavenTests action")

        if self._workspace.environment is None:
            raise ValueError("Environment is not set in workspace")
        
        maven_parser = MavenParser()
        
        # Check which test files exist
        non_existent_files = []
        directories = []
        test_files = []

        for test_file in args.test_files:
            # Check if file exists using environment
            try:
                # Try to read the file - if it succeeds, the file exists
                await self._workspace.environment.read_file(test_file)

                # Check if it's a directory (this is approximate)
                is_dir = False
                try:
                    output = await self._workspace.environment.execute(
                        f"[ -d {test_file} ] && echo 'true' || echo 'false'"
                    )
                    is_dir = output.strip() == "true"
                except Exception:
                    # If command fails, assume it's not a directory
                    pass

                if is_dir:
                    logger.warning(
                        f"Directory {test_file} provided instead of file, please use ListFiles to get the list of files in the directory and specify which files to run tests on"
                    )
                    directories.append(test_file)
                else:
                    test_files.append(test_file)
            except Exception:
                logger.warning(f"File {test_file} does not exist in repository")
                non_existent_files.append(test_file)

        error_details = []
        if non_existent_files:
            error_details.append(f"Files not found: {', '.join(non_existent_files)}")
        if directories:
            error_details.append(f"Directories provided instead of files: {', '.join(directories)}")

        if not test_files:
            return Observation.create(
                message="Unable to run Maven tests: " + "; ".join(error_details),
                properties={"test_results": [], "fail_reason": "no_test_files"},
            )

        start_time = time.time()
        # First run 'mvn compile' to verify the project compiles
        logger.info("Running 'mvn compile' to verify the project compiles")
        try:
            compile_output = await self._workspace.environment.execute(f"{self.maven_binary} compile test-compile")
            compile_success = "BUILD SUCCESS" in compile_output

            if not compile_success:
                logger.warning("Maven compilation failed")

                compile_results = maven_parser.parse_test_output(compile_output)

                if not compile_results:
                    # If no specific errors were parsed, create a generic error
                    compile_results = [
                        TestResult(
                            status=TestStatus.ERROR,
                            name="Maven compilation failed",
                            file_path=None,
                            failure_output=compile_output,
                        )
                    ]

                # Create a response with the compilation errors
                response_msg = "Maven compilation failed. Tests cannot be run until compilation errors are fixed.\n\n"
                response_msg += "Compilation errors:\n"
                for result in compile_results:
                    if result.name:
                        response_msg += f"- {result.name}\n"

                if any(result.failure_output for result in compile_results):
                    response_msg += "\nDetails:\n"
                    for result in compile_results:
                        if result.failure_output:
                            response_msg += f"{result.failure_output}\n"

                return Observation.create(
                    message=response_msg,
                    properties={"test_results": [], "fail_reason": "compilation_failed"},
                )

        except Exception as e:
            logger.warning(f"Maven compilation check failed: {str(e)}")
            # Continue with tests even if compilation check fails

        # Run tests
        test_results = []

        # Run tests directly on the current state of the files
        if test_files:
            for test_file in test_files:
                # Build test command
                test_command = self._build_test_command(test_file)
                logger.info(f"Test command: {test_command}")

                try:
                    # Execute Maven command - capture both stdout and stderr
                    try:
                        stdout = await self._workspace.environment.execute(test_command)
                        stderr = ""  # We don't have direct access to stderr through the environment interface
                    except Exception as cmd_error:
                        # If the command fails, capture the error message
                        logger.warning(f"Maven command failed: {str(cmd_error)}")
                        stdout = f"Command failed with return code 1: {test_command}"
                        stderr = str(cmd_error)

                    # Combine stdout and stderr for better error detection
                    output = stdout
                    if stderr:
                        if output:
                            output += "\n" + stderr
                        else:
                            output = stderr

                    logger.debug("Test completed")

                    # Parse test results
                    file_results = maven_parser.parse_test_output(output, test_file)

                    # Set file path if not set by parser
                    for result in file_results:
                        if not result.file_path:
                            result.file_path = test_file

                    test_results.extend(file_results)

                except Exception as e:
                    logger.warning(f"Test failed with error: {str(e)}")

                    # Create a failure result
                    error_result = TestResult(
                        status=TestStatus.ERROR,
                        name=f"Failed to run test: {test_file}",
                        file_path=test_file,
                        failure_output=str(e),
                    )
                    test_results.append(error_result)
        else:
            # Run all tests
            test_command = f"{self.maven_binary} test"
            logger.info(f"Running all tests with command: {test_command}")

            try:
                # Execute Maven command
                try:
                    stdout = await self._workspace.environment.execute(test_command)
                except EnvironmentExecutionError as cmd_error:
                    # If the command fails, capture the error message
                    logger.exception(f"Maven command failed: {test_command}")
                    stdout = cmd_error.stderr

                test_results = maven_parser.parse_test_output(stdout, repo_name)
            except Exception as e:
                logger.warning(f"Tests failed with error: {str(e)}")
                error_result = TestResult(
                    status=TestStatus.ERROR, name="Failed to run all tests", failure_output=str(e)
                )
                test_results.append(error_result)

        end_time = time.time()
        test_duration = end_time - start_time
        logger.info(f"Maven tests completed in {test_duration:.2f} seconds")

        # Create TestFile objects
        test_file_objects = self.create_test_files(test_file_paths=test_files, test_results=test_results)
        file_context.add_test_files(test_file_objects)

        # Build response message
        response_msg = ""
        failure_details = TestFile.get_test_failure_details(test_file_objects)
        if failure_details:
            response_msg += f"\n{failure_details}"

        summary = f"\n{TestFile.get_test_summary(test_file_objects)}"
        response_msg += summary

        # Create artifact changes
        artifact_changes = []
        for test_file in test_file_objects:
            if test_file.file_path not in args.test_files:
                continue

            # Calculate test counts for this file
            passed_count = sum(1 for r in test_file.test_results if r.status == TestStatus.PASSED)
            failed_count = sum(1 for r in test_file.test_results if r.status == TestStatus.FAILED)
            error_count = sum(1 for r in test_file.test_results if r.status == TestStatus.ERROR)
            skipped_count = sum(1 for r in test_file.test_results if r.status == TestStatus.SKIPPED)

            artifact_changes.append(
                ArtifactChange(
                    artifact_id=test_file.file_path,
                    artifact_type="test",
                    change_type="updated",
                    properties={
                        "passed": passed_count,
                        "failed": failed_count,
                        "errors": error_count,
                        "skipped": skipped_count,
                        "total": len(test_file.test_results),
                    },
                    actor="assistant",
                )
            )

        if error_details:
            response_msg += f"\n{error_details}"

        return Observation(message=response_msg, summary=summary, artifact_changes=artifact_changes)  # type: ignore

    def _build_test_command(self, test_file: str) -> str:
        """
        Build the Maven test command for a specific test file.

        Args:
            test_file: Path to the test file

        Returns:
            The Maven test command
        """
        # Extract class name from file path
        if test_file.endswith(".java"):
            # Convert file path to class name
            # src/test/java/com/example/MyTest.java -> com.example.MyTest
            file_without_ext = test_file[:-5]  # Remove ".java"

            # Handle common source directories
            for prefix in ["src/test/java/", "src/main/java/", "src/it/java/"]:
                if file_without_ext.startswith(prefix):
                    file_without_ext = file_without_ext[len(prefix) :]
                    break

            # Convert / to . for package name
            class_name = file_without_ext.replace("/", ".")

            # Build test command for specific class
            return f"{self.maven_binary} test -Dtest={class_name}"

        # If file path doesn't end with .java, it might be a module or package
        return f"{self.maven_binary} test -Dtest={test_file}"

    def create_test_files(self, test_file_paths: list[str], test_results: list[TestResult]) -> list[TestFile]:
        test_files = []

        # Parse the output and update test_file objects
        for test_file_path in test_file_paths:
            test_file = TestFile(file_path=test_file_path)

            # Filter test results for this file
            file_test_results = [
                result for result in test_results if result.file_path and result.file_path == test_file.file_path
            ]

            test_file.test_results = file_test_results
            if file_test_results:
                logger.info(
                    f"Test results for {test_file.file_path}: "
                    f"{sum(1 for r in file_test_results if r.status == TestStatus.PASSED)} passed, "
                    f"{sum(1 for r in file_test_results if r.status == TestStatus.FAILED)} failed, "
                    f"{sum(1 for r in file_test_results if r.status == TestStatus.ERROR)} errors"
                )
            else:
                logger.warning(f"No test results found for {test_file.file_path}")

            test_files.append(test_file)

        return test_files

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> list[str]:
        criteria = [
            "Test Result Evaluation: Analyze test results in conjunction with the proposed code changes.",
            "Test Failures Categorization: Differentiate between minor, foreseeable, and unforeseeable failures.",
            " * Minor, Easily Fixable Failures: Lightly penalize or treat as neutral.",
            " * Foreseeable Failures: Penalize appropriately based on the complexity of the fix.",
            " * Unforeseeable Failures: Penalize very lightly or reward for providing new insights.",
            "Impact of Failures: Consider the overall impact of test failures on the solution's viability.",
            "Iterative Improvement: Encourage fixing minor issues in subsequent iterations.",
            "Explanation Requirement: In your explanation, describe any test failures, their likely causes, and suggest potential next steps.",
        ]
        return criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> list[RewardScaleEntry]:
        return [
            RewardScaleEntry(
                min_value=90,
                max_value=100,
                description="All Maven tests pass successfully, confirming the solution's correctness.",
            ),
            RewardScaleEntry(
                min_value=75,
                max_value=89,
                description="Most Maven tests pass, with minor, easily fixable failures.",
            ),
            RewardScaleEntry(
                min_value=50,
                max_value=74,
                description="Maven tests have some failures, but they are minor or unforeseeable, and the agent shows understanding in interpreting results.",
            ),
            RewardScaleEntry(
                min_value=25,
                max_value=49,
                description="Maven tests have noticeable failures; some may have been foreseeable, but the agent can address them with effort.",
            ),
            RewardScaleEntry(
                min_value=0,
                max_value=24,
                description="Maven tests have significant failures; the agent's interpretation is minimal or incorrect.",
            ),
            RewardScaleEntry(
                min_value=-49,
                max_value=-1,
                description="Maven tests fail significantly; the agent misinterprets results or shows lack of progress, foreseeable failures are not addressed.",
            ),
            RewardScaleEntry(
                min_value=-100,
                max_value=-50,
                description="The action is counterproductive, demonstrating misunderstanding or causing setbacks, Maven test failures are severe and could have been anticipated.",
            ),
        ]
