import logging
import os
import re
import time
from typing import List, Optional, Dict, Any, Tuple

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
from moatless.testing.test_output_parser import TestOutputParser
from moatless.workspace import Workspace
from pydantic import ConfigDict, Field, PrivateAttr

logger = logging.getLogger(__name__)


class RunTestsArgs(ActionArguments):
    """
    Run the specified unit tests on the codebase.
    """

    test_files: list[str] = Field(..., description="The list of test files to run")

    model_config = ConfigDict(title="RunTests")

    @property
    def log_name(self):
        return f"RunTests({', '.join(self.test_files)})"

    def to_prompt(self):
        return "Running tests for the following files:\n" + "\n".join(f"* {file}" for file in self.test_files)

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Run the tests for our user authentication module",
                action=RunTestsArgs(
                    thoughts="Running authentication tests to verify the login functionality works as expected",
                    test_files=["tests/auth/test_authentication.py", "tests/auth/test_login.py"],
                ),
            ),
            FewShotExample.create(
                user_input="After fixing the data validation bug, I need to make sure the validation tests pass",
                action=RunTestsArgs(
                    thoughts="Running validation tests to confirm that the bug fix resolved the issues with input validation",
                    test_files=["tests/validation/test_input_validator.py"],
                ),
            ),
        ]


class RunTests(Action):
    args_schema = RunTestsArgs

    max_output_tokens: int = Field(
        2000,
        description="The maximum number of tokens in the test result output message",
    )

    async def initialize(self, workspace: Workspace):
        await super().initialize(workspace)
        if not workspace.runtime:
            raise ValueError("Runtime is not available for RunTests action")

    async def execute(
        self,
        args: RunTestsArgs,
        file_context: FileContext | None = None,
    ) -> Observation:
        """
        Run all tests found in file context or provided in args.
        """
        if file_context is None:
            raise ValueError("File context must be provided to execute the run tests action.")

        # Ensure workspace is set on file_context if needed
        if not hasattr(file_context, "workspace") or file_context.workspace is None:
            if self._workspace:
                file_context.workspace = self._workspace

        non_existent_files = []
        directories = []
        test_files = []

        for test_file in args.test_files:
            test_file = test_file.split("::")[0]
            if not self._repository.file_exists(test_file) and not file_context.file_exists(test_file):
                logger.warning(f"File {test_file} does not exist in repository")
                non_existent_files.append(test_file)
            elif self._repository.is_directory(test_file):
                logger.warning(
                    f"Directory {test_file} provided instead of file, please use ListFiles to get the list of files in the directory and specify which files to run tests on"
                )
                directories.append(test_file)
            else:
                test_files.append(test_file)

        error_details = []
        if non_existent_files:
            error_details.append(f"Files not found: {', '.join(non_existent_files)}")
        if directories:
            error_details.append(f"Directories provided instead of files: {', '.join(directories)}")

        if not test_files:
            return Observation.create(
                message="Unable to run tests: " + "; ".join(error_details),
                properties={"test_results": [], "fail_reason": "no_test_files"},
            )

        patch = file_context.generate_git_patch()
        start_time = time.time()
        test_results = await self._workspace.runtime.run_tests(test_files=test_files, patch=patch)

        end_time = time.time()
        test_duration = end_time - start_time
        logger.info(f"Tests completed in {test_duration:.2f} seconds")

        test_file_objects = self.create_test_files(test_file_paths=test_files, test_results=test_results)

        file_context.add_test_files(test_file_objects)

        response_msg = ""

        failure_details = TestFile.get_test_failure_details(test_file_objects)
        if failure_details:
            response_msg += f"\n{failure_details}"

        summary = f"\n{TestFile.get_test_summary(test_file_objects)}"
        response_msg += summary

        artifact_changes = []

        logger.info(f"Running tests for {len(args.test_files)} files")
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
                description="All tests pass successfully, confirming the solution's correctness.",
            ),
            RewardScaleEntry(
                min_value=75,
                max_value=89,
                description="Most tests pass, with minor, easily fixable failures.",
            ),
            RewardScaleEntry(
                min_value=50,
                max_value=74,
                description="Tests have some failures, but they are minor or unforeseeable, and the agent shows understanding in interpreting results.",
            ),
            RewardScaleEntry(
                min_value=25,
                max_value=49,
                description="Tests have noticeable failures; some may have been foreseeable, but the agent can address them with effort.",
            ),
            RewardScaleEntry(
                min_value=0,
                max_value=24,
                description="Tests have significant failures; the agent's interpretation is minimal or incorrect.",
            ),
            RewardScaleEntry(
                min_value=-49,
                max_value=-1,
                description="Tests fail significantly; the agent misinterprets results or shows lack of progress, foreseeable failures are not addressed.",
            ),
            RewardScaleEntry(
                min_value=-100,
                max_value=-50,
                description="The action is counterproductive, demonstrating misunderstanding or causing setbacks, test failures are severe and could have been anticipated.",
            ),
        ]
