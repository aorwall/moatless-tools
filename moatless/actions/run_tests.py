import logging
from typing import List, Any

from pydantic import Field, PrivateAttr

from moatless.actions.action import Action
from moatless.actions.model import (
    ActionArguments,
    FewShotExample,
    Observation,
    RewardScaleEntry,
)
from moatless.file_context import FileContext
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class RunTestsArgs(ActionArguments):
    """
    Run the specified unit tests on the codebase.
    """

    thoughts: str = Field(..., description="Your reasoning on what tests to run.")
    test_files: List[str] = Field(..., description="The list of test files to run")

    class Config:
        title = "RunTests"

    @property
    def log_name(self):
        return f"RunTests({', '.join(self.test_files)})"

    def to_prompt(self):
        return f"Running tests for the following files:\n" + "\n".join(
            f"* {file}" for file in self.test_files
        )


class RunTests(Action):
    args_schema = RunTestsArgs

    max_output_tokens: int = Field(
        2000,
        description="The maximum number of tokens in the test result output message",
    )

    _repository: Repository = PrivateAttr()
    _runtime: RuntimeEnvironment = PrivateAttr()

    def __init__(
        self,
        repository: Repository | None = None,
        runtime: RuntimeEnvironment | None = None,
        **data,
    ):
        super().__init__(**data)
        self._repository = repository
        self._runtime = runtime

    def execute(
        self,
        args: RunTestsArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        """
        Run all tests found in file context or provided in args.
        """
        if file_context is None:
            raise ValueError(
                "File context must be provided to execute the run tests action."
            )

        # Separate non-existent files and directories from valid test files
        non_existent_files = []
        directories = []
        test_files = []

        for test_file in args.test_files:
            if not file_context.file_exists(test_file):
                non_existent_files.append(test_file)
            elif file_context.is_directory(test_file):
                directories.append(test_file)
            else:
                test_files.append(test_file)

        if not test_files:
            error_details = []
            if non_existent_files:
                error_details.append(
                    f"Files not found: {', '.join(non_existent_files)}"
                )
            if directories:
                error_details.append(
                    f"Directories provided instead of files: {', '.join(directories)}"
                )

            return Observation(
                message="Unable to run tests: " + "; ".join(error_details),
                properties={"test_results": [], "fail_reason": "no_test_files"},
            )

        test_files = file_context.run_tests(test_files)

        response_msg = f"Running tests for the following files:\n"
        for test_file in test_files:
            response_msg += f"* {test_file.file_path}\n"

        failure_details = file_context.get_test_failure_details()
        if failure_details:
            response_msg += f"\n{failure_details}"

        summary = f"\n{file_context.get_test_summary()}"
        response_msg += summary

        return Observation(
            message=response_msg,
            summary=summary,
        )

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
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
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
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

    @classmethod
    def model_validate(cls, obj: Any) -> "RunTests":
        if isinstance(obj, dict):
            obj = obj.copy()
            repository = obj.pop("repository")
            code_index = obj.pop("code_index")
            runtime = obj.pop("runtime")
            return cls(
                code_index=code_index, repository=repository, runtime=runtime, **obj
            )
        return super().model_validate(obj)

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return []
