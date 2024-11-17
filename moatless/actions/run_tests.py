import logging
from typing import List, Any

from pydantic import Field, PrivateAttr

from moatless.actions.action import Action
from moatless.actions.model import (
    ActionArguments,
    FewShotExample,
    Observation
)
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex, is_test
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment, TestResult, TestStatus
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class RunTestsArgs(ActionArguments):
    """
    Run the specified unit tests on the codebase.
    """

    scratch_pad: str = Field(..., description="Your reasoning on what tests to run.")
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

    _code_index: CodeIndex = PrivateAttr()
    _repository: Repository = PrivateAttr()
    _runtime: RuntimeEnvironment = PrivateAttr()

    def __init__(
        self,
        code_index: CodeIndex | None = None,
        repository: Repository | None = None,
        runtime: RuntimeEnvironment | None = None,
        **data,
    ):
        super().__init__(**data)
        self._repository = repository
        self._code_index = code_index
        self._runtime = runtime

    def execute(
        self, args: RunTestsArgs, file_context: FileContext | None = None
    ) -> Observation:
        """
        Run all tests found in file context or provided in args.
        """
        if file_context is None:
            raise ValueError(
                "File context must be provided to execute the run tests action."
            )

        test_files = [
            test_file
            for test_file in args.test_files
            if file_context.get_file(test_file) is not None and is_test(test_file)
        ]

        if not test_files:
            file_paths = args.test_files
            if not file_paths:
                file_paths = [file.file_path for file in file_context.files]

            for file_path in file_paths:
                search_results = self._code_index.find_test_files(
                    file_path, query=file_path, max_results=2, max_spans=2
                )

                for search_result in search_results:
                    test_files.append(search_result.file_path)

        for test_file in test_files:
            if not file_context.has_file(test_file):
                logger.info(f"Adding test file: {test_file} to context")
                file_context.add_file(test_file, add_extra=False)

        test_files = [
            file.file_path for file in file_context.files if is_test(file.file_path)
        ]

        logger.info(f"Running tests: {test_files}")
        patch = file_context.generate_git_patch()
        test_results = self._runtime.run_tests(patch, test_files)
        failing_tests = [
            issue
            for issue in test_results
            if issue.status in [TestStatus.FAILED, TestStatus.ERROR]
        ]

        tests_with_output = [
            issue for issue in failing_tests if issue.message and issue.file_path
        ]

        if failing_tests:
            logger.info(
                f"{len(failing_tests)} out of {len(test_results)} tests failed. "
                f"Include spans for {len(tests_with_output)} tests with output."
            )

            # Add failing tests to context.
            failed_test_spans_by_file_path: dict = {}
            for test_result in tests_with_output:
                if test_result.file_path:
                    failed_test_spans_by_file_path.setdefault(
                        test_result.file_path, []
                    ).append(test_result.span_id)

            for test_file in test_files:
                failed_span_ids = failed_test_spans_by_file_path.get(test_file)
                if failed_span_ids:
                    test_context_file = file_context.get_file(test_file)
                    test_context_file.add_spans(failed_span_ids)

        return self.create_output(test_results, test_files)

    def create_output(
        self, test_results: List[TestResult], test_files: List[str]
    ) -> Observation:
        if not test_results:
            return Observation(
                message="No tests were run",
                properties={"test_results": [], "fail_reason": "no_tests"},
            )

        failure_count = sum(
            1 for issue in test_results if issue.status == TestStatus.FAILED
        )
        error_count = sum(
            1 for issue in test_results if issue.status == TestStatus.ERROR
        )

        passed_count = len(test_results) - failure_count - error_count

        test_result_strings = []
        token_count = 0

        for i, test_result in enumerate(test_results):
            if not test_result.message or test_result.status not in [
                TestStatus.FAILED,
                TestStatus.ERROR,
            ]:
                continue

            attributes = ""
            if test_result.file_path:
                attributes += f"{test_result.file_path}"

                if test_result.span_id:
                    attributes += f" {test_result.span_id}"

                if test_result.line:
                    attributes += f", line: {test_result.line}"

            test_output = f"* {test_result.status.value} {attributes}>\n```\n{test_result.message}\n```\n"
            test_output_tokens = count_tokens(test_output)
            if token_count + test_output_tokens > self.max_output_tokens:
                logger.warning(
                    f"Test output message exceeds max token limit ({self.max_output_tokens})."
                )
                break

            token_count += test_output_tokens

            test_result_strings.append(test_output)

        response_msg = f"Running {len(test_results)} tests in the following files:"
        for test_file in test_files:
            response_msg += f"\n * {test_file}"

        if test_result_strings:
            response_msg += "\n\n"
            response_msg += "\n".join(test_result_strings)

        response_msg += (
            f"\n\n{passed_count} passed. {failure_count} failed. {error_count} errors."
        )

        result_dicts = [result.model_dump() for result in test_results]

        return Observation(
            message=response_msg,
            properties={"test_results": result_dicts},
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Run the tests for our authentication module to verify the recent changes to the login flow",
                action=RunTestsArgs(
                    scratch_pad="We need to run the authentication tests to ensure the login flow changes haven't introduced any regressions.",
                    test_files=[
                        "tests/auth/test_authentication.py",
                        "tests/auth/test_login.py",
                    ],
                ),
            )
        ]
