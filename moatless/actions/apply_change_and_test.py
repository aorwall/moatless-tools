import logging
from enum import Enum
from typing import List, Any

from pydantic import Field, PrivateAttr, model_validator

from moatless.actions import RequestCodeChange, RunTests
from moatless.actions.model import ActionArguments, Observation
from moatless.actions.run_tests import RunTestsArgs
from moatless.completion.completion import CompletionModel
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    addition = "addition"
    modification = "modification"
    deletion = "deletion"


class RequestCodeChangeArgs(ActionArguments):
    """
    Apply a code change through an AI agent. This action instructs an AI assistant to
    modify code based on provided instructions and pseudo-code. The AI will analyze the existing code within
    the specified line range and apply changes while maintaining proper syntax, indentation, and context.

    After the change has been applied, relevant tests will be run.
    """

    file_path: str = Field(..., description="The file path of the code to be updated.")
    instructions: str = Field(
        ...,
        description="Natural language instructions for the AI assistant describing the required code changes.",
    )
    pseudo_code: str = Field(
        ...,
        description="Example code snippet illustrating the desired changes. The AI will use this as a reference for implementing the modifications.",
    )
    change_type: ChangeType = Field(
        ...,
        description="Type of change to perform: 'addition' (insert new code), 'modification' (update existing code), or 'deletion' (remove code).",
    )
    start_line: int = Field(
        ...,
        description="The line number where the code change should begin. For additions, specifies the insertion point.",
    )
    end_line: int = Field(
        ...,
        description="The line number where the code change should end. For additions, specifies the insertion point.",
    )

    class Config:
        title = "RequestCodeChange"

    @model_validator(mode="before")
    @classmethod
    def set_missing_end_line(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("end_line"):
                data["end_line"] = data["start_line"]

        return data

    def equals(self, other: "RequestCodeChangeArgs") -> bool:
        if not isinstance(other, RequestCodeChangeArgs):
            return False

        return (
            self.file_path == other.file_path
            and self.pseudo_code == other.pseudo_code
            and self.change_type == other.change_type
            and self.start_line == other.start_line
            and self.end_line == other.end_line
        )


class ApplyCodeChangeAndTest(RequestCodeChange):
    _runtime: RuntimeEnvironment = PrivateAttr()
    _code_index: CodeIndex = PrivateAttr()

    def __init__(
        self,
        repository: Repository | None = None,
        completion_model: CompletionModel | None = None,
        runtime: RuntimeEnvironment | None = None,
        code_index: CodeIndex | None = None,
        **data,
    ):
        super().__init__(
            repository=repository, completion_model=completion_model, **data
        )
        self._runtime = runtime
        self._code_index = code_index

    def execute(
        self, args: RequestCodeChangeArgs, file_context: FileContext
    ) -> Observation:
        observation = super().execute(args, file_context)

        if not observation.properties or not observation.properties.get("diff"):
            return observation

        run_tests = RunTests(
            repository=self._repository,
            runtime=self._runtime,
            code_index=self._code_index,
        )
        test_observation = run_tests.execute(
            RunTestsArgs(
                scratch_pad=args.scratch_pad,
                test_files=[args.file_path],
            ),
            file_context,
        )

        observation.properties.update(test_observation.properties)
        observation.message += "\n\n" + test_observation.message

        return observation

