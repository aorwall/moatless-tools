from typing import ClassVar, List

from litellm import Type
from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.model import (
    ActionArguments,
    Observation,
    RewardScaleEntry,
    FewShotExample,
)
from moatless.file_context import FileContext
from moatless.workspace import Workspace


class VerifiedFinishArgs(ActionArguments):
    """Indicate that the task is fully completed and verified with new or modified tests."""

    thoughts: str = Field(
        ...,
        description="Your reasoning about why the task is complete and verified with tests.",
    )
    finish_reason: str = Field(..., description="Explain why the task is complete.")
    test_verification: str = Field(
        ...,
        description="Detailed description of how the solution was verified, including: 1) Which tests were added/modified 2) What scenarios these tests cover 3) How the tests verify the changes work correctly",
    )

    class Config:
        title = "Finish"

    def to_prompt(self):
        files_str = "\n".join(f"- {file}" for file in self.test_files_changed)
        return (
            f"Finish with reason: {self.finish_reason}\n"
            f"Test verification: {self.test_verification}\n"
            f"Modified/Created test files:\n{files_str}"
        )

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, VerifiedFinishArgs)


class VerifiedFinish(Action):
    args_schema: ClassVar[Type[ActionArguments]] = VerifiedFinishArgs

    def execute(
        self,
        args: VerifiedFinishArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ):
        return Observation(message=args.finish_reason, terminal=True)

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int) -> List[str]:
        return [
            "**Full Trajectory Review:** Evaluate the complete sequence of actions taken by the agent leading to this finish action. Assess whether the trajectory represents an efficient and logical path to the solution.",
            "**Solution Correctness and Quality:** Verify that all changes made throughout the trajectory logically address the problem statement. Ensure the changes fit contextually within the existing codebase without introducing new issues.",
            "**Testing Requirements (Critical):**",
            " * **Test Implementation:** Verify that new tests were added or existing tests were modified to cover the changes.",
            " * **Test Coverage:** Evaluate if the tests adequately cover all modified functionality, including edge cases.",
            " * **Test Quality:** Assess if the tests are well-designed and effectively verify the intended behavior.",
            " * **Test Documentation:** Check if the test verification description accurately reflects the actual test changes.",
            "**Assessment of Complete Trajectory:** Evaluate if the sequence of actions taken represents the most optimal path to the solution.",
            "**Verification of Task Completion:** Confirm that all aspects of the original issue have been addressed and properly tested.",
        ]

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return cls.generate_reward_scale_entries(
            [
                (
                    90,
                    100,
                    "The complete trajectory perfectly resolves the issue with optimal code modifications AND includes comprehensive test updates/additions that thoroughly verify all changes. Tests are well-designed and cover all relevant scenarios.",
                ),
                (
                    75,
                    89,
                    "The trajectory successfully resolves the issue AND includes proper test updates/additions. Tests adequately cover the changes, though minor improvements to test coverage might be beneficial.",
                ),
                (
                    50,
                    74,
                    "The trajectory resolves the core issue but has gaps in test coverage OR the tests don't fully verify all aspects of the changes.",
                ),
                (
                    25,
                    49,
                    "The trajectory partially resolves the issue and includes some tests, but test coverage is inadequate or tests don't properly verify the changes.",
                ),
                (
                    0,
                    24,
                    "The trajectory shows some progress but fails to properly resolve the issue OR includes minimal/ineffective test coverage.",
                ),
                (
                    -49,
                    -1,
                    "The trajectory is inappropriate with major gaps in implementation and testing. Tests are missing or ineffective.",
                ),
                (
                    -100,
                    -50,
                    "The trajectory is entirely incorrect AND lacks proper test verification. The finish action is entirely premature.",
                ),
            ]
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Add input validation to the process_order function",
                action=VerifiedFinishArgs(
                    thoughts="I've added input validation and comprehensive tests to verify all validation cases",
                    finish_reason="Added robust input validation to process_order function with proper error handling",
                    test_verification=(
                        "Added new test file test_order_validation.py with comprehensive test cases:\n"
                        "1. test_valid_order: Verifies successful order processing with valid input\n"
                        "2. test_invalid_quantity: Tests rejection of negative and zero quantities\n"
                        "3. test_invalid_price: Verifies handling of invalid price formats\n"
                        "4. test_missing_fields: Ensures proper error messages for missing required fields\n"
                        "All tests pass and cover both success and error scenarios."
                    ),
                ),
            ),
            FewShotExample.create(
                user_input="Fix the bug in the date parsing logic",
                action=VerifiedFinishArgs(
                    thoughts="I've fixed the date parsing bug and added tests to prevent regression",
                    finish_reason="Fixed date parsing bug that was incorrectly handling timezone conversions",
                    test_verification=(
                        "Modified tests/test_date_utils.py to add new test cases:\n"
                        "1. Added test_timezone_conversion to verify correct timezone handling\n"
                        "2. Extended test_parse_date with edge cases:\n"
                        "   - Dates crossing DST boundaries\n"
                        "   - Various timezone formats (UTC, GMT, named zones)\n"
                        "   - Invalid date formats\n"
                        "3. Added test_date_validation for boundary conditions\n"
                        "All tests pass, confirming the bug is fixed and won't regress."
                    ),
                ),
            ),
        ]
