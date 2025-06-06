from typing import ClassVar

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
    RewardScaleEntry,
)
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext


class FinishArgs(ActionArguments):
    """Indicate that the task is fully completed."""

    finish_reason: str = Field(
        ...,
        description="Explain why the task is complete.",
    )
    is_terminal: bool = Field(default=True, description="Whether the action will finish the flow")

    model_config = ConfigDict(title="Finish")

    def to_prompt(self):
        return f"Finish with reason: {self.finish_reason}"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, FinishArgs)

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return []


class Finish(Action):
    args_schema: ClassVar[type[ActionArguments]] = FinishArgs

    async def execute(self, args: FinishArgs, file_context: FileContext | None = None):
        return Observation.create(message="Finished", terminal=True)

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int) -> list[str]:
        return [
            "**Full Trajectory Review:** Evaluate the complete sequence of actions taken by the agent leading to this finish action. Assess whether the trajectory represents an efficient and logical path to the solution.",
            "**Solution Correctness and Quality:** Verify that all changes made throughout the trajectory logically address the problem statement. Ensure the changes fit contextually within the existing codebase without introducing new issues. Confirm syntactic correctness and that there are no syntax errors or typos.",
            "**Testing Requirements (Critical):**",
            " * **Mandatory Test Updates:** The trajectory MUST include actions that either update existing tests or add new tests to verify the solution. A score of 75 or higher CANNOT be given without proper test coverage.",
            " * **Test Coverage Quality:** Evaluate whether the tests added or modified throughout the trajectory adequately cover the changes, including edge cases and error conditions.",
            " * **Test Execution Results:** Verify that all tests are passing after the complete sequence of changes.",
            "**Assessment of Complete Trajectory:** Evaluate if the sequence of actions taken represents the most optimal path to the solution, or if unnecessary steps were taken.",
            "**Verification of Task Completion:** Confirm that all aspects of the original issue have been addressed through the sequence of actions, including implementation, testing, and documentation where applicable.",
        ]

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> list[RewardScaleEntry]:
        return cls.generate_reward_scale_entries(
            [
                (
                    90,
                    100,
                    "The complete trajectory perfectly resolves the issue with optimal code modifications AND includes comprehensive test updates/additions. All tests pass and cover all relevant scenarios. No further improvements needed.",
                ),
                (
                    75,
                    89,
                    "The trajectory successfully resolves the issue AND includes proper test updates/additions. All tests pass, though minor improvements to test coverage might be beneficial. REQUIRES test modifications to qualify for this range.",
                ),
                (
                    50,
                    74,
                    "The trajectory resolves the core issue but has gaps in test coverage OR the solution path wasn't optimal. May include cases where implementation is correct but tests were not adequately updated.",
                ),
                (
                    25,
                    49,
                    "The trajectory partially resolves the issue but lacks proper test coverage AND has other significant gaps such as incomplete implementation or inefficient solution path.",
                ),
                (
                    0,
                    24,
                    "The trajectory shows some progress but fails to properly resolve the issue AND lacks necessary test updates. The finish action was premature.",
                ),
                (
                    -49,
                    -1,
                    "The trajectory is inappropriate with major gaps in both implementation and testing. The finish action indicates a clear misunderstanding of the requirements.",
                ),
                (
                    -100,
                    -50,
                    "The trajectory is entirely incorrect, potentially introducing new issues, and completely lacks test coverage. The finish action is entirely premature.",
                ),
            ]
        )

    @classmethod
    def get_value_function_prompt(cls) -> str:
        return """Your role is to evaluate the executed action of the search tree that our AI agents are traversing, with the goal of ensuring that a complete and verified solution is in place. The agent believes that it has finished solving the programming issue."""
