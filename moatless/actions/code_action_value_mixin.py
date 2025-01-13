from typing import List

from moatless.actions.action import Action
from moatless.actions.model import RewardScaleEntry
from moatless.actions.run_tests import RunTests


class CodeActionValueMixin:
    """
    A mixin class that provides common evaluation criteria and reward scales for code modification actions that run tests.
    This mixin helps standardize how we evaluate code changes and their test results across different types of code actions.
    """

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int) -> List[str]:
        base_criteria = Action.get_evaluation_criteria(trajectory_length)
        test_criteria = RunTests.get_evaluation_criteria(trajectory_length)

        base_criteria.extend(
            [
                "Code Modification Accuracy: Check for correct identification of code spans, accuracy of changes, syntax errors, logical flaws, and unintended modifications.",
                "Git Diff Quality: Verify that changes are precise and intentional, without unintended modifications.",
                "Test Results Integration: Evaluate how well test results are interpreted and addressed.",
            ]
        )
        base_criteria.extend(test_criteria)
        return base_criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length: int) -> List[RewardScaleEntry]:
        return cls.generate_reward_scale_entries(
            [
                (
                    90,
                    100,
                    "The code change is optimal with perfect implementation, AND all tests pass successfully confirming the solution's correctness.",
                ),
                (
                    75,
                    89,
                    "The code change significantly advances the solution, AND most tests pass with only minor, easily fixable failures.",
                ),
                (
                    50,
                    74,
                    "The code change is mostly correct but has minor issues; tests have some failures that are minor or unforeseeable, with the agent showing understanding in interpreting results.",
                ),
                (
                    25,
                    49,
                    "The code change is acceptable but has noticeable issues; tests have noticeable failures that may have been foreseeable but can be addressed with effort.",
                ),
                (
                    0,
                    24,
                    "The code change has minimal impact or introduces minor negative consequences, AND tests have significant failures with minimal or incorrect interpretation.",
                ),
                (
                    -49,
                    -1,
                    "The code change is inappropriate or unhelpful; tests fail significantly with misinterpreted results. Penalize attempts to modify non-existent code elements based on severity.",
                ),
                (
                    -100,
                    -50,
                    "The code change is counterproductive causing significant setbacks. Tests fail severely with failures that could have been anticipated. Heavily penalize severe hallucinations.",
                ),
            ]
        )
