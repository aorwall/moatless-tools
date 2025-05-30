import logging
from pathlib import Path
from typing import List, Optional

from opentelemetry import trace

from moatless.actions.schema import Observation
from moatless.actions.action import Action
from moatless.file_context import FileContext
from moatless.actions.schema import RewardScaleEntry

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


class CodeModificationMixin:
    """
    A mixin that provides common functionality for actions that modify code files.
    This includes path normalization, file validation, test running, and observation handling.
    """

    def normalize_path(self, file_path: str) -> str:
        """Normalize file path by removing /repo and leading /"""
        if file_path.startswith("/repo"):
            file_path = file_path[5:]
        if file_path.startswith("/"):
            file_path = file_path[1:]
        return file_path

    def validate_file_access(
        self, file_path: str, file_context: FileContext
    ) -> tuple[Optional[Path], Optional[Observation]]:
        """
        Validate file access and return either a valid Path object or an error Observation.

        Args:
            file_path: The path to validate
            file_context: The file context
        Returns:
            Tuple of (Path object if valid, Error observation if invalid)
        """
        path = Path(file_path)

        if not file_context.file_exists(str(path)):
            return None, Observation(
                message=f"File {path} not found.",
                properties={"fail_reason": "file_not_found"},
            )  # type: ignore

        if not file_context.has_file(str(path)):
            return None, Observation(
                message=f"You have not yet viewed the file {path}. Use ViewCode to view the parts of the file that you want to modify.",
                properties={"fail_reason": "file_not_in_context"},
            )  # type: ignore

        return path, None

    def format_snippet_with_lines(self, snippet: str, start_line: int) -> str:
        """Format a code snippet with line numbers"""
        return "\n".join(f"{i + start_line:6}\t{line}" for i, line in enumerate(snippet.split("\n")))

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        criteria = Action.get_evaluation_criteria(trajectory_length)
        criteria.extend(
            [
                "Instruction Clarity: Ensure that instructions and pseudocode are clear and actionable.",
                "Instruction Compliance: The git diff must *exactly* implement the provided pseudo_code. Identify any discrepancies, omissions, or additions. If discrepancies exist, you should lower the reward accordingly.",
                "Code Modification Accuracy and Quality: Check for correct identification of code spans, accuracy of changes, syntax errors, logical flaws, unintended modifications, and unintended side effects.",
                "Python-Specific Features Utilization: Assess whether the agent has appropriately utilized Python-specific features that enhance the solution.",
                "Common Git Diff Issues and Unintended Changes: Check for issues such as incorrect line numbers, unintended additions or deletions, formatting errors, changes to unrelated parts of the code, and heavily penalize unintended changes.",
                "Addressing Test Failures: Verify if the agent is properly addressing test failures from previous `RunTests` actions.",
            ]
        )
        return criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return Action.generate_reward_scale_entries(
            [
                (
                    90,
                    100,
                    "The code change is optimal, with a perfect Git diff exactly matching the pseudo code, and requires no further changes.",
                ),
                (
                    75,
                    89,
                    "The code change significantly advances the solution, with an accurate Git diff exactly matching the pseudo code,.",
                ),
                (
                    50,
                    74,
                    "The code change is mostly correct but has minor issues or opportunities for optimization; the Git diff exactly matching the pseudo code,.",
                ),
                (
                    25,
                    49,
                    "The code change is acceptable but has noticeable issues or is less effective than possible alternatives;",
                ),
                (
                    0,
                    24,
                    "The code change has minimal impact or introduces minor negative consequences",
                ),
                (
                    -49,
                    -1,
                    "The code change is inappropriate, unhelpful, or introduces new issues; the action did not result in any successful code changes. The Git diff does not match the pseud code and instructions, contains significant inaccuracies or shows no changes. Penalize attempts to modify non-existent code elements (hallucinations) based on severity.",
                ),
                (
                    -100,
                    -50,
                    "The code change is counterproductive, causing significant setbacks or demonstrating persistent repetition without learning. The Git diff is severely flawed or indicates that no effective changes were made. Heavily penalize severe hallucinations or continuous attempts to modify non-existent code elements.",
                ),
            ]
        )
