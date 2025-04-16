from asyncio.log import logger
from typing import Tuple, Optional

from moatless.evaluation.utils import (
    find_identified_files,
    find_identified_spans,
    get_swebench_instance,
    has_identified_files,
    has_identified_spans,
)
from moatless.file_context import FileContext
from moatless.node import Node, Reward
from moatless.utils.file import is_test
from moatless.value_function.base import BaseValueFunction
from moatless.context_data import current_trajectory_id


class SwebenchValueFunction(BaseValueFunction):
    """
    A value function for the Swebench benchmark which calculates a reward for code-generating
    agents based on their progress in identifying code spans, patching files, and achieving good
    test outcomes.

    The reward is calculated as a weighted sum of three components:
        - Identify Reward: Based on whether the correct files and code spans have been recognized.
        - Patch Reward: Based on whether the correct files have been updated.
        - Test Reward: Based on the improvements (or regressions) in test results.

    **Important:** This implementation is intended for evaluation purposes only. Using it outside of an evaluation
    scenario may contaminate the results.
    """

    async def get_reward(self, node: Node) -> Optional[Reward]:
        instance_id = current_trajectory_id.get()
        if not instance_id:
            logger.warning("No instance ID found in context data")
            return None

        swe_bench_instance = get_swebench_instance(instance_id)
        expected_spans = swe_bench_instance.get("expected_spans", {})
        solutions = [expected_spans]
        for resolved_by in swe_bench_instance.get("resolved_by", []):
            if "alternative_spans" in resolved_by and resolved_by["alternative_spans"] not in solutions:
                solutions.append(resolved_by["alternative_spans"])

        identify_reward = self._get_identify_reward(node.file_context, solutions)
        patch_reward = self._get_patch_reward(node.file_context, solutions)
        test_reward = self._get_test_reward(node, swe_bench_instance)

        weights = {
            "identify": 0.4,
            "patch": 0.3,
            "test": 0.3,
        }

        test_value = test_reward.value if test_reward is not None else 0

        final_reward_value = int(
            weights["identify"] * identify_reward.value
            + weights["patch"] * patch_reward.value
            + weights["test"] * test_value
        )
        final_explanation = (
            f"Identify: {identify_reward.explanation} \n"
            f"Patch: {patch_reward.explanation} \n"
            f"Test: {test_reward.explanation if test_reward else 'No test reward calculated.'}"
        )

        return Reward(value=final_reward_value, explanation=final_explanation)

    def _get_identify_reward(self, file_context: FileContext, solutions: list[dict]) -> Reward:
        """
        Reward the agent for identifying the correct files and code spans.
        """
        identified_spans = {}
        for file in file_context.files:
            identified_spans[file.file_path] = file.span_ids

        if not identified_spans:
            explanation = "No code spans found in file context."
            reward_value = -100
        elif identified_spans := find_identified_spans(solutions, identified_spans):
            span_str = "\n".join([f"{file}: {spans}" for file, spans in identified_spans.items()])
            explanation = f"Correct code spans found in file context.\n{span_str}"
            reward_value = 100
        elif identified_files := find_identified_files(solutions, identified_spans):
            file_str = "\n".join([f"{file}" for file in identified_files.keys()])
            explanation = f"Correct files identified in file context.\n{file_str}"
            reward_value = 50
        else:
            # find uniqe files in solutions
            unique_files = set()
            for solution in solutions:
                for file in solution.keys():
                    unique_files.add(file)
            unique_files_str = "\n".join([f"{file}" for file in unique_files])
            explanation = f"Incorrect files identified in file context. Expected one of: {unique_files_str}"
            reward_value = -50

        return Reward(value=reward_value, explanation=explanation)

    def _get_patch_reward(self, file_context: FileContext, solutions: list[dict]) -> Reward:
        """
        Reward the agent for applying patches to the correct files.
        """
        patched_files = []
        for file in file_context.files:
            if file.patch:
                patched_files.append(file.file_path)

        if not patched_files:
            explanation = "No files have been updated."
            reward_value = -50
        elif has_identified_files(solutions, patched_files):
            explanation = "The correct files have been updated."
            reward_value = 100
        else:
            explanation = "The wrong files have been updated."
            reward_value = -25

        return Reward(value=reward_value, explanation=explanation)

    def _get_test_reward(self, node: Node, swe_bench_instance: dict) -> Optional[Reward]:
        """
        Reward (or penalize) the agent based on the test outcomes.
        """
        has_changes = False
        has_test_changes = False

        test_files = node.file_context.test_files
        for file in node.file_context.files:
            if file.patch:
                has_changes = True
                if is_test(file.file_path):
                    has_test_changes = True

        passed_count, failure_count, error_count = node.file_context.get_test_counts()
        total_tests = passed_count + failure_count + error_count

        # Retrieve previous test results for incremental improvement
        previous_failure_count = 0
        previous_error_count = 0
        previous_reward = 0
        if node.parent and node.parent.file_context:
            _, previous_failure_count, previous_error_count = node.parent.file_context.get_test_counts()
            if node.parent.reward:
                previous_reward = node.parent.reward.value

        if total_tests == 0:
            return Reward(value=0, explanation="No tests have been run.")

        expected_test_files = list(swe_bench_instance.get("test_file_spans", {}).keys())
        missing_test_files = [file for file in expected_test_files if file not in test_files]

        if missing_test_files and failure_count == 0 and error_count == 0 and has_test_changes:
            return Reward(
                value=50,
                explanation=f"All {passed_count} tests are passing with changes applied. The expected test files have not been run: {' '.join(missing_test_files)}",
            )

        if missing_test_files and failure_count == 0 and error_count == 0:
            return Reward(
                value=0,
                explanation=f"All {passed_count} tests are passing but no changes were made to test files. The expected test files have not been run: {' '.join(missing_test_files)}",
            )

        # All tests passing and changes applied leads to the highest reward.
        if failure_count == 0 and error_count == 0 and has_test_changes:
            return Reward(value=100, explanation=f"All {passed_count} tests are passing with changes applied.")
        # All tests passing but no test file changes gets a moderate reward.
        elif failure_count == 0 and error_count == 0:
            return Reward(
                value=50, explanation=f"All {passed_count} tests are passing but no changes were made to test files. "
            )

        # If test failures or errors have increased, penalize the agent.
        elif failure_count > previous_failure_count or error_count > previous_error_count:
            new_value = max(-100, previous_reward - 50)
            return Reward(
                value=new_value,
                explanation=(
                    f"Test failures or errors increased "
                    f"(failures: {previous_failure_count} -> {failure_count}, "
                    f"errors: {previous_error_count} -> {error_count})."
                ),
            )
        # If there is an improvement in test results, provide a moderate reward.
        elif failure_count < previous_failure_count or error_count < previous_error_count:
            new_value = min(75, previous_reward + 25)
            return Reward(
                value=new_value,
                explanation=(
                    f"Test failures or errors decreased "
                    f"(failures: {previous_failure_count} -> {failure_count}, "
                    f"errors: {previous_error_count} -> {error_count})."
                ),
            )
        else:
            new_value = max(-100, previous_reward - 25)
            return Reward(
                value=new_value,
                explanation=f"No significant improvement in test results (failures: {failure_count}, errors: {error_count}).",
            )
