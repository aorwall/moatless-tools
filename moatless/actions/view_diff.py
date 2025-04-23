import logging

from pydantic import ConfigDict

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from moatless.environment.base import BaseEnvironment

logger = logging.getLogger(__name__)


class ViewDiffArgs(ActionArguments):
    """
    View the current git diff of all changes in the workspace.

    Notes:
    * Shows changes for all modified files
    * Uses git patch format
    """

    model_config = ConfigDict(title="ViewDiff")

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Show me the current changes in the workspace",
                action=ViewDiffArgs(thoughts="Viewing current git diff of all changes"),
            )
        ]


class ViewDiff(Action):
    """
    Action to view the current git diff of all changes.
    """

    args_schema = ViewDiffArgs

    async def execute(self, args: ViewDiffArgs, file_context: FileContext | None = None) -> Observation:
        if self.workspace.shadow_mode:
            if not file_context:
                raise ValueError("File context is required to view diff")

            diff = file_context.generate_git_patch()
        else:
            # Get the diff using git commands via the environment
            env = self.workspace.environment
            if not env:
                raise ValueError("Environment is required to view diff")
            try:
                # First try to get diff with main branch
                diff = await env.execute("git diff main", fail_on_error=False)
                # If that fails, try with master branch
                if not diff or "fatal: ambiguous argument 'main'" in diff:
                    diff = await env.execute("git diff master", fail_on_error=False)
                # If that also fails, just get uncommitted changes
                if not diff or "fatal: ambiguous argument 'master'" in diff:
                    diff = await env.execute("git diff", fail_on_error=True)
            except Exception as e:
                logger.error(f"Failed to get git diff: {e}")
                return Observation.create(message=f"Failed to get git diff: {str(e)}")

        if not diff:
            return Observation.create(message="No changes detected in the workspace.")

        return Observation.create(message=f"Current changes in workspace:\n{diff}")
