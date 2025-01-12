import logging
from typing import List

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation, FewShotExample
from moatless.file_context import FileContext
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class ViewDiffArgs(ActionArguments):
    """
    View the current git diff of all changes in the workspace.

    Notes:
    * Shows changes for all modified files
    * Uses git patch format
    """

    class Config:
        title = "ViewDiff"


class ViewDiff(Action):
    """
    Action to view the current git diff of all changes.
    """

    args_schema = ViewDiffArgs

    def execute(
        self,
        args: ViewDiffArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        diff = file_context.generate_git_patch()

        if not diff:
            return Observation(
                message="No changes detected in the workspace.",
                properties={"diff": "", "success": True},
            )

        return Observation(
            message="Current changes in workspace:",
            properties={"diff": diff, "success": True},
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Show me the current changes in the workspace",
                action=ViewDiffArgs(thoughts="Viewing current git diff of all changes"),
            )
        ]
