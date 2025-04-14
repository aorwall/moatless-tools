from typing import ClassVar, List

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments
from moatless.artifacts.coding_task import CodingTaskArtifact
from moatless.artifacts.task import TaskState
from moatless.file_context import FileContext


class RemoveCodingTasksArgs(ActionArguments):
    """Remove coding tasks by their IDs."""

    task_ids: List[str] = Field(
        ...,
        description="List of task IDs to remove.",
    )

    model_config = ConfigDict(title="RemoveCodingTasks")

    def to_prompt(self):
        return f"Remove coding tasks: {', '.join(self.task_ids)}"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, RemoveCodingTasksArgs)


class RemoveCodingTasks(Action):
    args_schema: ClassVar[type[ActionArguments]] = RemoveCodingTasksArgs

    async def _execute(self, args: RemoveCodingTasksArgs, file_context: FileContext | None = None) -> str:
        # Check if we have a coding task handler in the workspace
        if "coding_task" not in self.workspace.artifact_handlers:
            return "Error: Coding task handler not available"

        handler = self.workspace.artifact_handlers["coding_task"]

        # Get all tasks to find which ones exist
        all_tasks = await handler.get_all_artifacts()
        # Type cast to the proper type
        coding_tasks = [task for task in all_tasks if isinstance(task, CodingTaskArtifact)]

        # Filter for tasks that exist and those not found
        existing_task_ids = [task.id for task in coding_tasks]
        to_remove = [tid for tid in args.task_ids if tid in existing_task_ids]
        not_found = [tid for tid in args.task_ids if tid not in existing_task_ids]

        # Remove the tasks
        for task_id in to_remove:
            await handler.delete(task_id)

        # Build response message
        response = f"Removed {len(to_remove)} coding tasks."
        if not_found:
            response += f"\nTasks not found: {', '.join(not_found)}"

        # List remaining tasks with checkboxes
        remaining_artifacts = await handler.get_all_artifacts()
        remaining_tasks = [task for task in remaining_artifacts if isinstance(task, CodingTaskArtifact)]
        remaining_tasks.sort(key=lambda x: x.priority)

        if remaining_tasks:
            task_list = "\n".join(
                [
                    f"[{'x' if task.state == TaskState.COMPLETED else ' '}] {task.id} - {task.title}"
                    for task in remaining_tasks
                ]
            )
            response += f"\n\nRemaining tasks:\n{task_list}"

        return response
