from typing import ClassVar, List

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments
from moatless.artifacts.coding_task import CodingTaskArtifact
from moatless.artifacts.task import TaskState
from moatless.file_context import FileContext


class FinishCodingTasksArgs(ActionArguments):
    """Mark coding tasks as completed by their IDs."""

    task_ids: List[str] = Field(
        ...,
        description="List of task IDs to mark as completed.",
    )

    model_config = ConfigDict(title="FinishCodingTasks")

    def to_prompt(self):
        return f"Finish coding tasks: {', '.join(self.task_ids)}"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, FinishCodingTasksArgs)


class FinishCodingTasks(Action):
    args_schema: ClassVar[type[ActionArguments]] = FinishCodingTasksArgs

    async def _execute(self, args: FinishCodingTasksArgs, file_context: FileContext | None = None) -> str:
        # Check if we have a coding task handler in the workspace
        if "coding_task" not in self.workspace.artifact_handlers:
            return "Error: Coding task handler not available"

        handler = self.workspace.artifact_handlers["coding_task"]

        # Get all tasks to validate IDs and current state
        all_artifacts = await handler.get_all_artifacts()
        coding_tasks = [task for task in all_artifacts if isinstance(task, CodingTaskArtifact)]
        all_task_dict = {task.id: task for task in coding_tasks}

        completed = []
        not_found = []
        already_completed = []

        # Process each task ID
        for task_id in args.task_ids:
            if task_id not in all_task_dict:
                not_found.append(task_id)
                continue

            task = all_task_dict[task_id]

            if task.state == TaskState.COMPLETED:
                already_completed.append(task_id)
                continue

            # Update task state to completed
            updated_task = task.model_copy(update={"state": TaskState.COMPLETED})
            await handler.update(updated_task)
            completed.append(task_id)

        # Prepare response
        response_parts = []

        if completed:
            response_parts.append(f"Completed {len(completed)} coding tasks: {', '.join(completed)}")

        if not_found:
            response_parts.append(f"Tasks not found: {', '.join(not_found)}")

        if already_completed:
            response_parts.append(f"Tasks already completed: {', '.join(already_completed)}")

        # List all tasks with their current status
        updated_artifacts = await handler.get_all_artifacts()
        updated_tasks = [task for task in updated_artifacts if isinstance(task, CodingTaskArtifact)]
        updated_tasks.sort(key=lambda x: x.priority)

        task_list = "\n".join(
            [
                f"[{'x' if task.state == TaskState.COMPLETED else ' '}] {task.id} - {task.title}"
                for task in updated_tasks
            ]
        )

        response = "\n".join(response_parts)
        response += f"\n\nAll coding tasks:\n{task_list}"

        return response
