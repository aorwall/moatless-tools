from typing import ClassVar, Optional

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments
from moatless.artifacts.task import TaskState
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from pydantic import ConfigDict, Field


class UpdateTaskArgs(ActionArguments):
    """Update an existing task's state and optionally its result."""

    task_id: str = Field(
        ...,
        description="The ID of the task to update.",
    )

    state: Optional[TaskState] = Field(
        None,
        description="The new state of the task (open, completed, failed, deleted). If None, the state won't be changed.",
    )

    result: Optional[str] = Field(
        None,
        description="Optional result for the task (useful when completing or failing).",
    )

    priority: Optional[int] = Field(
        None,
        description="Optional new priority for the task (lower numbers = higher priority).",
    )

    model_config = ConfigDict(title="UpdateTask")

    def to_prompt(self):
        parts = [f"Update task {self.task_id}"]

        if self.state:
            parts.append(f"to state: {self.state}")

        if self.priority is not None:
            parts.append(f"with priority: {self.priority}")

        if self.result:
            parts.append(f"with result: {self.result}")

        return " ".join(parts)

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, UpdateTaskArgs)

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Mark the task for implementing user authentication as completed",
                action=cls(
                    thoughts="I'll update the task status to completed since the implementation is done.",
                    task_id="auth-flow",
                    state=TaskState.COMPLETED,
                    result="User authentication implementation completed with JWT token support.",
                    priority=None,
                ),
            ),
            FewShotExample.create(
                user_input="The database schema design task failed because of incompatible requirements",
                action=cls(
                    thoughts="I need to mark this task as failed with an explanation.",
                    task_id="db-schema",
                    state=TaskState.FAILED,
                    result="Failed due to incompatible requirements between the legacy system and new architecture.",
                    priority=None,
                ),
            ),
            FewShotExample.create(
                user_input="Change the priority of the registration API task to high priority",
                action=cls(
                    thoughts="I need to increase the priority of this task by giving it a lower number.",
                    task_id="registration-api",
                    state=None,
                    result=None,
                    priority=5,
                ),
            ),
        ]


class UpdateTask(Action):
    args_schema: ClassVar[type[ActionArguments]] = UpdateTaskArgs

    async def _execute(self, args: UpdateTaskArgs, file_context: FileContext | None = None) -> str:
        # Check if we have a task handler in the workspace
        if "task" not in self.workspace.artifact_handlers:
            return "Error: Task handler not available"

        handler = self.workspace.artifact_handlers["task"]

        try:
            # Load the task
            task = await handler.read(args.task_id)

            # Track changes for response message
            changes = []

            # Update task properties if provided
            if args.state:
                old_state = task.state
                task.state = args.state
                changes.append(f"state from '{old_state}' to '{args.state}'")

            if args.result is not None:
                task.result = args.result
                changes.append(f"result to '{args.result}'")

            if args.priority is not None:
                old_priority = task.priority
                task.priority = args.priority
                changes.append(f"priority from {old_priority} to {args.priority}")

            # Only update if there were changes
            if changes:
                # Save the updated task
                updated_task = await handler.update(task)

                # Return a message about the update
                return f"Updated task {args.task_id}: " + ", ".join(changes)
            else:
                return f"No changes specified for task {args.task_id}"

        except ValueError as e:
            # Handle case where task was not found
            return f"Error: {str(e)}"
