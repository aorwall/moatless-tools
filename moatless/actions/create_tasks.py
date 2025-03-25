from typing import ClassVar, List, Optional

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.artifacts.task import TaskArtifact, TaskHandler, TaskState
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from pydantic import BaseModel, ConfigDict, Field


class TaskItem(BaseModel):
    """A single task item to be created.

    Important: The 'id' field MUST be set by the AI agent to identify the task.
    """

    id: str = Field(
        ...,
        description="Identifier or short name for the task. This will be used as the task's ID in the system.",
    )

    content: str = Field(
        ...,
        description="The content/description of the task.",
    )

    priority: int = Field(
        default=100,
        description="Execution priority - determines the order in which tasks should be completed (lower numbers = higher priority). This is NOT severity - it indicates sequence/importance rather than impact level.",
    )


class CreateTasksArgs(ActionArguments):
    """Create new tasks with the given descriptions."""

    tasks: List[TaskItem] = Field(
        ...,
        description="List of tasks to create. Each task MUST have both 'id' and 'content' fields set.",
    )

    model_config = ConfigDict(title="CreateTasks")

    def to_prompt(self):
        task_list = "\n".join([f"- {task.id} (priority: {task.priority}): {task.content}" for task in self.tasks])
        return f"Create tasks:\n{task_list}"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, CreateTasksArgs)

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Create a list of things to do for implementing the new user authentication system",
                action=CreateTasksArgs(
                    thoughts="I'll create a set of tasks to track the work needed for the user authentication system.",
                    tasks=[
                        TaskItem(
                            id="db-schema",
                            content="Design the user authentication database schema",
                            priority=10,
                        ),
                        TaskItem(
                            id="registration-api",
                            content="Implement the user registration endpoint",
                            priority=20,
                        ),
                        TaskItem(
                            id="auth-flow",
                            content="Create the login and authentication flow",
                            priority=30,
                        ),
                        TaskItem(
                            id="password-reset",
                            content="Implement password reset functionality",
                            priority=40,
                        ),
                    ],
                ),
            ),
        ]


class CreateTasks(Action):
    args_schema: ClassVar[type[ActionArguments]] = CreateTasksArgs

    async def _execute(self, args: CreateTasksArgs, file_context: FileContext | None = None) -> str:
        # Check if we have a task handler in the workspace
        if "task" not in self.workspace.artifact_handlers:
            return "Error: Task handler not available"

        handler = self.workspace.artifact_handlers["task"]

        created_tasks = []
        for task_item in args.tasks:
            task = TaskArtifact(
                id=task_item.id,
                type="task",
                content=task_item.content,
                state=TaskState.OPEN,
                priority=task_item.priority,
            )

            created_task = await handler.create(task)
            created_tasks.append(created_task)

        # Sort tasks by priority for display
        created_tasks.sort(key=lambda x: x.priority)

        # Return a message with the created tasks
        task_details = "\n".join([f"- {task.id} (priority: {task.priority}): {task.content}" for task in created_tasks])
        return f"Created {len(created_tasks)} tasks:\n{task_details}"
