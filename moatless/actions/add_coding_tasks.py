from typing import ClassVar, List

from pydantic import BaseModel, ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments
from moatless.artifacts.coding_task import CodingTaskArtifact, FileLocation, FileRelationType
from moatless.artifacts.task import TaskState
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext


class CodingTaskItem(BaseModel):
    """A single coding task item to be created."""

    id: str = Field(
        ...,
        description="Identifier or short name for the task. This will be used as the task's ID in the system.",
    )

    title: str = Field(
        ...,
        description="Short title or description of the task.",
    )

    instructions: str = Field(
        ...,
        description="Detailed instructions for completing the task.",
    )

    related_files: List[FileLocation] = Field(
        default_factory=list,
        description="List of files related to this task with their locations and relationship types.",
    )

    priority: int = Field(
        default=100,
        description="Execution priority - lower numbers = higher priority.",
    )


class AddCodingTasksArgs(ActionArguments):
    """Create new coding tasks with the given descriptions and file locations."""

    tasks: List[CodingTaskItem] = Field(
        ...,
        description="List of coding tasks to create.",
    )

    model_config = ConfigDict(title="AddCodingTasks")

    def to_prompt(self):
        task_list = "\n".join([f"- {task.id} (priority: {task.priority}): {task.title}" for task in self.tasks])
        return f"Add coding tasks:\n{task_list}"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, AddCodingTasksArgs)

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Create coding tasks for implementing the new authentication module",
                action=AddCodingTasksArgs(
                    thoughts="I'll create a set of coding tasks to track the auth module implementation.",
                    tasks=[
                        CodingTaskItem(
                            id="auth-interface",
                            title="Create authentication interface",
                            instructions="Create a new interface for authentication that defines login and logout methods.",
                            related_files=[
                                FileLocation(file_path="src/auth/interface.ts", relation_type=FileRelationType.CREATE),
                                FileLocation(
                                    file_path="src/auth/existing.ts",
                                    start_line=10,
                                    end_line=25,
                                    relation_type=FileRelationType.REFERENCE,
                                ),
                            ],
                            priority=10,
                        ),
                        CodingTaskItem(
                            id="implement-auth",
                            title="Implement authentication service",
                            instructions="Implement the auth service based on the interface.",
                            related_files=[
                                FileLocation(file_path="src/services/auth.ts", relation_type=FileRelationType.UPDATE)
                            ],
                            priority=20,
                        ),
                    ],
                ),
            ),
        ]


class AddCodingTasks(Action):
    args_schema: ClassVar[type[ActionArguments]] = AddCodingTasksArgs

    async def _execute(self, args: AddCodingTasksArgs, file_context: FileContext | None = None) -> str:
        # Check if we have a coding task handler in the workspace
        if "coding_task" not in self.workspace.artifact_handlers:
            return "Error: Coding task handler not available"

        handler = self.workspace.artifact_handlers["coding_task"]

        created_tasks = []
        for task_item in args.tasks:
            task = CodingTaskArtifact(
                id=task_item.id,
                type="coding_task",
                title=task_item.title,
                instructions=task_item.instructions,
                related_files=task_item.related_files,
                state=TaskState.OPEN,
                priority=task_item.priority,
            )

            created_task = await handler.create(task)
            created_tasks.append(created_task)

        # Sort tasks by priority for display
        created_tasks.sort(key=lambda x: x.priority)

        # Format output with checkboxes
        task_details = "\n".join([f"[ ] {task.id} - {task.title}" for task in created_tasks])
        return f"Added {len(created_tasks)} coding tasks:\n{task_details}"
