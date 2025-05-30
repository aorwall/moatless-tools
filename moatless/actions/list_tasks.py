import logging
from typing import ClassVar, Optional

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments
from moatless.artifacts.artifact import SearchCriteria
from moatless.artifacts.task import TaskState
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class ListTasksArgs(ActionArguments):
    """List tasks with optional filtering by state."""

    state: Optional[TaskState] = Field(
        None,
        description="Filter tasks by state (open, completed, failed, deleted). If None, all tasks are listed.",
    )

    model_config = ConfigDict(title="ListTasks")

    def to_prompt(self):
        if self.state:
            return f"List tasks with state: {self.state}"
        else:
            return "List all tasks"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, ListTasksArgs)

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Show me all the open tasks",
                action=cls(
                    thoughts="I'll list all the tasks that are currently open.",
                    state=TaskState.OPEN,
                ),
            ),
            FewShotExample.create(
                user_input="Show me a list of all tasks",
                action=cls(
                    thoughts="I'll list all tasks regardless of their state.",
                    state=None,
                ),
            ),
        ]


class ListTasks(Action):
    args_schema: ClassVar[type[ActionArguments]] = ListTasksArgs

    async def _execute(self, args: ListTasksArgs, file_context: FileContext | None = None) -> str:
        # Check if we have a task handler in the workspace
        if "task" not in self.workspace.artifact_handlers:
            return "Error: Task handler not available"

        handler = self.workspace.artifact_handlers["task"]

        # Get all tasks
        criteria = []
        if args.state:
            criteria.append(SearchCriteria(field="state", value=args.state, operator="eq"))

        tasks = []
        artifact_items = await handler.search(criteria) if criteria else await handler.get_all_artifacts()

        # Convert to list of tasks
        for item in artifact_items:
            try:
                task = await handler.read(item.id)
                tasks.append(task)
            except ValueError:
                logger.exception(f"Error loading task {item.id}")
                pass  # Skip tasks that couldn't be loaded

        # Sort tasks by priority (lower numbers first) and then by ID for consistent output
        tasks.sort(key=lambda x: (x.priority, x.id))

        # Generate the response message
        if not tasks:
            state_str = f" with state '{args.state}'" if args.state else ""
            return f"No tasks found{state_str}."

        state_str = f" with state '{args.state}'" if args.state else ""
        message = f"Found {len(tasks)} tasks{state_str}:\n\n"

        # Format each task
        for task in tasks:
            message += f"ID: {task.id}\n"
            message += f"Priority: {task.priority}\n"
            message += f"State: {task.state}\n"
            message += f"Content: {task.content}\n"
            if task.result:
                message += f"Result: {task.result}\n"
            message += "\n"

        return message
