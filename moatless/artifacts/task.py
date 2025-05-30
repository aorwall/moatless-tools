import builtins
from enum import Enum
from typing import ClassVar, Optional

from pydantic import Field

from moatless.artifacts.artifact import Artifact
from moatless.artifacts.json_handler import JsonArtifactHandler
from moatless.completion.schema import MessageContentListBlock


class TaskState(str, Enum):
    OPEN = "open"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class TaskArtifact(Artifact):
    """
    An artifact that represents a task.
    """

    state: TaskState = TaskState.OPEN
    content: str = Field(description="The content of the task")
    result: Optional[str] = Field(default=None, description="The result of the task")
    priority: int = Field(
        default=100,
        description="Execution priority - determines the order in which tasks should be completed (lower numbers = higher priority)",
    )

    def to_prompt_message_content(self) -> MessageContentListBlock:
        # Create a ChatCompletionTextObject with the required fields
        # cache_control is optional and can be None
        return {"type": "text", "text": self.content, "cache_control": None}


class TaskHandler(JsonArtifactHandler[TaskArtifact]):
    """
    Handler for TaskArtifact objects.
    Stores tasks in a JSON file named "task.json" in the trajectory directory.
    """

    type: ClassVar[str] = "task"

    @classmethod
    def get_artifact_class(cls) -> builtins.type[TaskArtifact]:
        """Return the TaskArtifact class that this handler manages"""
        return TaskArtifact
