from enum import Enum
import builtins
from typing import ClassVar, List, Optional
from pydantic import BaseModel, Field

from moatless.artifacts.artifact import Artifact
from moatless.artifacts.json_handler import JsonArtifactHandler
from moatless.artifacts.task import TaskState, TaskHandler
from moatless.completion.schema import MessageContentListBlock


class FileRelationType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    REFERENCE = "reference"
    DEPENDENCY = "dependency"


class FileLocation(BaseModel):
    """Represents a file location with path and line range"""

    file_path: str = Field(description="Path to the file")
    start_line: Optional[int] = Field(default=None, description="Starting line number")
    end_line: Optional[int] = Field(default=None, description="Ending line number")
    relation_type: FileRelationType = Field(
        default=FileRelationType.REFERENCE, description="How this file relates to the task"
    )


class CodingTaskArtifact(Artifact):
    """
    An artifact that represents a coding task with related files.
    """

    state: TaskState = TaskState.OPEN
    title: str = Field(description="Short title or description of the task")
    instructions: str = Field(description="Detailed instructions for completing the task")
    related_files: List[FileLocation] = Field(default_factory=list, description="List of files related to this task")
    result: Optional[str] = Field(default=None, description="The result of the task")
    priority: int = Field(
        default=100,
        description="Execution priority - determines the order in which tasks should be completed (lower numbers = higher priority)",
    )

    def to_prompt_message_content(self) -> MessageContentListBlock:
        return {"type": "text", "text": self.instructions, "cache_control": None}


class CodingTaskHandler(TaskHandler):
    """
    Handler for CodingTaskArtifact objects.
    Stores coding tasks in a JSON file named "coding_task.json" in the trajectory directory.
    """

    type: ClassVar[str] = "coding_task"

    @classmethod
    def get_artifact_class(cls) -> builtins.type[CodingTaskArtifact]:
        """Return the CodingTaskArtifact class that this handler manages"""
        return CodingTaskArtifact
