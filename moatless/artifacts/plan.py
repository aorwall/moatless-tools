from pydantic import BaseModel

from moatless.artifacts.artifact import Artifact


class Task(BaseModel):
    description: str
    completed: bool = False


class PlanArtifact(Artifact):
    goal: str
    tasks: list[Task]
