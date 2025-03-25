from moatless.artifacts.artifact import Artifact
from pydantic import BaseModel


class Task(BaseModel):
    description: str
    completed: bool = False


class PlanArtifact(Artifact):
    goal: str
    tasks: list[Task]
