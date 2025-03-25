from abc import ABC, abstractmethod

from moatless.artifacts.artifact import Artifact
from pydantic import BaseModel


class ArtifactConnector(BaseModel, ABC):
    @abstractmethod
    def persist(self, artifact: Artifact) -> Artifact:
        """
        Persist the artifact to an external system.
        """
        pass
