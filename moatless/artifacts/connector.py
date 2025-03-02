from abc import ABC, abstractmethod

from pydantic import BaseModel

from moatless.artifacts.artifact import Artifact


class ArtifactConnector(BaseModel, ABC):
    @abstractmethod
    def persist(self, artifact: Artifact) -> Artifact:
        """
        Persist the artifact to an external system.
        """
        pass
