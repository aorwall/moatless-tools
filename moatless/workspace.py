from collections import defaultdict
from typing import List, Dict, Any

from pydantic import BaseModel, Field, PrivateAttr

from moatless.artifacts.artifact import Artifact, ArtifactHandler, ArtifactListItem, SearchCriteria


class Workspace(BaseModel):
    artifacts: List[Artifact] = Field(default_factory=list)
    artifact_handlers: Dict[str, ArtifactHandler] = Field(default_factory=dict, exclude=True)

    def __init__(self, artifact_handlers: List[ArtifactHandler], **data):
        super().__init__(**data)
        self.artifact_handlers = {handler.type: handler for handler in artifact_handlers}

    def create_artifact(self, artifact: Artifact) -> Artifact:
        if artifact.type in self.artifact_handlers:
            handler = self.artifact_handlers[artifact.type]
            artifact = handler.create(artifact)

        return artifact

    def get_artifact(self, artifact_type: str, artifact_id: str) -> Artifact | None:
        if artifact_type not in self.artifact_handlers:
            raise ValueError(f"No handler found for artifact type: {artifact_type}")

        handler = self.artifact_handlers[artifact_type]
        return handler.read(artifact_id)

    def get_artifact_by_id(self, artifact_id: str) -> Artifact | None:
        handler = self._get_handler(artifact.type)
        artifact = handler.read(artifact_id)
        return artifact

    def get_artifacts_by_type(self, artifact_type: str) -> List[Artifact]:
        handler = self._get_handler(artifact_type)
        return handler.get_all_artifacts()

    def search(self, artifact_type: str, criteria: List[SearchCriteria]) -> List[Artifact]:
        """
        Search for artifacts of a specific type using the provided criteria
        """
        if artifact_type not in self.artifact_handlers:
            raise ValueError(f"No handler found for artifact type: {artifact_type}")

        handler = self.artifact_handlers[artifact_type]
        return handler.search(criteria)

    def get_all_artifacts(self) -> List[ArtifactListItem]:
        """Get all artifacts from all handlers as list items"""
        all_artifacts = []
        for handler in self.artifact_handlers.values():
            try:
                artifacts = handler.get_all_artifacts()
                all_artifacts.extend(artifacts)
            except Exception as e:
                # Log error but continue with other handlers
                print(f"Error getting artifacts from handler {handler.type}: {e}")
        return all_artifacts

    def update_artifact(self, artifact: Artifact) -> None:
        handler = self.artifact_handlers[artifact.type]
        handler.update(artifact)

    def _get_handler(self, artifact_type: str) -> ArtifactHandler:
        return self.artifact_handlers[artifact_type]

    def model_post_init(self, __context) -> None:
        """Rebuild lookup dictionaries and handlers after loading from JSON"""

        self.artifact_handlers = {handler.type: handler for handler in self.artifact_handlers.values()}

    def dump_handlers(self) -> Dict[str, Any]:
        """Dump artifact handlers to a serializable format"""
        return {key: handler.model_dump() for key, handler in self.artifact_handlers.items()}

    def clone(self):
        cloned_workspace = Workspace(artifact_handlers=list(self.artifact_handlers.values()), artifacts=self.artifacts)
        return cloned_workspace
