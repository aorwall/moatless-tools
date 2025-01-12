from collections import defaultdict
from typing import List, Dict, Any

from pydantic import BaseModel, Field, PrivateAttr

from moatless.artifacts.artifact import Artifact, ArtifactHandler


class Workspace(BaseModel):
    artifacts: List[Artifact] = Field(default_factory=list)
    artifact_handlers: Dict[str, ArtifactHandler] = Field(
        default_factory=dict, exclude=True
    )

    _artifacts_by_id: Dict[str, Artifact] = PrivateAttr(default_factory=dict)
    _artifacts_by_type: Dict[str, List[Artifact]] = PrivateAttr(
        default_factory=lambda: defaultdict(list)
    )

    def __init__(self, artifact_handlers: List[ArtifactHandler], **data):
        super().__init__(**data)
        self.artifact_handlers = {
            handler.type: handler for handler in artifact_handlers
        }

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)
        self._artifacts_by_id[artifact.id] = artifact
        self._artifacts_by_type[artifact.type].append(artifact)
        if artifact.type in self.artifact_handlers:
            handler = self.artifact_handlers[artifact.type]
            handler.save(artifact)

    def get_artifact_by_id(self, artifact_id: str) -> Artifact | None:
        return self._artifacts_by_id.get(artifact_id)

    def get_artifacts_by_type(self, artifact_type: str) -> List[Artifact]:
        return self._artifacts_by_type[artifact_type]

    def update_artifact(self, artifact: Artifact) -> None:
        handler = self.artifact_handlers[artifact.type]
        handler.update(artifact)

    def model_post_init(self, __context) -> None:
        """Rebuild lookup dictionaries and handlers after loading from JSON"""
        self._artifacts_by_id.clear()
        self._artifacts_by_type.clear()
        for artifact in self.artifacts:
            self._artifacts_by_id[artifact.id] = artifact
            self._artifacts_by_type[artifact.type].append(artifact)

        self.artifact_handlers = {
            handler.type: handler for handler in self.artifact_handlers.values()
        }

    def dump_handlers(self) -> Dict[str, Any]:
        """Dump artifact handlers to a serializable format"""
        return {key: handler.dump() for key, handler in self.artifact_handlers.items()}

    def load_handlers(self, handlers_data: Dict[str, Any]) -> None:
        """Load artifact handlers from a serialized format"""
        handlers = [ArtifactHandler.load(data) for data in handlers_data.values()]
        self.artifact_handlers = {handler.type: handler for handler in handlers}
