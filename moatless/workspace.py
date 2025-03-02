from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field, PrivateAttr

from moatless.artifacts.artifact import (
    Artifact,
    ArtifactHandler,
    ArtifactListItem,
    SearchCriteria,
)
from moatless.environment.base import BaseEnvironment
from moatless.index.code_index import CodeIndex
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment


class Workspace(BaseModel):
    artifacts: list[Artifact] = Field(default_factory=list)
    artifact_handlers: dict[str, ArtifactHandler] = Field(default_factory=dict, exclude=True)

    _repository: Repository = PrivateAttr(default=None)
    _code_index: CodeIndex = PrivateAttr(default=None)
    _runtime: RuntimeEnvironment = PrivateAttr(default=None)  # TODO: Replace this with BaseEnvironment
    _environment: BaseEnvironment = PrivateAttr(default=None)

    def __init__(
        self,
        artifact_handlers: list[ArtifactHandler] | None = None,
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        runtime: RuntimeEnvironment | None = None,
        environment: BaseEnvironment | None = None,
        **data,
    ):
        super().__init__(**data)

        if not artifact_handlers:
            artifact_handlers = ArtifactHandler.initiate_handlers()
            self.artifact_handlers = {handler.type: handler for handler in artifact_handlers}

        self._environment = environment

        # TODO: These are for the legacy workspace and should evnetually be replaced by artifact handlers and environment
        self._repository = repository
        self._code_index = code_index
        self._runtime = runtime

    @property
    def repository(self) -> Repository:
        return self._repository

    @property
    def code_index(self) -> CodeIndex:
        return self._code_index

    @property
    def runtime(self) -> RuntimeEnvironment:
        return self._runtime

    @property
    def environment(self) -> BaseEnvironment:
        return self._environment

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

    def get_artifacts_by_type(self, artifact_type: str) -> list[Artifact]:
        handler = self._get_handler(artifact_type)
        return handler.get_all_artifacts()

    def search(self, artifact_type: str, criteria: list[SearchCriteria]) -> list[Artifact]:
        """
        Search for artifacts of a specific type using the provided criteria
        """
        if artifact_type not in self.artifact_handlers:
            raise ValueError(f"No handler found for artifact type: {artifact_type}")

        handler = self.artifact_handlers[artifact_type]
        return handler.search(criteria)

    def get_all_artifacts(self) -> list[ArtifactListItem]:
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
        if not artifact.id:
            raise ValueError("Artifact ID is required to update an artifact")
        handler = self.artifact_handlers[artifact.type]
        handler.update(artifact)

    async def persist_artifact(self, artifact_type: str, artifact_id: str) -> None:
        handler = self.artifact_handlers[artifact_type]
        await handler.persist(artifact_id)

    def _get_handler(self, artifact_type: str) -> ArtifactHandler:
        return self.artifact_handlers[artifact_type]

    def model_post_init(self, __context) -> None:
        """Rebuild lookup dictionaries and handlers after loading from JSON"""

        self.artifact_handlers = {handler.type: handler for handler in self.artifact_handlers.values()}

    def dump_handlers(self) -> dict[str, Any]:
        """Dump artifact handlers to a serializable format"""
        return {key: handler.model_dump() for key, handler in self.artifact_handlers.items()}

    def clone(self):
        cloned_workspace = Workspace(
            artifact_handlers=list(self.artifact_handlers.values()),
            artifacts=self.artifacts,
        )
        return cloned_workspace
