from dataclasses import Field
from typing import Dict
from moatless.file_context import ContextFile, FileContext
from pydantic import PrivateAttr

from moatless.artifacts.artifact import Artifact, ArtifactHandler
from moatless.repository.repository import Repository


class ContextFileArtifact(Artifact):
    file_context: FileContext = Field()


class FileContextArtifactHandler(ArtifactHandler):
    """
    Handle the legacy file context
    """

    type: str = "file_context"

    _repository: Repository = PrivateAttr(None)

    context_by_node_id: Dict[str, FileContext] = PrivateAttr(default_factory=dict)

    def __init__(self, repository: Repository | None = None, **kwargs):
        super().__init__(**kwargs)

        self._repository = repository

    async def read(self, artifact_id: str) -> ContextFileArtifact:
        current_node_id = current_node_id.get()
        if not current_node_id:
            raise RuntimeError("No current node id found")

        if current_node_id not in self.context_by_node_id:
            self.context_by_node_id[current_node_id] = FileContext.from_dir(
                self._repository.get_node(current_node_id).path
            )

        return ContextFileArtifact(context_file=self.context_by_node_id[current_node_id])

    async def create(self, artifact: ContextFileArtifact) -> ContextFileArtifact:
        pass

    async def update(self, artifact: ContextFileArtifact) -> None:
        pass
