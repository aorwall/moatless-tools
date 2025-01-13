import base64
import mimetypes
from pathlib import Path
from typing import Optional

from pydantic import Field

from moatless.artifacts.artifact import (
    Artifact,
    ArtifactHandler,
    TextPromptModel,
    ImageURLPromptModel,
    PromptModel,
)


class FileArtifact(Artifact):
    type: str = "file"
    file_path: str = Field(description="Path on disk where the artifact is stored")
    mime_type: Optional[str] = Field(
        default=None, description="MIME type of the file content"
    )
    content: bytes = Field(exclude=True)

    def to_prompt_format(self) -> PromptModel:
        if self.mime_type is None:
            self.mime_type = "text/plain"

        if self.mime_type.startswith("text/"):
            # Return TextPromptModel for text content
            text_str = self.content.decode("utf-8", errors="replace")
            return TextPromptModel(type="text", text=text_str)
        else:
            # Return ImageURLPromptModel for binary content
            encoded = base64.b64encode(self.content).decode("utf-8")
            return ImageURLPromptModel(
                type="image_url",
                image_url={"url": f"data:{self.mime_type};base64,{encoded}"},
            )


class FileArtifactHandler(ArtifactHandler[FileArtifact]):
    type: str = "file"
    directory_path: Path = Field(
        description="Base directory path for storing artifacts"
    )

    def _detect_mime_type(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    def load(self, artifact_id: str) -> FileArtifact:
        file_path = self.directory_path / artifact_id
        return FileArtifact(
            id=artifact_id,
            type=self.type,
            name=file_path.name,
            file_path=str(file_path),
            mime_type=self._detect_mime_type(str(file_path)),
            content=file_path.read_bytes() if file_path.exists() else None,
        )

    def save(self, artifact: FileArtifact) -> None:
        file_path = self.directory_path / artifact.file_path
        if artifact.content:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(artifact.content)

    def update(self, artifact: FileArtifact) -> None:
        self.save(artifact)

    def delete(self, artifact_id: str) -> None:
        file_path = self.directory_path / artifact_id
        if file_path.exists():
            file_path.unlink()
