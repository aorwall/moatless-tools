from abc import ABC, abstractmethod
import base64
import io
import logging
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from PIL import Image, ImageEnhance
import pymupdf as fitz

from dataclasses import dataclass

from pydantic import Field

from moatless.artifacts.artifact import Artifact, ArtifactHandler, ArtifactListItem
from moatless.completion.schema import ChatCompletionImageUrlObject, ChatCompletionTextObject, MessageContentListBlock


logger = logging.getLogger(__name__)


class FileArtifact(Artifact):
    type: str = "file"
    file_path: str = Field(description="Path on disk where the artifact is stored")
    mime_type: Optional[str] = Field(default=None, description="MIME type of the file content")
    content: Optional[bytes] = Field(default=None, description="Content of the file")
    parsed_content: Optional[str] = Field(default=None, description="Parsed content for PDFs and other parseable files")

    def to_prompt_message_content(self) -> MessageContentListBlock:
        return ChatCompletionTextObject(type="text", text=str(self.content))

    def to_ui_representation(self) -> Dict[str, Any]:
        """Convert file artifact to UI representation with binary content"""
        file_path = Path(self.file_path)
        content = file_path.read_bytes() if file_path.exists() else None

        # Special handling for PDFs to ensure proper binary data transfer
        if self.mime_type == "application/pdf" and content:
            content_b64 = base64.b64encode(content).decode("utf-8")
        else:
            content_b64 = base64.b64encode(content).decode("utf-8") if content else None

        base_repr = super().to_ui_representation()
        base_repr["data"].update(
            {"mime_type": self.mime_type, "content": content_b64, "parsed_content": self.parsed_content}
        )
        return base_repr


class TextFileArtifact(FileArtifact):
    content: str = Field(exclude=True)

    def to_prompt_message_content(self) -> MessageContentListBlock:
        return ChatCompletionTextObject(type="text", text=self.content)

    def to_ui_representation(self) -> Dict[str, Any]:
        """Convert text file to UI representation"""
        base_repr = super().to_ui_representation()
        base_repr["data"].update(
            {"mime_type": self.mime_type, "content": base64.b64encode(self.content.encode("utf-8")).decode("utf-8")}
        )
        return base_repr


class ImageFileArtifact(FileArtifact):
    base64_image: str = Field(exclude=True)

    def to_prompt_message_content(self) -> MessageContentListBlock:
        return ChatCompletionImageUrlObject(
            type="image_url", image_url={"url": f"data:{self.mime_type};base64,{self.base64_image}"}
        )

    def to_ui_representation(self) -> Dict[str, Any]:
        """Convert image to UI representation"""
        base_repr = super().to_ui_representation()
        base_repr["data"].update({"mime_type": self.mime_type, "content": self.base64_image})
        return base_repr


class FileArtifactHandler(ArtifactHandler[FileArtifact]):
    type: str = "file"
    directory_path: Path = Field(description="Base directory path for storing artifacts")

    max_image_size: Tuple[int, int] = Field(default=(1024, 1024), description="Maximum size of the image to save")
    quality: int = Field(default=85, description="Quality of the image to save")

    def _detect_mime_type(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    def get_file_path(self, artifact_id: str) -> Path:
        return self.directory_path / artifact_id

    def read(self, artifact_id: str) -> FileArtifact:
        file_path = self.directory_path / artifact_id

        mime_type = self._detect_mime_type(str(file_path))
        logger.info(f"Reading artifact {artifact_id} with MIME type {mime_type}")

        file_content = file_path.read_bytes()

        if mime_type.startswith("image/"):
            return ImageFileArtifact(
                id=artifact_id,
                type=self.type,
                name=file_path.name,
                file_path=str(file_path),
                mime_type=mime_type,
                base64_image=self.encode_image(file_content),
            )
        elif mime_type.startswith("application/pdf"):
            raw_content, parsed_content = self.read_pdf(str(file_path), file_content)
            return FileArtifact(
                id=artifact_id,
                type=self.type,
                name=file_path.name,
                file_path=str(file_path),
                mime_type=mime_type,
                content=raw_content,
                parsed_content=parsed_content,
            )
        else:
            # read content as text
            content = file_path.read_text()
            return TextFileArtifact(
                id=artifact_id,
                type=self.type,
                name=file_path.name,
                file_path=str(file_path),
                mime_type=mime_type,
                content=content,
            )

    def create(self, artifact: FileArtifact) -> Artifact:
        file_path = self.directory_path / artifact.file_path
        if artifact.content:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving artifact {artifact.id} to {file_path}")
            file_path.write_bytes(artifact.content)

        return artifact

    def update(self, artifact: FileArtifact) -> None:
        self.save(artifact)

    def delete(self, artifact_id: str) -> None:
        file_path = self.directory_path / artifact_id
        if file_path.exists():
            file_path.unlink()

    def get_all_artifacts(self) -> List[ArtifactListItem]:
        """Get all artifacts in the directory as list items"""
        artifacts = []
        for file_path in self.directory_path.glob("*"):
            if file_path.is_file():
                artifact_id = file_path.name
                try:
                    artifact = self.read(artifact_id)
                    artifacts.append(artifact.to_list_item())
                except Exception as e:
                    logger.error(f"Failed to read artifact {artifact_id}: {e}")
        return artifacts

    def encode_image(self, file_content: bytes) -> str:
        """Encodes image bytes to base64 string"""
        return base64.b64encode(file_content).decode("utf-8")

    def preprocess_image(self, file_content: bytes) -> bytes:
        """Enhance image for document processing with focus on text clarity"""
        image = Image.open(io.BytesIO(file_content))
        image = image.convert("L")

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
            image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)

        image = image.convert("RGB")

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=self.quality, optimize=True)
        return output.getvalue()

    def read_pdf(self, file_path: str, file_content: bytes) -> Tuple[bytes, str]:
        """Extract text content from PDF and return both raw PDF and parsed text"""
        file_name = Path(file_path).name
        pdf_content = f"Contents of file {file_name}:\n"
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                pdf_content += page.get_text()

        return file_content, pdf_content
