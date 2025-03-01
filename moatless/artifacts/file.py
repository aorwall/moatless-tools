import base64
import base64
import io
import json
import logging
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Type

from moatless.artifacts.json_handler import JsonArtifactHandler
import pymupdf as fitz
from PIL import Image, ImageEnhance
from pydantic import Field, PrivateAttr

from moatless.artifacts.artifact import Artifact, ArtifactHandler, ArtifactResponse
from moatless.completion.schema import (
    ChatCompletionImageUrlObject,
    ChatCompletionTextObject,
    MessageContentListBlock,
)

logger = logging.getLogger(__name__)


class FileArtifact(Artifact):
    type: str = "file"
    file_path: str = Field(description="Path on disk where the artifact is stored")
    mime_type: Optional[str] = Field(default=None, description="MIME type of the file content")
    content: Optional[bytes] = Field(default=None, description="Content of the file")
    parsed_content: Optional[str] = Field(default=None, description="Parsed content for PDFs and other parseable files")

    def to_prompt_message_content(self) -> MessageContentListBlock:
        return ChatCompletionTextObject(type="text", text=str(self.parsed_content))

    def to_ui_representation(self) -> ArtifactResponse:
        """Convert file artifact to UI representation with binary content"""
        file_path = Path(self.file_path)
        content = file_path.read_bytes() if file_path.exists() else None

        if not content:
            return ArtifactResponse(
                id=self.id,
                type=self.type,
                name=self.name,
                created_at=self.created_at,
                references=self.references,
                data={
                    "mime_type": self.mime_type,
                    "content": None,
                    "parsed_content": self.parsed_content if hasattr(self, 'parsed_content') else None,
                    "file_path": str(file_path)
                }
            )

        # For PDFs, return raw bytes as base64
        if self.mime_type == 'application/pdf':
            content_b64 = base64.b64encode(content).decode('utf-8')
        elif self.mime_type.startswith('image/'):
            # For images, optimize before sending
            try:
                image = Image.open(io.BytesIO(content))
                # Convert RGBA to RGB if needed
                if image.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                # Optimize size if needed
                max_size = (1024, 1024)  # Adjust as needed
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                # Save as optimized JPEG
                output = io.BytesIO()
                image.save(output, format='JPEG', quality=85, optimize=True)
                content_b64 = base64.b64encode(output.getvalue()).decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to optimize image {self.id}: {e}")
                content_b64 = base64.b64encode(content).decode('utf-8')
        elif self.mime_type.startswith('text/'):
            # For text files, decode and return as UTF-8
            try:
                text_content = content.decode('utf-8')
                content_b64 = base64.b64encode(text_content.encode('utf-8')).decode('utf-8')
            except UnicodeDecodeError:
                # Fallback to raw bytes if not valid UTF-8
                content_b64 = base64.b64encode(content).decode('utf-8')
        else:
            # Default handling for other file types
            content_b64 = base64.b64encode(content).decode('utf-8')

        return ArtifactResponse(
            id=self.id,
            type=self.type,
            name=self.name,
            created_at=self.created_at,
            references=self.references,
            status=self.status,
            can_persist=self.can_persist,
            data={
                "mime_type": self.mime_type,
                "content": content_b64,
                "parsed_content": self.parsed_content if hasattr(self, 'parsed_content') else None,
                "file_path": str(file_path)
            }
        )


class TextFileArtifact(FileArtifact):
    content: str = Field(exclude=True)

    def to_prompt_message_content(self) -> MessageContentListBlock:
        return ChatCompletionTextObject(type="text", text=self.content)

    def to_ui_representation(self) -> Dict[str, Any]:
        """Convert text file to UI representation"""
        base_repr = super().to_ui_representation()
        base_repr["data"].update(
            {
                "mime_type": self.mime_type,
                "content": base64.b64encode(self.content.encode("utf-8")).decode("utf-8"),
            }
        )
        return base_repr


class ImageFileArtifact(FileArtifact):
    base64_image: str = Field(exclude=True)

    def to_prompt_message_content(self) -> MessageContentListBlock:
        return ChatCompletionImageUrlObject(
            type="image_url",
            image_url={"url": f"data:{self.mime_type};base64,{self.base64_image}"},
        )

    def to_ui_representation(self) -> Dict[str, Any]:
        """Convert image to UI representation"""
        base_repr = super().to_ui_representation()
        base_repr["data"].update({"mime_type": self.mime_type, "content": self.base64_image})
        return base_repr


class FileArtifactHandler(JsonArtifactHandler[FileArtifact]):
    type: str = "file"

    max_image_size: Tuple[int, int] = Field(default=(1024, 1024), description="Maximum size of the image to save")
    quality: int = Field(default=85, description="Quality of the image to save")

    _artifacts: Dict[str, FileArtifact] = PrivateAttr(default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_type(cls) -> str:
        return "file"

    def _detect_mime_type(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    def get_file_path(self, artifact_id: str) -> Path:
        return self.trajectory_dir / "files" / artifact_id
    
    def get_artifact_class(self) -> Type[FileArtifact]:
        return FileArtifact

    async def read(self, artifact_id: str) -> FileArtifact:
        file_path = self.get_file_path(artifact_id)

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

    async def create(self, artifact: FileArtifact) -> Artifact:
        file_path = self.get_file_path(artifact.id)
        if artifact.content:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving artifact {artifact.id} to {file_path}")
            file_path.write_bytes(artifact.content)

        self._artifacts[artifact.id] = artifact
        self._save_artifacts()

        return artifact

    async def update(self, artifact: FileArtifact) -> None:
        file_path = self.get_file_path(artifact.id)
        if artifact.content:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving artifact {artifact.id} to {file_path}")
            file_path.write_bytes(artifact.content)

        self._artifacts[artifact.id] = artifact
        self._save_artifacts()

    async def delete(self, artifact_id: str) -> None:
        file_path = self.get_file_path(artifact_id)
        if file_path.exists():
            file_path.unlink()

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
        pdf_content = f"Contents of file {file_name}:\n\n"
        
        try:
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                # Add metadata if available
                metadata = doc.metadata
                if metadata:
                    pdf_content += "Document Information:\n"
                    for key, value in metadata.items():
                        if value:
                            pdf_content += f"- {key}: {value}\n"
                    pdf_content += "\n"

                # Extract text page by page
                pdf_content += "Document Content:\n"
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text().strip()
                    if text:
                        pdf_content += f"\n--- Page {page_num} ---\n{text}\n"
                
                # Add page count
                pdf_content += f"\nTotal Pages: {len(doc)}\n"
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_name}: {str(e)}")
            pdf_content += f"\nError: Failed to fully process PDF content. Error: {str(e)}"

        return file_content, pdf_content

    def _save_artifacts(self) -> None:
        artifact_dumps = []
        for artifact in self._artifacts.values():
            artifact_dumps.append(artifact.model_dump(exclude={"content", "parsed_content"}))

        with open(self.get_storage_path(), "w") as file:
            json.dump(artifact_dumps, file, indent=4)
