import logging
from enum import Enum
from typing import Optional, Literal, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MessageHistoryType(Enum):
    MESSAGES = "messages"  # Provides all messages in sequence
    SUMMARY = "summary"  # Generates one message with summarized history
    REACT = "react"
    MESSAGES_COMPACT = "messages_compact"

    @classmethod
    def _missing_(cls, value: str):
        """Handle case-insensitive enum lookup"""
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return None

    def __str__(self):
        return self.value

    def json(self):
        """Custom JSON serialization"""
        return self.value


class FileWithSpans(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    span_ids: list[str] = Field(
        default_factory=list,
        description="Span IDs identiying the relevant code spans. A span id is a unique identifier for a code sippet. It can be a class name or function name. For functions in classes separete with a dot like 'class.function'.",
    )

    def add_span_id(self, span_id):
        if span_id not in self.span_ids:
            self.span_ids.append(span_id)

    def add_span_ids(self, span_ids: list[str]):
        for span_id in span_ids:
            self.add_span_id(span_id)

    def __eq__(self, other: "FileWithSpans"):
        return self.file_path == other.file_path and self.span_ids == other.span_ids


class RankedFileSpan(BaseModel):
    file_path: str
    span_id: str
    rank: int = 0
    tokens: int = 0


class ActionView(BaseModel):
    name: str


class Attachment(BaseModel):
    """Represents a file attachment in a chat message"""

    file_name: str = Field(description="Original name of the uploaded file")
    content: bytes = Field(description="Raw binary content of the file")
    mime_type: Optional[str] = Field(
        default=None, description="MIME type of the file content"
    )


class Message(BaseModel):
    role: str = Field(description="Role of the message sender ('user' or 'assistant')")
    content: Optional[str] = Field(default=None, description="Content of the message")


class UserMessage(Message):
    role: Literal["user"] = Field(
        default="user", description="Role is always 'user' for user messages"
    )
    artifact_ids: Optional[List[str]] = Field(
        default=None, description="List of artifact ids associated with the message"
    )


class AssistantMessage(Message):
    role: Literal["assistant"] = Field(
        default="assistant",
        description="Role is always 'assistant' for assistant messages",
    )
    actions: Optional[List[ActionView]] = Field(
        default=None, description="List of actions performed by the assistant"
    )
