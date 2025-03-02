import builtins
from abc import abstractmethod
from enum import Enum
from typing import ClassVar, List, Optional, Type

from pydantic import BaseModel, Field

from moatless.artifacts.artifact import Artifact
from moatless.artifacts.json_handler import JsonArtifactHandler
from moatless.completion.schema import ChatCompletionTextObject, MessageContentListBlock


class Position(BaseModel):
    line: int
    column: Optional[int] = None


class Range(BaseModel):
    start: Position
    end: Optional[Position] = None


class DiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class Diagnostic(BaseModel):
    severity: DiagnosticSeverity = Field(description="The severity of the diagnostic")
    range: Range = Field(description="The range at which the message applies.")
    source: str = Field(
        description="A human-readable string describing the source of this diagnostic, e.g. 'typescript' or 'super lint'."
    )
    message: str = Field(description="The diagnostic's message.")
    code: str = Field(description="The diagnostic's code, which might appear in the user interface.")


class DiagnosticArtifact(Artifact):
    """
    An artifact that contains a list of diagnostics.
    """

    file_path: str = Field(description="The path to the file that the diagnostic is for")
    diagnostics: list[Diagnostic] = Field(description="The list of diagnostics")

    def __init__(self, file_path: str, diagnostics: list[Diagnostic], **kwargs):
        super().__init__(
            id=f"{file_path}-diagnostic", type="diagnostic", file_path=file_path, diagnostics=diagnostics, **kwargs
        )

    def to_prompt_message_content(self) -> MessageContentListBlock:
        """Convert the diagnostic to a message content block for prompts"""
        diagnostics_text = "\n".join(
            [
                f"{diag.severity.upper()} [{diag.source}] {diag.message} (code: {diag.code}) at line {diag.range.start.line}, column {diag.range.start.column}"
                for diag in self.diagnostics
            ]
        )

        return ChatCompletionTextObject(type="text", text=f"File: {self.file_path}\n{diagnostics_text}")  # type: ignore


class DiagnosticHandler(JsonArtifactHandler[DiagnosticArtifact]):
    """
    Handler for DiagnosticArtifact objects.
    Stores diagnostics in a JSON file named "diagnostic.json" in the trajectory directory.
    """

    type: ClassVar[str] = "diagnostic"

    @classmethod
    def get_artifact_class(cls) -> builtins.type[DiagnosticArtifact]:
        """Return the DiagnosticArtifact class that this handler manages"""
        return DiagnosticArtifact
