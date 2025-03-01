from abc import abstractmethod
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Type, List, ClassVar

from moatless.artifacts.artifact import Artifact
from moatless.artifacts.json_handler import JsonArtifactHandler
from moatless.completion.schema import MessageContentListBlock

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

class DiagnosticArtifact(Artifact):
    type: str = "diagnostic"

    severity: DiagnosticSeverity = Field(description="The severity of the diagnostic")
    range: Range = Field(description="The range at which the message applies.")
    source: str = Field(description="A human-readable string describing the source of this diagnostic, e.g. 'typescript' or 'super lint'.")
    message: str = Field(description="The diagnostic's message.")
    code: str = Field(description="The diagnostic's code, which might appear in the user interface.")
    
    def to_prompt_message_content(self) -> MessageContentListBlock:
        """Convert the diagnostic to a message content block for prompts"""
        return {
            "type": "text",
            "text": f"{self.severity.upper()} [{self.source}] {self.message} (code: {self.code}) at line {self.range.start.line}, column {self.range.start.column}"
        }


class DiagnosticHandler(JsonArtifactHandler[DiagnosticArtifact]):
    """
    Handler for DiagnosticArtifact objects.
    Stores diagnostics in a JSON file named "diagnostic.json" in the trajectory directory.
    """
    type: ClassVar[str] = "diagnostic"
    
    @classmethod
    def get_artifact_class(cls) -> Type[DiagnosticArtifact]:
        """Return the DiagnosticArtifact class that this handler manages"""
        return DiagnosticArtifact
    
