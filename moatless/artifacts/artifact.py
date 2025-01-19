from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Union, TypeVar, Generic

from pydantic import BaseModel, Field

from moatless.completion.schema import MessageContentListBlock

class Artifact(BaseModel, ABC):
    id: str = Field(description="Unique identifier for the artifact")
    type: str = Field(description="Type of artifact (e.g., 'receipt')")
    name: str = Field(description="Name of the artifact")

    @abstractmethod
    def to_prompt_message_content(self) -> MessageContentListBlock:
        pass


class ArtifactChange(BaseModel):
    artifact_id: str
    change_type: Literal["added", "updated", "removed"]
    diff_details: Optional[str] = None
    actor: Literal["user", "assistant"]


# Create a TypeVar for the specific Artifact type
T = TypeVar("T", bound="Artifact")


class ArtifactHandler(ABC, BaseModel, Generic[T]):
    """
    Defines how to load, save, update, and delete artifacts of a certain type.
    The type parameter T specifies which Artifact subclass this handler manages.
    """

    type: str = Field(description="Type of artifact this handler manages")

    @abstractmethod
    def load(self, artifact_id: str) -> T:
        pass

    @abstractmethod
    def save(self, artifact: T) -> None:
        pass

    @abstractmethod
    def update(self, artifact: T) -> None:
        pass

    @abstractmethod
    def delete(self, artifact_id: str) -> None:
        pass
