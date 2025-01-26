from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Literal, Optional, Union, TypeVar, Generic, List, Any, Callable

from pydantic import BaseModel, Field

from moatless.completion.schema import MessageContentListBlock


class ArtifactReference(BaseModel):
    id: str
    type: str


class ArtifactListItem(BaseModel):
    """Basic information about an artifact for listing purposes"""

    id: str
    type: str
    name: str | None
    created_at: datetime


class ArtifactResponse(BaseModel):
    """Standard response structure for artifacts"""

    id: str
    type: str
    name: Optional[str]
    created_at: datetime
    references: List[ArtifactReference]
    data: Dict[str, Any]


class Artifact(BaseModel, ABC):
    id: Optional[str] = Field(default=None, description="Unique identifier for the artifact")
    type: str = Field(description="Type of artifact (e.g., 'receipt')")
    name: Optional[str] = Field(default=None, description="Name of the artifact")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the artifact was created")
    references: List[ArtifactReference] = Field(default_factory=list, description="Reference to the artifacts")

    @abstractmethod
    def to_prompt_message_content(self) -> MessageContentListBlock:
        pass

    def to_list_item(self) -> ArtifactListItem:
        """Convert artifact to a list item representation"""
        return ArtifactListItem(id=self.id, type=self.type, name=self.name, created_at=self.created_at)

    def to_ui_representation(self) -> Dict[str, Any]:
        """Convert artifact to a UI-friendly representation with all necessary data for display.
        By default, returns the complete model data except for default fields in the data field."""
        model_data = self.model_dump(exclude={"id", "type", "name", "created_at", "references"})
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "created_at": self.created_at,
            "references": [ref.model_dump() for ref in self.references],
            "data": model_data,
        }


class ArtifactChange(BaseModel):
    artifact_id: str
    change_type: Literal["added", "updated", "removed"]
    diff_details: Optional[str] = None
    actor: Literal["user", "assistant"]


# Create a TypeVar for the specific Artifact type
T = TypeVar("T", bound="Artifact")


class SearchCriteria(BaseModel):
    """Base class for defining search criteria"""

    field: str
    value: Any
    operator: Literal["eq", "contains", "gt", "lt", "gte", "lte"] = "eq"
    case_sensitive: bool = False


class ArtifactHandler(ABC, BaseModel, Generic[T]):
    """
    Defines how to load, save, update, and delete artifacts of a certain type.
    The type parameter T specifies which Artifact subclass this handler manages.
    """

    type: str = Field(description="Type of artifact this handler manages")

    @abstractmethod
    def read(self, artifact_id: str) -> T:
        pass

    @abstractmethod
    def create(self, artifact: T) -> T:
        pass

    def update(self, artifact: T) -> None:
        raise NotImplementedError("Update is not supported for this artifact type")

    def delete(self, artifact_id: str) -> None:
        raise NotImplementedError("Delete is not supported for this artifact type")

    @abstractmethod
    def get_all_artifacts(self) -> List[ArtifactListItem]:
        """Get all artifacts managed by this handler as list items"""
        pass

    def search(self, criteria: List[SearchCriteria]) -> List[T]:
        """
        Search for artifacts based on the provided criteria.
        Each handler implements its own search logic.
        """
        raise NotImplementedError("Search is not supported for this artifact type")
