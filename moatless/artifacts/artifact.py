import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Type, TypeVar, Generic, List, Any

from pydantic import BaseModel, Field, PrivateAttr

from moatless.artifacts.content import ContentStructure
from moatless.completion.schema import MessageContentListBlock
from moatless.component import MoatlessComponent
from moatless.utils.class_loading import DynamicClassLoadingMixin

logger = logging.getLogger(__name__)

class ArtifactReference(BaseModel):
    id: str
    type: str


class ArtifactListItem(BaseModel):
    """Basic information about an artifact for listing purposes"""

    id: str
    type: str
    name: str | None
    created_at: datetime
    references: List[ArtifactReference]


class ArtifactResponse(BaseModel):
    """Standard response structure for artifacts"""

    id: str
    type: str
    name: Optional[str]
    created_at: datetime
    references: List[ArtifactReference]
    status: Literal["updated", "persisted", "new", "unchanged"]
    can_persist: bool = Field(default=False, description="Whether the artifact can be persisted")
    data: Dict[str, Any]
    content: Optional[ContentStructure] = None


class Artifact(BaseModel, ABC):
    id: Optional[str] = Field(default=None, description="Unique identifier for the artifact")
    type: str = Field(description="Type of artifact (e.g., 'receipt')")
    name: Optional[str] = Field(default=None, description="Name of the artifact")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the artifact was created")
    references: List[ArtifactReference] = Field(default_factory=list, description="Reference to the artifacts")
    status: Literal["new", "updated", "persisted", "unchanged"] = Field(default="new", description="Status of the artifact")

    @abstractmethod
    def to_prompt_message_content(self) -> MessageContentListBlock:
        pass

    @property
    def can_persist(self) -> bool:
        """Check if the artifact can be persisted based on its status and handler implementation"""
        return self.status in ["updated", "new"]

    def to_list_item(self) -> ArtifactListItem:
        """Convert artifact to a list item representation"""
        return ArtifactListItem(id=self.id, type=self.type, name=self.name, created_at=self.created_at, references=self.references)

    def to_ui_representation(self) -> ArtifactResponse:
        """Convert artifact to a UI-friendly representation with all necessary data for display.
        By default, returns the complete model data except for default fields in the data field."""
        model_data = self.model_dump(exclude={"id", "type", "name", "created_at", "references"})

        return ArtifactResponse(
            id=self.id,
            type=self.type,
            name=self.name,
            created_at=self.created_at,
            references=[ref.model_dump() for ref in self.references],
            status=self.status,
            can_persist=self.can_persist and self.status in ["updated", "new"],
            data=model_data,
        )
    
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Override model_dump to customize serialization."""
        data = super().model_dump(*args, **kwargs)
        # Convert created_at to epoch milliseconds
        if 'created_at' in data:
            data['created_at'] = int(self.created_at.timestamp() * 1000)
        return data
    
    @classmethod
    def model_validate(cls, data: Any):
        """Override model_validate to customize deserialization."""
        if 'created_at' in data:
            data['created_at'] = datetime.fromtimestamp(data['created_at'] / 1000)
        return super().model_validate(data)


class ArtifactChange(BaseModel):
    artifact_id: str
    artifact_type: str
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

class ArtifactHandler(MoatlessComponent, Generic[T]):
    """
    Defines how to load, save, update, and delete artifacts of a certain type.
    The type parameter T specifies which Artifact subclass this handler manages.
    """

    type: str = Field(description="Type of artifact this handler manages")
    trajectory_dir: Path = Field(description="Path to the trajectory directory")
    _artifacts: Dict[str, T] = PrivateAttr(default={})

    @classmethod
    def get_type(cls) -> str:
        pass
    
    @abstractmethod
    def read(self, artifact_id: str) -> T:
        """
        Read an existing artifact from the storage.
        """
        pass

    @abstractmethod
    def create(self, artifact: T) -> T:
        """
        Create a new artifact but do not persist it to the storage.
        """
        pass

    def update(self, artifact: T) -> None:
        """
        Update an existing artifact but do not persist it to the storage.
        """
        raise NotImplementedError("Update is not supported for this artifact type")

    def delete(self, artifact_id: str) -> None:
        """
        Delete an existing artifact but do not persist it to the storage.
        """
        raise NotImplementedError("Delete is not supported for this artifact type")
    
    def persist(self, artifact_id: str) -> None:
        """
        Finalize and save the artifact to its permanent storage (e.g., disk, remote server).
        Returns the updated artifact instance.
        """
        raise NotImplementedError("Persist is not supported for this artifact type")

    def get_all_artifacts(self) -> List[ArtifactListItem]:
        """Get all artifacts managed by this handler as list items"""
        list_items = []
        for artifact in self._artifacts.values():
            list_items.append(artifact.to_list_item())
        return list_items

    def search(self, criteria: List[SearchCriteria]) -> List[T]:
        """
        Search for artifacts based on the provided criteria.
        Each handler implements its own search logic.
        """
        raise NotImplementedError("Search is not supported for this artifact type")

    def _save_artifacts(self) -> None:
        artifact_dumps = []
        for artifact in self._artifacts.values():
            try:
                artifact_dict = artifact.model_dump()
                artifact_dumps.append(artifact_dict)
            except Exception as e:
                logger.error(f"Error dumping artifact {artifact}: {e}")

        try:        
            with open(self.get_storage_path(), "w") as file:
                json.dump(artifact_dumps, file, indent=4)
        except Exception as e:
            logger.exception(f"Error saving artifacts {artifact_dumps} to {self.get_storage_path()}: {e}")

    def get_storage_path(self) -> Path:
        return self.trajectory_dir / f"{self.type}.json"

    @classmethod
    def get_component_type(cls) -> str:
        return "artifact_handler"

    @classmethod
    def get_base_class(cls) -> Type:
        return ArtifactHandler

    @classmethod
    def get_name(cls) -> str:
        return cls.type

    