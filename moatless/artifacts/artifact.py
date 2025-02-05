import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Literal, Optional, Union, TypeVar, Generic, List, Any, Callable, Type

from pydantic import BaseModel, Field, PrivateAttr

from moatless.completion.schema import MessageContentListBlock
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

_handlers_by_type: Dict[str, "ArtifactHandler"] = {}

class ArtifactHandler(ABC, BaseModel, Generic[T], DynamicClassLoadingMixin):
    """
    Defines how to load, save, update, and delete artifacts of a certain type.
    The type parameter T specifies which Artifact subclass this handler manages.
    """

    type: str = Field(description="Type of artifact this handler manages")


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
    
    def persist(self, artifact: T) -> T:
        """
        Finalize and save the artifact to its permanent storage (e.g., disk, remote server).
        Returns the updated artifact instance.
        """
        raise NotImplementedError("Persist is not supported for this artifact type")

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

    @classmethod
    def get_handler_by_type(cls, type: str) -> Type["ArtifactHandler"]:
        """
        Get an ArtifactHandler class by name.
        
        Args:
            type: The name of the handler class
            
        Returns:
            The ArtifactHandler class
            
        Raises:
            ValueError: If the handler is not found
        """
        logger.info(f"Getting handler for type {type}")
        cls._initiate_handlers()
        
        if type not in _handlers_by_type:
            raise ValueError(f"Handler for type {type} not found, available handler types: {_handlers_by_type.keys()}")
        return _handlers_by_type[type]

    @classmethod
    def _initiate_handlers(cls):
        global _handlers_by_type
        if _handlers_by_type:
            return

        registered_classes = cls._load_classes("moatless.artifacts", ArtifactHandler)

        logger.info(f"Registered classes: {registered_classes.keys()}")
        _handlers_by_type = {}
        for name, handler in registered_classes.items():
            handler = handler()
            _handlers_by_type[handler.type] = handler

        logger.info(f"Initialized handlers: {_handlers_by_type.keys()}")