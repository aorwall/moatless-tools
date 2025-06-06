import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, PrivateAttr

from moatless.artifacts.content import ContentStructure
from moatless.completion.schema import MessageContentListBlock
from moatless.component import MoatlessComponent
from moatless.settings import get_storage
from moatless.storage.base import BaseStorage
from moatless.storage.file_storage import FileStorage

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
    references: list[ArtifactReference]


class ArtifactResponse(BaseModel):
    """Standard response structure for artifacts"""

    id: str
    type: str
    name: Optional[str]
    created_at: datetime
    references: list[ArtifactReference]
    status: Literal["updated", "persisted", "new", "unchanged"]
    can_persist: bool = Field(default=False, description="Whether the artifact can be persisted")
    data: dict[str, Any]
    content: Optional[ContentStructure] = None


class Artifact(BaseModel, ABC):
    id: str = Field(description="Unique identifier for the artifact")
    type: str = Field(description="Type of artifact (e.g., 'receipt')")
    name: Optional[str] = Field(default=None, description="Name of the artifact")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the artifact was created")
    references: list[ArtifactReference] = Field(default_factory=list, description="Reference to the artifacts")
    status: Literal["new", "updated", "persisted", "unchanged"] = Field(
        default="new", description="Status of the artifact"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def to_prompt_message_content(self) -> MessageContentListBlock:
        pass

    @property
    def can_persist(self) -> bool:
        """Check if the artifact can be persisted based on its status and handler implementation"""
        return self.status in ["updated", "new"]

    def to_list_item(self) -> ArtifactListItem:
        """Convert artifact to a list item representation"""
        return ArtifactListItem(
            id=self.id, type=self.type, name=self.name, created_at=self.created_at, references=self.references
        )

    def to_ui_representation(self) -> ArtifactResponse:
        """Convert artifact to a UI-friendly representation with all necessary data for display.
        By default, returns the complete model data except for default fields in the data field."""
        model_data = self.model_dump(exclude={"id", "type", "name", "created_at", "references"})

        return ArtifactResponse(
            id=self.id,
            type=self.type,
            name=self.name,
            created_at=self.created_at,
            references=self.references,
            status=self.status,
            can_persist=self.can_persist and self.status in ["updated", "new"],
            data=model_data,
        )

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Override model_dump to customize serialization."""
        data = super().model_dump(*args, **kwargs)
        # Convert created_at to epoch milliseconds
        if "created_at" in data:
            data["created_at"] = int(self.created_at.timestamp() * 1000)
        return data

    @classmethod
    def model_validate(cls, data: Any):
        """Override model_validate to customize deserialization."""
        if "created_at" in data:
            data["created_at"] = datetime.fromtimestamp(data["created_at"] / 1000)
        return super().model_validate(data)


class ArtifactChange(BaseModel):
    artifact_id: str
    artifact_type: str
    change_type: Literal["added", "updated", "removed"]
    properties: dict[str, Any]
    actor: Literal["user", "assistant"]

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data["properties"] = {k: v for k, v in data["properties"].items() if v is not None}
        return data

    @classmethod
    def model_validate(cls, data: Any):
        """Override model_validate to customize deserialization."""
        if "properties" in data:
            data["properties"] = {k: v for k, v in data["properties"].items() if v is not None}
        return super().model_validate(data)


T = TypeVar("T")  # TODO: Add bound="ArtifactHandler" once mypy doesnt crash when it's used


class SearchCriteria(BaseModel):
    """Base class for defining search criteria"""

    field: str
    value: Any
    operator: Literal["eq", "contains", "gt", "lt", "gte", "lte"] = "eq"
    case_sensitive: bool = False


class ArtifactHandler(MoatlessComponent[T]):
    """
    Defines how to load, save, update, and delete artifacts of a certain type.
    The type parameter T specifies which Artifact subclass this handler manages.
    """

    type: ClassVar[str]

    _storage: BaseStorage | None = PrivateAttr(default=None)

    def __init__(self, storage: BaseStorage | None = None, **data):
        super().__init__(**data)
        self._storage = storage

    @classmethod
    def get_component_type(cls) -> str:
        return "artifact_handler"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.artifacts"

    @classmethod
    def _get_base_class(cls) -> type:
        return ArtifactHandler

    @abstractmethod
    async def read(self, artifact_id: str) -> T:
        """
        Read an existing artifact from the storage.
        """
        pass

    @abstractmethod
    async def create(self, artifact: T) -> T:
        """
        Create a new artifact but do not persist it to the storage.
        """
        pass

    @abstractmethod
    async def update(self, artifact: T) -> None:
        """
        Update an existing artifact but do not persist it to the storage.
        """
        raise NotImplementedError("Update is not supported for this artifact type")

    @abstractmethod
    async def delete(self, artifact_id: str) -> None:
        """
        Delete an existing artifact but do not persist it to the storage.
        """
        raise NotImplementedError("Delete is not supported for this artifact type")

    async def persist(self, artifact_id: str) -> None:
        """
        Finalize and save the artifact to its permanent storage (e.g., disk, remote server).
        Returns the updated artifact instance.
        """
        raise NotImplementedError("Persist is not supported for this artifact type")

    async def get_all_artifacts(self) -> list[Artifact]:
        """Get all artifacts managed by this handler as list items"""
        raise NotImplementedError("Get all artifacts is not supported for this artifact type")

    async def search(self, criteria: list[SearchCriteria]) -> list[T]:
        """
        Search for artifacts based on the provided criteria.
        Each handler implements its own search logic.
        """
        raise NotImplementedError("Search is not supported for this artifact type")

    @classmethod
    def get_base_class(cls) -> type:
        return ArtifactHandler

    @classmethod
    def get_name(cls) -> str:
        return cls.type

    @classmethod
    def initiate_handlers(cls, storage: BaseStorage | None = None) -> list["ArtifactHandler"]:
        registered_classes = cls.get_available_components()

        logger.info(f"Registered classes: {list(registered_classes.keys())}")
        handlers = []
        for handler in registered_classes.values():
            handler = handler()
            handlers.append(handler)

            if hasattr(handler, "_storage"):
                handler._storage = storage

        logger.info(f"Initialized handlers: {list(map(lambda h: h.type, handlers))}")
        return handlers
