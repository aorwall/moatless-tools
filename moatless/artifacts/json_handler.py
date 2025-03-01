import json
import logging
import uuid
from abc import abstractmethod
from pydantic import PrivateAttr
from pathlib import Path
from typing import Dict, List, Type, TypeVar, Generic, Any, Optional, cast

from moatless.artifacts.artifact import Artifact, ArtifactHandler, SearchCriteria
from moatless.storage import BaseStorage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Artifact)

class JsonArtifactHandler(ArtifactHandler[T]):
    """
    Abstract base class for artifact handlers that store artifacts in JSON files.
    Implements common functionality for reading and writing artifacts to JSON files.
    
    The JSON file will be named "{type}.json" and stored in the trajectory directory.
    """

    _storage: BaseStorage = PrivateAttr()
    _artifacts: Dict[str, T] = PrivateAttr(default={})
        
    def __init__(self, storage: BaseStorage | None = None, **kwargs):
        """
        Initialize the JsonArtifactHandler.
        
        Args:
            storage: Optional storage backend to use. If not provided, a default storage will be created
                    based on the in_memory flag.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        
        super().__init__(**kwargs)
        self._storage = storage or BaseStorage.get_default_storage()
        
    
    @classmethod
    @abstractmethod
    def get_artifact_class(cls) -> Type[T]:
        """Return the Artifact class that this handler manages"""
        pass
    
    @classmethod
    def get_type(cls) -> str:
        """Return the type of artifact this handler manages"""
        return cls.type
    
    async def _load_artifacts(self) -> None:
        """Load artifacts from storage"""
        # Initialize empty artifacts dictionary

        if self._artifacts:
            return
            
        self._artifacts = {}
        
        # If storage doesn't have our key, return empty
        if not self._storage.exists(self.type):
            logger.info(f"No artifacts found for type {self.type}. Creating empty artifact store.")
            return
        
        try:
            # Load artifacts from storage
            artifact_data = await self._storage.read_json(self.type)
            artifact_class = self.get_artifact_class()
            
            for item in artifact_data:
                try:
                    artifact = artifact_class.model_validate(item)
                    artifact.status = "persisted"  # Mark as persisted since it was loaded from storage
                    self._artifacts[artifact.id] = artifact
                except Exception as e:
                    logger.error(f"Error loading artifact from {item}: {e}")
        
        except Exception as e:
            logger.exception(f"Error loading artifacts of type {self.type}: {e}")
    
    async def read(self, artifact_id: str) -> T:
        """Read an existing artifact from the storage"""
        await self._load_artifacts()

        if artifact_id not in self._artifacts:
            raise ValueError(f"Artifact with ID {artifact_id} not found")
        
        return self._artifacts[artifact_id]
    
    async def create(self, artifact: T) -> T:
        """Create a new artifact but do not persist it to storage"""
        await self._load_artifacts()

        if artifact.id is None:
            artifact.id = self.generate_id()
        
        artifact.status = "new"
        self._artifacts[artifact.id] = artifact
        await self._save_artifacts()
        return artifact
    
    async def update(self, artifact: T) -> T:
        """Update an existing artifact but do not persist it to storage"""
        await self._load_artifacts()

        if artifact.id not in self._artifacts:
            raise ValueError(f"Artifact with ID {artifact.id} not found")
        
        # Only update status if it's currently persisted
        if self._artifacts[artifact.id].status == "persisted":
            artifact.status = "updated"
        
        self._artifacts[artifact.id] = artifact
        await self._save_artifacts()
        return artifact
    
    async def delete(self, artifact_id: str) -> None:
        """Delete an existing artifact but do not persist the change to storage"""
        await self._load_artifacts()

        if artifact_id not in self._artifacts:
            raise ValueError(f"Artifact with ID {artifact_id} not found")
        
        del self._artifacts[artifact_id]
        await self._save_artifacts()
    
    async def search(self, criteria: List[SearchCriteria]) -> List[T]:
        """
        Search for artifacts based on the provided criteria.
        Implements a simple filtering mechanism based on the criteria.
        """
        results = list(self._artifacts.values())
        
        for criterion in criteria:
            filtered_results = []
            
            for artifact in results:
                artifact_dict = artifact.model_dump()
                
                if criterion.field not in artifact_dict:
                    continue
                
                field_value = artifact_dict[criterion.field]
                search_value = criterion.value
                
                # Handle string case sensitivity
                if isinstance(field_value, str) and isinstance(search_value, str) and not criterion.case_sensitive:
                    field_value = field_value.lower()
                    search_value = search_value.lower()
                
                # Apply the operator
                if criterion.operator == "eq" and field_value == search_value:
                    filtered_results.append(artifact)
                elif criterion.operator == "contains" and search_value in field_value:
                    filtered_results.append(artifact)
                elif criterion.operator == "gt" and field_value > search_value:
                    filtered_results.append(artifact)
                elif criterion.operator == "lt" and field_value < search_value:
                    filtered_results.append(artifact)
                elif criterion.operator == "gte" and field_value >= search_value:
                    filtered_results.append(artifact)
                elif criterion.operator == "lte" and field_value <= search_value:
                    filtered_results.append(artifact)
            
            results = filtered_results
        
        return results 
    
    async def _save_artifacts(self) -> None:
        """Save artifacts to storage"""
        artifact_data = [artifact.model_dump() for artifact in self._artifacts.values()]
        await self._storage.write_json(self.type, artifact_data)

    def generate_id(self) -> str:
        """Generate a unique ID for an artifact"""
        return f"{self.type}-{len(self._artifacts) + 1}"