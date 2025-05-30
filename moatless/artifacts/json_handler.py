import logging
from abc import abstractmethod
from typing import TypeVar

from pydantic import PrivateAttr

from moatless.artifacts.artifact import (
    Artifact,
    ArtifactHandler,
    SearchCriteria,
)
from moatless.storage import BaseStorage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Artifact)


class JsonArtifactHandler(ArtifactHandler[T]):
    """
    Abstract base class for artifact handlers that store artifacts in JSON files.
    Implements common functionality for reading and writing artifacts to JSON files.

    The JSON file will be named "{type}.json" and stored in the trajectory directory.
    """

    _storage: BaseStorage | None = PrivateAttr(default=None)
    _artifacts: dict[str, T] = PrivateAttr(default={})

    @classmethod
    @abstractmethod
    def get_artifact_class(cls) -> type[T]:
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
            logger.info(f"Artifacts already loaded for type {self.type}")
            return

        self._artifacts = {}
        if not self._storage:
            logger.warning(f"No storage set for {self.type} handler")
            return

        path = f"{self.type}.json"

        # If storage doesn't have our key, return empty
        if not await self._storage.exists_in_trajectory(path):
            logger.info(f"No artifacts found for type {self.type}. Creating empty artifact store.")
            return

        try:
            artifact_data = await self._storage.read_from_trajectory(path)
            artifact_class = self.get_artifact_class()
            logger.info(f"Loading artifacts for type {self.type} from {path}")

            # Fix: Ensure artifact_data is properly accessed as a dictionary
            if isinstance(artifact_data, dict) and "artifacts" in artifact_data:
                artifacts_list = artifact_data["artifacts"]
                for item in artifacts_list:
                    try:
                        artifact = artifact_class.model_validate(item)
                        artifact.status = "persisted"  # Mark as persisted since it was loaded from storage
                        self._artifacts[artifact.id] = artifact
                    except Exception as e:
                        logger.error(f"Error loading artifact from {item}: {e}")
            else:
                logger.error(f"Invalid artifact data format: {artifact_data}")

        except Exception as e:
            logger.exception(f"Error loading artifacts of type {self.type}: {e}")

    async def get_all_artifacts(self) -> list[Artifact]:
        await self._load_artifacts()
        return list(self._artifacts.values())

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
        # Ensure artifacts are saved after creation
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
        # Ensure artifacts are saved after update
        await self._save_artifacts()
        return artifact

    async def delete(self, artifact_id: str) -> None:
        """Delete an existing artifact but do not persist the change to storage"""
        await self._load_artifacts()

        if artifact_id not in self._artifacts:
            raise ValueError(f"Artifact with ID {artifact_id} not found")

        del self._artifacts[artifact_id]
        # Ensure artifacts are saved after deletion
        await self._save_artifacts()

    async def search(self, criteria: list[SearchCriteria]) -> list[T]:
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
                if (
                    criterion.operator == "eq"
                    and field_value == search_value
                    or criterion.operator == "contains"
                    and search_value in field_value
                    or (
                        criterion.operator == "gt"
                        and field_value > search_value
                        or criterion.operator == "lt"
                        and field_value < search_value
                    )
                    or (
                        criterion.operator == "gte"
                        and field_value >= search_value
                        or criterion.operator == "lte"
                        and field_value <= search_value
                    )
                ):
                    filtered_results.append(artifact)

            results = filtered_results

        return results

    async def _save_artifacts(self) -> None:
        """Save artifacts to storage"""
        if not self._storage:
            logger.warning(f"No storage set for {self.type} handler")
            return

        path = f"{self.type}.json"
        artifact_data = [artifact.model_dump() for artifact in self._artifacts.values()]
        artifacts_dict = {"artifacts": artifact_data}

        try:
            logger.info(f"Saving {len(artifact_data)} artifacts of type {self.type} to {path}")
            await self._storage.write_to_trajectory(path, artifacts_dict)
        except Exception as e:
            logger.exception(f"Error saving artifacts of type {self.type}: {e}")

    def generate_id(self) -> str:
        """Generate a unique ID for an artifact"""
        return f"{self.type}-{len(self._artifacts) + 1}"
