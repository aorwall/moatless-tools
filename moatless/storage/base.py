import abc
import json
import os
import logging
from typing import TypeVar, Optional, cast, Union, Type, Any, List
from pathlib import Path
from datetime import datetime

from moatless.context_data import get_project_dir, get_trajectory_dir, current_project_id, current_trajectory_id

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


T = TypeVar("T")


class BaseStorage(abc.ABC):
    """
    Abstract base class for storage operations.

    This class defines the interface for storage operations on artifacts, trajectories, configurations, etc.
    """

    _instance = None

    @classmethod
    def get_instance(cls, storage_impl: Optional[Type["BaseStorage"]] = None, **kwargs) -> "BaseStorage":
        """
        Get or create the singleton instance of Storage.

        Args:
            storage_impl: Optional storage implementation class to use, defaults to FileStorage
            **kwargs: Arguments to pass to the storage implementation constructor

        Returns:
            The singleton Storage instance
        """
        if cls._instance is None:
            # Import here to avoid circular imports
            from moatless.storage.file_storage import FileStorage

            # Use FileStorage as default implementation
            impl_class = storage_impl or FileStorage
            logger.info(f"Using {impl_class.__name__} as storage implementation")
            cls._instance = impl_class(**kwargs)

        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Mainly useful for testing."""
        cls._instance = None

    async def read(self, path: str) -> dict | list[dict] | str:
        """
        Read JSON data from a file.

        Args:
            path: The path to read

        Returns:
            The parsed JSON data or an empty dict if the file is empty

        Raises:
            KeyError: If the path does not exist
        """
        content = await self.read_raw(path)
        if content.strip().startswith("{") or content.strip().startswith("[") or path.endswith(".json"):
            return json.loads(content)
        else:
            return content

    @abc.abstractmethod
    async def read_raw(self, path: str) -> str:
        """
        Read raw data from storage without parsing.

        Args:
            path: The identifier for the data to read

        Returns:
            The raw data as a string

        Raises:
            KeyError: If the path does not exist
        """
        pass

    @abc.abstractmethod
    async def read_lines(self, path: str) -> List[dict]:
        """
        Read data from a JSONL or similar line-based format.

        Args:
            path: The identifier for the data to read

        Returns:
            A list of data objects, one per line

        Raises:
            KeyError: If the path does not exist
        """
        pass

    async def write(self, path: str, data: dict | list[dict] | str) -> None:
        """
        Write data to a file as JSON.

        Args:
            path: The path to write to
            data: The data to write
        """
        if isinstance(data, str):
            await self.write_raw(path, data)
        else:
            await self.write_raw(path, json.dumps(data, indent=2, cls=DateTimeEncoder))

    @abc.abstractmethod
    async def write_raw(self, path: str, data: str) -> None:
        """
        Write raw string data to storage.

        Args:
            path: The identifier for the data
            data: The string data to write
        """
        pass

    @abc.abstractmethod
    async def append(self, path: str, data: Union[dict, str]) -> None:
        """
        Append data to an existing file.

        Args:
            path: The identifier for the data
            data: The data to append. If dict, it will be serialized as JSON.
                 If string, it will be written as-is with a newline.
        """
        pass

    @abc.abstractmethod
    async def delete(self, path: str) -> None:
        """
        Delete data from storage.

        Args:
            path: The identifier for the data to delete

        Raises:
            KeyError: If the path does not exist
        """
        pass

    @abc.abstractmethod
    async def exists(self, path: str) -> bool:
        """
        Check if data exists in storage.

        Args:
            path: The identifier for the data to check

        Returns:
            True if the data exists, False otherwise
        """
        pass

    async def assert_exists(self, path: str) -> None:
        """
        Assert that data exists in storage.

        Args:
            path: The identifier for the data to check

        Raises:
            KeyError: If the path does not exist
        """
        if not await self.exists(path):
            raise KeyError(f"Path '{path}' does not exist")

    @abc.abstractmethod
    async def list_paths(self, prefix: str = "") -> list[str]:
        """
        List all paths in storage with the given prefix.

        Args:
            prefix: The prefix to filter paths by

        Returns:
            A list of paths
        """
        pass

    def normalize_path(self, path: str) -> str:
        """
        Normalize a path to ensure consistent format.

        Args:
            path: The path to normalize

        Returns:
            The normalized path
        """
        # Remove leading/trailing slashes and whitespace
        return path.strip().strip("/")

    def _get_project_id(self, project_id: Optional[str] = None) -> str:
        """
        Get a valid project ID from arguments or context.

        Args:
            project_id: Optional project ID

        Returns:
            A valid project ID

        Raises:
            ValueError: If no project ID is provided and none is in context
        """
        if project_id is not None:
            return project_id

        project_id_context = current_project_id.get()
        if project_id_context is None:
            raise ValueError("No project ID provided and no project context set")

        return project_id_context

    def _get_trajectory_id(self, trajectory_id: Optional[str] = None) -> str:
        """
        Get a valid trajectory ID from arguments or context.

        Args:
            trajectory_id: Optional trajectory ID

        Returns:
            A valid trajectory ID

        Raises:
            ValueError: If no trajectory ID is provided and none is in context
        """
        if trajectory_id is not None:
            return trajectory_id

        trajectory_id_context = current_trajectory_id.get()
        if trajectory_id_context is None:
            raise ValueError("No trajectory ID provided and no trajectory context set")

        return trajectory_id_context

    async def read_from_project(self, path: str, project_id: Optional[str] = None) -> dict | list[dict] | str:
        """
        Read data from a project.

        Args:
            path: The identifier for the data
            project_id: The project ID (defaults to current project context if None)

        Returns:
            The data associated with the path

        Raises:
            KeyError: If the path does not exist
            ValueError: If no project ID is provided and no project context is set
        """
        project_id = self._get_project_id(project_id)
        project_path = f"projects/{project_id}/{path}"
        try:
            return await self.read(project_path)
        except KeyError:
            logger.warning(f"No data found for project path: {project_path}")
            return {}

    async def write_to_project(self, path: str, data: dict, project_id: Optional[str] = None) -> None:
        """
        Write data to a project.

        Args:
            path: The identifier for the data
            data: The data to write
            project_id: The project ID (defaults to current project context if None)

        Raises:
            ValueError: If no project ID is provided and no project context is set
        """
        project_id = self._get_project_id(project_id)
        project_path = f"projects/{project_id}/{path}"
        await self.write(project_path, data)

    async def exists_in_project(self, path: str, project_id: Optional[str] = None) -> bool:
        """
        Check if a path exists in a project.

        Args:
            path: The identifier to check
            project_id: The project ID (defaults to current project context if None)

        Returns:
            True if the path exists, False otherwise

        Raises:
            ValueError: If no project ID is provided and no project context is set
        """
        project_id = self._get_project_id(project_id)
        project_path = f"projects/{project_id}/{path}"
        return await self.exists(project_path)

    async def read_from_trajectory(
        self, path: str, project_id: Optional[str] = None, trajectory_id: Optional[str] = None
    ) -> dict | list[dict] | str:
        """
        Read data from a trajectory.

        Args:
            path: The identifier for the data to read
            project_id: The project ID (defaults to current project context if None)
            trajectory_id: The trajectory ID (defaults to current trajectory context if None)

        Returns:
            The data associated with the path

        Raises:
            KeyError: If the path does not exist
            ValueError: If no project ID or trajectory ID is provided and none is in context
        """
        project_id = self._get_project_id(project_id)
        trajectory_id = self._get_trajectory_id(trajectory_id)

        trajectory_path = f"projects/{project_id}/trajs/{trajectory_id}/{path}"
        return await self.read(trajectory_path)

    async def write_to_trajectory(
        self, path: str, data: dict, project_id: Optional[str] = None, trajectory_id: Optional[str] = None
    ) -> None:
        """
        Write data to a trajectory.

        Args:
            path: The identifier for the data
            data: The data to write
            project_id: The project ID (defaults to current project context if None)
            trajectory_id: The trajectory ID (defaults to current trajectory context if None)

        Raises:
            ValueError: If no project ID or trajectory ID is provided and none is in context
        """
        project_id = self._get_project_id(project_id)
        trajectory_id = self._get_trajectory_id(trajectory_id)

        trajectory_path = f"projects/{project_id}/trajs/{trajectory_id}/{path}"
        await self.write(trajectory_path, data)

    async def exists_in_trajectory(
        self, path: str, project_id: Optional[str] = None, trajectory_id: Optional[str] = None
    ) -> bool:
        """
        Check if a path exists in a trajectory.

        Args:
            path: The identifier to check
            project_id: The project ID (defaults to current project context if None)
            trajectory_id: The trajectory ID (defaults to current trajectory context if None)

        Returns:
            True if the path exists, False otherwise

        Raises:
            ValueError: If no project ID or trajectory ID is provided and none is in context
        """
        project_id = self._get_project_id(project_id)
        trajectory_id = self._get_trajectory_id(trajectory_id)

        trajectory_path = f"projects/{project_id}/trajs/{trajectory_id}/{path}"
        return await self.exists(trajectory_path)

    async def assert_exists_in_trajectory(
        self, path: str, project_id: Optional[str] = None, trajectory_id: Optional[str] = None
    ) -> None:
        """
        Assert that a path exists in a trajectory.
        """

        project_id = self._get_project_id(project_id)
        trajectory_id = self._get_trajectory_id(trajectory_id)

        trajectory_path = f"projects/{project_id}/trajs/{trajectory_id}/{path}"
        await self.assert_exists(trajectory_path)

    def get_trajectory_path(self, project_id: str | None = None, trajectory_id: str | None = None) -> str:
        if project_id is None:
            project_id = self._get_project_id()

        if trajectory_id is None:
            trajectory_id = self._get_trajectory_id()

        return f"projects/{project_id}/trajs/{trajectory_id}"
