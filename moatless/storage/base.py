import abc
import logging
from typing import TypeVar, Optional, cast, Union, Type, Any, List
from pathlib import Path

from moatless.context_data import get_project_dir, get_trajectory_dir, current_project_id, current_trajectory_id

logger = logging.getLogger(__name__)

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

    @abc.abstractmethod
    async def read(self, key: str) -> dict:
        """
        Read data from storage.

        Args:
            key: The identifier for the data to read

        Returns:
            The data associated with the key

        Raises:
            KeyError: If the key does not exist
        """
        pass

    @abc.abstractmethod
    async def read_raw(self, key: str) -> str:
        """
        Read raw data from storage without parsing.

        Args:
            key: The identifier for the data to read

        Returns:
            The raw data as a string

        Raises:
            KeyError: If the key does not exist
        """
        pass

    @abc.abstractmethod
    async def read_lines(self, key: str) -> List[dict]:
        """
        Read data from a JSONL or similar line-based format.

        Args:
            key: The identifier for the data to read

        Returns:
            A list of data objects, one per line

        Raises:
            KeyError: If the key does not exist
        """
        pass

    @abc.abstractmethod
    async def write(self, key: str, data: dict) -> None:
        """
        Write data to storage.

        Args:
            key: The identifier for the data
            data: The data to write
        """
        pass

    @abc.abstractmethod
    async def write_raw(self, key: str, data: str) -> None:
        """
        Write raw string data to storage.

        Args:
            key: The identifier for the data
            data: The string data to write
        """
        pass

    @abc.abstractmethod
    async def append(self, key: str, data: Union[dict, str]) -> None:
        """
        Append data to an existing file.

        Args:
            key: The identifier for the data
            data: The data to append. If dict, it will be serialized as JSON.
                 If string, it will be written as-is with a newline.
        """
        pass

    @abc.abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete data from storage.

        Args:
            key: The identifier for the data to delete

        Raises:
            KeyError: If the key does not exist
        """
        pass

    @abc.abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if data exists in storage.

        Args:
            key: The identifier for the data to check

        Returns:
            True if the data exists, False otherwise
        """
        pass

    async def assert_exists(self, key: str) -> None:
        """
        Assert that data exists in storage.

        Args:
            key: The identifier for the data to check

        Raises:
            KeyError: If the key does not exist
        """
        if not await self.exists(key):
            raise KeyError(f"Key '{key}' does not exist")

    @abc.abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys in storage with the given prefix.

        Args:
            prefix: The prefix to filter keys by

        Returns:
            A list of keys
        """
        pass

    def normalize_key(self, key: str) -> str:
        """
        Normalize a key to ensure consistent format.

        Args:
            key: The key to normalize

        Returns:
            The normalized key
        """
        # Remove leading/trailing slashes and whitespace
        return key.strip().strip("/")

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

    async def read_from_project(self, key: str, project_id: Optional[str] = None) -> dict:
        """
        Read data from a project.

        Args:
            key: The identifier for the data to read
            project_id: The project ID (defaults to current project context if None)

        Returns:
            The data associated with the key

        Raises:
            KeyError: If the key does not exist
            ValueError: If no project ID is provided and no project context is set
        """
        project_id = self._get_project_id(project_id)
        project_key = f"projects/{project_id}/{key}"
        return await self.read(project_key)

    async def write_to_project(self, key: str, data: dict, project_id: Optional[str] = None) -> None:
        """
        Write data to a project.

        Args:
            key: The identifier for the data
            data: The data to write
            project_id: The project ID (defaults to current project context if None)

        Raises:
            ValueError: If no project ID is provided and no project context is set
        """
        project_id = self._get_project_id(project_id)
        project_key = f"projects/{project_id}/{key}"
        await self.write(project_key, data)

    async def delete_from_project(self, key: str, project_id: Optional[str] = None) -> None:
        """
        Delete data from a project.

        Args:
            key: The identifier for the data to delete
            project_id: The project ID (defaults to current project context if None)

        Raises:
            KeyError: If the key does not exist
            ValueError: If no project ID is provided and no project context is set
        """
        project_id = self._get_project_id(project_id)
        project_key = f"projects/{project_id}/{key}"
        await self.delete(project_key)

    async def exists_in_project(self, key: str, project_id: Optional[str] = None) -> bool:
        """
        Check if a key exists in a project.

        Args:
            key: The identifier to check
            project_id: The project ID (defaults to current project context if None)

        Returns:
            True if the key exists, False otherwise

        Raises:
            ValueError: If no project ID is provided and no project context is set
        """
        project_id = self._get_project_id(project_id)
        project_key = f"projects/{project_id}/{key}"
        return await self.exists(project_key)

    # Shorter aliases for project operations
    async def p_read(self, key: str, project_id: Optional[str] = None) -> dict:
        """Shorthand for read_from_project."""
        return await self.read_from_project(key, project_id)

    async def p_write(self, key: str, data: dict, project_id: Optional[str] = None) -> None:
        """Shorthand for write_to_project."""
        await self.write_to_project(key, data, project_id)

    async def p_delete(self, key: str, project_id: Optional[str] = None) -> None:
        """Shorthand for delete_from_project."""
        await self.delete_from_project(key, project_id)

    async def p_exists(self, key: str, project_id: Optional[str] = None) -> bool:
        """Shorthand for exists_in_project."""
        return await self.exists_in_project(key, project_id)

    async def read_from_trajectory(
        self, key: str, project_id: Optional[str] = None, trajectory_id: Optional[str] = None
    ) -> dict:
        """
        Read data from a trajectory.

        Args:
            key: The identifier for the data to read
            project_id: The project ID (defaults to current project context if None)
            trajectory_id: The trajectory ID (defaults to current trajectory context if None)

        Returns:
            The data associated with the key

        Raises:
            KeyError: If the key does not exist
            ValueError: If no project ID or trajectory ID is provided and none is in context
        """
        project_id = self._get_project_id(project_id)
        trajectory_id = self._get_trajectory_id(trajectory_id)

        trajectory_key = f"projects/{project_id}/trajs/{trajectory_id}/{key}"
        return await self.read(trajectory_key)

    async def write_to_trajectory(
        self, key: str, data: dict, project_id: Optional[str] = None, trajectory_id: Optional[str] = None
    ) -> None:
        """
        Write data to a trajectory.

        Args:
            key: The identifier for the data
            data: The data to write
            project_id: The project ID (defaults to current project context if None)
            trajectory_id: The trajectory ID (defaults to current trajectory context if None)

        Raises:
            ValueError: If no project ID or trajectory ID is provided and none is in context
        """
        project_id = self._get_project_id(project_id)
        trajectory_id = self._get_trajectory_id(trajectory_id)

        trajectory_key = f"projects/{project_id}/trajs/{trajectory_id}/{key}"
        await self.write(trajectory_key, data)

    async def delete_from_trajectory(
        self, key: str, project_id: Optional[str] = None, trajectory_id: Optional[str] = None
    ) -> None:
        """
        Delete data from a trajectory.

        Args:
            key: The identifier for the data to delete
            project_id: The project ID (defaults to current project context if None)
            trajectory_id: The trajectory ID (defaults to current trajectory context if None)

        Raises:
            KeyError: If the key does not exist
            ValueError: If no project ID or trajectory ID is provided and none is in context
        """
        project_id = self._get_project_id(project_id)
        trajectory_id = self._get_trajectory_id(trajectory_id)

        trajectory_key = f"projects/{project_id}/trajs/{trajectory_id}/{key}"
        await self.delete(trajectory_key)

    async def exists_in_trajectory(
        self, key: str, project_id: Optional[str] = None, trajectory_id: Optional[str] = None
    ) -> bool:
        """
        Check if a key exists in a trajectory.

        Args:
            key: The identifier to check
            project_id: The project ID (defaults to current project context if None)
            trajectory_id: The trajectory ID (defaults to current trajectory context if None)

        Returns:
            True if the key exists, False otherwise

        Raises:
            ValueError: If no project ID or trajectory ID is provided and none is in context
        """
        project_id = self._get_project_id(project_id)
        trajectory_id = self._get_trajectory_id(trajectory_id)

        trajectory_key = f"projects/{project_id}/trajs/{trajectory_id}/{key}"
        return await self.exists(trajectory_key)

    async def assert_exists_in_trajectory(
        self, key: str, project_id: Optional[str] = None, trajectory_id: Optional[str] = None
    ) -> None:
        """
        Assert that a key exists in a trajectory.
        """

        project_id = self._get_project_id(project_id)
        trajectory_id = self._get_trajectory_id(trajectory_id)

        trajectory_key = f"projects/{project_id}/trajs/{trajectory_id}/{key}"
        await self.assert_exists(trajectory_key)

    # Shorter aliases for trajectory operations
    async def t_read(self, key: str, project_id: Optional[str] = None, trajectory_id: Optional[str] = None) -> dict:
        """Shorthand for read_from_trajectory."""
        return await self.read_from_trajectory(key, project_id, trajectory_id)

    async def t_write(
        self, key: str, data: dict, project_id: Optional[str] = None, trajectory_id: Optional[str] = None
    ) -> None:
        """Shorthand for write_to_trajectory."""
        await self.write_to_trajectory(key, data, project_id, trajectory_id)

    async def t_delete(self, key: str, project_id: Optional[str] = None, trajectory_id: Optional[str] = None) -> None:
        """Shorthand for delete_from_trajectory."""
        await self.delete_from_trajectory(key, project_id, trajectory_id)

    async def t_exists(self, key: str, project_id: Optional[str] = None, trajectory_id: Optional[str] = None) -> bool:
        """Shorthand for exists_in_trajectory."""
        return await self.exists_in_trajectory(key, project_id, trajectory_id)

    async def list_projects(self) -> list[str]:
        """
        List all projects in storage.

        Returns:
            A list of project IDs
        """
        # Get all keys starting with "projects/"
        keys = await self.list_keys("projects")

        # Extract project IDs from keys
        project_ids = set()
        for key in keys:
            parts = key.split("/")
            if len(parts) > 1:
                project_ids.add(parts[1])

        return sorted(list(project_ids))

    async def list_trajectories(self, project_id: str) -> list[str]:
        """
        List all trajectories in a project.

        Args:
            project_id: The project ID

        Returns:
            A list of trajectory IDs
        """
        # Get all keys starting with "projects/{project_id}/trajs/"
        keys = await self.list_keys(f"projects/{project_id}/trajs")

        # Extract trajectory IDs from keys
        trajectory_ids = set()
        for key in keys:
            parts = key.split("/")
            if len(parts) > 3 and parts[0] == "projects" and parts[1] == project_id and parts[2] == "trajs":
                trajectory_ids.add(parts[3])

        return sorted(list(trajectory_ids))

    async def list_evaluations(self) -> list[dict]:
        """
        List all evaluations in storage.

        Returns:
            A list of evaluation data dictionaries
        """
        # Get all project IDs that could be evaluations (starting with "eval_")
        project_ids = await self.list_projects()
        evaluation_ids = [pid for pid in project_ids if pid.startswith("eval_")]

        # Get evaluation data for each evaluation ID
        evaluations = []
        for eval_id in evaluation_ids:
            try:
                # First try direct evaluation.json in project directory
                if await self.exists(f"projects/{eval_id}/evaluation.json"):
                    eval_data = await self.read(f"projects/{eval_id}/evaluation.json")
                    evaluations.append(eval_data)
                    continue

                # Then try other patterns without needing context
                try:
                    # Try using the evaluation ID as the explicit project ID
                    if await self.exists_in_project("evaluation", eval_id):
                        eval_data = await self.read_from_project("evaluation", eval_id)
                        evaluations.append(eval_data)
                        continue
                except ValueError:
                    # This is expected if there's no context
                    pass

                # Finally try the evaluation/{id} pattern under default
                try:
                    key = f"evaluation/{eval_id}"
                    # Use a temporary override for the project ID
                    eval_data = await self.read(f"projects/default/{key}")
                    evaluations.append(eval_data)
                except (KeyError, FileNotFoundError):
                    # This is also expected if the eval doesn't exist
                    pass

            except Exception as e:
                # Log but don't crash on evaluation read errors
                logger.error(f"Error reading evaluation {eval_id}: {e}")

        return evaluations

    @classmethod
    def get_default_storage(cls) -> "BaseStorage":
        """
        Get the default storage backend.
        """
        from moatless.storage.file_storage import FileStorage

        return FileStorage()
