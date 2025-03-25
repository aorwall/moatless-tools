"""
File-based storage implementation.

This module provides a storage implementation that reads and writes
data to files on disk.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, cast, TypeVar, Sequence, overload

import aiofiles
from moatless.storage.base import BaseStorage
from opentelemetry import trace

logger = logging.getLogger(__name__)


tracer = trace.get_tracer(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class FileStorage(BaseStorage):
    """
    Storage implementation that uses the filesystem.

    This class provides a storage implementation that reads and writes
    data to files on disk.
    """

    def __init__(self, base_dir: Union[str, Path] | None = None):
        """
        Initialize a FileStorage instance.

        Args:
            base_dir: The base directory for storage
        """

        if not base_dir:
            self.base_dir = Path(os.environ["MOATLESS_DIR"])
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"File storage initialized in {self.base_dir}")

    def __str__(self) -> str:
        return f"FileStorage(base_dir={self.base_dir})"

    def _get_path(self, path: str) -> Path:
        """
        Get the file path for a path.

        Handles hierarchical paths by creating subdirectories as needed.

        Args:
            path: The path to convert to a file path

        Returns:
            The file path for the path
        """
        normalized_path = self.normalize_path(path)
        return self.base_dir / normalized_path

    async def read_raw(self, path: str) -> str:
        """
        Read raw string data from a file without parsing.

        Args:
            path: The path to read

        Returns:
            The raw file contents as a string

        Raises:
            KeyError: If the path does not exist
        """
        file_path = self._get_path(path)
        if not file_path.exists():
            raise KeyError(f"No data found for path: {path}")

        async with aiofiles.open(str(file_path), "r", encoding="utf-8") as f:
            return await f.read()

    async def read_lines(self, path: str) -> List[dict]:
        """
        Read data from a JSONL file, parsing each line as a JSON object.

        Args:
            path: The path to read

        Returns:
            A list of parsed JSON objects, one per line

        Raises:
            KeyError: If the path does not exist
        """
        file_path = self._get_path(path)
        if not file_path.exists():
            raise KeyError(f"No data found for path: {path}")

        results = []
        async with aiofiles.open(str(file_path), "r", encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    results.append(json.loads(line))
        return results

    async def write_raw(self, path: str, data: str) -> None:
        """
        Write raw string data to a file.

        Args:
            path: The path to write to
            data: The string data to write
        """
        file_path = self._get_path(path)

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(str(file_path), "w", encoding="utf-8") as f:
            await f.write(data)

    async def append(self, path: str, data: Union[dict, str]) -> None:
        """
        Append data to an existing file.

        Args:
            path: The path to append to
            data: The data to append. If dict, it will be serialized as JSON.
                 If string, it will be written as-is with a newline.
        """
        file_path = self._get_path(path)

        # Create the file if it doesn't exist
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(str(file_path), "a", encoding="utf-8") as f:
            # Convert to JSON string if it's a dict
            if isinstance(data, dict):
                line = json.dumps(data, cls=DateTimeEncoder)
            else:
                line = data

            # Make sure the line ends with a newline
            if not line.endswith("\n"):
                line += "\n"

            await f.write(line)

    async def delete(self, path: str) -> None:
        """
        Delete a file.

        Args:
            path: The path to delete

        Raises:
            KeyError: If the path does not exist
        """
        file_path = self._get_path(path)
        if not file_path.exists():
            raise KeyError(f"No data found for path: {path}")

        os.remove(str(file_path))

    async def exists(self, path: str) -> bool:
        """
        Check if a file exists.

        Args:
            path: The path to check

        Returns:
            True if the file exists, False otherwise
        """
        file_path = self._get_path(path)
        exists = file_path.exists()
        if not exists:
            logger.info(f"File {file_path} does not exist")
        return exists

    @tracer.start_as_current_span("FileStorage.list_paths")
    async def list_paths(self, prefix: str = "") -> list[str]:
        """
        List all paths with the given prefix.

        Args:
            prefix: The prefix to filter paths by. Can be a directory path like '/path/to'
                   or a file prefix like '/path/to/file_'

        Returns:
            A list of paths
        """
        normalized_prefix = self.normalize_path(prefix)

        # Convert the normalized prefix into a filesystem path
        if normalized_prefix:
            prefix_path = self.base_dir / normalized_prefix.replace("/", os.path.sep)
        else:
            prefix_path = self.base_dir

        # If prefix_path points to an existing directory, list all files inside it
        if prefix_path.is_dir():
            return await self._list_paths_in_dir(prefix_path)

        # If not a directory, it might be a file prefix
        # Get the parent directory and the file prefix
        parent_dir = prefix_path.parent
        file_prefix = prefix_path.name

        if parent_dir.is_dir():
            # Only look in this specific directory for files with the given prefix
            return await self._list_paths_with_prefix(parent_dir, file_prefix)

        # Fallback to empty list if path doesn't exist at all
        return []

    async def _list_paths_in_dir(self, directory: Path) -> list[str]:
        """
        List all paths in a specific directory.

        Args:
            directory: Directory path to list paths from

        Returns:
            A list of paths
        """
        paths = []
        # Only search in the specified directory and its immediate subdirectories
        # This is more efficient than globbing the entire file system
        for path in directory.glob(f"**/*"):
            # Get relative path to base_dir
            rel_path = path.relative_to(self.base_dir)
            # Convert to string
            path_str = str(rel_path)
            # Replace path separators with forward slashes
            path_str = path_str.replace(os.path.sep, "/")
            paths.append(path_str)
        return paths

    async def _list_paths_with_prefix(self, directory: Path, prefix: str) -> list[str]:
        """
        List all paths in a directory that start with a specific prefix.

        Args:
            directory: Directory path to search in
            prefix: File prefix to match

        Returns:
            A list of paths
        """
        paths = []
        # Only search for files matching the prefix in the specified directory
        # This is much more efficient than scanning the entire file system
        for path in directory.glob(f"{prefix}*"):
            # Get relative path to base_dir
            rel_path = path.relative_to(self.base_dir)
            # Convert to string
            path_str = str(rel_path)
            # Replace path separators with forward slashes
            path_str = path_str.replace(os.path.sep, "/")
            paths.append(path_str)
        return paths

    async def _list_all_paths(self) -> list[str]:
        """
        List all paths in the storage.

        Returns:
            A list of all paths
        """
        return await self._list_paths_in_dir(self.base_dir)

    async def assert_exists(self, path: str) -> None:
        """
        Assert that a path exists in storage.
        """
        file_path = self._get_path(path)
        if not file_path.exists():
            raise KeyError(f"Key '{path}' does not exist, path: {file_path}")
