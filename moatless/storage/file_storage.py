"""
File-based storage implementation.

This module provides a storage implementation that reads and writes
data to files on disk.
"""

import json
import logging
import os
from pathlib import Path
from typing import Union, Optional

import aiofiles

from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)


class FileStorage(BaseStorage):
    """
    Storage implementation that uses the filesystem.

    This class provides a storage implementation that reads and writes
    data to files on disk.
    """

    def __init__(self, base_dir: Union[str, Path] | None = None, file_extension: str = "json"):
        """
        Initialize a FileStorage instance.

        Args:
            base_dir: The base directory for storage
            file_extension: The file extension to use for stored files
        """

        if not base_dir:
            self.base_dir = Path(os.environ["MOATLESS_DIR"])
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.file_extension = file_extension

    def _get_path(self, key: str) -> Path:
        """
        Get the file path for a key.

        Handles hierarchical keys by creating subdirectories as needed.

        Args:
            key: The key to convert to a file path

        Returns:
            The file path for the key
        """
        normalized_key = self.normalize_key(key)

        if "/" in normalized_key:
            # Handle hierarchical keys by creating appropriate directory structure
            parts = normalized_key.split("/")
            filename = parts[-1]

            # Create subdirectories if needed
            if len(parts) > 1:
                parent_dir = self.base_dir.joinpath(*parts[:-1])
                parent_dir.mkdir(parents=True, exist_ok=True)
                return parent_dir / f"{filename}.{self.file_extension}"

        # Simple case - just append extension
        return self.base_dir / f"{normalized_key}.{self.file_extension}"

    async def read(self, key: str) -> dict:
        """Read binary data from a file."""
        path = self._get_path(key)
        if not path.exists():
            logger.error(f"Path '{path}' does not exist")
            raise KeyError(f"Key '{key}' does not exist")

        try:
            async with aiofiles.open(path, "r") as f:
                return json.loads(await f.read())
        except Exception as e:
            logger.error(f"Error reading from {path}: {e}")
            raise

    async def write(self, key: str, data: dict) -> None:
        """Write binary data to a file."""
        path = self._get_path(key)
        try:
            async with aiofiles.open(path, "w") as f:
                await f.write(json.dumps(data))
        except Exception as e:
            logger.error(f"Error writing to {path}: {e}")
            raise

    async def delete(self, key: str) -> None:
        """Delete a file."""
        path = self._get_path(key)
        if not path.exists():
            raise KeyError(f"Key '{key}' does not exist")

        try:
            path.unlink()
        except Exception as e:
            logger.error(f"Error deleting {path}: {e}")
            raise

    async def exists(self, key: str) -> bool:
        """Check if a file exists."""
        path = self._get_path(key)
        return path.exists()

    async def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys with the given prefix.

        Args:
            prefix: The key prefix to search for

        Returns:
            A list of keys that match the prefix
        """
        normalized_prefix = self.normalize_key(prefix)

        # If prefix is empty, we need to search the entire base directory
        if not normalized_prefix:
            return await self._list_all_keys()

        # Handle file system path considerations
        if "/" in normalized_prefix:
            parts = normalized_prefix.split("/")
            search_dir = self.base_dir.joinpath(*parts)
        else:
            search_dir = self.base_dir / normalized_prefix

        # The directory might not exist yet
        if not search_dir.exists():
            return []

        # If it's a file, not a directory
        if search_dir.is_file():
            # Extract key from the file path
            rel_path = search_dir.relative_to(self.base_dir)
            # Remove file extension if present
            key = str(rel_path)
            if key.endswith(f".{self.file_extension}"):
                key = key[: -len(f".{self.file_extension}")]
            return [key]

        # It's a directory, so find all files recursively
        result = []

        # Walk through the directory
        for path in search_dir.glob(f"**/*.{self.file_extension}"):
            if path.is_file():
                # Get the relative path from the base directory
                rel_path = path.relative_to(self.base_dir)
                # Convert to key format and remove extension
                key = str(rel_path).replace("\\", "/")  # Normalize path separators
                if key.endswith(f".{self.file_extension}"):
                    key = key[: -len(f".{self.file_extension}")]
                result.append(key)

        return result

    async def _list_all_keys(self) -> list[str]:
        """
        List all keys in the storage.

        Returns:
            A list of all keys
        """
        result = []

        # Walk through the entire base directory
        for path in self.base_dir.glob(f"**/*.{self.file_extension}"):
            if path.is_file():
                # Get the relative path from the base directory
                rel_path = path.relative_to(self.base_dir)
                # Convert to key format and remove extension
                key = str(rel_path).replace("\\", "/")  # Normalize path separators
                if key.endswith(f".{self.file_extension}"):
                    key = key[: -len(f".{self.file_extension}")]
                result.append(key)

        return result

    async def assert_exists(self, key: str) -> None:
        """
        Assert that a key exists in storage.
        """
        path = self._get_path(key)
        if not path.exists():
            raise KeyError(f"Key '{key}' does not exist, path: {path}")
