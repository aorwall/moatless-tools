"""
File-based storage implementation.

This module provides a storage implementation that reads and writes
data to files on disk.
"""

import json
import logging
import os
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from datetime import datetime

import aiofiles

from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

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
            # For keys with slashes, create a directory structure
            parts = normalized_key.split("/")
            dir_path = self.base_dir.joinpath(*parts[:-1])
            dir_path.mkdir(parents=True, exist_ok=True)

            # Only add file extension to the last part
            filename = f"{parts[-1]}.{self.file_extension}"
            return dir_path / filename
        else:
            return self.base_dir / f"{normalized_key}.{self.file_extension}"

    async def read(self, key: str) -> dict:
        """
        Read JSON data from a file.

        Args:
            key: The key to read

        Returns:
            The parsed JSON data

        Raises:
            KeyError: If the key does not exist
        """
        path = self._get_path(key)
        if not path.exists():
            raise KeyError(f"No data found for key: {key}")

        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)

    async def read_raw(self, key: str) -> str:
        """
        Read raw string data from a file without parsing.

        Args:
            key: The key to read

        Returns:
            The raw file contents as a string

        Raises:
            KeyError: If the key does not exist
        """
        path = self._get_path(key)
        if not path.exists():
            raise KeyError(f"No data found for key: {key}")

        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()

    async def read_lines(self, key: str) -> List[dict]:
        """
        Read data from a JSONL file, parsing each line as a JSON object.

        Args:
            key: The key to read

        Returns:
            A list of parsed JSON objects, one per line

        Raises:
            KeyError: If the key does not exist
        """
        path = self._get_path(key)
        if not path.exists():
            raise KeyError(f"No data found for key: {key}")

        results = []
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    results.append(json.loads(line))
        return results

    async def write(self, key: str, data: dict) -> None:
        """
        Write data to a file as JSON.

        Args:
            key: The key to write to
            data: The data to write
        """
        path = self._get_path(key)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2, cls=DateTimeEncoder))

    async def write_raw(self, key: str, data: str) -> None:
        """
        Write raw string data to a file.

        Args:
            key: The key to write to
            data: The string data to write
        """
        path = self._get_path(key)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(data)

    async def append(self, key: str, data: Union[dict, str]) -> None:
        """
        Append data to an existing file.

        Args:
            key: The key to append to
            data: The data to append. If dict, it will be serialized as JSON.
                 If string, it will be written as-is with a newline.
        """
        path = self._get_path(key)

        # Create the file if it doesn't exist
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            # Convert to JSON string if it's a dict
            if isinstance(data, dict):
                line = json.dumps(data, cls=DateTimeEncoder)
            else:
                line = data

            # Make sure the line ends with a newline
            if not line.endswith("\n"):
                line += "\n"

            await f.write(line)

    async def delete(self, key: str) -> None:
        """
        Delete a file.

        Args:
            key: The key to delete

        Raises:
            KeyError: If the key does not exist
        """
        path = self._get_path(key)
        if not path.exists():
            raise KeyError(f"No data found for key: {key}")

        os.remove(path)

    async def exists(self, key: str) -> bool:
        """
        Check if a file exists.

        Args:
            key: The key to check

        Returns:
            True if the file exists, False otherwise
        """
        path = self._get_path(key)
        return path.exists()

    async def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys with the given prefix.

        Args:
            prefix: The prefix to filter keys by

        Returns:
            A list of keys
        """
        normalized_prefix = self.normalize_key(prefix)
        all_keys = await self._list_all_keys()

        if normalized_prefix:
            return [key for key in all_keys if key.startswith(normalized_prefix)]
        else:
            return all_keys

    async def _list_all_keys(self) -> list[str]:
        """
        List all keys in the storage.

        Returns:
            A list of all keys
        """
        keys = []
        for path in self.base_dir.glob(f"**/*.{self.file_extension}"):
            # Get relative path to base_dir
            rel_path = path.relative_to(self.base_dir)
            # Remove file extension
            key = str(rel_path.with_suffix(""))
            # Replace path separators with forward slashes
            key = key.replace(os.path.sep, "/")
            keys.append(key)
        return keys

    async def assert_exists(self, key: str) -> None:
        """
        Assert that a key exists in storage.
        """
        path = self._get_path(key)
        if not path.exists():
            raise KeyError(f"Key '{key}' does not exist, path: {path}")
