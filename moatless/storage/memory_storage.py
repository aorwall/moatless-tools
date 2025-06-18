"""
In-memory storage implementation.

This module provides a storage implementation that keeps all data
in memory, which is useful for testing or temporary storage.
"""

import json
import logging
from typing import Union

from moatless.storage.base import BaseStorage, DateTimeEncoder

logger = logging.getLogger(__name__)


class MemoryStorage(BaseStorage):
    """Simple in-memory implementation of :class:`BaseStorage`."""

    def __init__(self) -> None:
        """Initialize an empty in-memory storage."""
        self._data: dict[str, object] = {}

    async def read_raw(self, path: str) -> str:
        """Return the raw value stored under *path*."""
        normalized_key = self.normalize_path(path)
        if normalized_key not in self._data:
            raise KeyError(f"Key '{path}' does not exist")

        value = self._data[normalized_key]
        if isinstance(value, list):
            # Represent lists as JSONL
            return "\n".join(
                json.dumps(v, cls=DateTimeEncoder) if isinstance(v, dict) else str(v)
                for v in value
            )
        if isinstance(value, dict):
            return json.dumps(value, cls=DateTimeEncoder)
        return str(value)

    async def read_lines(self, path: str) -> list[dict]:
        """Return a list of objects stored under *path*."""
        normalized_key = self.normalize_path(path)
        if normalized_key not in self._data:
            raise KeyError(f"Key '{path}' does not exist")

        value = self._data[normalized_key]
        results: list[dict] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    item = item.strip()
                    if item:
                        results.append(json.loads(item))
                else:
                    results.append(item)
        elif isinstance(value, str):
            for line in value.splitlines():
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        elif isinstance(value, dict):
            results.append(value)
        return results

    async def write_raw(self, path: str, data: str) -> None:
        """Write raw string *data* to *path*."""
        normalized_key = self.normalize_path(path)
        self._data[normalized_key] = data

    async def append(self, path: str, data: Union[dict, str]) -> None:
        """Append *data* to the entry at *path*."""
        normalized_key = self.normalize_path(path)
        existing = self._data.get(normalized_key)

        if existing is None:
            self._data[normalized_key] = []
            existing = self._data[normalized_key]

        if not isinstance(existing, list):
            if isinstance(existing, str):
                lines = existing.splitlines()
                self._data[normalized_key] = lines
            else:
                self._data[normalized_key] = [existing]
            existing = self._data[normalized_key]

        assert isinstance(existing, list)
        if isinstance(data, dict):
            existing.append(data)
        else:
            existing.append(data.rstrip("\n"))

    async def delete(self, path: str) -> None:
        """Delete the value at *path*."""
        normalized_key = self.normalize_path(path)
        if normalized_key not in self._data:
            raise KeyError(f"Key '{path}' does not exist")
        del self._data[normalized_key]

    async def exists(self, path: str) -> bool:
        """Return ``True`` if *path* exists."""
        normalized_key = self.normalize_path(path)
        return normalized_key in self._data

    async def list_paths(self, prefix: str = "") -> list[str]:
        """List all keys starting with *prefix*."""
        normalized_prefix = self.normalize_path(prefix)

        if not normalized_prefix:
            return list(self._data.keys())

        return [
            key
            for key in self._data.keys()
            if key == normalized_prefix or key.startswith(normalized_prefix + "/")
        ]
