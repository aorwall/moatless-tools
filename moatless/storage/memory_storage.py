"""
In-memory storage implementation.

This module provides a storage implementation that keeps all data
in memory, which is useful for testing or temporary storage.
"""

import logging

from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)


class MemoryStorage(BaseStorage):
    """
    Storage implementation that uses in-memory dictionaries.

    This class provides a storage implementation that keeps all data
    in memory, which is useful for testing or temporary storage.
    """

    def __init__(self):
        """Initialize an empty in-memory storage."""
        self._data: dict[str, dict] = {}

    async def read(self, key: str) -> dict:
        """Read binary data from memory."""
        normalized_key = self.normalize_key(key)
        if normalized_key not in self._data:
            raise KeyError(f"Key '{key}' does not exist")
        return self._data[normalized_key]

    async def write(self, key: str, data: dict) -> None:
        """Write binary data to memory."""
        normalized_key = self.normalize_key(key)
        self._data[normalized_key] = data

    async def delete(self, key: str) -> None:
        """Delete data from memory."""
        normalized_key = self.normalize_key(key)
        if normalized_key not in self._data:
            raise KeyError(f"Key '{key}' does not exist")
        del self._data[normalized_key]

    async def exists(self, key: str) -> bool:
        """Check if a key exists in memory."""
        normalized_key = self.normalize_key(key)
        return normalized_key in self._data

    async def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys with the given prefix.

        Args:
            prefix: The key prefix to search for

        Returns:
            A list of keys that match the prefix
        """
        normalized_prefix = self.normalize_key(prefix)

        # When prefix is empty, return all keys
        if not normalized_prefix:
            return list(self._data.keys())

        # Filter keys that start with the prefix
        return [key for key in self._data.keys() if key == normalized_prefix or key.startswith(normalized_prefix + "/")]
