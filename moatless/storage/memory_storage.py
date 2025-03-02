"""
In-memory storage implementation.

This module provides a storage implementation that keeps all data
in memory, which is useful for testing or temporary storage.
"""

import logging
from typing import Dict

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
        self._data: Dict[str, dict] = {}

    async def read(self, key: str) -> dict:
        """Read binary data from memory."""
        if key not in self._data:
            raise KeyError(f"Key '{key}' does not exist")
        return self._data[key]

    async def write(self, key: str, data: dict) -> None:
        """Write binary data to memory."""
        self._data[key] = data

    async def delete(self, key: str) -> None:
        """Delete data from memory."""
        if key not in self._data:
            raise KeyError(f"Key '{key}' does not exist")
        del self._data[key]

    async def exists(self, key: str) -> bool:
        """Check if a key exists in memory."""
        return key in self._data
