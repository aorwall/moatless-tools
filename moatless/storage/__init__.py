"""
Storage module for handling file and data persistence.
"""

from moatless.storage.base import BaseStorage

try:
    from moatless.storage.file_storage import FileStorage
except Exception:  # pragma: no cover - optional dependency
    FileStorage = None  # type: ignore

from moatless.storage.memory_storage import MemoryStorage

__all__ = ["BaseStorage", "FileStorage", "MemoryStorage"]
