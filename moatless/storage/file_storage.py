"""
File-based storage implementation.

This module provides a storage implementation that reads and writes
data to files on disk.
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Optional, Union

import aiofiles
from moatless.storage.base import BaseStorage
from moatless.context_data import get_trajectory_dir
logger = logging.getLogger(__name__)


class FileStorage(BaseStorage):
    """
    Storage implementation that uses the filesystem.
    
    This class provides a storage implementation that reads and writes
    data to files on disk.
    """
    
    def __init__(self, 
                 base_dir: Union[str, Path] | None = None,
                 file_extension: str = "json"):
        """
        Initialize a FileStorage instance.
        
        Args:
            base_dir: The base directory for storage
            file_extension: The file extension to use for stored files
        """

        if not base_dir:
            self.base_dir = get_trajectory_dir()
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.file_extension = file_extension
    
    def _get_path(self, key: str) -> Path:
        """Get the file path for a key."""
        return self.base_dir / f"{key}{self.file_extension}"
    
    async def read(self, key: str) -> dict:
        """Read binary data from a file."""
        path = self._get_path(key)
        if not path.exists():
            raise KeyError(f"Key '{key}' does not exist")
        
        try:
            async with aiofiles.open(path, 'r') as f:
                return json.loads(await f.read())
        except Exception as e:
            logger.error(f"Error reading from {path}: {e}")
            raise
    
    async def write(self, key: str, data: dict) -> None:
        """Write binary data to a file."""
        path = self._get_path(key)
        try:
            async with aiofiles.open(path, 'w') as f:
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
        return self._get_path(key).exists()
    
    