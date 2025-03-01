import abc
import json
import logging
from typing import Any, List, Dict, Optional, TypeVar, Generic, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseStorage(abc.ABC):
    """
    Abstract base class for storage operations.
    
    This class defines the interface for storage operations on artifacts, trajectories, configurations, etc.
    """
    
    @abc.abstractmethod
    async def read(self, key: str) -> dict:
        """
        Read binary data from storage.
        
        Args:
            key: The identifier for the data to read
            
        Returns:
            The binary data associated with the key
            
        Raises:
            KeyError: If the key does not exist
        """
        pass
    
    @abc.abstractmethod
    async def write(self, key: str, data: dict) -> None:
        """
        Write binary data to storage.
        
        Args:
            key: The identifier for the data
            data: The binary data to write
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
        Check if a key exists in storage.
        
        Args:
            key: The identifier to check
            
        Returns:
            True if the key exists, False otherwise
        """
        pass
    
    @classmethod
    def get_default_storage(cls) -> "BaseStorage":
        """
        Get the default storage backend.
        """
        from moatless.storage.file_storage import FileStorage
        return FileStorage()
