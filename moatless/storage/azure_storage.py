"""
Azure Blob Storage implementation.

This module provides a storage implementation that reads and writes
data to Azure Blob Storage.
"""

import json
import logging
from datetime import datetime
import os
from typing import Union, List, Optional
from io import BytesIO

from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from opentelemetry import trace

from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Suppress Azure HTTP logging
logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class AzureBlobStorage(BaseStorage):
    """
    Storage implementation that uses Azure Blob Storage.

    This class provides a storage implementation that reads and writes
    data to Azure Blob Storage.
    """

    def __init__(self, connection_string: str | None = None, container_name: str = "moatless-tools"):
        """
        Initialize an AzureStorage instance.

        Args:
            connection_string: Azure Storage connection string
            container_name: Name of the container to use
        """
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("Azure Storage connection string is required")

        self.container_name = container_name

        try:
            self.blob_service_client = AsyncBlobServiceClient.from_connection_string(self.connection_string)
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
            logger.info(f"Successfully created Azure Blob Service client connection to {self.container_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Blob Storage: {e}")
            raise

    def __str__(self) -> str:
        return f"AzureStorage(container={self.container_name})"

    def _get_blob_name(self, path: str) -> str:
        """
        Get the blob name for a path.

        Args:
            path: The path to convert to a blob name

        Returns:
            The blob name for the path
        """
        return self.normalize_path(path)

    async def read_raw(self, path: str) -> str:
        """
        Read raw string data from a blob without parsing.

        Args:
            path: The path to read

        Returns:
            The raw blob contents as a string

        Raises:
            KeyError: If the path does not exist
        """
        blob_name = self._get_blob_name(path)
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            download_stream = await blob_client.download_blob()
            content = await download_stream.readall()
            return content.decode("utf-8")
        except ResourceNotFoundError:
            raise KeyError(f"No data found for path: {path}")

    async def read_lines(self, path: str) -> List[dict]:
        """
        Read data from a JSONL blob, parsing each line as a JSON object.

        Args:
            path: The path to read

        Returns:
            A list of parsed JSON objects, one per line

        Raises:
            KeyError: If the path does not exist
        """
        blob_name = self._get_blob_name(path)
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            download_stream = await blob_client.download_blob()
            content = await download_stream.readall()
            content_str = content.decode("utf-8")

            results = []
            for line in content_str.splitlines():
                line = line.strip()
                if line:  # Skip empty lines
                    results.append(json.loads(line))
            return results
        except ResourceNotFoundError:
            raise KeyError(f"No data found for path: {path}")

    async def write_raw(self, path: str, data: str) -> None:
        """
        Write raw string data to a blob.

        Args:
            path: The path to write to
            data: The string data to write
        """
        blob_name = self._get_blob_name(path)
        blob_client = self.container_client.get_blob_client(blob_name)
        await blob_client.upload_blob(data.encode("utf-8"), overwrite=True)

    async def append(self, path: str, data: Union[dict, str]) -> None:
        """
        Append data to an existing blob.

        Args:
            path: The path to append to
            data: The data to append. If dict, it will be serialized as JSON.
                 If string, it will be written as-is with a newline.
        """
        blob_name = self._get_blob_name(path)
        blob_client = self.container_client.get_blob_client(blob_name)

        # Convert to JSON string if it's a dict
        if isinstance(data, dict):
            line = json.dumps(data, cls=DateTimeEncoder)
        else:
            line = data

        # Make sure the line ends with a newline
        if not line.endswith("\n"):
            line += "\n"

        try:
            # Try to download existing content
            download_stream = await blob_client.download_blob()
            existing_content = await download_stream.readall()
            content = existing_content + line.encode("utf-8")
        except ResourceNotFoundError:
            # If blob doesn't exist, just use the new line
            content = line.encode("utf-8")

        await blob_client.upload_blob(content, overwrite=True)

    async def delete(self, path: str) -> None:
        """
        Delete a blob.

        Args:
            path: The path to delete

        Raises:
            KeyError: If the path does not exist
        """
        blob_name = self._get_blob_name(path)
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            await blob_client.delete_blob()
        except ResourceNotFoundError:
            raise KeyError(f"No data found for path: {path}")

    async def exists(self, path: str) -> bool:
        """
        Check if a blob exists.

        Args:
            path: The path to check

        Returns:
            True if the blob exists, False otherwise
        """
        blob_name = self._get_blob_name(path)
        blob_client = self.container_client.get_blob_client(blob_name)

        # Log the full URL being accessed for debugging
        logger.info(f"Checking existence of blob at: {blob_client.url}")

        try:
            await blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception as e:
            # Log any unexpected errors
            logger.error(f"Error checking if blob exists: {e}")
            # Re-raise non-ResourceNotFound errors
            if not isinstance(e, ResourceNotFoundError):
                raise

    async def list_paths(self, prefix: str = "") -> list[str]:
        """
        List all paths with the given prefix.

        Args:
            prefix: The prefix to filter paths by

        Returns:
            A list of paths
        """
        normalized_prefix = self.normalize_path(prefix)
        prefix_with_ext = f"{normalized_prefix}" if normalized_prefix else ""

        paths = []
        async for blob in self.container_client.list_blobs(name_starts_with=prefix_with_ext):
            paths.append(blob.name)

        return paths

    async def close(self) -> None:
        """
        Close the Azure Blob Storage connection.
        """
        await self.blob_service_client.close()
