"""
Azure Blob Storage implementation.

This module provides a storage implementation that reads and writes
data to Azure Blob Storage.
"""

import json
import logging
from datetime import datetime
from typing import Union, List, Optional
from io import BytesIO

from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from opentelemetry import trace

from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class AzureStorage(BaseStorage):
    """
    Storage implementation that uses Azure Blob Storage.

    This class provides a storage implementation that reads and writes
    data to Azure Blob Storage.
    """

    def __init__(self, connection_string: str, container_name: str = "moatless", file_extension: str = "json"):
        """
        Initialize an AzureStorage instance.

        Args:
            connection_string: Azure Storage connection string
            container_name: Name of the container to use
            file_extension: The file extension to use for stored files
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.file_extension = file_extension

        # Create async blob service client
        self.blob_service_client = AsyncBlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

        logger.info(f"Azure storage initialized with container {container_name}")

    def __str__(self) -> str:
        return f"AzureStorage(container={self.container_name}, file_extension={self.file_extension})"

    def _get_blob_name(self, key: str) -> str:
        """
        Get the blob name for a key.

        Args:
            key: The key to convert to a blob name

        Returns:
            The blob name for the key
        """
        normalized_key = self.normalize_key(key)
        return f"{normalized_key}.{self.file_extension}"

    @tracer.start_as_current_span("AzureStorage.read")
    async def read(self, key: str) -> dict | list[dict]:
        """
        Read JSON data from a blob.

        Args:
            key: The key to read

        Returns:
            The parsed JSON data or an empty dict if the blob is empty

        Raises:
            KeyError: If the key does not exist
        """
        blob_name = self._get_blob_name(key)
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            download_stream = await blob_client.download_blob()
            content = await download_stream.readall()
            content_str = content.decode("utf-8")

            if not content_str.strip():
                logger.warning(f"Empty content found for key: {key}, returning empty dict")
                return {}
            return json.loads(content_str)
        except ResourceNotFoundError:
            raise KeyError(f"No data found for key: {key}")

    async def read_raw(self, key: str) -> str:
        """
        Read raw string data from a blob without parsing.

        Args:
            key: The key to read

        Returns:
            The raw blob contents as a string

        Raises:
            KeyError: If the key does not exist
        """
        blob_name = self._get_blob_name(key)
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            download_stream = await blob_client.download_blob()
            content = await download_stream.readall()
            return content.decode("utf-8")
        except ResourceNotFoundError:
            raise KeyError(f"No data found for key: {key}")

    async def read_lines(self, key: str) -> List[dict]:
        """
        Read data from a JSONL blob, parsing each line as a JSON object.

        Args:
            key: The key to read

        Returns:
            A list of parsed JSON objects, one per line

        Raises:
            KeyError: If the key does not exist
        """
        blob_name = self._get_blob_name(key)
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
            raise KeyError(f"No data found for key: {key}")

    @tracer.start_as_current_span("AzureStorage.write")
    async def write(self, key: str, data: dict | list[dict]) -> None:
        """
        Write data to a blob as JSON.

        Args:
            key: The key to write to
            data: The data to write
        """
        blob_name = self._get_blob_name(key)
        blob_client = self.container_client.get_blob_client(blob_name)

        content = json.dumps(data, indent=2, cls=DateTimeEncoder)
        await blob_client.upload_blob(content.encode("utf-8"), overwrite=True)

    async def write_raw(self, key: str, data: str) -> None:
        """
        Write raw string data to a blob.

        Args:
            key: The key to write to
            data: The string data to write
        """
        blob_name = self._get_blob_name(key)
        blob_client = self.container_client.get_blob_client(blob_name)
        await blob_client.upload_blob(data.encode("utf-8"), overwrite=True)

    async def append(self, key: str, data: Union[dict, str]) -> None:
        """
        Append data to an existing blob.

        Args:
            key: The key to append to
            data: The data to append. If dict, it will be serialized as JSON.
                 If string, it will be written as-is with a newline.
        """
        blob_name = self._get_blob_name(key)
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

    async def delete(self, key: str) -> None:
        """
        Delete a blob.

        Args:
            key: The key to delete

        Raises:
            KeyError: If the key does not exist
        """
        blob_name = self._get_blob_name(key)
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            await blob_client.delete_blob()
        except ResourceNotFoundError:
            raise KeyError(f"No data found for key: {key}")

    async def exists(self, key: str) -> bool:
        """
        Check if a blob exists.

        Args:
            key: The key to check

        Returns:
            True if the blob exists, False otherwise
        """
        blob_name = self._get_blob_name(key)
        blob_client = self.container_client.get_blob_client(blob_name)
        try:
            await blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False

    async def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys with the given prefix.

        Args:
            prefix: The prefix to filter keys by

        Returns:
            A list of keys
        """
        normalized_prefix = self.normalize_key(prefix)
        prefix_with_ext = f"{normalized_prefix}" if normalized_prefix else ""

        keys = []
        async for blob in self.container_client.list_blobs(name_starts_with=prefix_with_ext):
            # Remove file extension and normalize path
            key = (
                blob.name[: -len(f".{self.file_extension}")]
                if blob.name.endswith(f".{self.file_extension}")
                else blob.name
            )
            keys.append(key)

        return keys

    async def close(self) -> None:
        """
        Close the Azure Blob Storage connection.
        """
        await self.blob_service_client.close()
