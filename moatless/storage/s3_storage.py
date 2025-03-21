"""
Amazon S3 Storage implementation.

This module provides a storage implementation that reads and writes
data to Amazon S3 buckets.
"""

import json
import logging
from datetime import datetime
import os
from typing import Union, List

import aioboto3
from botocore.exceptions import ClientError
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


class S3Storage(BaseStorage):
    """
    Storage implementation that uses Amazon S3.

    This class provides a storage implementation that reads and writes
    data to Amazon S3 buckets.
    """

    def __init__(
        self,
        bucket_name: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        endpoint_url: str | None = None,
        file_extension: str = "json",
    ):
        """
        Initialize an S3Storage instance.

        Args:
            bucket_name: Name of the S3 bucket to use
            region_name: AWS region name (optional, will use boto3 default)
            aws_access_key_id: AWS access key ID (optional, will use boto3 default)
            aws_secret_access_key: AWS secret access key (optional, will use boto3 default)
            endpoint_url: Custom endpoint URL for S3 (useful for Minio, etc.)
            file_extension: The file extension to use for stored files
        """
        self.bucket_name = bucket_name or os.getenv("MOATLESS_S3_BUCKET_NAME")
        self.region_name = region_name or os.getenv("MOATLESS_S3_REGION_NAME")
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url
        self.file_extension = file_extension

        # Connection params dict for aioboto3
        self.conn_params = {
            "service_name": "s3",
            "region_name": region_name,
        }

        if aws_access_key_id and aws_secret_access_key:
            self.conn_params["aws_access_key_id"] = aws_access_key_id
            self.conn_params["aws_secret_access_key"] = aws_secret_access_key

        if endpoint_url:
            self.conn_params["endpoint_url"] = endpoint_url

        # Create session for async operations
        self.session = aioboto3.Session()

        logger.info(f"S3 storage initialized with bucket {bucket_name}")

    def __str__(self) -> str:
        return f"S3Storage(bucket={self.bucket_name}, region={self.region_name}, file_extension={self.file_extension})"

    def _get_object_key(self, key: str) -> str:
        """
        Get the S3 object key for a storage key.

        Args:
            key: The key to convert to an S3 object key

        Returns:
            The S3 object key
        """
        normalized_key = self.normalize_key(key)
        return f"{normalized_key}.{self.file_extension}"

    @tracer.start_as_current_span("S3Storage.read")
    async def read(self, key: str) -> dict | list[dict]:
        """
        Read JSON data from an S3 object.

        Args:
            key: The key to read

        Returns:
            The parsed JSON data or an empty dict if the object is empty

        Raises:
            KeyError: If the key does not exist
        """
        object_key = self._get_object_key(key)

        try:
            async with self.session.resource(**self.conn_params) as s3:
                obj = await s3.Object(self.bucket_name, object_key)
                response = await obj.get()

                # Get the body as bytes and decode to string
                streaming_body = response["Body"]
                content = await streaming_body.read()
                content_str = content.decode("utf-8")

                if not content_str.strip():
                    logger.warning(f"Empty content found for key: {key}, returning empty dict")
                    return {}

                return json.loads(content_str)

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(f"No data found for key: {key}")
            else:
                # Re-raise other AWS errors
                raise

    async def read_raw(self, key: str) -> str:
        """
        Read raw string data from an S3 object without parsing.

        Args:
            key: The key to read

        Returns:
            The raw object contents as a string

        Raises:
            KeyError: If the key does not exist
        """
        object_key = self._get_object_key(key)

        try:
            async with self.session.resource(**self.conn_params) as s3:
                obj = await s3.Object(self.bucket_name, object_key)
                response = await obj.get()

                # Get the body as bytes and decode to string
                streaming_body = response["Body"]
                content = await streaming_body.read()
                return content.decode("utf-8")

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(f"No data found for key: {key}")
            else:
                # Re-raise other AWS errors
                raise

    async def read_lines(self, key: str) -> List[dict]:
        """
        Read data from a JSONL S3 object, parsing each line as a JSON object.

        Args:
            key: The key to read

        Returns:
            A list of parsed JSON objects, one per line

        Raises:
            KeyError: If the key does not exist
        """
        object_key = self._get_object_key(key)

        try:
            async with self.session.resource(**self.conn_params) as s3:
                obj = await s3.Object(self.bucket_name, object_key)
                response = await obj.get()

                # Get the body as bytes and decode to string
                streaming_body = response["Body"]
                content = await streaming_body.read()
                content_str = content.decode("utf-8")

                results = []
                for line in content_str.splitlines():
                    line = line.strip()
                    if line:  # Skip empty lines
                        results.append(json.loads(line))
                return results

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(f"No data found for key: {key}")
            else:
                # Re-raise other AWS errors
                raise

    @tracer.start_as_current_span("S3Storage.write")
    async def write(self, key: str, data: dict | list[dict]) -> None:
        """
        Write data to an S3 object as JSON.

        Args:
            key: The key to write to
            data: The data to write
        """
        object_key = self._get_object_key(key)
        content = json.dumps(data, indent=2, cls=DateTimeEncoder)

        async with self.session.resource(**self.conn_params) as s3:
            obj = await s3.Object(self.bucket_name, object_key)
            await obj.put(Body=content.encode("utf-8"), ContentType="application/json")

    async def write_raw(self, key: str, data: str) -> None:
        """
        Write raw string data to an S3 object.

        Args:
            key: The key to write to
            data: The string data to write
        """
        object_key = self._get_object_key(key)

        async with self.session.resource(**self.conn_params) as s3:
            obj = await s3.Object(self.bucket_name, object_key)
            await obj.put(Body=data.encode("utf-8"), ContentType="text/plain")

    async def append(self, key: str, data: Union[dict, str]) -> None:
        """
        Append data to an existing S3 object.

        Args:
            key: The key to append to
            data: The data to append. If dict, it will be serialized as JSON.
                 If string, it will be written as-is with a newline.
        """
        object_key = self._get_object_key(key)

        # Convert to JSON string if it's a dict
        if isinstance(data, dict):
            line = json.dumps(data, cls=DateTimeEncoder)
        else:
            line = data

        # Make sure the line ends with a newline
        if not line.endswith("\n"):
            line += "\n"

        # S3 doesn't support direct appends, so we need to download, modify, and re-upload
        try:
            # Try to get existing content
            existing_content = await self.read_raw(key)
            content = existing_content + line
        except KeyError:
            # If object doesn't exist, just use the new line
            content = line

        # Upload the combined content
        await self.write_raw(key, content)

    async def delete(self, key: str) -> None:
        """
        Delete an S3 object.

        Args:
            key: The key to delete

        Raises:
            KeyError: If the key does not exist
        """
        object_key = self._get_object_key(key)

        try:
            async with self.session.resource(**self.conn_params) as s3:
                obj = await s3.Object(self.bucket_name, object_key)
                await obj.load()  # This raises exception if object doesn't exist
                await obj.delete()

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey" or e.response["Error"]["Code"] == "404":
                raise KeyError(f"No data found for key: {key}")
            else:
                # Re-raise other AWS errors
                raise

    async def exists(self, key: str) -> bool:
        """
        Check if an S3 object exists.

        Args:
            key: The key to check

        Returns:
            True if the object exists, False otherwise
        """
        object_key = self._get_object_key(key)

        try:
            async with self.session.resource(**self.conn_params) as s3:
                obj = await s3.Object(self.bucket_name, object_key)
                await obj.load()
                return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey" or e.response["Error"]["Code"] == "404":
                return False
            else:
                # Re-raise other AWS errors
                raise

    @tracer.start_as_current_span("S3Storage.list_keys")
    async def list_keys(self, prefix: str = "", delimiter: str = "/") -> list[str]:
        """
        List keys with the given prefix.

        Args:
            prefix: The prefix to filter keys by
            delimiter: The delimiter character to use for directory-like structure (default: '/')

        Returns:
            A list of keys at the specified directory level (not recursive)
        """
        normalized_prefix = self.normalize_key(prefix)

        # Ensure prefix ends with delimiter if it represents a directory
        if normalized_prefix and not normalized_prefix.endswith(delimiter):
            normalized_prefix = f"{normalized_prefix}{delimiter}"

        keys = []
        directory_paths = set()

        logger.info(f"Listing keys with prefix: {normalized_prefix}")

        async with self.session.resource(**self.conn_params) as s3:
            bucket = await s3.Bucket(self.bucket_name)

            # Get all objects with the prefix
            async for obj in bucket.objects.filter(Prefix=normalized_prefix):
                # Extract key
                key = obj.key

                # Skip the prefix directory itself
                if key == normalized_prefix:
                    continue

                # Check if this is a direct child of the prefix or a deeper descendant
                rel_key = key[len(normalized_prefix) :] if normalized_prefix else key

                # Skip entries that don't have content (directory markers)
                if not rel_key:
                    continue

                # For non-recursive listing, we need to handle directories
                if delimiter in rel_key:
                    # This is a deeper path, extract just the top-level directory
                    child_path = rel_key.split(delimiter)[0]
                    directory_path = f"{normalized_prefix}{child_path}"
                    directory_paths.add(directory_path)
                else:
                    # This is a direct child, include it
                    file_path = key
                    # Remove file extension if present
                    if file_path.endswith(f".{self.file_extension}"):
                        file_path = file_path[: -len(f".{self.file_extension}")]
                    keys.append(file_path)

        # Add directories to the result
        keys.extend(directory_paths)

        return keys

    async def close(self) -> None:
        """
        Close any connections to S3.
        """
        # aioboto3 manages connections automatically with context managers,
        # but we provide this method for consistency with other storage implementations
        pass
