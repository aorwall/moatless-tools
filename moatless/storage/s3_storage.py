"""
Amazon S3 Storage implementation.

This module provides a storage implementation that reads and writes
data to Amazon S3 buckets.
"""

import json
import logging
import os
from datetime import datetime
from typing import Union, List

import aioboto3
from botocore.exceptions import ClientError
from moatless.storage.base import BaseStorage
from opentelemetry import trace

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
    ):
        """
        Initialize an S3Storage instance.

        Args:
            bucket_name: Name of the S3 bucket to use
            region_name: AWS region name (optional, will use boto3 default)
            aws_access_key_id: AWS access key ID (optional, will use boto3 default)
            aws_secret_access_key: AWS secret access key (optional, will use boto3 default)
            endpoint_url: Custom endpoint URL for S3 (useful for Minio, etc.)
        """
        self.bucket_name = bucket_name or os.getenv("MOATLESS_S3_BUCKET_NAME")
        self.region_name = region_name or os.getenv("MOATLESS_S3_REGION_NAME")
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url

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
        return f"S3Storage(bucket={self.bucket_name}, region={self.region_name})"

    def _get_object_path(self, path: str) -> str:
        """
        Get the S3 object path for a storage path.

        Args:
            path: The path to convert to an S3 object path

        Returns:
            The S3 object path
        """
        return self.normalize_path(path)

    async def read_raw(self, path: str) -> str:
        """
        Read raw string data from an S3 object without parsing.

        Args:
            path: The path to read

        Returns:
            The raw object contents as a string

        Raises:
            KeyError: If the path does not exist
        """
        object_path = self._get_object_path(path)

        try:
            async with self.session.resource(**self.conn_params) as s3:
                obj = await s3.Object(self.bucket_name, object_path)
                response = await obj.get()

                # Get the body as bytes and decode to string
                streaming_body = response["Body"]
                content = await streaming_body.read()
                return content.decode("utf-8")

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(f"No data found for path: {path}")
            else:
                # Re-raise other AWS errors
                raise

    async def read_lines(self, path: str) -> List[dict]:
        """
        Read data from a JSONL S3 object, parsing each line as a JSON object.

        Args:
            path: The path to read

        Returns:
            A list of parsed JSON objects, one per line

        Raises:
            KeyError: If the path does not exist
        """
        object_path = self._get_object_path(path)

        try:
            async with self.session.resource(**self.conn_params) as s3:
                obj = await s3.Object(self.bucket_name, object_path)
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
                raise KeyError(f"No data found for path: {path}")
            else:
                # Re-raise other AWS errors
                raise

    async def write_raw(self, path: str, data: str) -> None:
        """
        Write raw string data to an S3 object.

        Args:
            path: The path to write to
            data: The string data to write
        """
        object_path = self._get_object_path(path)

        async with self.session.resource(**self.conn_params) as s3:
            obj = await s3.Object(self.bucket_name, object_path)
            await obj.put(Body=data.encode("utf-8"), ContentType="text/plain")

    async def append(self, path: str, data: Union[dict, str]) -> None:
        """
        Append data to an existing S3 object.

        Args:
            path: The path to append to
            data: The data to append. If dict, it will be serialized as JSON.
                 If string, it will be written as-is with a newline.
        """
        object_path = self._get_object_path(path)

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
            existing_content = await self.read_raw(path)
            content = existing_content + line
        except KeyError:
            # If object doesn't exist, just use the new line
            content = line

        # Upload the combined content
        await self.write_raw(path, content)

    async def delete(self, path: str) -> None:
        """
        Delete an S3 object.

        Args:
            path: The path to delete

        Raises:
            KeyError: If the path does not exist
        """
        object_path = self._get_object_path(path)

        try:
            async with self.session.resource(**self.conn_params) as s3:
                obj = await s3.Object(self.bucket_name, object_path)
                await obj.load()  # This raises exception if object doesn't exist
                await obj.delete()

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey" or e.response["Error"]["Code"] == "404":
                raise KeyError(f"No data found for path: {path}")
            else:
                # Re-raise other AWS errors
                raise

    async def exists(self, path: str) -> bool:
        """
        Check if an S3 object exists.

        Args:
            path: The path to check

        Returns:
            True if the object exists, False otherwise
        """
        object_path = self._get_object_path(path)

        try:
            async with self.session.resource(**self.conn_params) as s3:
                obj = await s3.Object(self.bucket_name, object_path)
                await obj.load()
                return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey" or e.response["Error"]["Code"] == "404":
                return False
            else:
                # Re-raise other AWS errors
                raise

    @tracer.start_as_current_span("S3Storage.list_paths")
    async def list_paths(self, prefix: str = "", delimiter: str = "/") -> list[str]:
        """
        List paths with the given prefix.

        Args:
            prefix: The prefix to filter paths by
            delimiter: The delimiter character to use for directory-like structure (default: '/')

        Returns:
            A list of paths at the specified directory level (not recursive)
        """
        normalized_prefix = self.normalize_path(prefix)

        # Ensure prefix ends with delimiter if it represents a directory
        if normalized_prefix and not normalized_prefix.endswith(delimiter):
            normalized_prefix = f"{normalized_prefix}{delimiter}"

        paths = []
        directory_paths = set()

        logger.info(f"Listing paths with prefix: {normalized_prefix}")

        async with self.session.resource(**self.conn_params) as s3:
            bucket = await s3.Bucket(self.bucket_name)

            # Get all objects with the prefix
            async for obj in bucket.objects.filter(Prefix=normalized_prefix):
                # Extract path
                path = obj.key
                logger.info(f"Listing key: {path}")

                # Skip the prefix directory itself
                if path == normalized_prefix:
                    continue

                # Check if this is a direct child of the prefix or a deeper descendant
                rel_path = path[len(normalized_prefix) :] if normalized_prefix else path

                # Skip entries that don't have content (directory markers)
                if not rel_path:
                    continue

                # For non-recursive listing, we need to handle directories
                if delimiter in rel_path:
                    # This is a deeper path, extract just the top-level directory
                    child_path = rel_path.split(delimiter)[0]
                    directory_path = f"{normalized_prefix}{child_path}{delimiter}"
                    directory_paths.add(directory_path)
                else:
                    paths.append(path)

        # No need to add directory paths when listing contents of a specific folder
        if normalized_prefix:
            return paths

        # Add directories to the result only for top-level browsing
        paths.extend(directory_paths)

        return paths

    async def close(self) -> None:
        """
        Close any connections to S3.
        """
        # aioboto3 manages connections automatically with context managers,
        # but we provide this method for consistency with other storage implementations
        pass
