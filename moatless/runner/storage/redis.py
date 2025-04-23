"""Redis storage implementation for job scheduler."""

import json
from datetime import datetime
from typing import List, Optional, Union


from moatless.runner.runner import JobInfo, JobStatus
from moatless.runner.storage.storage import JobStorage


class RedisJobStorage(JobStorage):
    """Redis implementation of job storage."""

    def __init__(self, redis_url: str, prefix: str = "moatless:jobs:"):
        """Initialize Redis job storage.

        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for Redis storage
        """
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("redis is not installed. Please install it with `pip install redis`.")

        self._redis = redis.from_url(redis_url)
        self._prefix = prefix

    async def add_job(self, job_info: JobInfo) -> None:
        """Add a job to storage.

        Args:
            job_info: Information about the job to add
        """
        job_key = self._get_job_key(job_info.project_id, job_info.trajectory_id)
        job_data = self._serialize_job_info(job_info)

        # Store job data
        await self._redis.set(job_key, job_data)

        # Add to project set
        project_key = f"{self._prefix}project:{job_info.project_id}"
        await self._redis.sadd(project_key, job_key)

        # Add to status set
        status_key = f"{self._prefix}status:{job_info.status.value}"
        await self._redis.sadd(status_key, job_key)

    async def update_job(self, job_info: JobInfo) -> None:
        """Update job information in storage.

        Args:
            job_info: Updated job information
        """
        job_key = self._get_job_key(job_info.project_id, job_info.trajectory_id)

        # Get current job data to check if status has changed
        current_job_data = await self._redis.get(job_key)
        if current_job_data:
            current_job = self._deserialize_job_info(current_job_data)

            # If status changed, update status sets
            if current_job.status != job_info.status:
                old_status_key = f"{self._prefix}status:{current_job.status.value}"
                new_status_key = f"{self._prefix}status:{job_info.status.value}"

                await self._redis.srem(old_status_key, job_key)
                await self._redis.sadd(new_status_key, job_key)

        # Update job data
        job_data = self._serialize_job_info(job_info)
        await self._redis.set(job_key, job_data)

    async def get_job(self, project_id: str, trajectory_id: str) -> Optional[JobInfo]:
        """Get job information from storage.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobInfo object if the job exists, None otherwise
        """
        job_key = self._get_job_key(project_id, trajectory_id)
        job_data = await self._redis.get(job_key)

        if job_data:
            return self._deserialize_job_info(job_data)
        return None

    async def get_jobs(self, project_id: Optional[str] = None) -> List[JobInfo]:
        """Get all jobs or jobs for a specific project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            List of JobInfo objects
        """
        if project_id is None:
            # Get all job keys
            pattern = f"{self._prefix}job:*"
            keys = await self._redis.keys(pattern)
        else:
            # Get keys for specific project
            project_key = f"{self._prefix}project:{project_id}"
            keys = await self._redis.smembers(project_key)

        jobs = []
        for key in keys:
            job_data = await self._redis.get(key)
            if job_data:
                try:
                    jobs.append(self._deserialize_job_info(job_data))
                except Exception as e:
                    # Log error but continue processing other jobs
                    import logging

                    logging.getLogger(__name__).error(f"Error deserializing job data for key {key}: {e}")

        return jobs

    async def remove_job(self, project_id: str, trajectory_id: str) -> None:
        """Remove a job from storage.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
        """
        job_key = self._get_job_key(project_id, trajectory_id)

        # Get job data to remove from sets
        job_data = await self._redis.get(job_key)
        if job_data:
            try:
                job = self._deserialize_job_info(job_data)

                # Remove from project set
                project_key = f"{self._prefix}project:{project_id}"
                await self._redis.srem(project_key, job_key)

                # Remove from status set
                status_key = f"{self._prefix}status:{job.status.value}"
                await self._redis.srem(status_key, job_key)
            except Exception as e:
                # Log error but continue with deletion
                import logging

                logging.getLogger(__name__).error(f"Error removing job references: {e}")

        # Remove job data
        await self._redis.delete(job_key)

    async def get_running_jobs_count(self, project_id: Optional[str] = None) -> int:
        """Get count of running jobs, optionally filtered by project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            Count of running jobs
        """
        if project_id is None:
            # Count all running jobs
            running_key = f"{self._prefix}status:{JobStatus.RUNNING.value}"

            running_count = await self._redis.scard(running_key)

            return running_count

        # For specific project, we need to get intersection of project set and status sets
        project_key = f"{self._prefix}project:{project_id}"
        running_key = f"{self._prefix}status:{JobStatus.RUNNING.value}"

        running_count = await self._redis.sinter(project_key, running_key)

        return len(running_count)

    async def get_queued_jobs_count(self, project_id: Optional[str] = None) -> int:
        """Get count of queued jobs, optionally filtered by project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            Count of queued jobs
        """
        if project_id is None:
            # Count all pending jobs
            pending_key = f"{self._prefix}status:{JobStatus.PENDING.value}"
            return await self._redis.scard(pending_key)

        # For specific project, get intersection of project set and pending status set
        project_key = f"{self._prefix}project:{project_id}"
        pending_key = f"{self._prefix}status:{JobStatus.PENDING.value}"

        pending_count = await self._redis.sinter(project_key, pending_key)
        return len(pending_count)

    async def delete_jobs(self, project_id: Optional[str] = None) -> int:
        """Delete all jobs or jobs for a specific project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            Number of jobs deleted
        """
        import logging

        logger = logging.getLogger(__name__)

        if project_id is None:
            # Get all job keys
            pattern = f"{self._prefix}job:*"
            keys = await self._redis.keys(pattern)

            # Get all project and status keys
            project_keys = await self._redis.keys(f"{self._prefix}project:*")
            status_keys = await self._redis.keys(f"{self._prefix}status:*")

            count = len(keys)

            # Delete all job data, project sets, and status sets
            pipeline = self._redis.pipeline()

            if keys:
                pipeline.delete(*keys)

            if project_keys:
                pipeline.delete(*project_keys)

            if status_keys:
                pipeline.delete(*status_keys)

            await pipeline.execute()

            return count
        else:
            # Get keys for specific project
            project_key = f"{self._prefix}project:{project_id}"
            job_keys = await self._redis.smembers(project_key)

            count = len(job_keys)

            if not job_keys:
                return 0

            # Get job info for each key to remove from status sets
            pipeline = self._redis.pipeline()

            # Delete all job data
            for job_key in job_keys:
                try:
                    job_data = await self._redis.get(job_key)
                    if job_data:
                        job = self._deserialize_job_info(job_data)
                        status_key = f"{self._prefix}status:{job.status.value}"
                        pipeline.srem(status_key, job_key)
                    pipeline.delete(job_key)
                except Exception as e:
                    logger.error(f"Error processing job {job_key} for deletion: {e}")

            # Delete project set
            pipeline.delete(project_key)

            await pipeline.execute()

            return count

    def _get_job_key(self, project_id: str, trajectory_id: str) -> str:
        """Create a unique key for a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            A unique Redis key
        """
        return f"{self._prefix}job:{project_id}:{trajectory_id}"

    def _serialize_job_info(self, job_info: JobInfo) -> str:
        """Serialize JobInfo to JSON string.

        Args:
            job_info: The JobInfo object to serialize

        Returns:
            JSON string representation
        """
        # Convert datetime objects to ISO format strings
        data = job_info.model_dump()

        if data.get("enqueued_at"):
            data["enqueued_at"] = data["enqueued_at"].isoformat()
        if data.get("started_at"):
            data["started_at"] = data["started_at"].isoformat()
        if data.get("ended_at"):
            data["ended_at"] = data["ended_at"].isoformat()

        return json.dumps(data)

    def _deserialize_job_info(self, job_data: Union[str, bytes]) -> JobInfo:
        """Deserialize JSON string to JobInfo.

        Args:
            job_data: JSON string or bytes representation

        Returns:
            JobInfo object
        """
        if isinstance(job_data, bytes):
            job_data = job_data.decode("utf-8")

        data = json.loads(job_data)

        # Convert ISO format strings to datetime objects
        if data.get("enqueued_at"):
            data["enqueued_at"] = datetime.fromisoformat(data["enqueued_at"])
        if data.get("started_at"):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("ended_at"):
            data["ended_at"] = datetime.fromisoformat(data["ended_at"])

        return JobInfo.model_validate(data)
