"""In-memory storage implementation for job scheduler."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from moatless.runner.runner import JobInfo, JobStatus
from moatless.runner.storage.storage import JobStorage


class InMemoryJobStorage(JobStorage):
    """In-memory implementation of job storage."""

    def __init__(self):
        """Initialize in-memory job storage."""
        self._jobs: Dict[str, JobInfo] = {}
        self._lock = asyncio.Lock()

    async def add_job(self, job_info: JobInfo) -> None:
        """Add a job to storage.

        Args:
            job_info: Information about the job to add
        """
        async with self._lock:
            job_key = self._get_job_key(job_info.project_id, job_info.trajectory_id)
            self._jobs[job_key] = job_info

    async def update_job(self, job_info: JobInfo) -> None:
        """Update job information in storage.

        Args:
            job_info: Updated job information
        """
        async with self._lock:
            job_key = self._get_job_key(job_info.project_id, job_info.trajectory_id)
            self._jobs[job_key] = job_info

    async def get_job(self, project_id: str, trajectory_id: str) -> Optional[JobInfo]:
        """Get job information from storage.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobInfo object if the job exists, None otherwise
        """
        async with self._lock:
            job_key = self._get_job_key(project_id, trajectory_id)
            return self._jobs.get(job_key)

    async def get_jobs(self, project_id: Optional[str] = None) -> List[JobInfo]:
        """Get all jobs or jobs for a specific project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            List of JobInfo objects
        """
        async with self._lock:
            if project_id is None:
                return list(self._jobs.values())

            return [job for job in self._jobs.values() if job.project_id == project_id]

    async def remove_job(self, project_id: str, trajectory_id: str) -> None:
        """Remove a job from storage.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
        """
        async with self._lock:
            job_key = self._get_job_key(project_id, trajectory_id)
            if job_key in self._jobs:
                del self._jobs[job_key]

    async def get_running_jobs_count(self, project_id: Optional[str] = None) -> int:
        """Get count of running jobs, optionally filtered by project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            Count of running jobs
        """
        async with self._lock:
            if project_id is None:
                return sum(1 for job in self._jobs.values() if job.status in [JobStatus.RUNNING])

            return sum(
                1 for job in self._jobs.values() if job.project_id == project_id and job.status in [JobStatus.RUNNING]
            )

    async def get_queued_jobs_count(self, project_id: Optional[str] = None) -> int:
        """Get count of queued jobs, optionally filtered by project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            Count of queued jobs
        """
        async with self._lock:
            if project_id is None:
                return sum(1 for job in self._jobs.values() if job.status == JobStatus.PENDING)

            return sum(
                1 for job in self._jobs.values() if job.project_id == project_id and job.status == JobStatus.PENDING
            )

    async def delete_jobs(self, project_id: Optional[str] = None) -> int:
        """Delete all jobs or jobs for a specific project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            Number of jobs deleted
        """
        async with self._lock:
            # If no project_id is specified, delete all jobs
            if project_id is None:
                count = len(self._jobs)
                self._jobs.clear()
                return count

            # Otherwise, delete jobs for the specified project
            keys_to_delete = [key for key, job in self._jobs.items() if job.project_id == project_id]

            for key in keys_to_delete:
                del self._jobs[key]

            return len(keys_to_delete)

    def _get_job_key(self, project_id: str, trajectory_id: str) -> str:
        """Create a unique key for a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            A unique string key
        """
        return f"{project_id}:{trajectory_id}"
