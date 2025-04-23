"""Storage interface for the job scheduler."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict

from moatless.runner.runner import JobInfo, JobStatus


class JobStorage(ABC):
    """Interface for storing job states."""

    @abstractmethod
    async def add_job(self, job_info: JobInfo) -> None:
        """Add a job to storage.

        Args:
            job_info: Information about the job to add
        """
        pass

    @abstractmethod
    async def update_job(self, job_info: JobInfo) -> None:
        """Update job information in storage.

        Args:
            job_info: Updated job information
        """
        pass

    @abstractmethod
    async def get_job(self, project_id: str, trajectory_id: str) -> Optional[JobInfo]:
        """Get job information from storage.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobInfo object if the job exists, None otherwise
        """
        pass

    @abstractmethod
    async def get_jobs(self, project_id: Optional[str] = None) -> List[JobInfo]:
        """Get all jobs or jobs for a specific project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            List of JobInfo objects
        """
        pass

    @abstractmethod
    async def remove_job(self, project_id: str, trajectory_id: str) -> None:
        """Remove a job from storage.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
        """
        pass

    @abstractmethod
    async def get_running_jobs_count(self, project_id: Optional[str] = None) -> int:
        """Get count of running jobs, optionally filtered by project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            Count of running jobs
        """
        pass

    @abstractmethod
    async def get_queued_jobs_count(self, project_id: Optional[str] = None) -> int:
        """Get count of queued jobs, optionally filtered by project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            Count of queued jobs
        """
        pass

    @abstractmethod
    async def delete_jobs(self, project_id: Optional[str] = None) -> int:
        """Delete all jobs or jobs for a specific project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            Number of jobs deleted
        """
        pass
