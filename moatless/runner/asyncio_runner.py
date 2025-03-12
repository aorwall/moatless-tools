import asyncio
import logging
from asyncio import Task
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, cast
from uuid import uuid4

from opentelemetry import trace

from moatless.runner.runner import (
    JobInfo,
    JobStatus,
    JobsStatusSummary,
    BaseRunner,
    RunnerInfo,
    RunnerStatus,
)

tracer = trace.get_tracer(__name__)


class AsyncioRunner(BaseRunner):
    """A simple runner implementation using asyncio tasks."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Dictionary to store all tasks by their job_id
        self.tasks: Dict[str, Task] = {}
        # Dictionary to store job metadata
        self.job_metadata: Dict[str, Dict[str, Any]] = {}

    def _job_id(self, project_id: str, trajectory_id: str) -> str:
        """Generate a job ID from project ID and trajectory ID."""
        return f"{project_id}:{trajectory_id}"

    @tracer.start_as_current_span("AsyncioRunner.start_job")
    async def start_job(self, project_id: str, trajectory_id: str, job_func: Callable) -> bool:
        """Start a job for the given project and trajectory.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run

        Returns:
            True if the job was started, False otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)

        # Check if job already exists
        if await self.job_exists(project_id, trajectory_id):
            self.logger.warning(f"Job {job_id} already exists")
            return False

        # Create metadata for the job
        self.job_metadata[job_id] = {
            "id": job_id,
            "status": JobStatus.QUEUED,
            "enqueued_at": datetime.now(),
            "started_at": None,
            "ended_at": None,
            "exc_info": None,
            "project_id": project_id,
            "trajectory_id": trajectory_id,
        }

        # Create and start the task
        task = asyncio.create_task(self._run_job(job_id, job_func))
        self.tasks[job_id] = task

        return True

    async def _run_job(self, job_id: str, job_func: Callable) -> None:
        """Run a job and update its status.

        Args:
            job_id: The job ID
            job_func: The function to run
        """
        meta = self.job_metadata[job_id]
        try:
            # Update job status to running
            meta["status"] = JobStatus.RUNNING
            meta["started_at"] = datetime.now()

            # Run the job
            await job_func()

            # Update job status to completed
            meta["status"] = JobStatus.COMPLETED

        except asyncio.CancelledError:
            # Handle cancellation
            meta["status"] = JobStatus.CANCELED
            self.logger.info(f"Job {job_id} was cancelled")

        except Exception as exc:
            # Handle failure
            meta["status"] = JobStatus.FAILED
            meta["exc_info"] = str(exc)
            self.logger.exception(f"Job {job_id} failed: {exc}")

        finally:
            # Update end time
            meta["ended_at"] = datetime.now()

            # Clean up task reference (but keep metadata for status queries)
            if job_id in self.tasks:
                del self.tasks[job_id]

    async def get_jobs(self, project_id: str | None = None) -> List[JobInfo]:
        """Get all jobs for the given project, or all jobs if project_id is None.

        Args:
            project_id: The project ID to filter by, or None for all jobs

        Returns:
            List of JobInfo objects
        """
        jobs = []

        for job_id, meta in self.job_metadata.items():
            # Filter by project_id if provided
            if project_id is not None and not job_id.startswith(f"{project_id}:"):
                continue

            jobs.append(
                JobInfo(
                    id=meta["id"],
                    status=cast(JobStatus, meta["status"]),
                    enqueued_at=meta["enqueued_at"],
                    started_at=meta["started_at"],
                    ended_at=meta["ended_at"],
                    exc_info=meta["exc_info"],
                )
            )

        return jobs

    @tracer.start_as_current_span("AsyncioRunner.cancel_job")
    async def cancel_job(self, project_id: str, trajectory_id: str | None = None) -> None:
        """Cancel a job for the given project and trajectory, or all jobs for the project.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID, or None to cancel all jobs for the project
        """
        if trajectory_id is None:
            # Cancel all jobs for the project
            for job_id in list(self.tasks.keys()):
                if job_id.startswith(f"{project_id}:"):
                    await self._cancel_job_by_id(job_id)
        else:
            # Cancel a specific job
            job_id = self._job_id(project_id, trajectory_id)
            await self._cancel_job_by_id(job_id)

    async def _cancel_job_by_id(self, job_id: str) -> None:
        """Cancel a job by its ID.

        Args:
            job_id: The job ID
        """
        if job_id in self.tasks:
            task = self.tasks[job_id]
            task.cancel()

            # Wait for the task to be properly cancelled
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Update job metadata
            if job_id in self.job_metadata:
                self.job_metadata[job_id]["status"] = JobStatus.CANCELED
                self.job_metadata[job_id]["ended_at"] = datetime.now()

            self.logger.info(f"Job {job_id} cancelled")

    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists for the given project and trajectory.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job exists, False otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)
        return job_id in self.job_metadata

    @tracer.start_as_current_span("AsyncioRunner.retry_job")
    async def retry_job(self, project_id: str, trajectory_id: str) -> bool:
        """Retry a failed job for the given project and trajectory.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job was requeued, False otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)

        # Check if job exists and is failed
        if job_id not in self.job_metadata:
            self.logger.warning(f"Job {job_id} not found for retry")
            return False

        meta = self.job_metadata[job_id]
        if meta["status"] != JobStatus.FAILED:
            self.logger.warning(f"Job {job_id} is not failed, cannot retry")
            return False

        # Cannot retry without the original job function
        self.logger.error(f"Retrying job {job_id} is not implemented for AsyncioRunner")
        return False

    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        """Get the status of a job for the given project and trajectory.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            The job status
        """
        job_id = self._job_id(project_id, trajectory_id)

        if job_id in self.job_metadata:
            return cast(JobStatus, self.job_metadata[job_id]["status"])

        return JobStatus.NOT_FOUND

    async def get_runner_info(self) -> RunnerInfo:
        """Get information about the runner.

        Returns:
            RunnerInfo object with runner status information
        """
        try:
            active_tasks = len(self.tasks)

            return RunnerInfo(
                runner_type="asyncio",
                status=RunnerStatus.RUNNING if active_tasks > 0 else RunnerStatus.STOPPED,
                data={"active_tasks": active_tasks},
            )
        except Exception as exc:
            self.logger.exception(f"Error checking runner status: {exc}")
            return RunnerInfo(runner_type="asyncio", status=RunnerStatus.ERROR, data={"error": str(exc)})

    async def get_job_status_summary(self, project_id: str) -> JobsStatusSummary:
        """Get a summary of job statuses for the given project.

        Args:
            project_id: The project ID

        Returns:
            A JobsStatusSummary object
        """
        # Initialize counters
        total = 0
        queued = 0
        running = 0
        completed = 0
        failed = 0
        canceled = 0
        pending = 0

        # Collect job IDs by status
        job_ids = {
            "queued": [],
            "running": [],
            "completed": [],
            "failed": [],
            "canceled": [],
            "pending": [],
        }

        # Count jobs by status
        for job_id, meta in self.job_metadata.items():
            if job_id.startswith(f"{project_id}:"):
                total += 1
                status = meta["status"]

                if status == JobStatus.QUEUED:
                    queued += 1
                    job_ids["queued"].append(job_id)
                elif status == JobStatus.RUNNING:
                    running += 1
                    job_ids["running"].append(job_id)
                elif status == JobStatus.COMPLETED:
                    completed += 1
                    job_ids["completed"].append(job_id)
                elif status == JobStatus.FAILED:
                    failed += 1
                    job_ids["failed"].append(job_id)
                elif status == JobStatus.CANCELED:
                    canceled += 1
                    job_ids["canceled"].append(job_id)
                elif status == JobStatus.PENDING:
                    pending += 1
                    job_ids["pending"].append(job_id)

        return JobsStatusSummary(
            project_id=project_id,
            total_jobs=total,
            queued_jobs=queued,
            running_jobs=running,
            completed_jobs=completed,
            failed_jobs=failed,
            canceled_jobs=canceled,
            pending_jobs=pending,
            job_ids=job_ids,
        )
