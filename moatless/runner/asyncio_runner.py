import asyncio
import logging
from asyncio import Task
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, cast
from uuid import uuid4
import importlib
import traceback

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
    async def start_job(self, project_id: str, trajectory_id: str, job_func: Callable | str) -> bool:
        """Start a job for the given project and trajectory.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run or a string with the fully qualified function name

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
        }

        # Start the job in a separate task
        task = asyncio.create_task(
            self._run_job(job_id, job_func),
            name=f"job-{job_id}",
        )
        self.tasks[job_id] = task

        # Add a callback to handle task completion
        task.add_done_callback(lambda t: self._handle_task_done(job_id, t))

        return True

    def _handle_task_done(self, job_id: str, task: asyncio.Task) -> None:
        """Handle task completion.

        Args:
            job_id: The job ID
            task: The completed task
        """
        # Clean up task reference (but keep metadata for status queries)
        if job_id in self.tasks:
            del self.tasks[job_id]

        # Check if the task raised an exception
        if task.exception() and job_id in self.job_metadata:
            self.job_metadata[job_id]["status"] = JobStatus.FAILED
            self.job_metadata[job_id]["ended_at"] = datetime.now()
            self.job_metadata[job_id]["exc_info"] = str(task.exception())

    async def _run_job(self, job_id: str, job_func: Callable | str) -> None:
        """Run a job in a separate task.

        Args:
            job_id: The job ID
            job_func: The function to run or a string with the fully qualified function name
        """
        # Extract project_id and trajectory_id from job_id
        project_id, trajectory_id = job_id.split(":")

        # Update job status to running
        self.job_metadata[job_id]["status"] = JobStatus.RUNNING
        self.job_metadata[job_id]["started_at"] = datetime.now()

        try:
            # If job_func is a string, import it
            if isinstance(job_func, str):
                self.logger.info(f"Importing function {job_func}")
                module_path, func_name = job_func.rsplit(".", 1)
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                job_func = func

            # Run the job function
            # Use a direct call instead of asyncio.to_thread to avoid type issues
            if callable(job_func):
                job_func(project_id=project_id, trajectory_id=trajectory_id)
            else:
                raise TypeError(f"job_func must be callable, got {type(job_func)}")

            # Update job status to completed
            self.job_metadata[job_id]["status"] = JobStatus.COMPLETED
            self.job_metadata[job_id]["ended_at"] = datetime.now()
        except asyncio.CancelledError:
            # Job was cancelled
            self.logger.info(f"Job {job_id} was cancelled")
            self.job_metadata[job_id]["status"] = JobStatus.CANCELED
            self.job_metadata[job_id]["ended_at"] = datetime.now()
        except Exception as e:
            # Job failed with an exception
            self.logger.exception(f"Job {job_id} failed with exception: {e}")
            self.job_metadata[job_id]["status"] = JobStatus.FAILED
            self.job_metadata[job_id]["ended_at"] = datetime.now()
            self.job_metadata[job_id]["exc_info"] = traceback.format_exc()

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
        summary = JobsStatusSummary(project_id=project_id)

        # Count jobs for the project by status
        for job_id, meta in self.job_metadata.items():
            # Check if job belongs to this project
            if not job_id.startswith(f"{project_id}:"):
                continue

            summary.total_jobs += 1
            status = meta["status"]

            # Update counts and job ID lists based on status
            if status == JobStatus.PENDING:
                summary.pending_jobs += 1
                summary.job_ids["pending"].append(job_id)
            elif status == JobStatus.INITIALIZING:
                summary.initializing_jobs += 1
                summary.job_ids["initializing"].append(job_id)
            elif status == JobStatus.RUNNING:
                summary.running_jobs += 1
                summary.job_ids["running"].append(job_id)
            elif status == JobStatus.COMPLETED:
                summary.completed_jobs += 1
                summary.job_ids["completed"].append(job_id)
            elif status == JobStatus.FAILED:
                summary.failed_jobs += 1
                summary.job_ids["failed"].append(job_id)
            elif status == JobStatus.CANCELED:
                summary.canceled_jobs += 1
                summary.job_ids["canceled"].append(job_id)

        return summary

    async def get_job_logs(self, project_id: str, trajectory_id: str) -> Optional[str]:
        """Get logs for a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            String containing the logs if available, None otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)

        # Check if the job exists
        if job_id not in self.job_metadata:
            return None

        # For AsyncioRunner, we don't capture logs directly, so we'll return basic job status information
        meta = self.job_metadata[job_id]
        status = meta["status"]

        logs = [
            f"Job ID: {job_id}",
            f"Status: {status}",
            f"Enqueued at: {meta['enqueued_at']}",
        ]

        if meta["started_at"]:
            logs.append(f"Started at: {meta['started_at']}")

        if meta["ended_at"]:
            logs.append(f"Ended at: {meta['ended_at']}")

        if meta.get("exc_info") and status == JobStatus.FAILED:
            logs.append("\nError Information:")
            logs.append(meta["exc_info"])

        return "\n".join(logs)
