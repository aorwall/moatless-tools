import asyncio
import importlib
import inspect
import logging
import traceback
import signal
import sys
from asyncio import Task
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, cast

from opentelemetry import trace

from moatless.runner.runner import (
    JobInfo,
    JobStatus,
    JobsStatusSummary,
    BaseRunner,
    RunnerInfo,
    RunnerStatus,
    JobDetails,
    JobDetailSection,
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
        # Register signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Start a periodic job cleanup task
        self._start_cleanup_task()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            # Use a lambda to call cleanup_tasks via the default event loop
            try:
                signal.signal(sig, self._signal_handler)
                self.logger.info(f"Registered signal handler for {sig.name}")
            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Failed to register signal handler for {getattr(sig, 'name', sig)}: {e}")

    def _signal_handler(self, signum, frame):
        """Handle signals by scheduling task cleanup and then exiting."""
        sig_name = signal.Signals(signum).name
        self.logger.info(f"Received {sig_name} signal, cancelling all tasks...")

        # Get the event loop and schedule the cleanup
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._cleanup_tasks())

            # Add a callback to stop the loop after cleanup
            loop.call_soon(lambda: sys.exit(0))
        except RuntimeError as e:
            self.logger.error(f"Error in signal handler: {e}")
            # If we can't get the event loop, just exit
            sys.exit(1)

    async def _cleanup_tasks(self):
        """Cancel all running tasks."""
        self.logger.info(f"Cleaning up {len(self.tasks)} tasks...")

        # Create a list of tasks to cancel to avoid modifying dict during iteration
        tasks_to_cancel = list(self.tasks.items())

        for job_id, task in tasks_to_cancel:
            self.logger.info(f"Cancelling task for job {job_id}")
            await self._cancel_job_by_id(job_id)

        self.logger.info("All tasks cancelled")

    def _job_id(self, project_id: str, trajectory_id: str) -> str:
        """Generate a job ID from project ID and trajectory ID."""
        return f"{project_id}:{trajectory_id}"

    def _start_cleanup_task(self):
        """Start a background task to clean up stale jobs periodically."""
        try:
            loop = asyncio.get_event_loop()
            self.cleanup_task = loop.create_task(self._periodic_cleanup())

            # Make sure we don't get warnings when the task is destroyed
            self.cleanup_task.add_done_callback(lambda _: None)
        except RuntimeError:
            self.logger.warning("Couldn't start cleanup task: no event loop available")

    async def _periodic_cleanup(self):
        """Periodically check for and clean up any stale jobs."""
        try:
            while True:
                await asyncio.sleep(60)  # Run cleanup every minute
                await self._cleanup_stale_jobs()
        except asyncio.CancelledError:
            # Task was cancelled, no need to log this
            pass
        except Exception as e:
            self.logger.error(f"Error in periodic cleanup: {e}")

    async def _cleanup_stale_jobs(self):
        """Check for and clean up any stale jobs that should be removed."""
        try:
            stale_tasks_count = 0

            # Look for jobs that are in terminal states but still in tasks dict
            for job_id, metadata in list(self.job_metadata.items()):
                status = metadata["status"]
                if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]:
                    if job_id in self.tasks:
                        self.logger.warning(
                            f"Found stale job {job_id} in terminal state {status} but still in tasks dictionary"
                        )
                        try:
                            # Cancel the task if it's still running
                            self.tasks[job_id].cancel()
                            del self.tasks[job_id]
                            stale_tasks_count += 1
                        except Exception as e:
                            self.logger.error(f"Error cleaning up stale task {job_id}: {e}")

            if stale_tasks_count > 0:
                self.logger.info(f"Cleaned up {stale_tasks_count} stale tasks")
        except Exception as e:
            self.logger.error(f"Error in cleanup_stale_jobs: {e}")

    @tracer.start_as_current_span("AsyncioRunner.start_job")
    async def start_job(
        self, project_id: str, trajectory_id: str, job_func: Callable | str, node_id: int | None = None
    ) -> bool:
        """Start a job for the given project and trajectory.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run or a string with the fully qualified function name
            node_id: The node ID
        Returns:
            True if the job was started, False otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)

        # Check if job already exists
        if await self.job_exists(project_id, trajectory_id):
            # Check if the job is in a terminal state but still in the tasks dict (stale job)
            if job_id in self.job_metadata:
                status = self.job_metadata[job_id]["status"]
                if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]:
                    if job_id in self.tasks:
                        self.logger.warning(
                            f"Found stale job {job_id} in terminal state {status}. Cleaning up before restart."
                        )
                        try:
                            # Remove the stale task
                            self.tasks[job_id].cancel()
                            del self.tasks[job_id]
                            # Delete the old metadata and continue with starting the job
                            del self.job_metadata[job_id]
                        except Exception as e:
                            self.logger.error(f"Error cleaning up stale job {job_id}: {e}")
                            return False
                    else:
                        # Job is in terminal state but not in tasks, it's safe to restart
                        self.logger.info(f"Removing completed job {job_id} for restart")
                        del self.job_metadata[job_id]
                else:
                    # Job exists and is not in a terminal state
                    self.logger.warning(f"Job {job_id} already exists")
                    return False
            else:
                # This shouldn't happen, but handle it just in case
                self.logger.warning(f"Job {job_id} already exists")
                return False

        # Create metadata for the job
        self.job_metadata[job_id] = {
            "id": job_id,
            "node_id": node_id,
            "project_id": project_id,
            "trajectory_id": trajectory_id,
            "status": JobStatus.PENDING,
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
            self.logger.info(f"Removing job {job_id} from active tasks")
            del self.tasks[job_id]
        else:
            self.logger.warning(f"Task done callback called for job {job_id}, but job not found in tasks")

        # Skip if job metadata no longer exists
        if job_id not in self.job_metadata:
            self.logger.warning(f"Task done callback called for job {job_id}, but job not found in metadata")
            return

        # Get the current job status
        current_status = self.job_metadata[job_id]["status"]

        # Check if the job is already in a terminal state (COMPLETED, FAILED, CANCELED)
        # We don't want to override these statuses as they might have been set by _run_job
        terminal_states = [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]
        if current_status in terminal_states:
            self.logger.info(f"Job {job_id} already in terminal state {current_status}, not updating status")
            return

        # If not in a terminal state, determine the final state based on the task result
        if task.exception():
            self.logger.info(f"Job {job_id} failed with exception: {task.exception()}")
            self.job_metadata[job_id]["status"] = JobStatus.FAILED
            self.job_metadata[job_id]["ended_at"] = datetime.now()
            self.job_metadata[job_id]["exc_info"] = str(task.exception())
        else:
            self.logger.info(f"Job {job_id} completed successfully")
            self.job_metadata[job_id]["status"] = JobStatus.COMPLETED
            self.job_metadata[job_id]["ended_at"] = datetime.now()

    async def _run_job(self, job_id: str, job_func: Callable | str) -> None:
        """Run a job in a separate task.

        Args:
            job_id: The job ID
            job_func: The function to run or a string with the fully qualified function name
            node_id: The node ID
        """

        # Extract project_id and trajectory_id from job_id
        job = self.job_metadata.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        project_id = job["project_id"]
        trajectory_id = job["trajectory_id"]
        node_id = job["node_id"]

        self.logger.info(
            f"Running job {job_id} with project_id {project_id}, trajectory_id {trajectory_id} and node_id {node_id}"
        )

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
                sig = inspect.signature(job_func)
                parameters = sig.parameters

                # Determine how to call the function based on its parameters
                if not "project_id" in parameters or not "trajectory_id" in parameters or not "node_id" in parameters:
                    raise ValueError("Function must accept project_id, trajectory_id and node_id as arguments")

                # Function accepts both project_id and trajectory_id
                result = job_func(project_id=project_id, trajectory_id=trajectory_id, node_id=node_id)

                # Check if the result is a coroutine and await it if necessary
                if inspect.iscoroutine(result):
                    await result
            else:
                raise TypeError(f"job_func must be callable, got {type(job_func)}")

            # _handle_task_done will update the status to COMPLETED
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
                    project_id=meta["project_id"],
                    trajectory_id=meta["trajectory_id"],
                    status=cast(JobStatus, meta["status"]),
                    enqueued_at=meta["enqueued_at"],
                    started_at=meta["started_at"],
                    ended_at=meta["ended_at"],
                    metadata={"exc_info": meta["exc_info"]} if meta["exc_info"] else {},
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
        exists_in_metadata = job_id in self.job_metadata
        exists_in_tasks = job_id in self.tasks

        if exists_in_metadata:
            status = self.job_metadata[job_id]["status"]
            self.logger.info(f"Job {job_id} exists in metadata with status: {status}")

            # If the job is in a terminal state but still in tasks, log a warning
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED] and exists_in_tasks:
                self.logger.warning(f"Job {job_id} is in terminal state {status} but still in tasks dictionary")

        return exists_in_metadata

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
        summary = JobsStatusSummary()

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
            elif status == JobStatus.RUNNING:
                summary.running_jobs += 1
            elif status == JobStatus.COMPLETED:
                summary.completed_jobs += 1
            elif status == JobStatus.FAILED:
                summary.failed_jobs += 1
            elif status == JobStatus.CANCELED:
                summary.canceled_jobs += 1

        return summary

    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobStatus enum representing the current status, or JobStatus.PENDING if the job does not exist
        """
        job_id = self._job_id(project_id, trajectory_id)
        
        # Check if the job exists in metadata
        if job_id in self.job_metadata:
            return cast(JobStatus, self.job_metadata[job_id]["status"])
        
        # If job doesn't exist, return PENDING as default status
        return JobStatus.PENDING

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

    async def get_job_details(self, project_id: str, trajectory_id: str) -> Optional[JobDetails]:
        """Get detailed information about a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobDetails object containing detailed information about the job if available
        """
        job_id = self._job_id(project_id, trajectory_id)

        # Check if the job exists
        if job_id not in self.job_metadata:
            return None

        # Get job metadata
        meta = self.job_metadata[job_id]
        status = cast(JobStatus, meta["status"])

        # Create the job details object
        details = JobDetails(
            id=job_id,
            status=status,
            project_id=project_id,
            trajectory_id=trajectory_id,
            enqueued_at=meta["enqueued_at"],
            started_at=meta["started_at"],
            ended_at=meta["ended_at"],
            raw_data=meta,
        )

        # Add sections with detailed information
        basic_info_section = JobDetailSection(
            name="basic_info",
            display_name="Basic Information",
            data={
                "job_id": job_id,
                "status": status.value,
                "project_id": project_id,
                "trajectory_id": trajectory_id,
            },
        )
        details.sections.append(basic_info_section)

        # Add timing information section
        timing_section = JobDetailSection(
            name="timing",
            display_name="Timing Information",
            data={
                "enqueued_at": meta["enqueued_at"].isoformat() if meta["enqueued_at"] else None,
                "started_at": meta["started_at"].isoformat() if meta["started_at"] else None,
                "ended_at": meta["ended_at"].isoformat() if meta["ended_at"] else None,
            },
        )
        details.sections.append(timing_section)

        # Add error information if applicable
        if meta.get("exc_info") and status == JobStatus.FAILED:
            details.error = meta["exc_info"]
            error_section = JobDetailSection(
                name="error",
                display_name="Error Information",
                data={"error": meta["exc_info"]},
            )
            details.sections.append(error_section)

        return details

    async def cleanup(self):
        """Cleanup all resources used by the runner.

        This method can be called explicitly when the application is shutting down
        to ensure all tasks are properly cancelled and resources are released.
        """
        self.logger.info("Explicitly cleaning up AsyncioRunner resources...")

        # Cancel the cleanup task if it exists
        if hasattr(self, "cleanup_task") and self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await asyncio.wait_for(self.cleanup_task, timeout=0.1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        await self._cleanup_tasks()

        # Also clean up any tasks that might have been completed but not removed
        await self._cleanup_stale_jobs()

        # Clear any remaining job metadata
        job_count = len(self.job_metadata)
        if job_count > 0:
            self.logger.info(f"Clearing {job_count} job metadata entries")
            self.job_metadata.clear()

        self.logger.info("AsyncioRunner cleanup completed")
