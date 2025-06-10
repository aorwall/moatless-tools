"""Scheduler runner implementation for Moatless."""

import asyncio
import logging
import importlib
import os
from collections.abc import Callable
from datetime import datetime
from typing import Optional, Type

from moatless.runner.runner import (
    BaseRunner,
    JobInfo,
    JobStatus,
    RunnerInfo,
    JobDetails,
    JobFunction,
)
from moatless.runner.storage.storage import JobStorage
from moatless.runner.storage.memory import InMemoryJobStorage


logger = logging.getLogger(__name__)


class SchedulerRunner(BaseRunner):
    """Runner implementation that schedules jobs with configurable limits."""

    def __init__(
        self,
        runner_impl: Type[BaseRunner],
        storage_type: str = "memory",
        redis_url: Optional[str] = None,
        max_jobs_per_project: Optional[int] = None,
        max_total_jobs: Optional[int] = None,
        scheduler_interval_seconds: int = 5,
        auto_cleanup_completed: bool = True,
        **runner_kwargs,
    ):
        """Initialize the scheduler runner.

        Args:
            runner_impl: Implementation class for the underlying runner
            storage_type: Type of storage to use ("memory" or "redis")
            redis_url: Redis URL, required if storage_type is "redis"
            max_jobs_per_project: Maximum number of concurrent jobs per project
            max_total_jobs: Maximum number of total concurrent jobs
            scheduler_interval_seconds: Interval in seconds for job scheduling
            auto_cleanup_completed: Whether to automatically clean up completed jobs from storage
            **runner_kwargs: Additional arguments to pass to the runner implementation
        """
        self.runner = runner_impl(**runner_kwargs)

        # Read job limits from environment variables, fall back to provided values if not set
        # A value of 0 means unlimited jobs
        self.max_jobs_per_project = max_jobs_per_project or int(os.environ.get("MOATLESS_MAX_JOBS_PER_PROJECT", 0))
        self.max_total_jobs = max_total_jobs or int(os.environ.get("MOATLESS_MAX_TOTAL_JOBS", 1))
        self.scheduler_interval_seconds = scheduler_interval_seconds
        self.auto_cleanup_completed = auto_cleanup_completed
        self.logger = logging.getLogger(__name__)

        # Initialize storage
        # Use JobStorage as the base type for self.storage
        self.storage: JobStorage
        if storage_type == "redis":
            if not redis_url:
                raise ValueError("Redis URL is required for Redis storage")
            from moatless.runner.storage.redis import RedisJobStorage
            self.storage = RedisJobStorage(redis_url)
        else:
            self.storage = InMemoryJobStorage()

        # Initialize scheduler
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_running = False

        # Start the scheduler
        self.start_scheduler()

    async def start_job(
        self, project_id: str, trajectory_id: str, job_func: Callable, node_id: int | None = None
    ) -> bool:
        """Queue a job for execution.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run
            node_id: Optional node ID

        Returns:
            True if the job was queued successfully, False otherwise
        """
        # Validate inputs
        if not project_id or not trajectory_id:
            self.logger.error(f"Invalid job parameters: project_id={project_id}, trajectory_id={trajectory_id}")
            return False
            
        if not job_func or not callable(job_func):
            self.logger.error(f"Invalid job function provided for {project_id}-{trajectory_id}")
            return False
            
        try:
            # Check if job already exists
            existing_job = await self.storage.get_job(project_id, trajectory_id)
            if existing_job:
                job_status = await self.runner.get_job_status(project_id, trajectory_id)

                # If job already exists and is not in a terminal state, just return True
                if job_status in [JobStatus.PENDING, JobStatus.RUNNING]:
                    self.logger.info(f"Job {project_id}-{trajectory_id} already exists with status {job_status}")
                    return True

                # If job exists but is in a terminal state, we'll re-queue it
                self.logger.info(
                    f"Re-queueing job {project_id}-{trajectory_id} with previous status {existing_job.status} and job status {job_status}"
                )

                # Remove the job from storage
                # TODO: Just update the existing one?
                await self.storage.remove_job(project_id, trajectory_id)

            # Create job info and set it to PENDING
            job_info = JobInfo(
                id=f"{project_id}-{trajectory_id}",
                status=JobStatus.PENDING,
                project_id=project_id,
                trajectory_id=trajectory_id,
                enqueued_at=datetime.now(),
                job_func=JobFunction(module=job_func.__module__, name=job_func.__name__),
                node_id=node_id,
            )

            # Store job in storage
            await self.storage.add_job(job_info)

            # Try to start the job immediately if possible
            try:
                return await self._try_start_job(job_info)
            except Exception as e:
                self.logger.exception(f"Error trying to start job {project_id}-{trajectory_id}: {e}")
                # Mark job as failed if we can't start it
                job_info.status = JobStatus.FAILED
                job_info.ended_at = datetime.now()
                if not job_info.metadata:
                    job_info.metadata = {}
                job_info.metadata["error"] = f"Failed to start job: {str(e)}"
                await self.storage.update_job(job_info)
                return False
                
        except Exception as e:
            self.logger.exception(f"Unexpected error starting job {project_id}-{trajectory_id}: {e}")
            return False

    async def get_jobs(self, project_id: str | None = None) -> list[JobInfo]:
        """Get a list of jobs for the given project.

        Args:
            project_id: The project ID, or None for all projects

        Returns:
            List of JobInfo objects
        """
        logger.debug(f"Getting jobs for project {project_id}")
        jobs = await self.storage.get_jobs(project_id)

        # Filter out only completed and canceled jobs (these should have been cleaned up)
        # Keep failed jobs visible for manual inspection
        # But also clean up any completed/canceled jobs that still exist in storage
        active_jobs = []
        for job in jobs:
            if job.status in [JobStatus.COMPLETED, JobStatus.CANCELED]:
                # These should have been cleaned up, remove them now
                self.logger.info(f"Cleaning up terminal job {job.id} from storage")
                await self.storage.remove_job(job.project_id, job.trajectory_id)
            else:
                # Include PENDING, RUNNING, FAILED, and other statuses
                active_jobs.append(job)

        return active_jobs

    async def cancel_job(self, project_id: str, trajectory_id: str | None = None) -> None:
        """Cancel a job or all jobs for a project.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID, or None for all jobs in the project
        """
        if trajectory_id is None:
            # Get all jobs for the project
            jobs = await self.storage.get_jobs(project_id)

            # Cancel each job
            for job in jobs:
                if job.project_id is not None and job.trajectory_id is not None:
                    await self._cancel_job(job.project_id, job.trajectory_id)
        else:
            # Cancel the specific job
            await self._cancel_job(project_id, trajectory_id)

    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job exists, False otherwise
        """
        job = await self.storage.get_job(project_id, trajectory_id)
        return job is not None

    async def get_job_status(self, project_id: str, trajectory_id: str) -> Optional[JobStatus]:
        """Get the status of a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobStatus enum representing the current status, or None if the job does not exist
        """
        job = await self.storage.get_job(project_id, trajectory_id)
        if job is None:
            # Job not found in our storage, check if it exists in the underlying runner
            underlying_status = await self.runner.get_job_status(project_id, trajectory_id)
            if underlying_status is not None:
                self.logger.warning(
                    f"Job {project_id}-{trajectory_id} exists in kubernetes but not in scheduler storage"
                )
            return underlying_status

        # For non-terminal jobs, sync with the underlying runner
        if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]:
            await self._sync_job(job)
            # Check if job was removed during sync
            job = await self.storage.get_job(project_id, trajectory_id)
            if job is None:
                return None
            return job.status
        return job.status

    async def get_runner_info(self) -> RunnerInfo:
        """Get information about the runner.

        Returns:
            RunnerInfo object with runner status information
        """
        # Get the underlying runner info
        return await self.runner.get_runner_info()

    async def get_job_details(self, project_id: str, trajectory_id: str) -> Optional[JobDetails]:
        """Get detailed information about a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobDetails object or None if job not found
        """
        # Get the job from storage
        job = await self.storage.get_job(project_id, trajectory_id)
        if job is None:
            return None

        return await self.runner.get_job_details(project_id, trajectory_id)

    async def get_job_logs(self, project_id: str, trajectory_id: str) -> Optional[str]:
        """Get logs for a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            String containing the logs if available, None otherwise
        """
        # Get job status
        job = await self.storage.get_job(project_id, trajectory_id)
        if job is None:
            return None

        # If job is running or failed, delegate to the underlying runner
        if job.status in [JobStatus.RUNNING, JobStatus.FAILED]:
            try:
                return await self.runner.get_job_logs(project_id, trajectory_id)
            except Exception as e:
                self.logger.warning(f"Error getting job logs from underlying runner: {e}")
                return None

        # For pending jobs, return a message indicating the job is queued
        if job.status in [JobStatus.PENDING]:
            queue_position = "unknown"
            if job.metadata and "queue_position" in job.metadata:
                queue_position = job.metadata["queue_position"]

            enqueued_at = "unknown time"
            if job.enqueued_at:
                enqueued_at = str(job.enqueued_at)

            return f"Job is queued (position: {queue_position}) since {enqueued_at}"

        # For canceled jobs
        if job.status == JobStatus.CANCELED:
            canceled_at = "unknown time"
            if job.ended_at:
                canceled_at = job.ended_at.isoformat()

            return f"Job was canceled at {canceled_at}"

        return None

    async def get_queue_size(self) -> int:
        """Get the current queue size.

        Returns:
            Number of jobs in the queue
        """
        # Create a synchronous wrapper for the async method
        return await self.storage.get_queued_jobs_count()

    async def reset_jobs(self, project_id: str | None = None) -> bool:
        """Reset all jobs or jobs for a specific project.

        Args:
            project_id: Optional project ID to limit reset to specific project

        Returns:
            True if jobs were reset successfully, False otherwise
        """
        self.logger.info(f"Resetting jobs{f' for project {project_id}' if project_id else ''}")

        try:
            # Get jobs to reset
            jobs = await self.storage.get_jobs(project_id)

            # Cancel all non-terminal jobs first
            for job in jobs:
                if job.status in [JobStatus.RUNNING, JobStatus.PENDING]:
                    # Only attempt to cancel with underlying runner if job is actually running
                    if job.status in [JobStatus.RUNNING] and job.project_id and job.trajectory_id:
                        try:
                            await self.runner.cancel_job(job.project_id, job.trajectory_id)
                        except Exception as e:
                            self.logger.warning(f"Error canceling job {job.id} with underlying runner: {e}")

                await self.storage.remove_job(job.project_id, job.trajectory_id)

            self.logger.info(f"Reset {len(jobs)} jobs{f' for project {project_id}' if project_id else ''}")
            return True

        except Exception as e:
            self.logger.exception(f"Error resetting jobs: {e}")
            return False

    def start_scheduler(self) -> None:
        """Start the job scheduler."""
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            self.logger.info("Started job scheduler")

    def stop_scheduler(self) -> None:
        """Stop the job scheduler."""
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_running = False
            self._scheduler_task.cancel()
            self.logger.info("Stopped job scheduler")
    
    async def cleanup(self) -> None:
        """Cleanup resources used by the scheduler.
        
        This method should be called when the scheduler is no longer needed
        to properly cleanup the scheduler task and avoid RuntimeWarnings.
        """
        self.stop_scheduler()
        
        # Wait for the scheduler task to complete if it exists
        if self._scheduler_task and not self._scheduler_task.done():
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                # Expected when task is cancelled
                pass
            except Exception as e:
                self.logger.warning(f"Error while waiting for scheduler task to complete: {e}")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that periodically tries to start queued jobs."""
        try:
            self.logger.info("Job scheduler started")

            while self._scheduler_running:
                try:
                    await self._sync_jobs_with_runner()

                    # Clean up completed/terminal jobs that still exist in storage
                    await self._cleanup_terminal_jobs()

                    # Get all pending jobs
                    all_jobs = await self.storage.get_jobs()
                    pending_jobs = [job for job in all_jobs if job.status == JobStatus.PENDING]

                    if pending_jobs:
                        self.logger.debug(f"Found {len(pending_jobs)} pending jobs")

                        # Sort jobs by enqueued_at (oldest first)
                        pending_jobs.sort(key=lambda j: j.enqueued_at or datetime.max)

                        # Update queue position for each job
                        for i, job in enumerate(pending_jobs):
                            if not job.metadata:
                                job.metadata = {}
                            job.metadata["queue_position"] = i + 1
                            await self.storage.update_job(job)

                        # Try to start jobs
                        for job in pending_jobs:
                            await self._try_start_job(job)

                except Exception as e:
                    self.logger.exception(f"Error in scheduler loop: {e}")

                # Wait before next iteration
                await asyncio.sleep(self.scheduler_interval_seconds)

        except asyncio.CancelledError:
            self.logger.info("Job scheduler task cancelled")
        except Exception as e:
            self.logger.exception(f"Error in job scheduler: {e}")

    async def _cleanup_terminal_jobs(self) -> None:
        """Clean up jobs that are in terminal states from storage."""
        try:
            all_jobs = await self.storage.get_jobs()
            # Only clean up completed and canceled jobs automatically
            # Keep failed jobs for manual inspection unless explicitly canceled
            terminal_jobs = [job for job in all_jobs if job.status in [JobStatus.COMPLETED, JobStatus.CANCELED]]

            if terminal_jobs:
                self.logger.debug(f"Cleaning up {len(terminal_jobs)} terminal jobs from storage")

                for job in terminal_jobs:
                    # For completed jobs, delete from kubernetes if they still exist
                    if job.status == JobStatus.COMPLETED:
                        job_exists_in_runner = await self.runner.job_exists(job.project_id, job.trajectory_id)
                        if job_exists_in_runner:
                            try:
                                self.logger.debug(f"Deleting completed job {job.id} from kubernetes during cleanup")
                                await self.runner.cancel_job(job.project_id, job.trajectory_id)
                            except Exception as e:
                                self.logger.warning(f"Failed to delete completed job {job.id} during cleanup: {e}")
                        await self.storage.remove_job(job.project_id, job.trajectory_id)
                        self.logger.debug(f"Removed completed job {job.id} from storage")
                    elif job.status == JobStatus.CANCELED:
                        # For canceled jobs, also clean up from kubernetes if they exist
                        job_exists_in_runner = await self.runner.job_exists(job.project_id, job.trajectory_id)
                        if job_exists_in_runner:
                            try:
                                self.logger.debug(f"Deleting canceled job {job.id} from kubernetes during cleanup")
                                await self.runner.cancel_job(job.project_id, job.trajectory_id)
                                self.logger.debug(f"Successfully deleted canceled job {job.id} from kubernetes")
                            except Exception as e:
                                self.logger.warning(f"Failed to delete canceled job {job.id} during cleanup: {e}")
                        await self.storage.remove_job(job.project_id, job.trajectory_id)
                        self.logger.debug(f"Removed canceled job {job.id} from storage")

        except Exception as e:
            self.logger.exception(f"Error cleaning up terminal jobs: {e}")

    async def _try_start_job(self, job: JobInfo) -> bool:
        """Try to start a job with the underlying runner.

        Args:
            job: The job to start
            job_func: The function to run
            node_id: Optional node ID
        """
        # Check if the job is in PENDING state
        if job.status != JobStatus.PENDING:
            self.logger.warning(f"Cannot start job {job.id}: not in PENDING state, current state: {job.status}")
            return False

        # Check if we can start more jobs in total
        total_running = await self.storage.get_running_jobs_count()
        if self.max_total_jobs and total_running >= self.max_total_jobs:
            self.logger.debug(f"Cannot start job {job.id}: total limit reached ({total_running}/{self.max_total_jobs})")
            return False

        # Check if we can start more jobs for this project
        project_running = await self.storage.get_running_jobs_count(job.project_id)
        if self.max_jobs_per_project and project_running >= self.max_jobs_per_project:
            self.logger.debug(
                f"Cannot start job {job.id}: project limit reached ({project_running}/{self.max_jobs_per_project})"
            )
            return False

        # Import the function
        try:
            if job.job_func is None:
                self.logger.error(f"Cannot start job {job.id}: job_func is None")
                job.status = JobStatus.FAILED
                job.ended_at = datetime.now()
                if job.metadata is None:
                    job.metadata = {}
                job.metadata["error"] = "Error importing job function: job_func is None"
                await self.storage.update_job(job)
                return False

            module = importlib.import_module(job.job_func.module)
            job_func = getattr(module, job.job_func.name)
        except Exception as e:
            self.logger.exception(f"Error importing job function {job.job_func}: {e}")

            # Update job status to FAILED
            job.status = JobStatus.FAILED
            job.ended_at = datetime.now()
            if job.metadata is None:
                job.metadata = {}
            job.metadata["error"] = f"Error importing job function: {str(e)}"
            await self.storage.update_job(job)
            return False

        # Try to start the job with the underlying runner
        try:
            success = await self.runner.start_job(job.project_id, job.trajectory_id, job_func, job.node_id)

            if success:
                # Job started successfully
                self.logger.info(f"Started job {job.id} in RUNNING state")

                # Update job status to RUNNING
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                await self.storage.update_job(job)

            else:
                self.logger.warning(f"Runner could not start job {job.id} at this time.")

            return success

        except Exception as e:
            # Error starting job, update status to FAILED
            job.status = JobStatus.FAILED
            job.ended_at = datetime.now()
            if job.metadata is None:
                job.metadata = {}
            job.metadata["error"] = f"Error starting job: {str(e)}"
            await self.storage.update_job(job)
            self.logger.exception(f"Error starting job {job.id}: {e}")
            return False

    async def _sync_jobs_with_runner(self) -> None:
        """Synchronize job status with the underlying runner.

        This method only checks active jobs (PENDING or RUNNING) to reduce unnecessary calls.
        """
        try:
            # Get only PENDING or RUNNING jobs from our storage
            all_jobs = await self.storage.get_jobs()
            active_jobs = [job for job in all_jobs if job.status in [JobStatus.PENDING, JobStatus.RUNNING]]

            if not active_jobs:
                self.logger.debug("No active jobs found")
                return

            sync_count = 0
            for job in active_jobs:
                try:
                    await self._sync_job(job)
                    sync_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to sync job {job.id}: {e}")
                    # Continue syncing other jobs even if one fails

            if sync_count > 0:
                self.logger.debug(f"Successfully synced {sync_count}/{len(active_jobs)} jobs with runner")
        except Exception as e:
            self.logger.exception(f"Error syncing jobs with runner: {e}")

    async def _sync_job(self, job: JobInfo) -> None:
        """Synchronize the status of a job with the underlying runner.

        Args:
            job: The job info to sync
        """
        # Validate job has required fields
        if not job.project_id or not job.trajectory_id:
            self.logger.error(f"Job {job.id} missing required fields for sync")
            return
            
        try:
            job_status = await self.runner.get_job_status(job.project_id, job.trajectory_id)

            if job_status is None:
                if job.status == JobStatus.RUNNING:
                    self.logger.warning(
                        f"Job {job.id} is marked as RUNNING but doesn't exist in runner, marking as STOPPED"
                    )
                    job.status = JobStatus.STOPPED
                    job.ended_at = datetime.now()
                    if job.metadata is None:
                        job.metadata = {}
                    job.metadata["error"] = "Job disappeared from the underlying runner while RUNNING"
                    await self.storage.update_job(job)
                elif job.status == JobStatus.COMPLETED:
                    # Job was completed and removed from kubernetes, remove from our storage too
                    self.logger.info(
                        f"Job {job.id} was completed and removed from kubernetes, removing from scheduler storage"
                    )
                    await self.storage.remove_job(job.project_id, job.trajectory_id)
            elif job_status != job.status:
                self.logger.info(f"Job {job.id} status changed from {job.status} to {job_status}")
                job.status = job_status

                # Update timestamps based on status
                if job_status == JobStatus.RUNNING and job.started_at is None:
                    job.started_at = datetime.now()
                elif job_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED, JobStatus.STOPPED]:
                    if job.ended_at is None:
                        job.ended_at = datetime.now()

                    # For completed jobs, delete them from kubernetes to clean up cluster
                    if job_status == JobStatus.COMPLETED:
                        try:
                            self.logger.info(f"Job {job.id} completed, deleting from kubernetes cluster")
                            await self.runner.cancel_job(job.project_id, job.trajectory_id)
                            self.logger.info(f"Successfully deleted completed job {job.id} from kubernetes")
                            
                            # Only remove from storage if auto_cleanup_completed is True
                            if self.auto_cleanup_completed:
                                await self.storage.remove_job(job.project_id, job.trajectory_id)
                                return
                        except Exception as e:
                            self.logger.warning(f"Failed to delete completed job {job.id} from kubernetes: {e}")
                            # Still update the job status even if deletion failed

                    # For failed jobs, keep them in Kubernetes and storage for manual inspection
                    # Only delete them when explicitly canceled through the API
                    elif job_status == JobStatus.FAILED:
                        self.logger.info(
                            f"Job {job.id} failed, keeping in Kubernetes for manual inspection. Use the cancel API to delete it."
                        )
                        # Add error message to metadata for failed jobs
                        if job.metadata is None:
                            job.metadata = {}
                        if "error" not in job.metadata:
                            job.metadata["error"] = "Job failed in the underlying runner"
                        # Don't delete failed jobs automatically - let them be manually inspected and canceled

                await self.storage.update_job(job)
                
        except Exception as e:
            self.logger.exception(f"Error syncing job {job.id}: {e}")
            # Don't update job status on sync errors to avoid corrupting state

    async def _update_job_status(self, job: JobInfo) -> None:
        """Update the status of a job by syncing with the underlying runner.
        
        This method handles the finite state machine transitions for job status updates.
        Terminal states (COMPLETED, FAILED, CANCELED, STOPPED) cannot transition to other states.
        
        Args:
            job: The job to update
        """
        # Don't update terminal states
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED, JobStatus.STOPPED]:
            self.logger.debug(f"Job {job.id} is in terminal state {job.status}, not updating")
            return
            
        # Sync the job with the underlying runner
        await self._sync_job(job)

    async def _cancel_job(self, project_id: str, trajectory_id: str) -> None:
        """Cancel a specific job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
        """
        # Get the job from storage
        job = await self.storage.get_job(project_id, trajectory_id)
        if job is None:
            self.logger.warning(f"Cannot cancel job {project_id}-{trajectory_id}: job not found")
            return

        # Check for None values before calling job_exists
        if job.project_id is None or job.trajectory_id is None:
            self.logger.warning("Cannot check job existence: project_id or trajectory_id is None")
            job_exists = False
        else:
            job_exists = await self.runner.job_exists(job.project_id, job.trajectory_id)

        # If job exists in kubernetes, cancel/remove it with the underlying runner
        if job_exists:
            try:
                await self.runner.cancel_job(project_id, trajectory_id)
                self.logger.info(f"Removed job {job.id} from kubernetes runner")
            except Exception as e:
                self.logger.exception(f"Error canceling job {job.id} with underlying runner: {e}")

        # Handle different job states
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]:
            # For terminal jobs (including failed ones), delete from kubernetes if it still exists and remove from our storage
            self.logger.info(f"Cleaning up terminal job {job.id} (status: {job.status})")
            if job_exists:
                self.logger.info(f"Deleting terminal job {job.id} from kubernetes")
            await self.storage.remove_job(project_id, trajectory_id)
        elif job.status == JobStatus.PENDING:
            # For pending jobs, mark as canceled and remove from storage
            self.logger.info(f"Canceling pending job {job.id}")
            job.status = JobStatus.CANCELED
            job.ended_at = datetime.now()
            await self.storage.update_job(job)
            # Remove from storage after updating status
            await self.storage.remove_job(project_id, trajectory_id)
        else:
            # For running jobs, mark as canceled and remove from storage
            self.logger.info(f"Canceling running job {job.id}")
            job.status = JobStatus.CANCELED
            job.ended_at = datetime.now()
            await self.storage.update_job(job)
            # Remove from storage after updating status
            await self.storage.remove_job(project_id, trajectory_id)
