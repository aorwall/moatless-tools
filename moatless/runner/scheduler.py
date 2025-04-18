"""Scheduler runner implementation for Moatless."""

import asyncio
import logging
import importlib
from collections.abc import Callable
from datetime import datetime
from typing import Dict, List, Optional, Type, Any, Union

from moatless.runner.runner import (
    BaseRunner,
    JobInfo,
    JobStatus,
    RunnerInfo,
    RunnerStatus,
    JobsStatusSummary,
    JobDetails,
    JobDetailSection, JobFunction,
)
from moatless.runner.storage.storage import JobStorage
from moatless.runner.storage.memory import InMemoryJobStorage
from moatless.runner.storage.redis import RedisJobStorage

logger = logging.getLogger(__name__)

class SchedulerRunner(BaseRunner):
    """Runner implementation that schedules jobs with configurable limits."""
    
    def __init__(
        self,
        runner_impl: Type[BaseRunner],
        storage_type: str = "memory",
        redis_url: Optional[str] = None,
        max_jobs_per_project: int = 3,
        max_total_jobs: int = 10,
        scheduler_interval_seconds: int = 5,
        **runner_kwargs
    ):
        """Initialize the scheduler runner.
        
        Args:
            runner_impl: Implementation class for the underlying runner
            storage_type: Type of storage to use ("memory" or "redis")
            redis_url: Redis URL, required if storage_type is "redis"
            max_jobs_per_project: Maximum number of concurrent jobs per project
            max_total_jobs: Maximum number of total concurrent jobs
            scheduler_interval_seconds: Interval in seconds for job scheduling
            **runner_kwargs: Additional arguments to pass to the runner implementation
        """
        self.runner = runner_impl(**runner_kwargs)
        self.max_jobs_per_project = max_jobs_per_project
        self.max_total_jobs = max_total_jobs
        self.scheduler_interval_seconds = scheduler_interval_seconds
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        # Use JobStorage as the base type for self.storage
        self.storage: JobStorage
        if storage_type == "redis":
            if not redis_url:
                raise ValueError("Redis URL is required for Redis storage")
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
        # Check if job already exists
        existing_job = await self.storage.get_job(project_id, trajectory_id)
        if existing_job:
            job_status = await self.runner.get_job_status(project_id, trajectory_id)
            
            # If job already exists and is not in a terminal state, just return True
            if job_status in [JobStatus.PENDING, JobStatus.RUNNING]:
                self.logger.info(f"Job {project_id}-{trajectory_id} already exists with status {job_status}")
                return True
            
            # If job exists but is in a terminal state, we'll re-queue it
            self.logger.info(f"Re-queueing job {project_id}-{trajectory_id} with previous status {existing_job.status} and job status {job_status}")

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
            job_func=JobFunction(
                module=job_func.__module__,
                name=job_func.__name__
            )
        )
        
        # Store job in storage
        await self.storage.add_job(job_info)
        
        # Try to start the job immediately if possible
        try:
            await self._try_start_job(job_info)
        except Exception as e:
            self.logger.exception(f"Error trying to start job {project_id}-{trajectory_id}: {e}")
        
        return True
    
    async def get_jobs(self, project_id: str | None = None) -> list[JobInfo]:
        """Get a list of jobs for the given project.
        
        Args:
            project_id: The project ID, or None for all projects

        Returns:
            List of JobInfo objects
        """
        logger.debug(f"Getting jobs for project {project_id}")
        jobs = await self.storage.get_jobs(project_id)
        jobs = [job for job in jobs if job.status not in [JobStatus.COMPLETED, JobStatus.CANCELED]]
        return jobs
    
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
    
    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        """Get the status of a job.
        
        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobStatus enum representing the current status
        """
        job = await self.storage.get_job(project_id, trajectory_id)
        if job is None:
            return JobStatus.UNKNOWN
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
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that periodically tries to start queued jobs."""
        try:
            self.logger.info("Job scheduler started")

            while self._scheduler_running:
                try:
                    await self._sync_jobs_with_runner()

                    # Get all pending jobs
                    all_jobs = await self.storage.get_jobs()
                    pending_jobs = [job for job in all_jobs if job.status == JobStatus.PENDING]
                    
                    if pending_jobs:
                        self.logger.info(f"Found {len(pending_jobs)} pending jobs")
                        
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
                            await self._try_start_queued_job(job)

                except Exception as e:
                    self.logger.exception(f"Error in scheduler loop: {e}")
                
                # Wait before next iteration
                await asyncio.sleep(self.scheduler_interval_seconds)
                
        except asyncio.CancelledError:
            self.logger.info("Job scheduler task cancelled")
        except Exception as e:
            self.logger.exception(f"Error in job scheduler: {e}")
    
    async def _try_start_queued_job(self, job: JobInfo) -> None:
        """Try to start a queued job if limits allow.
        
        Args:
            job: The job to try to start
        """
        # Check if we can start more jobs in total
        total_running = await self.storage.get_running_jobs_count()
        if total_running >= self.max_total_jobs:
            self.logger.debug(f"Cannot start job {job.id}: total limit reached ({total_running}/{self.max_total_jobs})")
            return
        
        # Check if we can start more jobs for this project
        project_running = await self.storage.get_running_jobs_count(job.project_id)
        if project_running >= self.max_jobs_per_project:
            self.logger.debug(
                f"Cannot start job {job.id}: project limit reached ({project_running}/{self.max_jobs_per_project})"
            )
            return

        # Try to start the job
        await self._try_start_job(job)
    
    async def _try_start_job(self, job: JobInfo) -> None:
        """Try to start a job with the underlying runner.
        
        Args:
            job: The job to start
            job_func: The function to run
            node_id: Optional node ID
        """
        # Check if the job is in PENDING state
        if job.status != JobStatus.PENDING:
            self.logger.warning(f"Cannot start job {job.id}: not in PENDING state, current state: {job.status}")
            return

        # Check if we can start more jobs in total
        total_running = await self.storage.get_running_jobs_count()
        if total_running >= self.max_total_jobs:
            self.logger.info(f"Cannot start job {job.id}: total limit reached ({total_running}/{self.max_total_jobs})")
            return
        
        # Check if we can start more jobs for this project
        project_running = await self.storage.get_running_jobs_count(job.project_id)
        if project_running >= self.max_jobs_per_project:
            self.logger.info(
                f"Cannot start job {job.id}: project limit reached ({project_running}/{self.max_jobs_per_project})"
            )
            return

        # Import the function
        try:
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
            return

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


        except Exception as e:
            # Error starting job, update status to FAILED
            job.status = JobStatus.FAILED
            job.ended_at = datetime.now()
            if job.metadata is None:
                job.metadata = {}
            job.metadata["error"] = f"Error starting job: {str(e)}"
            await self.storage.update_job(job)
            self.logger.exception(f"Error starting job {job.id}: {e}")
    
    async def _update_job_status(self, job: JobInfo) -> None:
        """Update the status of a running job.
        
        Args:
            job: The job to update
        """
        try:
            # Only update jobs that are pending or running
            if job.status not in [JobStatus.RUNNING, JobStatus.PENDING, JobStatus.UNKNOWN]:
                self.logger.info(f"Not updating job {job.id} with status {job.status} - not in RUNNING, PENDING or UNKNOWN state")
                return
            
            # Job exists, get current status from the underlying runner
            current_status = await self.runner.get_job_status(job.project_id, job.trajectory_id)
            
            if current_status == JobStatus.COMPLETED:
                job.status = JobStatus.COMPLETED
                job.ended_at = datetime.now()
                await self.storage.update_job(job)
                self.logger.info(f"Job {job.id} completed")
            
            elif current_status == JobStatus.FAILED:
                job.status = JobStatus.FAILED
                job.ended_at = datetime.now()
                job.metadata["error"] = "Job failed during execution"
                await self.storage.update_job(job)
                self.logger.info(f"Job {job.id} failed")

            elif current_status == JobStatus.UNKNOWN and job.status in [JobStatus.RUNNING, JobStatus.PENDING]:
                job.status = JobStatus.UNKNOWN
                await self.storage.update_job(job)
                self.logger.info(f"Job {job.id} unknown")
            
            elif job.status == JobStatus.UNKNOWN:
                job.status = current_status
                await self.storage.update_job(job)
                self.logger.info(f"Job {job.id} unknown to runner, updating to {current_status}")

            elif current_status != JobStatus.RUNNING and job.status == JobStatus.RUNNING:
                # Any other status for a RUNNING job means it's stopped unexpectedly
                
                job.status = JobStatus.STOPPED
                job.ended_at = datetime.now()
                job.metadata["error"] = f"Job stopped unexpectedly with status: {current_status}"
                await self.storage.update_job(job)
                self.logger.info(f"Job {job.id} stopped")
                
        except Exception as e:
            self.logger.exception(f"Error updating status for job {job.id}: {e}")
            
            # Only transition to FAILED if job was in an active state
            if job.status == JobStatus.RUNNING:
                try:
                    job.status = JobStatus.FAILED
                    job.ended_at = datetime.now()
                    if job.metadata is None:
                        job.metadata = {}
                    job.metadata["error"] = f"Error updating job status: {str(e)}"
                    await self.storage.update_job(job)
                except Exception as inner_e:
                    self.logger.exception(f"Failed to mark job {job.id} as failed: {inner_e}")
    
    async def _sync_jobs_with_runner(self) -> None:
        """Synchronize job status with the underlying runner.
        
        This method only checks active jobs (PENDING or RUNNING) to reduce unnecessary calls.
        """
        try:
            
            # Get only PENDING or RUNNING jobs from our storage
            all_jobs = await self.storage.get_jobs()
            active_jobs = [job for job in all_jobs 
                          if job.status in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.UNKNOWN]]
            
            if not active_jobs:
                self.logger.debug("No active jobs found")
                return
            
            self.logger.info(f"Syncing status for {len(active_jobs)} active jobs")
            
            # For each active job, directly check its status
            for job in active_jobs:
                # Check if job exists in runner
                job_status = await self.runner.get_job_status(job.project_id, job.trajectory_id)
                
                if job_status == JobStatus.PENDING and job.status == JobStatus.RUNNING:
                    self.logger.warning(f"Job {job.id} is marked as RUNNING but doesn't exist in runner, marking as STOPPED")
                    job.status = JobStatus.STOPPED
                    job.ended_at = datetime.now()
                    job.metadata["error"] = "Job disappeared from the underlying runner while RUNNING"
                    await self.storage.update_job(job)
                elif job_status != job.status:
                    self.logger.info(f"Job {job.id} status changed from {job.status} to {job_status}")
                    await self._update_job_status(job)
        
            self.logger.info("Job sync completed")
        except Exception as e:
            self.logger.exception(f"Error syncing jobs with runner: {e}")
    
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
            self.logger.warning(f"Cannot check job existence: project_id or trajectory_id is None")
        else:
            job_exists = await self.runner.job_exists(job.project_id, job.trajectory_id)

            # If job is running, cancel with the underlying runner
            if job_exists:
                try:
                    await self.runner.cancel_job(project_id, trajectory_id)
                except Exception as e:
                    self.logger.exception(f"Error canceling job {job.id} with underlying runner: {e}")

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]:
            self.logger.info(f"Job {job.id} is already in terminal state {job.status}, no need to cancel")
        else:
            job.status = JobStatus.CANCELED
            job.ended_at = datetime.now()
            await self.storage.update_job(job)
            self.logger.info(f"Canceled job {job.id}")
    