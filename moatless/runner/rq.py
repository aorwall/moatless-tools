import asyncio
import json
import logging
import os
import shutil
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from redis import Redis
from rq import Queue, Worker
from rq.command import send_stop_job_command
from rq.job import Dependency, Job

from moatless.runner.runner import JobInfo, JobsStatusSummary, JobStatus, RunnerInfo, RunnerStatus
from moatless.telemetry import extract_context_data, extract_trace_context

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.runner.rq")


class MoatlessQueue(Queue):
    def enqueue(self, f, *args, **kwargs):
        """Enqueue a job with OpenTelemetry context propagation."""
        carrier = extract_trace_context()

        # Add trace context and context data to job metadata
        if "meta" not in kwargs:
            kwargs["meta"] = {}
        kwargs["meta"]["otel_context"] = carrier

        job = super().enqueue(f, *args, **kwargs)

        return job


class RQRunner:
    """Runner for managing jobs with RQ."""

    def __init__(self, redis_url: str | None = None):
        """Initialize the runner with Redis connection.

        Args:
            redis_url: URL for Redis connection
        """
        if redis_url:
            self.redis_url = redis_url
        elif os.getenv("REDIS_URL"):
            self.redis_url = os.getenv("REDIS_URL")
        else:
            self.redis_url = "redis://localhost:6379"
        self.redis_conn = Redis.from_url(self.redis_url)
        self.queue = MoatlessQueue(connection=self.redis_conn, default_timeout=3600)
        self.logger = logging.getLogger(__name__)

    @tracer.start_as_current_span("RQRunner.start_job")
    async def start_job(self, project_id: str, trajectory_id: str, job_func: Callable) -> bool:
        """Start a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job was scheduled successfully, False otherwise
        """

        if await self.job_exists(project_id, trajectory_id):
            return False

        job_id = self._job_id(project_id, trajectory_id)
        try:
            run_job = self.queue.enqueue(
                job_func,
                kwargs={
                    "project_id": project_id,
                    "trajectory_id": trajectory_id,
                },
                job_id=job_id,
                result_ttl=3600,
                job_timeout=3600,
            )
            return True
        except Exception as exc:
            self.logger.exception(f"Error starting job {project_id}_{trajectory_id}")
            raise exc

    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
        """
        job_id = self._job_id(project_id, trajectory_id)

        if self._is_queued(job_id):
            return JobStatus.QUEUED
        elif self._is_running(job_id):
            return JobStatus.RUNNING
        else:
            return JobStatus.PENDING

    def _is_queued(self, job_id: str) -> bool:
        # if get_job_ids has an item that contains job_id, return True
        for item in self.queue.get_job_ids():
            if job_id in item:
                return True
        return False

    def _is_running(self, job_id: str) -> bool:
        # if started_job_registry has an item that contains job_id, return True
        for item in self.queue.started_job_registry.get_job_ids():
            if job_id in item:
                return True
        return False

    async def get_jobs(self, project_id: str | None = None) -> list[JobInfo]:
        """Get all jobs for a project.

        Args:
            project_id: The project ID

        Returns:
            List of JobInfo objects with job status information
        """

        try:
            result = []
            job_ids = self.get_job_ids()

            # Filter by project_id if specified
            if project_id:
                job_prefix = f"run_{project_id}_"
                job_ids = [job_id for job_id in job_ids if job_id.startswith(job_prefix)]

            if not job_ids:
                return []

            # Use fetch_many for better performance with multiple jobs
            jobs = Job.fetch_many(job_ids, connection=self.redis_conn)

            for job in jobs:
                if job:
                    result.append(
                        JobInfo(
                            id=job.id,
                            status=self._map_job_status(job.get_status()),
                            enqueued_at=job.enqueued_at,
                            started_at=job.started_at,
                            ended_at=job.ended_at,
                            exc_info=job.exc_info,
                        )
                    )
                else:
                    self.logger.warning("One of the jobs could not be fetched")

            return result
        except Exception:
            self.logger.exception(f"Error getting jobs for project {project_id}")
            return []

    @tracer.start_as_current_span("RQRunner.cancel_job")
    async def cancel_job(self, project_id: str, trajectory_id: str | None = None):
        """Cancel a job or all jobs for a project.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID. If None, cancels all jobs for the project.

        Returns:
            None
        """
        if trajectory_id is None:
            # Cancel all jobs for the project
            self.logger.info(f"Canceling all jobs for project {project_id}")
            job_prefix = f"run_{project_id}_"

            queued_job_ids = self.queue.get_job_ids()
            started_job_ids = self.queue.started_job_registry.get_job_ids()
            queued_job_ids = [
                job_id
                for job_id in queued_job_ids
                if job_id.startswith(job_prefix) and (not trajectory_id or trajectory_id in job_id)
            ]
            started_job_ids = [
                job_id
                for job_id in started_job_ids
                if job_id.startswith(job_prefix) and (not trajectory_id or trajectory_id in job_id)
            ]

            logger.info(f"Canceling {len(queued_job_ids)} queued jobs and {len(started_job_ids)} started jobs")

            # Process started jobs with send_stop_job_command
            for job_id in started_job_ids:
                try:
                    self.logger.info(f"Stopping running job {job_id}")
                    send_stop_job_command(self.redis_conn, job_id)
                except Exception as exc:
                    self.logger.warning(f"Error stopping job {job_id}: {exc}")

            # Process queued and other jobs with Job.fetch_many and job.cancel()
            if queued_job_ids:
                try:
                    # Fetch jobs in batch for better performance
                    jobs = Job.fetch_many(queued_job_ids, connection=self.redis_conn)
                    for job in jobs:
                        if job:
                            status = job.get_status()
                            self.logger.info(f"Canceling job {job.id} with status {status}")
                            job.cancel()
                        else:
                            self.logger.warning("One of the jobs could not be fetched")
                except Exception as exc:
                    self.logger.exception(f"Error batch canceling jobs: {exc}")
        else:
            # Cancel a specific job
            job_id = self._job_id(project_id, trajectory_id)
            try:
                job = Job.fetch(job_id, connection=self.redis_conn)
                if job:
                    status = job.get_status()
                    if status == "started":
                        self.logger.info(f"Stopping running job {job_id}")
                        send_stop_job_command(self.redis_conn, job_id)
                    else:
                        self.logger.info(f"Canceling job {job_id} with status {status}")
                        job.cancel()
            except Exception as exc:
                self.logger.warning(f"Error canceling job {job_id}: {exc}")

    @tracer.start_as_current_span("RQRunner.retry_job")
    async def retry_job(self, project_id: str, trajectory_id: str):
        """Retry a failed job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            None
        """
        job_id = self._job_id(project_id, trajectory_id)
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            if job and job.get_status() == "failed":
                self.logger.info(f"Retrying failed job {job_id}")
                job.requeue()
                return True
            else:
                self.logger.warning(f"Job {job_id} not found or not in failed state")
                return False
        except Exception as exc:
            self.logger.exception(f"Error retrying job {job_id}: {exc}")
            return False

    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists in Redis.

        Args:
            job_id: ID of the job to check

        Returns:
            True if the job exists, False otherwise
        """
        run_job_id = self._job_id(project_id, trajectory_id)
        return run_job_id in self.get_job_ids()

    def get_job_ids(self) -> list[str]:
        queued_job_ids = self.queue.get_job_ids()
        started_job_ids = self.queue.started_job_registry.get_job_ids()
        return queued_job_ids + started_job_ids

    async def get_runner_info(self) -> RunnerInfo:
        """Check if any RQ workers are currently running.

        This method checks if there are any active workers connected to the Redis instance
        that can process jobs from the queue.

        Returns:
            RunnerInfo object with runner status information
        """
        try:
            workers = Worker.all(connection=self.redis_conn)

            active_workers = [w for w in workers if w.state != "suspended"]

            return RunnerInfo(
                runner_type="rq",
                status=RunnerStatus.RUNNING if len(active_workers) > 0 else RunnerStatus.STOPPED,
                data={"active_workers": len(active_workers), "total_workers": len(workers)},
            )

        except Exception as exc:
            self.logger.exception(f"Error checking if runner is up: {exc}")
            return RunnerInfo(runner_type="rq", status=RunnerStatus.ERROR, data={"error": str(exc)})

    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        """Get the status by checking if job_id is part of job id in queued or started job ids

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
        """
        job_prefix = self._job_id(project_id, trajectory_id)

        if any(
            [
                job_id in job_id
                for job_id in self.queue.started_job_registry.get_job_ids()
                if job_id.startswith(job_prefix)
            ]
        ):
            return JobStatus.RUNNING

        if any([job_id in job_id for job_id in self.queue.get_job_ids() if job_id.startswith(job_prefix)]):
            return JobStatus.QUEUED

        if any(
            [
                job_id in job_id
                for job_id in self.queue.failed_job_registry.get_job_ids()
                if job_id.startswith(job_prefix)
            ]
        ):
            return JobStatus.FAILED

        if any(
            [
                job_id in job_id
                for job_id in self.queue.finished_job_registry.get_job_ids()
                if job_id.startswith(job_prefix)
            ]
        ):
            return JobStatus.COMPLETED

        if any(
            [
                job_id in job_id
                for job_id in self.queue.canceled_job_registry.get_job_ids()
                if job_id.startswith(job_prefix)
            ]
        ):
            return JobStatus.CANCELED

        return JobStatus.NOT_FOUND

    def _job_id(self, project_id: str, trajectory_id: str) -> str:
        return f"run_{project_id}_{trajectory_id}"

    def _map_job_status(self, status: str) -> JobStatus:
        """Map RQ job status to JobStatus enum.

        Args:
            status: RQ job status string

        Returns:
            Corresponding JobStatus enum value
        """
        if status == "queued":
            return JobStatus.QUEUED
        elif status == "started":
            return JobStatus.RUNNING
        elif status == "finished":
            return JobStatus.COMPLETED
        elif status == "failed":
            return JobStatus.FAILED
        elif status == "canceled":
            return JobStatus.CANCELED
        elif status == "stopped":
            return JobStatus.FAILED  # Map stopped to failed as per RQ behavior
        else:
            return JobStatus.PENDING

    async def get_job_status_summary(self, project_id: str) -> JobsStatusSummary:
        """Get a summary of job statuses for a project.

        Args:
            project_id: The project ID

        Returns:
            JobsStatusSummary with counts and IDs of jobs in different states
        """
        try:
            summary = JobsStatusSummary(project_id=project_id)
            job_ids = self.get_job_ids()

            # Filter by project_id
            job_prefix = f"run_{project_id}_"
            project_job_ids = [job_id for job_id in job_ids if job_id.startswith(job_prefix)]

            if not project_job_ids:
                return summary

            # Separate jobs by state for counting
            queued_job_ids = [job_id for job_id in project_job_ids if job_id in self.queue.get_job_ids()]
            started_job_ids = [
                job_id for job_id in project_job_ids if job_id in self.queue.started_job_registry.get_job_ids()
            ]

            # Set counts for queued and running jobs
            summary.queued_jobs = len(queued_job_ids)
            summary.running_jobs = len(started_job_ids)
            summary.total_jobs = len(project_job_ids)

            # Store job IDs by status
            summary.job_ids["queued"] = queued_job_ids
            summary.job_ids["running"] = started_job_ids

            # For other statuses, we need to fetch the jobs
            other_job_ids = [
                job_id for job_id in project_job_ids if job_id not in queued_job_ids and job_id not in started_job_ids
            ]

            if other_job_ids:
                jobs = Job.fetch_many(other_job_ids, connection=self.redis_conn)
                for job in jobs:
                    if job:
                        status = job.get_status()
                        if status == "finished":
                            summary.completed_jobs += 1
                            summary.job_ids["completed"].append(job.id)
                        elif status == "failed":
                            summary.failed_jobs += 1
                            summary.job_ids["failed"].append(job.id)
                        elif status == "canceled":
                            summary.canceled_jobs += 1
                            summary.job_ids["canceled"].append(job.id)
                        else:
                            summary.pending_jobs += 1
                            summary.job_ids["pending"].append(job.id)

            return summary
        except Exception as exc:
            self.logger.exception(f"Error getting job status summary for project {project_id}: {exc}")
            # Return empty summary on error
            return JobsStatusSummary(project_id=project_id)
