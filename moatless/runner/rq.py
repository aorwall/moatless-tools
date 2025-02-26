import json
import logging
import os
import shutil
import time
import traceback
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Any, Dict, List
import uuid
from functools import wraps
from redis import Redis

from moatless.runner.runner import JobInfo, JobStatus, RunnerInfo, RunnerStatus
from moatless.telemetry import extract_context_data, extract_trace_context

from opentelemetry import trace

from rq.job import Dependency, Job        
from rq import Queue
from rq import Worker

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

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the runner with Redis connection.
        
        Args:
            redis_url: URL for Redis connection
        """
        self.redis_url = redis_url
        self.redis_conn = Redis.from_url(redis_url)
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
                job_timeout=3600
            )
            self.logger.info(f"Scheduled run job for {project_id}_{trajectory_id}, job_id: {run_job.id}")
            return True
        except Exception as exc:
            self.logger.exception(f"Error starting job {project_id}_{trajectory_id}")
            raise exc
    
    async def get_job(self, project_id: str, trajectory_id: str) -> JobInfo:
        """Get the status of a job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            JobInfo object with job status information
        """
        if not await self.job_exists(project_id, trajectory_id):
            return None
        
        job_id = self._job_id(project_id, trajectory_id)
        run_job = Job.fetch(job_id, connection=self.redis_conn)
        
        if not run_job:
            return None
        
        return JobInfo(
            id=run_job.id,
            status=self._map_job_status(run_job),
            enqueued_at=run_job.enqueued_at,
            started_at=run_job.started_at,
            ended_at=run_job.ended_at,
            exc_info=run_job.exc_info
        )
    
    async def get_jobs(self, project_id: str | None = None) -> List[JobInfo]:
        """Get all jobs for a project.
        
        Args:
            project_id: The project ID
            
        Returns:
            List of JobInfo objects with job status information
        """
        
        try:
            result = []

            for job_id in self.get_job_ids():
                job = Job.fetch(job_id, connection=self.redis_conn)
                if job:
                    result.append(JobInfo(
                        id=job.id,
                        status=self._map_job_status(job),
                        enqueued_at=job.enqueued_at,
                        started_at=job.started_at,
                        ended_at=job.ended_at,
                        exc_info=job.exc_info
                    ))
                else:
                    self.logger.warning(f"Job {job_id} not found")

            return result
        except Exception as exc:
            self.logger.exception(f"Error getting jobs for project {project_id}")
            return []
    
    
    @tracer.start_as_current_span("RQRunner.cancel_job")
    async def cancel_job(self, project_id: str, trajectory_id: str):
        """Cancel a job.
        
        Args:
            job_id: The job ID
            
        Returns:
            CancellationResult object with cancellation results
        """
        
        run_job_id = self._job_id(project_id, trajectory_id)
            
        if Job.exists(run_job_id, connection=self.redis_conn):
            job = Job.fetch(run_job_id, connection=self.redis_conn)
            if job.get_status() in ['queued', 'started']:
                job.cancel()
    

    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists in Redis.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            True if the job exists, False otherwise
        """
        run_job_id = self._job_id(project_id, trajectory_id)
        return run_job_id in self.get_job_ids()
    
    def get_job_ids(self) -> List[str]:
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
                        
            active_workers = [w for w in workers if w.state != 'suspended']
            
            self.logger.info(f"Found {len(active_workers)} active workers out of {len(workers)} total workers")

            return RunnerInfo(
                runner_type="rq",
                status=RunnerStatus.RUNNING if len(active_workers) > 0 else RunnerStatus.STOPPED,
                data={
                    "active_workers": len(active_workers),
                    "total_workers": len(workers)
                }
            )

        except Exception as exc:
            self.logger.exception(f"Error checking if runner is up: {exc}")
            return RunnerInfo(
                runner_type="rq",
                status=RunnerStatus.ERROR,
                data={
                    "error": str(exc)
                }
            )
        
    def _job_id(self, project_id: str, trajectory_id: str) -> str:
        return f"run_{project_id}_{trajectory_id}"

    def _map_job_status(self, status: str) -> JobStatus:
        if status == "queued":
            return JobStatus.QUEUED
        elif status == "started":
            return JobStatus.RUNNING
        elif status == "completed":
            return JobStatus.COMPLETED
        elif status == "failed":
            return JobStatus.FAILED
        else:
            return JobStatus.PENDING

