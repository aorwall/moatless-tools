"""Integration tests for the SchedulerRunner with real Docker and Redis."""

import asyncio
import os
import subprocess
import time
import uuid
from datetime import datetime, timedelta

import pytest

from moatless.runner.docker_runner import DockerRunner
from moatless.runner.scheduler import SchedulerRunner
from moatless.runner.runner import JobStatus


def is_docker_available():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def is_redis_available():
    """Check if Redis is available locally."""
    try:
        # Try to ping Redis on default port
        result = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True)
        return result.returncode == 0 and "PONG" in result.stdout
    except Exception:
        return False


# Only run these tests if both Docker and Redis are available
pytestmark = pytest.mark.skipif(
    not (is_docker_available() and is_redis_available()), reason="Docker and/or Redis is not available"
)


def echo_test_func():
    """Simple echo function that's compatible with Docker runner.

    This function just echoes text to stdout and doesn't rely on importing
    anything that might not be available in the Docker container.
    """
    print("Echo function called successfully")


@pytest.mark.asyncio
async def test_scheduler_job_queuing_and_execution():
    """
    Integration test that verifies:
    1. Scheduler properly queues jobs based on limits
    2. Jobs get started when limits allow
    3. Redis storage correctly tracks job states
    4. Scheduler updates job status correctly
    """
    # Create a unique project ID for this test run
    project_id = f"test-{uuid.uuid4().hex[:8]}"

    # Create a scheduler with Docker runner and Redis storage
    # Use a low max_jobs_per_project to easily test queuing
    scheduler = SchedulerRunner(
        runner_impl=DockerRunner,
        storage_type="redis",
        redis_url="redis://localhost:6379/0",
        max_jobs_per_project=1,
        max_total_jobs=3,
        scheduler_interval_seconds=2,
        default_image_name="busybox:latest",  # Use busybox which is lightweight but has enough tools
    )

    try:
        # Function to wait for a job to reach a specific status or any of the target statuses
        async def wait_for_job_status(project_id, trajectory_id, target_statuses, timeout=30):
            """Wait for a job to reach any of the target statuses."""
            if not isinstance(target_statuses, list):
                target_statuses = [target_statuses]

            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < timeout:
                status = await scheduler.get_job_status(project_id, trajectory_id)
                print(f"Job {trajectory_id} status: {status}")
                if status in target_statuses:
                    return True
                await asyncio.sleep(1)
            return False

        # Start first job
        job1_id = "job1"
        print(f"Starting job 1 ({job1_id})")
        result1 = await scheduler.start_job(project_id, job1_id, echo_test_func)
        assert result1 is True, "Failed to start job 1"

        # Check if job exists
        job_exists = await scheduler.job_exists(project_id, job1_id)
        assert job_exists, "Job 1 should exist"

        # Start second job for the same project - should be queued due to project limit
        job2_id = "job2"
        print(f"Starting job 2 ({job2_id})")
        result2 = await scheduler.start_job(project_id, job2_id, echo_test_func)
        assert result2 is True, "Failed to queue job 2"

        # Check that job 2 is queued
        job2_status = await scheduler.get_job_status(project_id, job2_id)
        assert job2_status == JobStatus.PENDING, f"Job 2 should be pending, status is {job2_status}"

        # Get all jobs and check their statuses
        jobs = await scheduler.get_jobs(project_id)
        assert len(jobs) == 2, f"Expected 2 jobs, found {len(jobs)}"

        # Start a third job from a different project
        different_project_id = f"test-{uuid.uuid4().hex[:8]}"
        job3_id = "job3"
        print(f"Starting job 3 ({job3_id}) in different project")
        result3 = await scheduler.start_job(different_project_id, job3_id, echo_test_func)
        assert result3 is True, "Failed to start job 3"

        # Wait for job status changes (they will either complete or fail quickly in Docker)
        await wait_for_job_status(project_id, job1_id, [JobStatus.COMPLETED, JobStatus.FAILED], timeout=10)

    finally:
        # Clean up
        print("Cleaning up...")
        scheduler.stop_scheduler()

        # Cancel any remaining jobs
        try:
            await scheduler.cancel_job(project_id)
            await scheduler.cancel_job(different_project_id)
        except Exception as e:
            print(f"Error during cleanup: {e}")

        print("Test completed")


@pytest.mark.asyncio
async def test_job_cancellation():
    """Test job cancellation via the scheduler."""
    # Create a unique project ID
    project_id = f"test-{uuid.uuid4().hex[:8]}"

    # Create scheduler with Docker and Redis
    scheduler = SchedulerRunner(
        runner_impl=DockerRunner,
        storage_type="redis",
        redis_url="redis://localhost:6379/0",
        max_jobs_per_project=3,
        max_total_jobs=5,
        scheduler_interval_seconds=2,
        default_image_name="busybox:latest",
    )

    try:
        # Start jobs with a simple echo command
        job1_id = "job1"
        job2_id = "job2"

        # Start the jobs
        await scheduler.start_job(project_id, job1_id, echo_test_func)
        await scheduler.start_job(project_id, job2_id, echo_test_func)

        # Verify jobs exist
        job1_exists = await scheduler.job_exists(project_id, job1_id)
        job2_exists = await scheduler.job_exists(project_id, job2_id)

        assert job1_exists, "Job 1 should exist"
        assert job2_exists, "Job 2 should exist"

        # Cancel job 1
        await scheduler.cancel_job(project_id, job1_id)

        # Wait briefly for cancellation to take effect
        await asyncio.sleep(2)

        # Get job statuses
        job1_status = await scheduler.get_job_status(project_id, job1_id)
        job2_status = await scheduler.get_job_status(project_id, job2_id)

        # The jobs may complete or fail quickly, so we check for all possible terminal states
        assert job1_status in [
            JobStatus.CANCELED,
            JobStatus.FAILED,
            JobStatus.COMPLETED,
            JobStatus.STOPPED,
        ], f"Job 1 should be in a terminal state, status is {job1_status}"
        # Job 2 might complete quickly in test environment, so it could be RUNNING or in a terminal state
        assert job2_status in [
            JobStatus.RUNNING,
            JobStatus.FAILED,
            JobStatus.COMPLETED,
        ], f"Job 2 should be running or completed, status is {job2_status}"

        # Cancel all jobs for the project
        await scheduler.cancel_job(project_id)

        # Verify job status summary after cancellation
        summary = await scheduler.get_job_status_summary()
        assert summary.total_jobs > 0, "Summary should include the jobs we created"

    finally:
        # Clean up
        scheduler.stop_scheduler()
        await scheduler.cancel_job(project_id)
        print("Test completed")
