# type: ignore
"""Tests for the AsyncioRunner implementation."""

import asyncio
import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timedelta
from typing import AsyncGenerator

import pytest_asyncio

from moatless.runner.asyncio_runner import AsyncioRunner
from moatless.runner.runner import BaseRunner, JobStatus, RunnerStatus


# mypy: disable-error-code="misc"
@pytest_asyncio.fixture  # type: ignore
async def asyncio_runner() -> AsyncGenerator[AsyncioRunner, None]:
    """Fixture to create an AsyncioRunner instance."""
    runner = AsyncioRunner()
    yield runner
    
    # Clean up any running tasks to avoid warnings
    for task in list(runner.tasks.values()):
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=0.1)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


@pytest.mark.asyncio
async def test_start_job(asyncio_runner):
    """Test that a job can be started successfully."""
    # Create a mock job function
    mock_job = AsyncMock()
    mock_job.return_value = None
    
    # Start the job
    result = await asyncio_runner.start_job("test-project", "test-trajectory", mock_job)
    
    # Check the result
    assert result is True
    
    # Check that the job was stored
    job_id = asyncio_runner._job_id("test-project", "test-trajectory")
    assert job_id in asyncio_runner.job_metadata
    
    # Verify initial job status
    assert asyncio_runner.job_metadata[job_id]["status"] == JobStatus.PENDING
    
    # Wait for the job to complete
    await asyncio.sleep(0.1)
    
    # Verify the job was called
    mock_job.assert_called_once()
    
    # Verify final job status
    assert asyncio_runner.job_metadata[job_id]["status"] == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_start_job_already_exists(asyncio_runner):
    """Test that starting a job that already exists returns False."""
    # Create a mock job function
    mock_job = AsyncMock()
    
    # Start the job once
    await asyncio_runner.start_job("test-project", "test-trajectory", mock_job)
    
    # Try to start it again
    result = await asyncio_runner.start_job("test-project", "test-trajectory", mock_job)
    
    # Check that the second attempt failed
    assert result is False


@pytest.mark.asyncio
async def test_job_failure(asyncio_runner):
    """Test that a job failure is properly handled."""
    # Create a job function that raises an exception
    async def failing_job():
        raise ValueError("Test error")
    
    # Start the job
    await asyncio_runner.start_job("test-project", "test-trajectory", failing_job)
    
    # Wait for the job to fail
    await asyncio.sleep(0.1)
    
    # Get the job ID
    job_id = asyncio_runner._job_id("test-project", "test-trajectory")
    
    # Verify job status
    assert asyncio_runner.job_metadata[job_id]["status"] == JobStatus.FAILED
    assert "Test error" in asyncio_runner.job_metadata[job_id]["exc_info"]


@pytest.mark.asyncio
async def test_cancel_job(asyncio_runner):
    """Test that a job can be cancelled."""
    # Create a job function that sleeps
    async def long_running_job():
        await asyncio.sleep(10)
    
    # Start the job
    await asyncio_runner.start_job("test-project", "test-trajectory", long_running_job)
    
    # Wait for the job to start
    await asyncio.sleep(0.1)
    
    # Cancel the job
    await asyncio_runner.cancel_job("test-project", "test-trajectory")
    
    # Wait for cancellation to complete
    await asyncio.sleep(0.1)
    
    # Get the job ID
    job_id = asyncio_runner._job_id("test-project", "test-trajectory")
    
    # Verify job status
    assert asyncio_runner.job_metadata[job_id]["status"] == JobStatus.CANCELED


@pytest.mark.asyncio
async def test_cancel_all_project_jobs(asyncio_runner):
    """Test that all jobs for a project can be cancelled."""
    # Create a job function that sleeps
    async def long_running_job():
        await asyncio.sleep(10)
    
    # Start multiple jobs for the same project
    await asyncio_runner.start_job("test-project", "trajectory-1", long_running_job)
    await asyncio_runner.start_job("test-project", "trajectory-2", long_running_job)
    await asyncio_runner.start_job("other-project", "trajectory-3", long_running_job)
    
    # Wait for jobs to start
    await asyncio.sleep(0.1)
    
    # Cancel all jobs for test-project
    await asyncio_runner.cancel_job("test-project", None)
    
    # Wait for cancellation to complete
    await asyncio.sleep(0.1)
    
    # Get job IDs
    job_id1 = asyncio_runner._job_id("test-project", "trajectory-1")
    job_id2 = asyncio_runner._job_id("test-project", "trajectory-2")
    job_id3 = asyncio_runner._job_id("other-project", "trajectory-3")
    
    # Verify job statuses
    assert asyncio_runner.job_metadata[job_id1]["status"] == JobStatus.CANCELED
    assert asyncio_runner.job_metadata[job_id2]["status"] == JobStatus.CANCELED
    assert asyncio_runner.job_metadata[job_id3]["status"] == JobStatus.RUNNING
    
    # Cleanup the other-project job to prevent warnings
    await asyncio_runner.cancel_job("other-project", "trajectory-3")


@pytest.mark.asyncio
async def test_get_jobs(asyncio_runner):
    """Test retrieving jobs with various filters."""
    # Create a mock job function
    mock_job = AsyncMock()
    
    # Start jobs for different projects
    await asyncio_runner.start_job("project-1", "trajectory-1", mock_job)
    await asyncio_runner.start_job("project-1", "trajectory-2", mock_job)
    await asyncio_runner.start_job("project-2", "trajectory-3", mock_job)
    
    # Wait for jobs to complete
    await asyncio.sleep(0.1)
    
    # Get all jobs
    all_jobs = await asyncio_runner.get_jobs()
    assert len(all_jobs) == 3
    
    # Get jobs for project-1
    project1_jobs = await asyncio_runner.get_jobs("project-1")
    assert len(project1_jobs) == 2
    assert all(job.id.startswith("project-1:") for job in project1_jobs)
    
    # Get jobs for project-2
    project2_jobs = await asyncio_runner.get_jobs("project-2")
    assert len(project2_jobs) == 1
    assert project2_jobs[0].id.startswith("project-2:")


@pytest.mark.asyncio
async def test_job_exists(asyncio_runner):
    """Test checking if a job exists."""
    # Create a mock job function
    mock_job = AsyncMock()
    
    # Start a job
    await asyncio_runner.start_job("test-project", "test-trajectory", mock_job)
    
    # Check if jobs exist
    assert await asyncio_runner.job_exists("test-project", "test-trajectory") is True
    assert await asyncio_runner.job_exists("test-project", "nonexistent") is False
    assert await asyncio_runner.job_exists("nonexistent", "test-trajectory") is False


@pytest.mark.asyncio
async def test_get_job_status(asyncio_runner):
    """Test getting the status of a job."""
    # Create a mock job function
    mock_job = AsyncMock()
    
    # Start a job
    await asyncio_runner.start_job("test-project", "test-trajectory", mock_job)
    
    # Check status
    status = await asyncio_runner.get_job_status("test-project", "test-trajectory")
    assert status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED)
    
    # Wait for job to complete
    await asyncio.sleep(0.1)
    
    # Check final status
    status = await asyncio_runner.get_job_status("test-project", "test-trajectory")
    assert status == JobStatus.COMPLETED
    
    # Check status of nonexistent job
    status = await asyncio_runner.get_job_status("nonexistent", "nonexistent")
    assert status == JobStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_retry_job(asyncio_runner):
    """Test that retry_job returns False since it's not implemented."""
    # Create a job function that fails
    async def failing_job():
        raise ValueError("Test error")
    
    # Start the job
    await asyncio_runner.start_job("test-project", "test-trajectory", failing_job)
    
    # Wait for the job to fail
    await asyncio.sleep(0.1)
    
    # Try to retry the job
    result = await asyncio_runner.retry_job("test-project", "test-trajectory")
    
    # Check that retry returns False
    assert result is False


@pytest.mark.asyncio
async def test_get_runner_info(asyncio_runner):
    """Test getting runner information."""
    # Create a job function that sleeps for a longer time
    async def long_running_job():
        await asyncio.sleep(1)
    
    # Get info when no jobs are running
    info = await asyncio_runner.get_runner_info()
    assert info.runner_type == "asyncio"
    assert info.status == RunnerStatus.STOPPED
    assert info.data["active_tasks"] == 0
    
    # Start a job
    await asyncio_runner.start_job("test-project", "long-job", long_running_job)
    
    # Get info when jobs are running
    info = await asyncio_runner.get_runner_info()
    assert info.runner_type == "asyncio"
    assert info.status == RunnerStatus.RUNNING
    assert info.data["active_tasks"] > 0
    
    # Wait for the job to complete to avoid warnings
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_get_job_status_summary(asyncio_runner):
    """Test getting job status summary."""
    # Create job functions with different outcomes
    async def successful_job():
        return
    
    async def failing_job():
        raise ValueError("Test error")
    
    async def long_running_job():
        await asyncio.sleep(0.5)
    
    # Start various jobs
    await asyncio_runner.start_job("test-project", "success-1", successful_job)
    await asyncio_runner.start_job("test-project", "success-2", successful_job)
    await asyncio_runner.start_job("test-project", "failure-1", failing_job)
    await asyncio_runner.start_job("test-project", "long-1", long_running_job)
    
    # Let jobs process
    await asyncio.sleep(0.1)
    
    # Get summary
    summary = await asyncio_runner.get_job_status_summary("test-project")
    
    # Verify summary
    assert summary.project_id == "test-project"
    assert summary.total_jobs == 4
    assert summary.completed_jobs == 2
    assert summary.failed_jobs == 1
    assert summary.running_jobs == 1
    
    # Check job_ids collections
    assert len(summary.job_ids["completed"]) == 2
    assert len(summary.job_ids["failed"]) == 1
    assert len(summary.job_ids["running"]) == 1
    
    # Cancel the long-running job
    await asyncio_runner.cancel_job("test-project", "long-1")
    await asyncio.sleep(0.1)
    
    # Get updated summary
    summary = await asyncio_runner.get_job_status_summary("test-project")
    assert summary.canceled_jobs == 1
    assert len(summary.job_ids["canceled"]) == 1 