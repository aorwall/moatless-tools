"""Tests for the scheduler runner implementation."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from moatless.runner.runner import (
    BaseRunner,
    JobInfo,
    JobStatus,
    RunnerInfo,
    RunnerStatus,
    JobsStatusSummary,
    JobDetails,
    JobDetailSection,
)
from moatless.runner.scheduler import SchedulerRunner


class MockRunner(BaseRunner):
    """Mock runner for testing the scheduler."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Dictionary of jobs by key "project_id:trajectory_id"
        self.jobs = {}
        # Keep track of canceled jobs
        self.canceled_jobs = []
        self.job_exists_return_value = True
    
    async def start_job(self, project_id, trajectory_id, job_func, node_id=None):
        """Start a job (mock implementation)."""
        job_key = f"{project_id}:{trajectory_id}"
        
        # Create job record
        self.jobs[job_key] = {
            "project_id": project_id,
            "trajectory_id": trajectory_id,
            "status": JobStatus.RUNNING,
            "started_at": datetime.now(),
        }
        
        return True
    
    async def get_jobs(self, project_id=None):
        """Get all jobs (mock implementation)."""
        result = []
        
        for job_key, job_data in self.jobs.items():
            if project_id is None or job_data["project_id"] == project_id:
                job_info = JobInfo(
                    id=job_key,
                    status=job_data["status"],
                    project_id=job_data["project_id"],
                    trajectory_id=job_data["trajectory_id"],
                    started_at=job_data.get("started_at"),
                    ended_at=job_data.get("ended_at"),
                    enqueued_at=job_data.get("enqueued_at", datetime.now()),
                    metadata=job_data.get("metadata", {})
                )
                result.append(job_info)
                
        return result
        
    async def cancel_job(self, project_id, trajectory_id=None):
        """Mock canceling a job."""
        if trajectory_id is None:
            # Cancel all jobs for the project
            to_cancel = []
            for key, data in self.jobs.items():
                if data["project_id"] == project_id:
                    to_cancel.append(key)
            
            for key in to_cancel:
                self.jobs[key]["status"] = JobStatus.CANCELED
                self.canceled_jobs.append(key)
        else:
            # Cancel specific job
            job_key = f"{project_id}:{trajectory_id}"
            if job_key in self.jobs:
                self.jobs[job_key]["status"] = JobStatus.CANCELED
                self.canceled_jobs.append(job_key)
                
    async def job_exists(self, project_id, trajectory_id):
        """Mock checking if a job exists."""
        job_key = f"{project_id}:{trajectory_id}"
        return job_key in self.jobs
        
    async def get_job_status(self, project_id, trajectory_id):
        """Mock getting job status."""
        job_key = f"{project_id}:{trajectory_id}"
        if job_key in self.jobs:
            return self.jobs[job_key]["status"]
        return JobStatus.PENDING  # Changed from NOT_STARTED to PENDING
        
    async def get_runner_info(self):
        """Mock getting runner info."""
        return RunnerInfo(
            runner_type="mock",
            status=RunnerStatus.RUNNING,
            data={"mock": True}
        )
        
    async def get_job_status_summary(self, project_id):
        """Mock getting job status summary."""
        summary = JobsStatusSummary(project_id=project_id)
        
        for key, data in self.jobs.items():
            if data["project_id"] == project_id:
                summary.total_jobs += 1
                if data["status"] == JobStatus.RUNNING:
                    summary.running_jobs += 1
                    summary.job_ids["running"].append(key)
                elif data["status"] == JobStatus.COMPLETED:
                    summary.completed_jobs += 1
                    summary.job_ids["completed"].append(key)
                elif data["status"] == JobStatus.FAILED:
                    summary.failed_jobs += 1
                    summary.job_ids["failed"].append(key)
                elif data["status"] == JobStatus.CANCELED:
                    summary.canceled_jobs += 1
                    summary.job_ids["canceled"].append(key)
                    
        return summary
        
    async def get_job_details(self, project_id, trajectory_id):
        """Mock getting job details."""
        job_key = f"{project_id}:{trajectory_id}"
        if job_key not in self.jobs:
            return None
            
        data = self.jobs[job_key]
        return JobDetails(
            id=job_key,
            status=data["status"],
            project_id=project_id,
            trajectory_id=trajectory_id,
            started_at=data["started_at"],
            sections=[
                JobDetailSection(
                    name="mock",
                    display_name="Mock Section",
                    data={"mock": True}
                )
            ]
        )
        
    async def get_job_logs(self, project_id, trajectory_id):
        """Mock getting job logs."""
        job_key = f"{project_id}:{trajectory_id}"
        if job_key not in self.jobs:
            return None
            
        return f"Mock logs for job {job_key}"
        
    async def reset_jobs(self, project_id=None):
        """Mock resetting jobs."""
        if project_id is None:
            # Reset all jobs
            self.jobs = {}
            self.canceled_jobs = []
        else:
            # Reset jobs for a specific project
            keys_to_remove = []
            for key, data in self.jobs.items():
                if data["project_id"] == project_id:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self.jobs[key]
                if key in self.canceled_jobs:
                    self.canceled_jobs.remove(key)
                    
        return True
        
    async def get_queue_size(self):
        """Get queue size (mock implementation)."""
        return 0
        
    async def cleanup_job(self, project_id: str, trajectory_id: str):
        """Clean up a job (mock implementation)."""
        job_key = f"{project_id}:{trajectory_id}"
        # Just remove from our dictionary
        if job_key in self.jobs:
            del self.jobs[job_key]
            
        return True


@pytest.fixture
def mock_runner():
    """Fixture to create a mock runner."""
    return MockRunner()


@pytest.fixture
def scheduler(mock_runner):
    """Fixture to create a scheduler with mock runner."""
    # Patch the asyncio.create_task to prevent actual scheduling
    with patch('asyncio.create_task') as mock_create_task:
        # Create a scheduler with our mock runner
        scheduler = SchedulerRunner(
            runner_impl=MockRunner,
            max_jobs_per_project=2,
            max_total_jobs=5,
            scheduler_interval_seconds=1
        )
        
        # Set scheduler as not running to prevent actual scheduling
        scheduler._scheduler_running = False
        
        yield scheduler


@pytest.mark.asyncio
async def test_start_job(scheduler):
    """Test starting a new job."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Start a job
    project_id = "test-project"
    trajectory_id = "test-trajectory"
    result = await scheduler.start_job(project_id, trajectory_id, mock_job_func)
    assert result is True
    
    # Make sure the job exists in the storage
    job_exists = await scheduler.job_exists(project_id, trajectory_id)
    assert job_exists is True
    
    # Check the job status
    job = await scheduler.storage.get_job(project_id, trajectory_id)
    assert job is not None
    
    # Job should exist and be in PENDING or RUNNING state
    assert job.id == f"{project_id}-{trajectory_id}"
    assert job.project_id == project_id
    assert job.trajectory_id == trajectory_id
    assert job.status in [JobStatus.PENDING, JobStatus.RUNNING]
    assert job.enqueued_at is not None


@pytest.mark.asyncio
async def test_job_exists(scheduler):
    """Test checking if a job exists."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Start a job
    await scheduler.start_job("test-project", "test-trajectory", mock_job_func)
    
    # Check if job exists
    exists = await scheduler.job_exists("test-project", "test-trajectory")
    assert exists is True
    
    # Check if non-existent job exists
    exists = await scheduler.job_exists("nonexistent", "nonexistent")
    assert exists is False


@pytest.mark.asyncio
async def test_get_job_status(scheduler):
    """Test getting job status."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Start a job
    project_id = "test-project"
    trajectory_id = "test-trajectory"
    await scheduler.start_job(project_id, trajectory_id, mock_job_func)
    
    # Check the job status
    status = await scheduler.get_job_status(project_id, trajectory_id)
    
    # Initially job should be in PENDING or RUNNING state
    assert status in [JobStatus.PENDING, JobStatus.RUNNING]
    
    # Change the job status
    job = await scheduler.storage.get_job(project_id, trajectory_id)
    job.status = JobStatus.RUNNING
    await scheduler.storage.update_job(job)
    
    # Check the status again
    status = await scheduler.get_job_status(project_id, trajectory_id)
    assert status == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_get_jobs(scheduler):
    """Test getting jobs."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Start some jobs
    await scheduler.start_job("test-project1", "test-trajectory1", mock_job_func)
    await scheduler.start_job("test-project1", "test-trajectory2", mock_job_func)
    await scheduler.start_job("test-project2", "test-trajectory3", mock_job_func)
    
    # Get all jobs
    all_jobs = await scheduler.get_jobs()
    assert len(all_jobs) == 3
    
    # Get jobs for a specific project
    project1_jobs = await scheduler.get_jobs("test-project1")
    assert len(project1_jobs) == 2
    
    project2_jobs = await scheduler.get_jobs("test-project2")
    assert len(project2_jobs) == 1


@pytest.mark.asyncio
async def test_cancel_job(scheduler):
    """Test canceling a job."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Start some jobs
    await scheduler.start_job("test-project", "test-trajectory1", mock_job_func)
    await scheduler.start_job("test-project", "test-trajectory2", mock_job_func)
    
    # Cancel one job
    await scheduler.cancel_job("test-project", "test-trajectory1")
    
    # Check the job status
    status = await scheduler.get_job_status("test-project", "test-trajectory1")
    assert status == JobStatus.CANCELED
    
    # The other job should not be canceled
    status = await scheduler.get_job_status("test-project", "test-trajectory2")
    assert status != JobStatus.CANCELED


@pytest.mark.asyncio
async def test_cancel_all_project_jobs(scheduler):
    """Test canceling all jobs for a project."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Start some jobs for different projects
    await scheduler.start_job("test-project1", "test-trajectory1", mock_job_func)
    await scheduler.start_job("test-project1", "test-trajectory2", mock_job_func)
    await scheduler.start_job("test-project2", "test-trajectory3", mock_job_func)
    
    # Cancel all jobs for project1
    await scheduler.cancel_job("test-project1")
    
    # Check statuses for project1 jobs
    status1 = await scheduler.get_job_status("test-project1", "test-trajectory1")
    status2 = await scheduler.get_job_status("test-project1", "test-trajectory2")
    assert status1 == JobStatus.CANCELED
    assert status2 == JobStatus.CANCELED
    
    # Project2 job should not be canceled
    status3 = await scheduler.get_job_status("test-project2", "test-trajectory3")
    assert status3 != JobStatus.CANCELED


@pytest.mark.asyncio
async def test_get_runner_info(scheduler):
    """Test getting runner info."""
    info = await scheduler.get_runner_info()
    
    # Check basic info
    assert info.runner_type == "mock"
    assert info.status == RunnerStatus.RUNNING
    
    # Check scheduler info
    assert "scheduler" in info.data
    assert info.data["scheduler"]["max_jobs_per_project"] == 2
    assert info.data["scheduler"]["max_total_jobs"] == 5
    assert "queued_jobs" in info.data["scheduler"]
    assert "running_jobs" in info.data["scheduler"]


@pytest.mark.asyncio
async def test_get_job_details(scheduler):
    """Test getting job details."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Start a job and mark it as running directly in storage
    await scheduler.start_job("test-project", "test-trajectory", mock_job_func)
    job = await scheduler.storage.get_job("test-project", "test-trajectory")
    job.status = JobStatus.RUNNING
    await scheduler.storage.update_job(job)
    
    # Get job details
    details = await scheduler.get_job_details("test-project", "test-trajectory")
    
    # Should get details
    assert details is not None
    assert details.project_id == "test-project"
    assert details.trajectory_id == "test-trajectory"
    assert len(details.sections) > 0


@pytest.mark.asyncio
async def test_queue_limits_per_project(scheduler):
    """Test that jobs are queued when project limit is reached."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Patch the storage count methods to simulate running jobs
    with patch.object(scheduler.storage, 'get_running_jobs_count') as mock_running_count:
        # Make it return 2 running jobs for the project, below total limit (5)
        mock_running_count.side_effect = lambda project_id=None: 2 if project_id == "test-project" else 2
        
        # Now start another job - it should be queued because project limit (2) is reached
        await scheduler.start_job("test-project", "new-trajectory", mock_job_func)
        
        # The new job should be in PENDING state
        job = await scheduler.storage.get_job("test-project", "new-trajectory")
        assert job is not None
        assert job.status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_queue_limits_total(scheduler):
    """Test that jobs are queued when total limit is reached."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Patch the storage count methods to simulate running jobs
    with patch.object(scheduler.storage, 'get_running_jobs_count') as mock_running_count:
        # Make it return 5 total running jobs (at the limit)
        mock_running_count.return_value = 5
        
        # Now start another job - it should be queued because total limit (5) is reached
        await scheduler.start_job("new-project", "new-trajectory", mock_job_func)
        
        # The new job should be in PENDING state
        job = await scheduler.storage.get_job("new-project", "new-trajectory")
        assert job is not None
        assert job.status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_restart_pending_jobs(scheduler):
    """Test that pending jobs remain in storage when the scheduler is restarted.
    
    Note: This test used to verify that pending jobs were automatically started
    on scheduler restart, but that functionality has been removed.
    """
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Create a job and set it to PENDING directly in storage
    job_info = JobInfo(
        id="test-project-pending-job",
        status=JobStatus.PENDING,
        project_id="test-project",
        trajectory_id="pending-job",
        enqueued_at=datetime.now(),
        metadata={
            "job_func": {
                "module": mock_job_func.__module__,
                "name": mock_job_func.__name__,
            }
        }
    )
    await scheduler.storage.add_job(job_info)
    
    # Verify job is in PENDING state
    job = await scheduler.storage.get_job("test-project", "pending-job")
    assert job is not None
    assert job.status == JobStatus.PENDING
    
    # Verify the job will be included in the next scheduler loop
    # Start the scheduler manually
    scheduler._scheduler_running = True
    
    # Call sync directly to simulate a scheduler loop
    await scheduler._sync_jobs_with_runner()
    
    # Job should still be in PENDING state in storage
    job = await scheduler.storage.get_job("test-project", "pending-job")
    assert job is not None
    assert job.status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_job_status_synchronization(scheduler):
    """Test that scheduler properly synchronizes job status with the underlying runner."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Start a job
    await scheduler.start_job("test-project", "test-trajectory", mock_job_func)
    
    # Get the job from storage
    job = await scheduler.storage.get_job("test-project", "test-trajectory")
    assert job is not None
    
    # Modify the underlying runner to report a different status
    job_key = f"test-project:test-trajectory"
    scheduler.runner.jobs[job_key] = {
        "project_id": "test-project",
        "trajectory_id": "test-trajectory",
        "status": JobStatus.FAILED,  # Simulate a job that has failed in the runner
        "started_at": datetime.now(),
    }
    
    # Now call the _update_job_status method to sync status
    await scheduler._update_job_status(job)
    
    # Get the updated job
    updated_job = await scheduler.storage.get_job("test-project", "test-trajectory")
    
    # The job should either be marked as FAILED or be PENDING/RUNNING due to auto-restart
    assert updated_job.status in [JobStatus.FAILED, JobStatus.PENDING, JobStatus.RUNNING]
    
    # If failed, it should have an error message
    if updated_job.status == JobStatus.FAILED:
        assert updated_job.metadata and "error" in updated_job.metadata
    
    # If pending/running, it should have retry information
    if updated_job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
        assert updated_job.metadata and "retry_count" in updated_job.metadata
        assert updated_job.metadata["retry_count"] >= 1


@pytest.mark.asyncio
async def test_sync_with_runner(scheduler):
    """Test the _sync_jobs_with_runner method."""
    # Define a mock job function
    def mock_job_func():
        pass

    # Add a RUNNING job to storage directly
    storage_job = JobInfo(
        id="storage-only-job",
        status=JobStatus.RUNNING,
        project_id="test-project",
        trajectory_id="storage-only",
        started_at=datetime.now(),
        metadata={"job_func": {"module": mock_job_func.__module__, "name": mock_job_func.__name__}}
    )
    await scheduler.storage.add_job(storage_job)

    # Add a COMPLETED job to storage
    completed_job = JobInfo(
        id="test-project-completed-job",
        status=JobStatus.COMPLETED,
        project_id="test-project",
        trajectory_id="completed-job",
        started_at=datetime.now(),
        ended_at=datetime.now(),
        metadata={"job_func": {"module": mock_job_func.__module__, "name": mock_job_func.__name__}}
    )
    await scheduler.storage.add_job(completed_job)

    # Add a job to the runner directly that's not in storage
    runner_job_key = "test-project:runner-only"
    scheduler.runner.jobs[runner_job_key] = {
        "project_id": "test-project",
        "trajectory_id": "runner-only",
        "status": JobStatus.RUNNING,
        "started_at": datetime.now(),
    }

    # Mock job_exists and get_job_status to track which jobs are checked
    original_job_exists = scheduler.runner.job_exists
    job_exists_calls = []

    async def mock_job_exists(project_id, trajectory_id):
        job_exists_calls.append((project_id, trajectory_id))
        if project_id == "test-project" and trajectory_id == "storage-only":
            return False
        return project_id == "test-project" and trajectory_id in ["runner-only", "completed-job"]

    scheduler.runner.job_exists = mock_job_exists

    try:
        # Call the sync method - first time will increment not_found_count to 1
        await scheduler._sync_jobs_with_runner()
        
        # With the new retry logic, the job should still be marked as RUNNING but with not_found_count=1
        updated_storage_job = await scheduler.storage.get_job("test-project", "storage-only")
        assert updated_storage_job.status == JobStatus.RUNNING
        assert updated_storage_job.metadata and "not_found_count" in updated_storage_job.metadata
        assert updated_storage_job.metadata["not_found_count"] == 1
        
        # Run sync again - should increment not_found_count to 2
        await scheduler._sync_jobs_with_runner()
        updated_storage_job = await scheduler.storage.get_job("test-project", "storage-only")
        assert updated_storage_job.status == JobStatus.RUNNING
        assert updated_storage_job.metadata["not_found_count"] == 2
        
        # Run sync a third time - should increment not_found_count to 3 and mark as STOPPED
        await scheduler._sync_jobs_with_runner()
        updated_storage_job = await scheduler.storage.get_job("test-project", "storage-only")
        assert updated_storage_job.status == JobStatus.STOPPED
        assert updated_storage_job.metadata and "error" in updated_storage_job.metadata
        assert updated_storage_job.metadata["not_found_count"] == 3
        
        # The completed job should NOT be checked with job_exists
        assert ("test-project", "completed-job") not in job_exists_calls
        
        # The runner-only job should NOT be added to storage
        runner_job_in_storage = await scheduler.storage.get_job("test-project", "runner-only")
        assert runner_job_in_storage is None
        
        # Verify only RUNNING/PENDING jobs are checked in each sync
        assert len(job_exists_calls) == 3  # Once per sync call
        assert job_exists_calls[0] == ("test-project", "storage-only")
        assert job_exists_calls[1] == ("test-project", "storage-only")
        assert job_exists_calls[2] == ("test-project", "storage-only")
    
    finally:
        scheduler.runner.job_exists = original_job_exists


@pytest.mark.asyncio
async def test_sync_only_processes_active_jobs(scheduler):
    """Test that _sync_jobs_with_runner only processes PENDING and RUNNING jobs."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # Add jobs with different states to storage
    job_states = [
        (JobStatus.PENDING, "pending-job"),
        (JobStatus.RUNNING, "running-job"),
        (JobStatus.COMPLETED, "completed-job"),
        (JobStatus.FAILED, "failed-job"),
        (JobStatus.CANCELED, "canceled-job"),
        (JobStatus.STOPPED, "stopped-job")
    ]
    
    for status, job_id in job_states:
        job_info = JobInfo(
            id=f"test-project-{job_id}",
            status=status,
            project_id="test-project",
            trajectory_id=job_id,
            enqueued_at=datetime.now(),
            started_at=datetime.now() if status != JobStatus.PENDING else None,
            ended_at=datetime.now() if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED, JobStatus.STOPPED] else None,
            metadata={"job_func": {"module": mock_job_func.__module__, "name": mock_job_func.__name__}}
        )
        await scheduler.storage.add_job(job_info)
    
    # Set up tracking for which jobs are checked
    update_job_status_calls = []
    original_update_job_status = scheduler._update_job_status
    
    async def mock_update_job_status(job):
        update_job_status_calls.append(job.trajectory_id)
        await original_update_job_status(job)
    
    scheduler._update_job_status = mock_update_job_status
    
    # All jobs exist in the runner for this test
    scheduler.runner.job_exists = AsyncMock(return_value=True)
    
    try:
        # Call the sync method
        await scheduler._sync_jobs_with_runner()
        
        # Only PENDING and RUNNING jobs should be checked
        assert sorted(update_job_status_calls) == ["pending-job", "running-job"]
        assert len(update_job_status_calls) == 2
        
    finally:
        scheduler._update_job_status = original_update_job_status


@pytest.mark.asyncio
async def test_fsm_job_transitions(scheduler):
    """Test that jobs follow the correct Finite State Machine transitions."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # 1. Start a job - should go to PENDING first, but might directly go to RUNNING in some configurations
    # Patch the _try_start_job method to prevent immediate running
    with patch.object(scheduler, '_try_start_job', return_value=None):
        await scheduler.start_job("test-project", "test-trajectory", mock_job_func)
    
        # Get the job from storage
        job = await scheduler.storage.get_job("test-project", "test-trajectory")
        assert job.status == JobStatus.PENDING
    
    # 2. Transition from PENDING to RUNNING
    # _try_start_job will be called on the job
    job_key = f"test-project:test-trajectory"
    scheduler.runner.jobs[job_key] = {
        "project_id": "test-project",
        "trajectory_id": "test-trajectory",
        "status": JobStatus.RUNNING,
        "started_at": datetime.now(),
    }
    
    # Manually call _try_start_job with unlimited capacity to trigger the state change
    with patch.object(scheduler.storage, 'get_running_jobs_count', return_value=0):
        await scheduler._try_start_job(job, mock_job_func)
    
    # Job should now be in RUNNING state
    job = await scheduler.storage.get_job("test-project", "test-trajectory")
    assert job.status == JobStatus.RUNNING
    
    # 3. Transition from RUNNING to COMPLETED
    # Update underlying runner status to COMPLETED
    scheduler.runner.jobs[job_key]["status"] = JobStatus.COMPLETED
    
    # Call update_job_status to detect the change
    await scheduler._update_job_status(job)
    
    # Job should have been removed from storage after completion
    job = await scheduler.storage.get_job("test-project", "test-trajectory")
    assert job is None
    
    # 4. Test transition from RUNNING to FAILED
    # Add a new job and set it to RUNNING
    await scheduler.start_job("test-project", "failed-job", mock_job_func)
    job = await scheduler.storage.get_job("test-project", "failed-job")
    job.status = JobStatus.RUNNING
    await scheduler.storage.update_job(job)
    
    # Set underlying runner job to FAILED
    failed_job_key = f"test-project:failed-job"
    scheduler.runner.jobs[failed_job_key] = {
        "project_id": "test-project",
        "trajectory_id": "failed-job",
        "status": JobStatus.FAILED,
        "started_at": datetime.now(),
    }
    
    # Call update_job_status
    await scheduler._update_job_status(job)
    
    # Job should now be in FAILED state
    job = await scheduler.storage.get_job("test-project", "failed-job")
    assert job.status == JobStatus.FAILED
    assert job.ended_at is not None
    assert job.metadata and "error" in job.metadata
    
    # 5. Test transition from RUNNING to STOPPED (job disappears)
    # Add a new job and set it to RUNNING
    await scheduler.start_job("test-project", "stopped-job", mock_job_func)
    job = await scheduler.storage.get_job("test-project", "stopped-job")
    job.status = JobStatus.RUNNING
    await scheduler.storage.update_job(job)
    
    # The runner returns that the job doesn't exist
    scheduler.runner.job_exists = AsyncMock(return_value=False)
    
    # Call update_job_status
    await scheduler._update_job_status(job)
    
    # Job should now be in STOPPED state
    job = await scheduler.storage.get_job("test-project", "stopped-job")
    assert job.status == JobStatus.STOPPED
    assert job.ended_at is not None
    assert job.metadata and "error" in job.metadata


@pytest.mark.asyncio
async def test_invalid_fsm_transitions_blocked(scheduler):
    """Test that invalid FSM transitions are blocked."""
    # Define a mock job function
    def mock_job_func():
        pass
    
    # 1. Test that COMPLETED jobs can't transition to other states
    await scheduler.start_job("test-project", "completed-job", mock_job_func)
    job = await scheduler.storage.get_job("test-project", "completed-job")
    job.status = JobStatus.COMPLETED
    job.ended_at = datetime.now()
    await scheduler.storage.update_job(job)
    
    # Try to update the job status
    old_status = job.status
    old_ended_at = job.ended_at
    
    # Call update_job_status - should do nothing for COMPLETED jobs
    await scheduler._update_job_status(job)
    
    # Status should remain unchanged
    job = await scheduler.storage.get_job("test-project", "completed-job")
    assert job.status == old_status
    assert job.ended_at == old_ended_at
    
    # 2. Test that FAILED jobs can't transition to other states
    await scheduler.start_job("test-project", "failed-job", mock_job_func)
    job = await scheduler.storage.get_job("test-project", "failed-job")
    job.status = JobStatus.FAILED
    job.ended_at = datetime.now()
    if job.metadata is None:
        job.metadata = {}
    job.metadata["error"] = "Test error"
    await scheduler.storage.update_job(job)
    
    # Try to update the job status
    old_status = job.status
    old_ended_at = job.ended_at
    old_error = job.metadata["error"]
    
    # Call update_job_status - should do nothing for FAILED jobs
    await scheduler._update_job_status(job)
    
    # Status should remain unchanged
    job = await scheduler.storage.get_job("test-project", "failed-job")
    assert job.status == old_status
    assert job.ended_at == old_ended_at
    assert job.metadata["error"] == old_error
    
    # 3. Test that STOPPED jobs can't transition to other states
    await scheduler.start_job("test-project", "stopped-job", mock_job_func)
    job = await scheduler.storage.get_job("test-project", "stopped-job")
    job.status = JobStatus.STOPPED
    job.ended_at = datetime.now()
    if job.metadata is None:
        job.metadata = {}
    job.metadata["error"] = "Job stopped"
    await scheduler.storage.update_job(job)
    
    # Try to update the job status
    old_status = job.status
    old_ended_at = job.ended_at
    old_error = job.metadata["error"]
    
    # Call update_job_status - should do nothing for STOPPED jobs
    await scheduler._update_job_status(job)
    
    # Status should remain unchanged
    job = await scheduler.storage.get_job("test-project", "stopped-job")
    assert job.status == old_status
    assert job.ended_at == old_ended_at
    assert job.metadata["error"] == old_error
    
    # 4. Test that CANCELED jobs can't transition to other states
    await scheduler.start_job("test-project", "canceled-job", mock_job_func)
    job = await scheduler.storage.get_job("test-project", "canceled-job")
    job.status = JobStatus.CANCELED
    job.ended_at = datetime.now()
    await scheduler.storage.update_job(job)
    
    # Try to update the job status
    old_status = job.status
    old_ended_at = job.ended_at
    
    # Call update_job_status - should do nothing for CANCELED jobs
    await scheduler._update_job_status(job)
    
    # Status should remain unchanged
    job = await scheduler.storage.get_job("test-project", "canceled-job")
    assert job.status == old_status
    assert job.ended_at == old_ended_at 