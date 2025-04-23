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
    JobFunction,
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
        self.start_job_called = []
        self.cancel_job_called = []
        self.job_status_override = {}
        self.job_exists_override = {}
    
    async def start_job(self, project_id, trajectory_id, job_func, node_id=None):
        """Start a job and mark it as running."""
        self.start_job_called.append((project_id, trajectory_id))
        # Add the job to the dictionary with a RUNNING status
        self.jobs[f"{project_id}-{trajectory_id}"] = {
            "status": JobStatus.RUNNING, 
            "project_id": project_id, 
            "trajectory_id": trajectory_id
        }
        return True

    async def cancel_job(self, project_id, trajectory_id=None):
        """Cancel a job."""
        self.cancel_job_called.append((project_id, trajectory_id))
        if trajectory_id:
            # Mark the job as CANCELED
            job_key = f"{project_id}-{trajectory_id}"
            if job_key in self.jobs:
                self.jobs[job_key]["status"] = JobStatus.CANCELED
        else:
            # Cancel all jobs for the project
            for job_key, job in list(self.jobs.items()):
                if job["project_id"] == project_id:
                    self.jobs[job_key]["status"] = JobStatus.CANCELED

    async def job_exists(self, project_id, trajectory_id):
        """Check if a job exists."""
        # Override for testing specific cases
        if f"{project_id}-{trajectory_id}" in self.job_exists_override:
            return self.job_exists_override[f"{project_id}-{trajectory_id}"]
        return f"{project_id}-{trajectory_id}" in self.jobs

    async def get_job_status(self, project_id, trajectory_id):
        """Get the status of a job."""
        # Override for testing specific cases
        if f"{project_id}-{trajectory_id}" in self.job_status_override:
            return self.job_status_override[f"{project_id}-{trajectory_id}"]
        
        job_key = f"{project_id}-{trajectory_id}"
        if job_key in self.jobs:
            return self.jobs[job_key]["status"]
        return None  # Return None for jobs that don't exist
        
    async def get_runner_info(self):
        """Mock getting runner info."""
        return RunnerInfo(
            runner_type="mock",
            status=RunnerStatus.RUNNING,
            data={
                "mock": True,
                "scheduler": {
                    "max_jobs_per_project": 2,
                    "max_total_jobs": 5,
                    "queued_jobs": 0,
                    "running_jobs": 0
                }
            }
        )

    async def get_jobs(self, project_id: str | None = None) -> list[JobInfo]:
       return list(self.jobs.values())

    async def get_job_details(self, project_id, trajectory_id):
        """Mock getting job details."""
        job_key = f"{project_id}:{trajectory_id}"
        if job_key not in self.jobs:
            return None
            
        job_info = self.jobs[job_key]
        return JobDetails(
            id=job_key,
            status=job_info.status,
            project_id=project_id,
            trajectory_id=trajectory_id,
            started_at=job_info.started_at,
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
            for key, job_info in self.jobs.items():
                if job_info.project_id == project_id:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self.jobs[key]
                if key in self.canceled_jobs:
                    self.canceled_jobs.remove(key)
                    
        return True

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

def mock_job_func():
    pass

@pytest.mark.asyncio
async def test_start_job(scheduler):
    """Test starting a new job."""
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
    
    # Check scheduler info - MockRunner needs to include this in get_runner_info
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
    
    # Create a mock JobDetails object that the runner will return
    mock_details = JobDetails(
        id="test-project-test-trajectory",
        status=JobStatus.RUNNING,
        project_id="test-project",
        trajectory_id="test-trajectory",
        enqueued_at=datetime.now(),
        sections=[
            JobDetailSection(
                name="overview",
                display_name="Overview",
                data={"status": "running"}
            )
        ]
    )
    
    # Mock the runner's get_job_details method to return our mock details
    with patch.object(scheduler.runner, "get_job_details", AsyncMock(return_value=mock_details)):
        # Get job details
        details = await scheduler.get_job_details("test-project", "test-trajectory")
        
        # Should get details
        assert details is not None
        assert details.id == "test-project-test-trajectory"
        assert details.status == JobStatus.RUNNING
        assert len(details.sections) == 1
        assert details.sections[0].name == "overview"


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
    scheduler.runner.job_status_override[f"test-project-test-trajectory"] = JobStatus.FAILED

    # Now call the _update_job_status method to sync status
    await scheduler._update_job_status(job)

    # Get the updated job
    updated_job = await scheduler.storage.get_job("test-project", "test-trajectory")

    # The job should be marked as FAILED
    assert updated_job.status == JobStatus.FAILED

    # It should have an error message
    assert updated_job.metadata and "error" in updated_job.metadata
    assert updated_job.ended_at is not None


@pytest.mark.asyncio
async def test_sync_with_runner(scheduler):
    """Test that the scheduler syncs job status with the underlying runner."""
    mock_runner = scheduler.runner

    def mock_job_func():
        pass

    # Add some jobs with different statuses
    await scheduler.storage.add_job(JobInfo(
        id="test-project-completed-job",
        status=JobStatus.RUNNING,  # Initially RUNNING but will be marked as COMPLETED
        project_id="test-project",
        trajectory_id="completed-job",
        enqueued_at=datetime.now(),
        started_at=datetime.now()
    ))

    await scheduler.storage.add_job(JobInfo(
        id="test-project-failed-job",
        status=JobStatus.RUNNING,  # Initially RUNNING but will be marked as FAILED
        project_id="test-project",
        trajectory_id="failed-job",
        enqueued_at=datetime.now(),
        started_at=datetime.now()
    ))

    await scheduler.storage.add_job(JobInfo(
        id="test-project-disappeared-job",
        status=JobStatus.RUNNING,  # Initially RUNNING but will disappear
        project_id="test-project",
        trajectory_id="disappeared-job",
        enqueued_at=datetime.now(),
        started_at=datetime.now()
    ))

    # Override get_job_status to return different statuses for different jobs
    async def mock_get_job_status(project_id, trajectory_id):
        if trajectory_id == "completed-job":
            return JobStatus.COMPLETED
        elif trajectory_id == "failed-job":
            return JobStatus.FAILED
        elif trajectory_id == "disappeared-job":
            return None  # Job no longer exists
        return None

    with patch.object(mock_runner, "get_job_status", side_effect=mock_get_job_status):
        # Run the sync
        await scheduler._sync_jobs_with_runner()

        # Check if job statuses were updated
        job1 = await scheduler.storage.get_job("test-project", "completed-job")
        assert job1.status == JobStatus.COMPLETED

        job2 = await scheduler.storage.get_job("test-project", "failed-job")
        assert job2.status == JobStatus.FAILED

        job3 = await scheduler.storage.get_job("test-project", "disappeared-job")
        assert job3.status == JobStatus.STOPPED  # Should be marked as STOPPED if it disappeared


@pytest.mark.asyncio
async def test_sync_only_processes_active_jobs(scheduler):
    """Test that the scheduler only processes active jobs during sync."""
    mock_runner = scheduler.runner

    def mock_job_func():
        pass

    # Add some jobs with different statuses
    await scheduler.storage.add_job(JobInfo(
        id="test-project-running-job",
        status=JobStatus.RUNNING,
        project_id="test-project",
        trajectory_id="running-job",
        enqueued_at=datetime.now(),
        started_at=datetime.now()
    ))

    await scheduler.storage.add_job(JobInfo(
        id="test-project-completed-job",
        status=JobStatus.COMPLETED,  # Terminal state, should not be processed
        project_id="test-project",
        trajectory_id="completed-job",
        enqueued_at=datetime.now(),
        started_at=datetime.now(),
        ended_at=datetime.now()
    ))

    # Mock function to track which jobs are processed
    processed_jobs = []

    # Override get_job_status to return status based on job name
    async def modified_get_job_status(project_id, trajectory_id):
        # For testing purposes, we want to return status directly
        processed_jobs.append((project_id, trajectory_id))
        if trajectory_id == "running-job":
            return JobStatus.RUNNING
        return None

    # Create a mock for _update_job_status to track calls
    update_job_status_calls = []
    async def mock_update_job_status(job):
        update_job_status_calls.append(job.trajectory_id)

    with patch.object(mock_runner, "get_job_status", side_effect=modified_get_job_status), \
         patch.object(scheduler, "_update_job_status", side_effect=mock_update_job_status):
        # Run the sync
        await scheduler._sync_jobs_with_runner()

        # Check that only active jobs were processed
        assert len(processed_jobs) == 1
        assert processed_jobs[0] == ("test-project", "running-job")

        # Completed jobs should not be updated
        assert "completed-job" not in update_job_status_calls


@pytest.mark.asyncio
async def test_fsm_job_transitions(scheduler):
    """Test job state transitions in the scheduler."""
    # Define a mock job function
    def mock_job_func():
        pass

    # Create a job with an initial PENDING state
    job_info = JobInfo(
        id="test-project-test-job",
        status=JobStatus.PENDING,
        project_id="test-project",
        trajectory_id="test-job",
        enqueued_at=datetime.now(),
        job_func=JobFunction(
            module=mock_job_func.__module__,
            name=mock_job_func.__name__
        )
    )
    await scheduler.storage.add_job(job_info)

    # Mock the runner's get_job_status method to return different statuses for testing
    async def mock_get_job_status(project_id, trajectory_id):
        # For tracking job status transitions
        current_job = await scheduler.storage.get_job(project_id, trajectory_id)
        
        # Return based on current job status
        if current_job.status == JobStatus.PENDING:
            return None  # Job doesn't exist yet in the runner
        elif current_job.status == JobStatus.RUNNING:
            # We can test the RUNNING -> COMPLETED transition
            return JobStatus.COMPLETED
        elif current_job.status == JobStatus.COMPLETED:
            # Job is already complete
            return JobStatus.COMPLETED
        
        # Default - shouldn't reach here
        return None

    with patch.object(scheduler.runner, 'get_job_status', side_effect=mock_get_job_status), \
         patch.object(scheduler.runner, 'start_job', return_value=True):
        
        # 1. Try to start the job - it should transition from PENDING to RUNNING
        await scheduler._try_start_job(job_info)
        job = await scheduler.storage.get_job("test-project", "test-job")
        assert job.status == JobStatus.RUNNING
        
        # 2. Sync with runner - should transition from RUNNING to COMPLETED
        await scheduler._sync_jobs_with_runner()
        job = await scheduler.storage.get_job("test-project", "test-job")
        assert job.status == JobStatus.COMPLETED
        
        # 3. Try to start the job again - should not start because it's in a terminal state
        await scheduler._try_start_job(job)
        job = await scheduler.storage.get_job("test-project", "test-job")
        assert job.status == JobStatus.COMPLETED  # Still COMPLETED


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