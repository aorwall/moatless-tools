"""Tests for the DockerRunner implementation."""

import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import pytest_asyncio
from moatless.runner.docker_runner import DockerRunner
from moatless.runner.runner import BaseRunner, JobInfo, JobStatus, RunnerStatus, JobsStatusSummary


@pytest_asyncio.fixture  # type: ignore
async def docker_runner() -> AsyncGenerator[DockerRunner, None]:
    """Fixture to create a DockerRunner instance with mocked subprocess calls."""
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Configure the mock subprocess to return successful results by default
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"container-id\n", b"")
        mock_subprocess.return_value = mock_process
        
        # Create runner with test settings
        runner = DockerRunner(
            job_ttl_seconds=60,  # Short TTL for tests
            timeout_seconds=10,   # Short timeout for tests
            moatless_source_dir="/test/moatless"  # Test source dir
        )
        
        yield runner


@pytest.mark.asyncio
async def test_start_job(docker_runner):
    """Test that a job can be started successfully."""
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Mock process for container existence check (returns non-zero to indicate container doesn't exist)
        mock_existence_process = AsyncMock()
        mock_existence_process.returncode = 1
        mock_existence_process.communicate.return_value = (b"", b"")
        
        # Mock process for container creation (returns zero and container ID)
        mock_creation_process = AsyncMock()
        mock_creation_process.returncode = 0
        mock_creation_process.communicate.return_value = (b"container-id\n", b"")
        
        # Configure subprocess to return different mocks for different calls
        mock_subprocess.side_effect = [mock_existence_process, mock_creation_process]
        
        # Start the job
        result = await docker_runner.start_job("test-project", "test-repo__instance", lambda: None)
        
        # Check the result
        assert result is True
        
        # Verify docker run was called
        assert mock_subprocess.call_count == 2
        
        # The second call should be to create the container
        args = mock_subprocess.call_args_list[1][0]
        assert args[0] == "docker"
        assert "run" in args
        assert "--name" in args
        
        # Check container was stored in running_containers
        container_name = docker_runner._container_name("test-project", "test-repo__instance")
        assert container_name in docker_runner.running_containers
        assert docker_runner.running_containers[container_name]["status"] == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_start_job_already_exists(docker_runner):
    """Test that starting a job that already exists returns False."""
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Mock process for container existence check (returns zero to indicate container exists)
        mock_existence_process = AsyncMock()
        mock_existence_process.returncode = 0
        mock_existence_process.communicate.return_value = (b"container-id\n", b"")
        
        # Mock process for container status check
        mock_status_process = AsyncMock()
        mock_status_process.returncode = 0
        mock_status_process.communicate.return_value = (b"running\n", b"")
        
        # Configure subprocess to return different mocks for different calls
        mock_subprocess.side_effect = [mock_existence_process, mock_status_process]
        
        # Start the job
        result = await docker_runner.start_job("test-project", "test-repo__instance", lambda: None)
        
        # Check that the attempt failed because container already exists
        assert result is False


@pytest.mark.asyncio
async def test_get_jobs(docker_runner):
    """Test that jobs can be retrieved."""
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Mock process for docker ps
        mock_process = AsyncMock()
        mock_process.returncode = 0
        # Format: container_name|project_id|trajectory_id|state|started_at
        mock_process.communicate.return_value = (
            b"moatless-test-project-test-repo__instance|test-project|test-repo__instance|running|2023-01-01T00:00:00\n"
            b"moatless-other-project-other-repo__instance|other-project|other-repo__instance|exited|2023-01-01T00:00:00\n",
            b""
        )
        mock_subprocess.return_value = mock_process
        
        # Get all jobs
        jobs = await docker_runner.get_jobs()
        
        # Check jobs
        assert len(jobs) == 2
        
        # Check first job
        assert jobs[0].project_id == "test-project"
        assert jobs[0].trajectory_id == "test-repo__instance"
        assert jobs[0].status == JobStatus.RUNNING
        
        # Check second job
        assert jobs[1].project_id == "other-project"
        assert jobs[1].trajectory_id == "other-repo__instance"
        
        # For the second part of the test, we need to handle exit code checks
        # that might happen during job status determination
        mock_subprocess.reset_mock()
        
        # First create a mock for the initial docker ps call with project filter
        mock_ps_process = AsyncMock()
        mock_ps_process.returncode = 0
        mock_ps_process.communicate.return_value = (
            b"moatless-test-project-test-repo__instance|test-project|test-repo__instance|running|2023-01-01T00:00:00\n",
            b""
        )
        
        # Second create a mock for any exit code check that might happen
        mock_inspect_process = AsyncMock()
        mock_inspect_process.returncode = 0
        mock_inspect_process.communicate.return_value = (b"0\n", b"")
        
        # Set up the sequence of returns
        mock_subprocess.side_effect = [mock_ps_process, mock_inspect_process, mock_inspect_process]
        
        # Get jobs filtered by project
        jobs = await docker_runner.get_jobs("test-project")
        
        # Verify the first call was to docker ps with filter
        first_call = mock_subprocess.call_args_list[0]
        cmd_str = " ".join(str(arg) for arg in first_call[0])
        assert "docker" in cmd_str
        # The command could be either ps or some other format, 
        # but it should have the filter for the project
        assert "moatless.project_id=test-project" in cmd_str


@pytest.mark.asyncio
async def test_cancel_job(docker_runner):
    """Test that a job can be cancelled."""
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Mock process for container existence check
        mock_existence_process = AsyncMock()
        mock_existence_process.returncode = 0
        mock_existence_process.communicate.return_value = (b"container-info\n", b"")
        
        # Mock processes for stop and remove
        mock_stop_process = AsyncMock()
        mock_stop_process.returncode = 0
        mock_stop_process.communicate.return_value = (b"", b"")
        
        mock_rm_process = AsyncMock()
        mock_rm_process.returncode = 0
        mock_rm_process.communicate.return_value = (b"", b"")
        
        # Configure subprocess to return different mocks for different calls
        mock_subprocess.side_effect = [mock_existence_process, mock_stop_process, mock_rm_process]
        
        # Cancel the job
        await docker_runner.cancel_job("test-project", "test-repo__instance")
        
        # Verify docker stop and rm were called
        assert mock_subprocess.call_count == 3
        
        # Check the commands that were run
        stop_call = mock_subprocess.call_args_list[1]
        rm_call = mock_subprocess.call_args_list[2]
        
        assert stop_call[0][0] == "docker"
        assert stop_call[0][1] == "stop"
        
        assert rm_call[0][0] == "docker"
        assert rm_call[0][1] == "rm"


@pytest.mark.asyncio
async def test_cancel_all_jobs_for_project(docker_runner):
    """Test that all jobs for a project can be cancelled."""
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Mock process for listing containers
        mock_list_process = AsyncMock()
        mock_list_process.returncode = 0
        mock_list_process.communicate.return_value = (b"container1\ncontainer2\n", b"")
        
        # Mock processes for stop and remove
        mock_stop_process = AsyncMock()
        mock_stop_process.returncode = 0
        mock_stop_process.communicate.return_value = (b"", b"")
        
        mock_rm_process = AsyncMock()
        mock_rm_process.returncode = 0
        mock_rm_process.communicate.return_value = (b"", b"")
        
        # Configure subprocess to return different mocks for different calls
        mock_subprocess.side_effect = [mock_list_process, mock_stop_process, mock_rm_process]
        
        # Cancel all jobs for the project
        await docker_runner.cancel_job("test-project", None)
        
        # Verify docker commands were called
        assert mock_subprocess.call_count == 3
        
        # Check the filter that was used in the ps command
        ps_call = mock_subprocess.call_args_list[0]
        cmd_str = " ".join(str(arg) for arg in ps_call[0])
        assert "docker" in cmd_str
        assert "ps" in cmd_str
        assert "filter" in cmd_str
        assert "moatless.project_id=test-project" in cmd_str


@pytest.mark.asyncio
async def test_job_exists(docker_runner):
    """Test checking if a job exists."""
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Case 1: Job exists
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"container-name\n", b"")
        mock_subprocess.return_value = mock_process
        
        result = await docker_runner.job_exists("test-project", "test-repo__instance")
        assert result is True
        
        # Case 2: Job doesn't exist
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")
        
        result = await docker_runner.job_exists("nonexistent", "nonexistent")
        assert result is False


@pytest.mark.asyncio
async def test_get_job_status(docker_runner):
    """Test getting the status of a job."""
    container_name = docker_runner._container_name("test-project", "test-repo__instance")
    
    with patch.object(docker_runner, '_get_container_status') as mock_get_status:
        # Test different statuses
        for status in [
            JobStatus.RUNNING,
            JobStatus.PENDING,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELED,
            JobStatus.NOT_STARTED
        ]:
            mock_get_status.return_value = status
            result = await docker_runner.get_job_status("test-project", "test-repo__instance")
            assert result == status
            mock_get_status.assert_called_with(container_name)


@pytest.mark.asyncio
async def test_get_runner_info(docker_runner):
    """Test getting runner information."""
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Case 1: Docker is running
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Docker info\n", b"")
        mock_subprocess.return_value = mock_process
        
        info = await docker_runner.get_runner_info()
        assert info.status == RunnerStatus.RUNNING
        assert info.runner_type == "docker"
        
        # Case 2: Docker is not accessible
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Error\n")
        
        info = await docker_runner.get_runner_info()
        assert info.status == RunnerStatus.ERROR
        assert "Docker is not accessible" in info.data.get("error", "")


@pytest.mark.asyncio
async def test_get_job_status_summary(docker_runner):
    """Test getting job status summary."""
    with patch.object(docker_runner, 'get_jobs') as mock_get_jobs:
        # Create some test jobs with different statuses
        mock_get_jobs.return_value = [
            JobInfo(
                id="moatless-test-project-job1",
                status=JobStatus.RUNNING,
                project_id="test-project",
                trajectory_id="job1",
                enqueued_at=datetime.now(),
                started_at=datetime.now(),
                ended_at=None,
                metadata={}
            ),
            JobInfo(
                id="moatless-test-project-job2",
                status=JobStatus.COMPLETED,
                project_id="test-project",
                trajectory_id="job2",
                enqueued_at=datetime.now() - timedelta(minutes=5),
                started_at=datetime.now() - timedelta(minutes=5),
                ended_at=datetime.now() - timedelta(minutes=1),
                metadata={}
            ),
            JobInfo(
                id="moatless-test-project-job3",
                status=JobStatus.FAILED,
                project_id="test-project",
                trajectory_id="job3",
                enqueued_at=datetime.now() - timedelta(minutes=10),
                started_at=datetime.now() - timedelta(minutes=10),
                ended_at=datetime.now() - timedelta(minutes=8),
                metadata={}
            )
        ]
        
        # Get summary
        summary = await docker_runner.get_job_status_summary("test-project")
        
        # Check counts
        assert summary.project_id == "test-project"
        assert summary.total_jobs == 3
        assert summary.running_jobs == 1
        assert summary.completed_jobs == 1
        assert summary.failed_jobs == 1
        
        # Check job IDs using JobStatus enum values as string
        assert "moatless-test-project-job1" in summary.job_ids[JobStatus.RUNNING.name.lower()]
        assert "moatless-test-project-job2" in summary.job_ids[JobStatus.COMPLETED.name.lower()]
        assert "moatless-test-project-job3" in summary.job_ids[JobStatus.FAILED.name.lower()]


@pytest.mark.asyncio
async def test_get_job_logs(docker_runner):
    """Test getting job logs."""
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Mock process for docker logs
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Test log output\n", b"")
        mock_subprocess.return_value = mock_process
        
        # Get logs
        logs = await docker_runner.get_job_logs("test-project", "test-repo__instance")
        
        # Check logs
        assert logs == "Test log output\n"
        
        # Verify docker logs was called with the right container name
        container_name = docker_runner._container_name("test-project", "test-repo__instance")
        call_args = mock_subprocess.call_args[0]
        assert call_args[0] == "docker"
        assert call_args[1] == "logs"
        assert container_name in call_args
        
        # Test case where logs command fails
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Error\n")
        
        logs = await docker_runner.get_job_logs("test-project", "test-repo__instance")
        assert logs is None

@pytest.mark.asyncio
async def test_get_image_name(docker_runner):
    """Test that Docker image names are generated correctly."""
    # Test with standard trajectory ID format
    image_name = docker_runner._get_image_name("repo__instance")
    assert image_name == "aorwall/sweb.eval.x86_64.repo_moatless_instance"
    
    # Test with different repository name
    image_name = docker_runner._get_image_name("other-repo__test-instance")
    assert image_name == "aorwall/sweb.eval.x86_64.other-repo_moatless_test-instance" 