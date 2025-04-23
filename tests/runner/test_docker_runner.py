"""Tests for the DockerRunner implementation."""

import asyncio
import subprocess
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
            timeout_seconds=10,  # Short timeout for tests
            moatless_source_dir="/test/moatless",  # Test source dir
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
    with patch("asyncio.create_subprocess_exec") as mock_subprocess, \
         patch.object(docker_runner, "_get_container_status") as mock_get_status:
        # Mock process for docker ps
        mock_process = AsyncMock()
        mock_process.returncode = 0
        # Format: container_name|project_id|trajectory_id|state|started_at
        mock_process.communicate.return_value = (
            b"moatless-test-project-test-repo__instance|test-project|test-repo__instance|running|2023-01-01T00:00:00\n"
            b"moatless-other-project-other-repo__instance|other-project|other-repo__instance|exited|2023-01-01T00:00:00\n",
            b"",
        )
        mock_subprocess.return_value = mock_process

        # Mock container status queries
        mock_get_status.side_effect = lambda container_name: JobStatus.RUNNING if "test-project" in container_name else JobStatus.COMPLETED

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
        assert jobs[1].status == JobStatus.COMPLETED

        # For the second part of the test, we need to handle exit code checks
        # that might happen during job status determination
        mock_subprocess.reset_mock()
        mock_get_status.reset_mock()

        # First create a mock for the initial docker ps call with project filter
        mock_ps_process = AsyncMock()
        mock_ps_process.returncode = 0
        mock_ps_process.communicate.return_value = (
            b"moatless-test-project-test-repo__instance|test-project|test-repo__instance|running|2023-01-01T00:00:00\n",
            b"",
        )

        # Set up the side effect for _get_container_status to return RUNNING for test-project
        mock_get_status.side_effect = lambda container_name: JobStatus.RUNNING

        # Set up the sequence of returns
        mock_subprocess.return_value = mock_ps_process

        # Get jobs filtered by project
        jobs = await docker_runner.get_jobs("test-project")

        # Verify the first call was to docker ps with filter
        first_call = mock_subprocess.call_args_list[0]
        cmd_str = " ".join(str(arg) for arg in first_call[0])
        assert "docker" in cmd_str
        # The command should have the filter for the project with the new label format
        assert "label=project_id=test-project" in cmd_str


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
        assert "label=project_id=test-project" in cmd_str


@pytest.mark.asyncio
async def test_job_exists(docker_runner):
    """Test checking if a job exists."""
    with patch.object(docker_runner, "get_job_status") as mock_get_status:
        # Case 1: Job exists with RUNNING status
        mock_get_status.return_value = JobStatus.RUNNING
        result = await docker_runner.job_exists("test-project", "test-repo__instance")
        assert result is True

        # Case 2: Job exists with COMPLETED status
        mock_get_status.return_value = JobStatus.COMPLETED
        result = await docker_runner.job_exists("test-project", "test-repo__instance")
        assert result is True

        # Case 3: Job doesn't exist (status is None)
        mock_get_status.return_value = None
        result = await docker_runner.job_exists("nonexistent", "nonexistent")
        assert result is False


@pytest.mark.asyncio
async def test_get_job_status(docker_runner):
    """Test getting the status of a job."""
    container_name = docker_runner._container_name("test-project", "test-repo__instance")

    with patch.object(docker_runner, "_get_container_status") as mock_get_status:
        # Test different statuses
        for status in [
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELED,
            None,  # Test None status for non-existent jobs
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
    with patch.object(docker_runner, "get_jobs") as mock_get_jobs:
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
                metadata={},
            ),
            JobInfo(
                id="moatless-test-project-job2",
                status=JobStatus.COMPLETED,
                project_id="test-project",
                trajectory_id="job2",
                enqueued_at=datetime.now() - timedelta(minutes=5),
                started_at=datetime.now() - timedelta(minutes=5),
                ended_at=datetime.now() - timedelta(minutes=1),
                metadata={},
            ),
            JobInfo(
                id="moatless-test-project-job3",
                status=JobStatus.FAILED,
                project_id="test-project",
                trajectory_id="job3",
                enqueued_at=datetime.now() - timedelta(minutes=10),
                started_at=datetime.now() - timedelta(minutes=10),
                ended_at=datetime.now() - timedelta(minutes=8),
                metadata={},
            ),
        ]

        # Get summary
        summary = await docker_runner.get_job_status_summary()

        # Check counts
        assert summary.total_jobs == 3
        assert summary.running_jobs == 1
        assert summary.completed_jobs == 1
        assert summary.failed_jobs == 1


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


@pytest.mark.asyncio
async def test_labels_include_managed_flag(docker_runner):
    """Test that containers are created with the moatless.managed=true label."""
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
        await docker_runner.start_job("test-project", "test-repo__instance", lambda: None)

        # Verify docker run was called
        assert mock_subprocess.call_count == 2

        # The second call should be to create the container
        args = mock_subprocess.call_args_list[1][0]
        
        # Check that the moatless.managed=true label is included
        # Find where the --label arguments are
        label_args = []
        for i, arg in enumerate(args):
            if arg == "--label" and i + 1 < len(args):
                label_args.append(args[i + 1])
        
        # Verify the managed label is present
        assert any("moatless.managed=true" in label for label in label_args), "moatless.managed=true label not found"
        
        # Also check for project_id and trajectory_id labels
        assert any("project_id=" in label for label in label_args), "project_id label not found"
        assert any("trajectory_id=" in label for label in label_args), "trajectory_id label not found"


@pytest.mark.asyncio
async def test_image_name_override():
    """Test that default_image_name and per-job image_name parameters work correctly."""
    default_image = "default-image:latest"
    custom_image = "custom-image:latest"
    
    # Test with default_image_name set in constructor
    runner_with_default = DockerRunner(default_image_name=default_image)
    assert runner_with_default._get_image_name("any-trajectory-id") == default_image
    
    # Test with default runner (no default image)
    runner_without_default = DockerRunner()
    
    # Should raise an error with invalid trajectory format
    with pytest.raises(IndexError):
        runner_without_default._get_image_name("invalid-format")
    
    # Should construct a proper image name when trajectory_id has the expected format
    valid_trajectory_id = "repo__instance"
    expected_image = "aorwall/sweb.eval.x86_64.repo_moatless_instance"
    assert runner_without_default._get_image_name(valid_trajectory_id) == expected_image
    
    # Test that per-job image name works when starting a job
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Mock for container existence check and creation
        mock_existence_process = AsyncMock()
        mock_existence_process.returncode = 1
        mock_existence_process.communicate.return_value = (b"", b"")
        
        mock_creation_process = AsyncMock()
        mock_creation_process.returncode = 0
        mock_creation_process.communicate.return_value = (b"container-id\n", b"")
        
        mock_subprocess.side_effect = [mock_existence_process, mock_creation_process]
        
        # Start a job with a custom image
        runner = DockerRunner(default_image_name=default_image)
        await runner.start_job("test-project", "job2", lambda: None, image_name=custom_image)
        
        # Check that the custom image was used in the docker run command
        docker_run_call = mock_subprocess.call_args_list[1][0]
        cmd_str = " ".join(str(arg) for arg in docker_run_call)
        assert custom_image in cmd_str, f"Custom image {custom_image} not found in Docker command"


@pytest.mark.asyncio
async def test_get_container_status():
    """Test the _get_container_status method with proper docker inspect command."""
    # Create a DockerRunner instance
    runner = DockerRunner()
    
    # Create a mock for the create_subprocess_exec function
    process_mock = AsyncMock()
    process_mock.returncode = 0
    process_mock.communicate.return_value = (b"running,true,0,test_project,test_trajectory", b"")
    
    # Patch the asyncio.create_subprocess_exec function
    with patch("asyncio.create_subprocess_exec", return_value=process_mock) as mock_exec:
        # Call the method
        container_name = "moatless_test_project_test_trajectory"
        status = await runner._get_container_status(container_name)
        
        # Verify the correct command was executed
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        
        # Ensure there's only one --format flag in the command
        format_flags = [arg for arg in args if arg == "--format"]
        assert len(format_flags) == 1
        
        # Verify the correct status is returned
        assert status == JobStatus.RUNNING
        
        # Ensure the process.communicate method was called
        process_mock.communicate.assert_called_once()
