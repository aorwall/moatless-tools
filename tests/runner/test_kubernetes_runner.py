# type: ignore
"""Tests for the KubernetesRunner implementation."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import AsyncGenerator

import pytest_asyncio
from kubernetes.client import (
    V1Job,
    V1JobStatus,
    V1ObjectMeta,
    V1Pod,
    V1PodStatus,
    V1Container,
    V1ContainerState,
    V1ContainerStateTerminated,
    V1ContainerStatus,
    V1PodCondition,
)

from moatless.runner.kubernetes_runner import KubernetesRunner
from moatless.runner.runner import BaseRunner, JobStatus, RunnerStatus


@pytest_asyncio.fixture  # type: ignore
async def mock_k8s_api():
    """Fixture to mock Kubernetes API clients."""
    with patch("kubernetes.client.BatchV1Api") as mock_batch_api, \
         patch("kubernetes.client.CoreV1Api") as mock_core_api, \
         patch("kubernetes.config.load_incluster_config"), \
         patch("kubernetes.config.load_kube_config"):
        # Set up batch API mock
        batch_api = mock_batch_api.return_value
        
        # Set up core API mock
        core_api = mock_core_api.return_value
        
        # Configure mocks for basic API functionality
        api_resources = MagicMock()
        api_resources.group_version = "batch/v1"
        batch_api.get_api_resources.return_value = api_resources
        
        # Set up node list mock
        node_list = MagicMock()
        node = MagicMock()
        condition = MagicMock()
        condition.type = "Ready"
        condition.status = "True"
        node.status.conditions = [condition]
        node_list.items = [node]
        core_api.list_node.return_value = node_list
        
        yield {
            "batch_api": batch_api,
            "core_api": core_api
        }


@pytest_asyncio.fixture  # type: ignore
async def kubernetes_runner(mock_k8s_api) -> AsyncGenerator[KubernetesRunner, None]:
    """Fixture to create a KubernetesRunner instance with mocked API."""
    runner = KubernetesRunner(namespace="test-namespace", image="test-image")
    yield runner


def create_mock_job(name, status="Running", succeeded=0, failed=0, active=1):
    """Helper to create a mock Kubernetes job."""
    job = MagicMock(spec=V1Job)
    job.metadata = MagicMock(spec=V1ObjectMeta)
    job.metadata.name = name
    job.metadata.namespace = "test-namespace"
    job.metadata.creation_timestamp = datetime.now()
    
    # Extract project_id from the job name
    # Names now use format "run-{project_id}-{trajectory_id}"
    parts = name.split("-")
    if len(parts) >= 3:  # Ensure we have enough parts
        project_id = parts[1]
        trajectory_id = parts[2]
    else:
        # Fallback for tests
        project_id = "unknown"
        trajectory_id = "unknown"
        
    job.metadata.labels = {"app": "moatless-worker", "project_id": project_id, "trajectory_id": trajectory_id}
    job.metadata.deletion_timestamp = None
    
    job.status = MagicMock(spec=V1JobStatus)
    job.status.succeeded = succeeded
    job.status.failed = failed
    job.status.active = active
    job.status.start_time = datetime.now() if active or succeeded or failed else None
    job.status.completion_time = datetime.now() if succeeded else None
    
    # Setup container with environment variables
    container = MagicMock(spec=V1Container)
    container.env = [
        MagicMock(name="PROJECT_ID", value=project_id),
        MagicMock(name="TRAJECTORY_ID", value=trajectory_id),
        MagicMock(name="JOB_FUNC", value="test_module.test_function"),
    ]
    
    # Create spec to hold container
    job.spec = MagicMock()
    job.spec.template = MagicMock()
    job.spec.template.spec = MagicMock()
    job.spec.template.spec.containers = [container]
    
    return job


@pytest.mark.asyncio
async def test_start_job(kubernetes_runner, mock_k8s_api):
    """Test that a job can be started successfully."""
    # Mock the job creation response
    batch_api = mock_k8s_api["batch_api"]
    core_api = mock_k8s_api["core_api"]
    created_job = create_mock_job("run-test-project-test-trajectory")
    batch_api.create_namespaced_job.return_value = created_job
    
    # Create pod list with running container
    pod_list = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Running"
    container_status = MagicMock()
    container_status.ready = True
    pod.status.container_statuses = [container_status]
    pod_list.items = [pod]
    core_api.list_namespaced_pod.return_value = pod_list
    
    # Mock _get_image_name method to avoid IndexError
    with patch.object(kubernetes_runner, "_get_image_name", return_value="test-image"), \
         patch.object(kubernetes_runner, "job_exists", AsyncMock(return_value=False)):
        # Start the job
        result = await kubernetes_runner.start_job("test-project", "test-trajectory", "test_module.test_function")
        
        # Check the result
        assert result is True
        
        # Verify API was called with correct arguments
        batch_api.create_namespaced_job.assert_called_once()
        call_args = batch_api.create_namespaced_job.call_args
        assert call_args[1]["namespace"] == "test-namespace"
        assert "body" in call_args[1]
        
        # Verify job body has correct labels and environment variables
        job_body = call_args[1]["body"]
        assert job_body.metadata.labels["app"] == "moatless-worker"
        assert job_body.metadata.labels["project_id"] == "test-project"
        assert job_body.metadata.labels["trajectory_id"] == "test-trajectory"


@pytest.mark.asyncio
async def test_start_job_already_exists(kubernetes_runner):
    """Test that starting a job that already exists returns False."""
    # Mock job_exists to return True
    with patch.object(kubernetes_runner, "job_exists", AsyncMock(return_value=True)):
        # Try to start the job
        result = await kubernetes_runner.start_job("test-project", "test-trajectory", "test_module.test_function")
        
        # Check that the attempt failed
        assert result is False


@pytest.mark.asyncio
async def test_get_jobs(kubernetes_runner, mock_k8s_api):
    """Test retrieving jobs with various filters."""
    batch_api = mock_k8s_api["batch_api"]
    
    # Create mock job list
    job_list = MagicMock()
    job_list.items = [
        create_mock_job("run-project1-trajectory1"),
        create_mock_job("run-project1-trajectory2"),
        create_mock_job("run-project2-trajectory3"),
    ]
    batch_api.list_namespaced_job.return_value = job_list
    
    # Get all jobs
    all_jobs = await kubernetes_runner.get_jobs()
    batch_api.list_namespaced_job.assert_called_with(
        namespace="test-namespace",
        label_selector="app=moatless-worker",
    )
    assert len(all_jobs) == 3
    
    # Reset mock and set up for project filter
    batch_api.list_namespaced_job.reset_mock()
    filtered_job_list = MagicMock()
    filtered_job_list.items = [
        create_mock_job("run-project1-trajectory1"),
        create_mock_job("run-project1-trajectory2"),
    ]
    batch_api.list_namespaced_job.return_value = filtered_job_list
    
    # Get jobs for project-1
    project1_jobs = await kubernetes_runner.get_jobs("project1")
    batch_api.list_namespaced_job.assert_called_with(
        namespace="test-namespace",
        label_selector="app=moatless-worker,project_id=project1",
    )
    assert len(project1_jobs) == 2


@pytest.mark.asyncio
async def test_cancel_job(kubernetes_runner, mock_k8s_api):
    """Test that a job can be cancelled."""
    batch_api = mock_k8s_api["batch_api"]
    
    # Cancel a specific job
    await kubernetes_runner.cancel_job("test-project", "test-trajectory")
    
    # Verify API was called to delete the job
    batch_api.delete_namespaced_job.assert_called_once()
    call_args = batch_api.delete_namespaced_job.call_args
    assert call_args[1]["name"].startswith("run-test-project-test-trajectory")
    assert call_args[1]["namespace"] == "test-namespace"
    assert "body" in call_args[1]


@pytest.mark.asyncio
async def test_cancel_all_project_jobs(kubernetes_runner, mock_k8s_api):
    """Test that all jobs for a project can be cancelled."""
    batch_api = mock_k8s_api["batch_api"]
    
    # Create mock job list
    job_list = MagicMock()
    job_list.items = [
        create_mock_job("run-test-project-trajectory1"),
        create_mock_job("run-test-project-trajectory2"),
    ]
    batch_api.list_namespaced_job.return_value = job_list
    
    # Cancel all jobs for test-project
    await kubernetes_runner.cancel_job("test-project", None)
    
    # Verify list_namespaced_job was called with correct selector
    batch_api.list_namespaced_job.assert_called_once_with(
        namespace="test-namespace",
        label_selector="app=moatless-worker,project_id=test-project",
    )
    
    # Verify delete was called for each job
    assert batch_api.delete_namespaced_job.call_count == 2


@pytest.mark.asyncio
async def test_job_exists(kubernetes_runner, mock_k8s_api):
    """Test checking if a job exists."""
    batch_api = mock_k8s_api["batch_api"]
    
    # Job exists case
    batch_api.read_namespaced_job.return_value = create_mock_job("run-test-project-test-trajectory")
    assert await kubernetes_runner.job_exists("test-project", "test-trajectory") is True
    
    # Job doesn't exist case
    from kubernetes.client.rest import ApiException
    error_response = ApiException()
    error_response.status = 404
    batch_api.read_namespaced_job.side_effect = error_response
    
    assert await kubernetes_runner.job_exists("test-project", "nonexistent") is False


@pytest.mark.asyncio
async def test_get_job_status(kubernetes_runner, mock_k8s_api):
    """Test getting the status of a job."""
    batch_api = mock_k8s_api["batch_api"]
    core_api = mock_k8s_api["core_api"]
    
    # Create running pod
    pod_list = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Running"
    container_status = MagicMock()
    container_status.ready = True
    pod.status.container_statuses = [container_status]
    pod_list.items = [pod]
    core_api.list_namespaced_pod.return_value = pod_list
    
    # Running job
    running_job = create_mock_job("run-test-project-test-trajectory", active=1, succeeded=0, failed=0)
    batch_api.read_namespaced_job.return_value = running_job
    
    status = await kubernetes_runner.get_job_status("test-project", "test-trajectory")
    assert status == JobStatus.RUNNING
    
    # Completed job
    completed_job = create_mock_job("run-test-project-test-trajectory", active=0, succeeded=1, failed=0)
    batch_api.read_namespaced_job.return_value = completed_job
    
    status = await kubernetes_runner.get_job_status("test-project", "test-trajectory")
    assert status == JobStatus.COMPLETED
    
    # Failed job
    failed_job = create_mock_job("run-test-project-test-trajectory", active=0, succeeded=0, failed=1)
    batch_api.read_namespaced_job.return_value = failed_job
    
    status = await kubernetes_runner.get_job_status("test-project", "test-trajectory")
    assert status == JobStatus.FAILED
    
    # Job doesn't exist
    from kubernetes.client.rest import ApiException
    error_response = ApiException()
    error_response.status = 404
    batch_api.read_namespaced_job.side_effect = error_response
    
    status = await kubernetes_runner.get_job_status("test-project", "nonexistent")
    assert status == JobStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_retry_job(kubernetes_runner, mock_k8s_api):
    """Test that a failed job can be retried."""
    batch_api = mock_k8s_api["batch_api"]
    
    # Create a failed job with JOB_FUNC in env vars
    failed_job = create_mock_job("run-test-project-test-trajectory", active=0, succeeded=0, failed=1)
    
    # Make sure the container has the JOB_FUNC env var properly set
    for container in failed_job.spec.template.spec.containers:
        # Replace the env list with a properly mocked one that has correct name attributes
        env_vars = []
        for var in [
            {"name": "PROJECT_ID", "value": "test-project"},
            {"name": "TRAJECTORY_ID", "value": "test-trajectory"},
            {"name": "JOB_FUNC", "value": "test_module.test_function"},
        ]:
            mock_env = MagicMock()
            mock_env.name = var["name"]
            mock_env.value = var["value"]
            env_vars.append(mock_env)
        container.env = env_vars
    
    batch_api.read_namespaced_job.return_value = failed_job
    
    # Add annotations to the job metadata
    failed_job.metadata.annotations = {
        "moatless.ai/project-id": "test-project",
        "moatless.ai/trajectory-id": "test-trajectory",
        "moatless.ai/function": "test_module.test_function"
    }
    
    # Set up create job response
    new_job = create_mock_job("run-test-project-test-trajectory", active=1, succeeded=0, failed=0)
    batch_api.create_namespaced_job.return_value = new_job
    
    # Mock _get_image_name method to avoid IndexError
    with patch.object(kubernetes_runner, "_get_image_name", return_value="test-image"):
        # Retry the job
        result = await kubernetes_runner.retry_job("test-project", "test-trajectory")
        
        # Verify the job was deleted and recreated
        assert result is True
        batch_api.delete_namespaced_job.assert_called_once()
        batch_api.create_namespaced_job.assert_called_once()


@pytest.mark.asyncio
async def test_get_runner_info(kubernetes_runner, mock_k8s_api):
    """Test getting runner info."""
    runner_info = await kubernetes_runner.get_runner_info()
    
    assert runner_info.runner_type == "kubernetes"
    assert runner_info.status == RunnerStatus.RUNNING
    assert runner_info.data["nodes"] == 1
    assert runner_info.data["ready_nodes"] == 1
    assert runner_info.data["api_version"] == "batch/v1"


@pytest.mark.asyncio
async def test_get_job_status_summary(kubernetes_runner, mock_k8s_api):
    """Test getting job status summary."""
    batch_api = mock_k8s_api["batch_api"]
    core_api = mock_k8s_api["core_api"]
    
    # Create running pod (for job status detection)
    pod_list = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Running"
    container_status = MagicMock()
    container_status.ready = True
    pod.status.container_statuses = [container_status]
    pod_list.items = [pod]
    core_api.list_namespaced_pod.return_value = pod_list
    
    # Create jobs in various states
    job_list = MagicMock()
    job_list.items = [
        create_mock_job("run-test-project-running", active=1, succeeded=0, failed=0),
        create_mock_job("run-test-project-completed", active=0, succeeded=1, failed=0),
        create_mock_job("run-test-project-failed", active=0, succeeded=0, failed=1),
    ]
    
    # Add annotations to each job
    for job in job_list.items:
        job.metadata.annotations = {
            "moatless.ai/project-id": "test-project",
            "moatless.ai/trajectory-id": job.metadata.name.split("-")[-1]
        }
    
    batch_api.list_namespaced_job.return_value = job_list
    
    # Get summary
    summary = await kubernetes_runner.get_job_status_summary("test-project")
    
    # Verify summary contains all jobs
    assert summary.project_id == "test-project"
    assert summary.total_jobs == 3
    assert summary.running_jobs == 1
    assert summary.completed_jobs == 1
    assert summary.failed_jobs == 1
    assert len(summary.job_ids["running"]) == 1
    assert len(summary.job_ids["completed"]) == 1
    assert len(summary.job_ids["failed"]) == 1


@pytest.mark.asyncio
async def test_get_job_error(kubernetes_runner, mock_k8s_api):
    """Test getting error details from a failed pod."""
    core_api = mock_k8s_api["core_api"]
    
    # Create a failed job
    failed_job = create_mock_job("run-test-project-test-trajectory", active=0, succeeded=0, failed=1)
    
    # Create failed pod
    pod_list = MagicMock()
    pod = MagicMock(spec=V1Pod)
    pod.status = MagicMock(spec=V1PodStatus)
    pod.status.phase = "Failed"
    pod.status.message = "Pod failed: Out of memory"
    
    # Create container status with OOM error
    container_status = MagicMock(spec=V1ContainerStatus)
    container_state = MagicMock(spec=V1ContainerState)
    container_terminated = MagicMock(spec=V1ContainerStateTerminated)
    container_terminated.reason = "OOMKilled"
    container_terminated.message = "Killed for using too much memory"
    container_state.terminated = container_terminated
    container_status.state = container_state
    pod.status.container_statuses = [container_status]
    
    pod_list.items = [pod]
    core_api.list_namespaced_pod.return_value = pod_list
    
    # Get error details
    error_message = kubernetes_runner._get_job_error(failed_job)
    
    # Verify error message
    assert "OOMKilled" in error_message
    assert "Killed for using too much memory" in error_message 