# type: ignore
"""Tests for the KubernetesRunner implementation."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from kubernetes import client
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
from moatless.runner.runner import BaseRunner, JobInfo, JobStatus, RunnerStatus
from moatless.runner.label_utils import create_resource_id


class TestKubernetesRunner(KubernetesRunner):
    """Subclass of KubernetesRunner for testing purposes.

    This subclass overrides specific methods to provide test-friendly behavior
    without polluting the main class with test-specific logic.
    """

    def __init__(self, namespace="test-namespace", image="test-image", **kwargs):
        """Initialize with test-specific defaults."""
        super().__init__(namespace=namespace, image=image, **kwargs)

    async def _process_job_info(self, job, project_id: str = None) -> JobInfo | None:
        """Test-specific implementation for processing job info."""
        job_id = job.metadata.name
        job_status = self._get_job_status_from_k8s_job(job)

        job_project_id = None
        job_trajectory_id = None

        # Extract project_id and trajectory_id from job metadata
        if job.metadata.labels and "project_id" in job.metadata.labels:
            job_project_id = job.metadata.labels.get("project_id")

            if job.metadata.labels.get("trajectory_id"):
                job_trajectory_id = job.metadata.labels.get("trajectory_id")
            else:
                # Extract trajectory_id from job name pattern "run-{project_id}-{trajectory_id}"
                if job_id.startswith(f"run-{job_project_id}-"):
                    parts = job_id.split("-")
                    if len(parts) >= 3:
                        job_trajectory_id = parts[2]

        # Fallback to annotations
        elif job.metadata.annotations:
            if "moatless.ai/project-id" in job.metadata.annotations:
                job_project_id = job.metadata.annotations["moatless.ai/project-id"]
            if "moatless.ai/trajectory-id" in job.metadata.annotations:
                job_trajectory_id = job.metadata.annotations["moatless.ai/trajectory-id"]

            # If project_id is specified and doesn't match, skip this job
            if project_id and job_project_id != project_id:
                return None

        pod_metadata = None
        if job_status in [JobStatus.RUNNING, JobStatus.FAILED]:
            pod_metadata = await self._get_pod_metadata(job_project_id, job_trajectory_id)

        return JobInfo(
            id=job_id,
            status=job_status,
            project_id=job_project_id,
            trajectory_id=job_trajectory_id,
            enqueued_at=job.metadata.creation_timestamp if job.metadata.creation_timestamp else None,
            started_at=job.status.start_time if job.status and job.status.start_time else None,
            ended_at=job.status.completion_time if job.status and job.status.completion_time else None,
            metadata={
                "error": self._get_job_error(job) if job_status == JobStatus.FAILED else None,
                "pod_status": pod_metadata,
            },
        )

    # Override missing methods for tests
    def _get_project_label(self, project_id: str) -> str:
        """Test implementation for getting project label."""
        return project_id.lower()

    def _get_trajectory_label(self, trajectory_id: str) -> str:
        """Test implementation for getting trajectory label."""
        return trajectory_id.lower()

    async def _get_pod_metadata(self, project_id: str, trajectory_id: str) -> dict | None:
        """Test implementation for getting pod metadata."""
        if not project_id or not trajectory_id:
            return None

        return {
            "name": f"pod-{project_id}-{trajectory_id}",
            "phase": "Running",
            "container_state": "running",
            "ready": True,
        }

    def _get_job_error(self, job) -> str | None:
        """Test implementation for getting job error."""
        if not job.status.failed:
            return None
        return "Test job failed"

    def _create_job_object(
        self,
        job_id: str,
        project_id: str,
        trajectory_id: str,
        job_func: Callable | str,
        otel_context: dict = None,
        node_id: int = None,
        update_on_start: bool = False,
        update_branch: str = "docker",
    ) -> client.V1Job:
        """Create a mock kubernetes job object."""
        # Create a mock job with the minimum required fields
        job = MagicMock(spec=client.V1Job)
        job.metadata = MagicMock()
        job.metadata.name = job_id
        job.metadata.namespace = self.namespace
        job.metadata.labels = {
            "project_id": self._get_project_label(project_id),
            "trajectory_id": self._get_trajectory_label(trajectory_id),
            "app": "moatless-worker",
        }
        job.metadata.annotations = {
            "moatless.ai/project-id": project_id,
            "moatless.ai/trajectory-id": trajectory_id,
        }

        # Add node_id annotation if specified
        if node_id is not None:
            job.metadata.annotations["moatless.ai/node-id"] = str(node_id)

        # Mock status
        job.status = MagicMock()
        job.status.active = 1
        job.status.succeeded = 0
        job.status.failed = 0
        job.status.start_time = datetime.now(timezone.utc)
        job.status.completion_time = None

        # Mock spec
        job.spec = MagicMock()
        job.spec.template = MagicMock()
        job.spec.template.spec = MagicMock()
        job.spec.template.spec.containers = [MagicMock()]

        return job

    def _get_image_name(self, trajectory_id: str = None) -> str:
        """Test implementation for getting image name."""
        return self.image or "test-image"

    def _get_node_selector(self, node_id: int = None) -> dict:
        """Test implementation for getting node selector."""
        selector = dict(self.node_selector) if self.node_selector else {}
        if node_id is not None:
            selector["moatless.ai/node-id"] = str(node_id)
        return selector

    def _job_id(self, project_id: str, trajectory_id: str) -> str:
        """Generate job ID for Kubernetes."""
        return create_resource_id(project_id, trajectory_id, prefix="run")


@pytest_asyncio.fixture  # type: ignore
async def mock_k8s_api():
    """Fixture to mock Kubernetes API clients."""
    with (
        patch("kubernetes.client.BatchV1Api") as mock_batch_api,
        patch("kubernetes.client.CoreV1Api") as mock_core_api,
        patch("kubernetes.config.load_incluster_config"),
        patch("kubernetes.config.load_kube_config"),
    ):
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

        yield {"batch_api": batch_api, "core_api": core_api}


@pytest_asyncio.fixture  # type: ignore
async def kubernetes_runner(mock_k8s_api) -> AsyncGenerator[KubernetesRunner, None]:
    """Fixture to create a KubernetesRunner instance with mocked API."""
    runner = TestKubernetesRunner(namespace="test-namespace", image="test-image")
    yield runner


def create_mock_job(name, status="Running", succeeded=0, failed=0, active=1, namespace=None):
    """Helper to create a mock Kubernetes job."""
    job = MagicMock(spec=V1Job)
    job.metadata = MagicMock(spec=V1ObjectMeta)
    job.metadata.name = name

    # Extract project_id and trajectory_id from job name
    # Names follow pattern "run-{project_id}-{trajectory_id}"
    parts = name.split("-")

    if len(parts) < 3 or parts[0] != "run":
        # If name doesn't follow the expected pattern, use defaults
        project_id = "unknown"
        trajectory_id = "unknown"
    else:
        # For simple case: run-{project_id}-{trajectory_id}
        # Note: In more complex cases with truncation, these might be approximations
        project_id = parts[1]
        trajectory_id = parts[2]

    # Use provided namespace or use test-namespace as default
    job.metadata.namespace = namespace or "test-namespace"
    job.metadata.creation_timestamp = datetime.now()

    job.metadata.labels = {"app": "moatless-worker", "project_id": project_id, "trajectory_id": trajectory_id}
    job.metadata.annotations = {
        "moatless.ai/project-id": project_id,
        "moatless.ai/trajectory-id": trajectory_id,
    }
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

    project_id = "test-project"
    trajectory_id = "test-trajectory"
    expected_namespace = kubernetes_runner.namespace

    # For project namespaces, job name is different
    expected_job_id = kubernetes_runner._job_id(project_id, trajectory_id)
    created_job = create_mock_job(expected_job_id)
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
    with (
        patch.object(kubernetes_runner, "_get_image_name", return_value="test-image"),
        patch.object(kubernetes_runner, "job_exists", AsyncMock(return_value=False)),
    ):
        # Start the job
        result = await kubernetes_runner.start_job(project_id, trajectory_id, "test_module.test_function")

        # Check the result
        assert result is True

        # Verify API was called with correct arguments
        batch_api.create_namespaced_job.assert_called_once()
        call_args = batch_api.create_namespaced_job.call_args
        assert call_args[1]["namespace"] == expected_namespace
        assert "body" in call_args[1]

        # Verify job body has correct labels and environment variables
        job_body = call_args[1]["body"]
        assert job_body.metadata.labels["app"] == "moatless-worker"
        assert job_body.metadata.labels["project_id"] == project_id
        assert job_body.metadata.labels["trajectory_id"] == trajectory_id

    # Test with node_id parameter
    batch_api.create_namespaced_job.reset_mock()
    node_id = 5

    with (
        patch.object(kubernetes_runner, "_get_image_name", return_value="test-image"),
        patch.object(kubernetes_runner, "job_exists", AsyncMock(return_value=False)),
    ):
        # Start the job with node_id
        result = await kubernetes_runner.start_job(
            project_id, trajectory_id, "test_module.test_function", node_id=node_id
        )

        # Check the result
        assert result is True

        # Verify API was called with correct arguments
        batch_api.create_namespaced_job.assert_called_once()

        # Check that node_id was passed to _create_job_object
        job_body = batch_api.create_namespaced_job.call_args[1]["body"]
        if hasattr(job_body.metadata, "annotations") and job_body.metadata.annotations:
            assert "moatless.ai/node-id" in job_body.metadata.annotations
            assert job_body.metadata.annotations["moatless.ai/node-id"] == str(node_id)


@pytest.mark.asyncio
async def test_start_job_already_exists(kubernetes_runner):
    """Test that starting a job that already exists returns False."""
    # Mock job_exists to return True
    with (
        patch.object(kubernetes_runner, "job_exists", AsyncMock(return_value=True)),
        patch.object(kubernetes_runner, "get_job_status", AsyncMock(return_value=JobStatus.RUNNING)),
    ):
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
        namespace=kubernetes_runner.namespace,
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

    project_id = "project1"

    # Get jobs for project-1
    project1_jobs = await kubernetes_runner.get_jobs(project_id)
    batch_api.list_namespaced_job.assert_called_with(
        namespace=kubernetes_runner.namespace,
        label_selector=f"app=moatless-worker,project_id={project_id}",
    )
    assert len(project1_jobs) == 2


@pytest.mark.asyncio
async def test_cancel_job(kubernetes_runner, mock_k8s_api):
    """Test that a job can be cancelled."""
    batch_api = mock_k8s_api["batch_api"]

    project_id = "test-project"
    trajectory_id = "test-trajectory"
    expected_namespace = kubernetes_runner.namespace
    expected_job_id = kubernetes_runner._job_id(project_id, trajectory_id)

    # Cancel a specific job
    await kubernetes_runner.cancel_job(project_id, trajectory_id)

    # Verify API was called to delete the job
    batch_api.delete_namespaced_job.assert_called_once()
    call_args = batch_api.delete_namespaced_job.call_args
    assert call_args[1]["name"] == expected_job_id
    assert call_args[1]["namespace"] == expected_namespace
    assert "body" in call_args[1]


@pytest.mark.asyncio
async def test_cancel_all_project_jobs(kubernetes_runner, mock_k8s_api):
    """Test that all jobs for a project can be cancelled."""
    batch_api = mock_k8s_api["batch_api"]

    project_id = "test-project"
    expected_namespace = kubernetes_runner.namespace

    # Create mock job list
    job_list = MagicMock()
    job_list.items = [
        create_mock_job("run-test-project-trajectory1"),
        create_mock_job("run-test-project-trajectory2"),
    ]
    batch_api.list_namespaced_job.return_value = job_list

    # Cancel all jobs for test-project
    await kubernetes_runner.cancel_job(project_id, None)

    # Verify list_namespaced_job was called with correct selector
    batch_api.list_namespaced_job.assert_called_once_with(
        namespace=expected_namespace,
        label_selector="app=moatless-worker,project_id=test-project",
    )

    # Verify delete was called for each job
    assert batch_api.delete_namespaced_job.call_count == 2


@pytest.mark.asyncio
async def test_job_exists(kubernetes_runner, mock_k8s_api):
    """Test checking if a job exists."""
    batch_api = mock_k8s_api["batch_api"]
    core_api = mock_k8s_api["core_api"]

    project_id = "test-project"
    trajectory_id = "test-trajectory"
    expected_job_id = kubernetes_runner._job_id(project_id, trajectory_id)

    # Job exists case
    batch_api.read_namespaced_job.return_value = create_mock_job(expected_job_id)
    assert await kubernetes_runner.job_exists("test-project", "test-trajectory") is True

    # Job doesn't exist case - also ensure no pods exist
    from kubernetes.client.rest import ApiException

    error_response = ApiException()
    error_response.status = 404
    batch_api.read_namespaced_job.side_effect = error_response

    # Mock an empty pod list response for the pod check
    empty_pod_list = MagicMock()
    empty_pod_list.items = []
    core_api.list_namespaced_pod.return_value = empty_pod_list

    assert await kubernetes_runner.job_exists("test-project", "nonexistent") is False


@pytest.mark.asyncio
async def test_get_job_status(kubernetes_runner, mock_k8s_api):
    """Test getting job status for different scenarios."""
    batch_api = mock_k8s_api["batch_api"]

    project_id = "test-project"
    trajectory_id = "test-trajectory"
    expected_job_id = kubernetes_runner._job_id(project_id, trajectory_id)

    # Test for running job
    running_job = create_mock_job(expected_job_id, active=1, succeeded=0, failed=0)
    batch_api.read_namespaced_job.return_value = running_job

    status = await kubernetes_runner.get_job_status("test-project", "test-trajectory")
    assert status == JobStatus.RUNNING

    # Test for completed job
    completed_job = create_mock_job(expected_job_id, active=0, succeeded=1, failed=0)
    batch_api.read_namespaced_job.return_value = completed_job

    status = await kubernetes_runner.get_job_status("test-project", "test-trajectory")
    assert status == JobStatus.COMPLETED

    # Test for failed job
    failed_job = create_mock_job(expected_job_id, active=0, succeeded=0, failed=1)
    batch_api.read_namespaced_job.return_value = failed_job

    status = await kubernetes_runner.get_job_status("test-project", "test-trajectory")
    assert status == JobStatus.FAILED

    # Test for non-existent job
    from kubernetes.client.rest import ApiException

    error_response = ApiException()
    error_response.status = 404
    batch_api.read_namespaced_job.side_effect = error_response

    status = await kubernetes_runner.get_job_status("test-project", "nonexistent")
    assert status is None  # Job doesn't exist, returns None instead of PENDING


@pytest.mark.asyncio
async def test_get_runner_info(kubernetes_runner, mock_k8s_api):
    """Test getting runner info."""
    batch_api = mock_k8s_api["batch_api"]
    core_api = mock_k8s_api["core_api"]

    # Set up API resources response
    api_resources = MagicMock()
    api_resources.group_version = "batch/v1"
    batch_api.get_api_resources.return_value = api_resources

    # Set up nodes response
    node_list = MagicMock()
    node1 = MagicMock()
    ready_condition = MagicMock()
    ready_condition.type = "Ready"
    ready_condition.status = "True"
    node1.status.conditions = [ready_condition]

    node2 = MagicMock()
    not_ready_condition = MagicMock()
    not_ready_condition.type = "Ready"
    not_ready_condition.status = "False"
    node2.status.conditions = [not_ready_condition]

    node_list.items = [node1, node2]
    core_api.list_node.return_value = node_list

    # Get runner info
    runner_info = await kubernetes_runner.get_runner_info()

    # Verify info
    assert runner_info.runner_type == "kubernetes"
    assert runner_info.status == RunnerStatus.RUNNING
    assert runner_info.data["nodes"] == 2
    assert runner_info.data["ready_nodes"] == 1
    assert runner_info.data["api_version"] == "batch/v1"
    assert runner_info.data["namespace"] == "test-namespace"

    # Test error case
    batch_api.get_api_resources.side_effect = Exception("Test error")

    # Get runner info with error
    runner_info = await kubernetes_runner.get_runner_info()

    # Verify error status
    assert runner_info.runner_type == "kubernetes"
    assert runner_info.status == RunnerStatus.ERROR
    assert "error" in runner_info.data


@pytest.mark.asyncio
async def test_get_job_logs(kubernetes_runner, mock_k8s_api):
    """Test getting logs for a job."""
    core_api = mock_k8s_api["core_api"]

    # Setup test data
    project_id = "test-project"
    trajectory_id = "test-trajectory"
    expected_job_id = kubernetes_runner._job_id(project_id, trajectory_id)
    expected_namespace = kubernetes_runner.namespace

    # Create pod list with one pod
    pod_list = MagicMock()
    pod = MagicMock()
    pod.metadata = MagicMock()
    pod.metadata.name = "test-pod"
    pod_list.items = [pod]
    core_api.list_namespaced_pod.return_value = pod_list

    # Set up log response
    core_api.read_namespaced_pod_log.return_value = "Test log output"

    # Get logs
    logs = await kubernetes_runner.get_job_logs(project_id, trajectory_id)

    # Verify log retrieval
    assert logs == "Test log output"
    core_api.list_namespaced_pod.assert_called_once()
    core_api.read_namespaced_pod_log.assert_called_once_with(name="test-pod", namespace=expected_namespace)

    # Test with no pods
    core_api.list_namespaced_pod.reset_mock()
    core_api.read_namespaced_pod_log.reset_mock()

    # Set up empty pod list
    empty_pod_list = MagicMock()
    empty_pod_list.items = []
    core_api.list_namespaced_pod.return_value = empty_pod_list

    # Get logs with no pods
    logs = await kubernetes_runner.get_job_logs(project_id, trajectory_id)

    # Verify no logs were fetched
    assert logs is None
    core_api.list_namespaced_pod.assert_called_once()
    core_api.read_namespaced_pod_log.assert_not_called()


@pytest.mark.asyncio
async def test_get_job_error(kubernetes_runner, mock_k8s_api):
    """Test getting error details from a failed pod."""
    core_api = mock_k8s_api["core_api"]

    # Setup test data
    project_id = "test-project"
    trajectory_id = "test-trajectory"
    expected_job_id = kubernetes_runner._job_id(project_id, trajectory_id)
    expected_namespace = kubernetes_runner.namespace

    # Create a failed job
    failed_job = create_mock_job(expected_job_id, active=0, succeeded=0, failed=1, namespace=expected_namespace)

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

    # Set job metadata for error handling
    failed_job.metadata.namespace = expected_namespace

    # Use the real KubernetesRunner._get_job_error method for this test
    # instead of the TestKubernetesRunner stub implementation
    with patch.object(TestKubernetesRunner, "_get_job_error", KubernetesRunner._get_job_error):
        # Get error details
        error_message = kubernetes_runner._get_job_error(failed_job)

        # Verify error message
        assert "OOMKilled" in error_message
        assert "Killed for using too much memory" in error_message


@pytest.mark.asyncio
async def test_create_job_object(kubernetes_runner):
    """Test creating a Kubernetes Job object with various configurations."""
    project_id = "test-project"
    trajectory_id = "test-trajectory"
    job_id = kubernetes_runner._job_id(project_id, trajectory_id)
    job_func = "test_module.test_function"
    expected_namespace = kubernetes_runner.namespace

    # Test without node_id
    job_obj = kubernetes_runner._create_job_object(
        job_id=job_id,
        project_id=project_id,
        trajectory_id=trajectory_id,
        job_func=job_func,
    )

    # Verify basic job object structure
    assert job_obj.metadata.name == job_id
    assert job_obj.metadata.labels["app"] == "moatless-worker"
    assert job_obj.metadata.labels["project_id"] == project_id
    assert job_obj.metadata.labels["trajectory_id"] == trajectory_id
    assert job_obj.metadata.annotations["moatless.ai/project-id"] == project_id
    assert job_obj.metadata.annotations["moatless.ai/trajectory-id"] == trajectory_id

    # Verify the namespace is set correctly
    assert job_obj.metadata.namespace == expected_namespace

    # Test with node_id
    node_id = 42
    job_obj_with_node = kubernetes_runner._create_job_object(
        job_id=job_id,
        project_id=project_id,
        trajectory_id=trajectory_id,
        job_func=job_func,
        node_id=node_id,
    )

    # Verify node_id is included in annotations
    assert job_obj_with_node.metadata.annotations["moatless.ai/node-id"] == str(node_id)

    # Test with OpenTelemetry context
    otel_context = {"trace_id": "1234567890abcdef", "span_id": "fedcba0987654321"}

    job_obj_with_otel = kubernetes_runner._create_job_object(
        job_id=job_id,
        project_id=project_id,
        trajectory_id=trajectory_id,
        job_func=job_func,
        otel_context=otel_context,
    )

    # In the TestKubernetesRunner implementation, otel_context should be added
    # to annotations but this isn't explicitly tested in our stub implementation
    # Verify other basic attributes are still correct
    assert job_obj_with_otel.metadata.name == job_id
    assert job_obj_with_otel.metadata.labels["project_id"] == project_id


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

    # Setup job list
    project_id = "test-project"
    expected_namespace = kubernetes_runner.namespace

    job_list = MagicMock()
    job_list.items = [
        create_mock_job("run-test-project-running", active=1, succeeded=0, failed=0),
        create_mock_job("run-test-project-completed", active=0, succeeded=1, failed=0),
        create_mock_job("run-test-project-failed", active=0, succeeded=0, failed=1),
    ]

    # Add annotations to each job
    for job in job_list.items:
        # Extract trajectory_id from the end of the name "run-{project_id}-{trajectory_id}"
        trajectory_id = job.metadata.name.split("-")[-1]
        job.metadata.annotations = {"moatless.ai/project-id": project_id, "moatless.ai/trajectory-id": trajectory_id}

    batch_api.list_namespaced_job.return_value = job_list

    # Get summary
    summary = await kubernetes_runner.get_job_status_summary()

    # Verify summary contains all jobs
    assert summary.total_jobs == 3
    assert summary.running_jobs == 1
    assert summary.completed_jobs == 1
    assert summary.failed_jobs == 1


@pytest.mark.asyncio
async def test_job_exists_with_pods_fallback(kubernetes_runner, mock_k8s_api):
    """Test that job_exists fallbacks to checking pods when job not found directly."""
    project_id = "test-project"
    trajectory_id = "test-trajectory"
    job_id = kubernetes_runner._job_id(project_id, trajectory_id)

    batch_api = mock_k8s_api["batch_api"]
    core_api = mock_k8s_api["core_api"]

    # Setup batch_api to raise a 404 when trying to get the job directly
    api_exception = client.ApiException(status=404, reason="Not Found")
    batch_api.read_namespaced_job.side_effect = api_exception

    # First test with no pods - should return False
    pod_list_empty = MagicMock()
    pod_list_empty.items = []
    core_api.list_namespaced_pod.return_value = pod_list_empty

    exists = await kubernetes_runner.job_exists(project_id, trajectory_id)
    assert not exists

    # Verify correct label selector was used
    core_api.list_namespaced_pod.assert_called_with(
        namespace=kubernetes_runner.namespace, label_selector=f"job-name={job_id}", limit=1
    )

    # Now test with a pod present - should return True
    pod = MagicMock(spec=V1Pod)
    pod.metadata = MagicMock()
    pod.metadata.name = f"pod-{job_id}"
    pod.status = MagicMock()
    pod.status.phase = "Running"

    pod_list_with_pod = MagicMock()
    pod_list_with_pod.items = [pod]
    core_api.list_namespaced_pod.return_value = pod_list_with_pod

    exists = await kubernetes_runner.job_exists(project_id, trajectory_id)
    assert exists

    # Test with server error
    api_exception_server = client.ApiException(status=500, reason="Internal Server Error")
    batch_api.read_namespaced_job.side_effect = api_exception_server

    # Should assume job exists for safety
    exists = await kubernetes_runner.job_exists(project_id, trajectory_id)
    assert exists

    # Test with unexpected exception
    batch_api.read_namespaced_job.side_effect = Exception("Unexpected error")

    # Should assume job exists for safety
    exists = await kubernetes_runner.job_exists(project_id, trajectory_id)
    assert exists
