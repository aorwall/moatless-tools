import logging
import os
from collections.abc import Callable
from typing import Any, Optional
from datetime import datetime, timezone

from kubernetes import client, config
from kubernetes.client import ApiException
from opentelemetry import trace

from moatless.runner.runner import (
    BaseRunner,
    JobInfo,
    JobsStatusSummary,
    JobStatus,
    RunnerInfo,
    RunnerStatus,
    JobDetails,
    JobDetailSection,
)
from moatless.telemetry import extract_trace_context
from moatless.runner.label_utils import sanitize_label, get_project_label, get_trajectory_label, create_metadata

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.runner.kubernetes")


class KubernetesRunner(BaseRunner):
    """Runner for managing jobs with Kubernetes."""

    def __init__(
        self,
        namespace: str | None = None,
        image: str | None = None,
        job_ttl_seconds: int = 3600,
        timeout_seconds: int = 3600,
        service_account: str | None = None,
        kubernetes_provider: str | None = None,
        node_selector: dict | None = None,
        max_jobs_per_project: int = 5,
    ):
        """Initialize the runner with Kubernetes client configuration.

        Args:
            namespace: Kubernetes namespace to use for jobs
            image: Docker image to use for worker jobs
            job_ttl_seconds: Time-to-live for completed jobs
            timeout_seconds: Timeout for jobs
            service_account: Service account to use for jobs
            kubernetes_provider: Kubernetes provider to use
            node_selector: Node selector labels for job pods (default: {"node-purpose": "testbeds"})
            max_jobs_per_project: Maximum number of jobs allowed per project
        """
        self.namespace = namespace or os.getenv("KUBERNETES_RUNNER_NAMESPACE", "moatless-tools")
        self.image = image
        self.job_ttl_seconds = job_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.service_account = service_account
        self.kubernetes_provider = kubernetes_provider or os.getenv("KUBERNETES_RUNNER_PROVIDER", "in-cluster")
        logger.info(f"Using Kubernetes provider: {self.kubernetes_provider}")
        self.node_selector = node_selector or {"node-purpose": "testbed-dev"}
        self.logger = logging.getLogger(__name__)
        self.max_jobs_per_project = max_jobs_per_project
        self.use_project_namespaces = os.getenv("USE_PROJECT_NAMESPACES", "true").lower() == "true"

        # Load the Kubernetes configuration
        try:
            config.load_incluster_config()
            self.logger.info("Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            try:
                config.load_kube_config()
                self.logger.info("Loaded Kubernetes configuration from default location")
            except config.ConfigException:
                self.logger.error("Could not load Kubernetes configuration")
                raise

        # Initialize Kubernetes API clients
        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()

    async def _get_or_create_project_namespace(self, project_id: str) -> str:
        """Get or create a namespace for a project and set up resource quota.

        Args:
            project_id: The project ID

        Returns:
            Namespace name
        """
        if not self.use_project_namespaces:
            return self.namespace

        # Create a valid namespace name
        namespace_name = self._create_namespace_name(project_id)

        try:
            # Check if namespace exists
            self.core_v1.read_namespace(name=namespace_name)
            self.logger.info(f"Using existing namespace {namespace_name} for project {project_id}")

            # Check if the required resources exist in the namespace
            await self._ensure_namespace_resources(namespace_name)
        except ApiException as e:
            if e.status == 404:
                # Create namespace
                await self._create_project_namespace(namespace_name, project_id)
            else:
                self.logger.exception(f"Error checking namespace {namespace_name}: {e}")
                # Fall back to default namespace
                return self.namespace

        return namespace_name

    def _create_namespace_name(self, project_id: str) -> str:
        """Create a valid Kubernetes namespace name for a project.

        Args:
            project_id: The project ID

        Returns:
            Valid namespace name
        """
        # Create a valid namespace name that complies with RFC 1123
        # Must be lowercase alphanumeric chars or '-', max 63 chars, start/end with alphanumeric
        project_hash = str(hash(project_id) % 10000000)  # Use hash to create a shorter identifier
        base_name = f"moatless-proj-{self._sanitize_kubernetes_label(project_id)}"

        # Ensure it's not too long (K8s has a 63 character limit)
        if len(base_name) > 57:  # Allow room for hash
            # Use just the start of the name plus hash to maintain uniqueness
            namespace_name = f"moatless-proj-{project_hash}"
        else:
            namespace_name = base_name

        # Ensure namespace name is at most 63 chars
        namespace_name = namespace_name[:63]

        # Ensure it starts and ends with alphanumeric character
        if not namespace_name[0].isalnum():
            namespace_name = f"x{namespace_name[1:]}"
        if not namespace_name[-1].isalnum():
            namespace_name = f"{namespace_name[:-1]}x"

        return namespace_name

    async def _create_project_namespace(self, namespace_name: str, project_id: str) -> None:
        """Create a namespace for a project.

        Args:
            namespace_name: Name for the namespace
            project_id: The project ID
        """
        project_hash = str(hash(project_id) % 10000000)

        # Create namespace
        namespace = client.V1Namespace(
            metadata=client.V1ObjectMeta(
                name=namespace_name,
                labels={"moatless-project": project_hash},  # Use hash for label value too
                annotations={"moatless.ai/project-id": project_id},
            )
        )

        self.core_v1.create_namespace(body=namespace)
        self.logger.info(f"Created namespace {namespace_name} for project {project_id}")

        # Create resource quota for the namespace
        await self._create_resource_quota(namespace_name)

        # Copy shared ConfigMap and Secret to the new namespace
        await self._copy_shared_resources(namespace_name)

    async def _ensure_namespace_resources(self, namespace: str) -> None:
        """Ensure the namespace has all required resources.

        Args:
            namespace: Namespace to check
        """
        try:
            await self._ensure_resource_quota(namespace)
            await self._ensure_config_map(namespace)
            await self._ensure_secret(namespace)
        except Exception as exc:
            self.logger.exception(f"Error ensuring resources in namespace {namespace}: {exc}")

    async def _ensure_resource_quota(self, namespace: str) -> None:
        """Ensure resource quota exists in namespace.

        Args:
            namespace: Namespace to check
        """
        try:
            self.core_v1.read_namespaced_resource_quota(name="project-job-limit", namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                # Create resource quota
                self.logger.info(f"Resource quota not found in namespace {namespace}, creating it")
                await self._create_resource_quota(namespace)

    async def _ensure_config_map(self, namespace: str) -> None:
        """Ensure ConfigMap exists in namespace.

        Args:
            namespace: Namespace to check
        """
        try:
            self.core_v1.read_namespaced_config_map(name="moatless-tools-env", namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                # ConfigMap not found, copy it
                self.logger.info(f"ConfigMap not found in namespace {namespace}, copying it")
                await self._copy_shared_resources(namespace)

    async def _ensure_secret(self, namespace: str) -> None:
        """Ensure Secret exists in namespace.

        Args:
            namespace: Namespace to check
        """
        try:
            self.core_v1.read_namespaced_secret(name="moatless-tools-secrets", namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                # Secret not found, copy it
                self.logger.info(f"Secret not found in namespace {namespace}, copying it")
                await self._copy_secret_to_namespace(namespace)

    async def _copy_shared_resources(self, namespace: str) -> None:
        """Copy shared ConfigMap and Secret from default namespace to project namespace.

        Args:
            namespace: Target namespace to copy resources to
        """
        try:
            await self._copy_config_map_to_namespace(namespace)
            await self._copy_secret_to_namespace(namespace)
        except Exception as exc:
            self.logger.exception(f"Error copying shared resources to namespace {namespace}: {exc}")

    async def _copy_config_map_to_namespace(self, namespace: str) -> None:
        """Copy ConfigMap to a namespace.

        Args:
            namespace: Target namespace
        """
        try:
            # Get the ConfigMap from the source namespace
            config_map = self.core_v1.read_namespaced_config_map("moatless-tools-env", self.namespace)

            # Create a new ConfigMap in the target namespace
            new_config_map = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(name="moatless-tools-env", namespace=namespace), data=config_map.data
            )

            self.core_v1.create_namespaced_config_map(namespace, new_config_map)
            self.logger.info(f"Copied ConfigMap moatless-tools-env to namespace {namespace}")
        except ApiException as e:
            self.logger.exception(f"Error copying ConfigMap to namespace {namespace}: {e}")

    async def _copy_secret_to_namespace(self, namespace: str) -> None:
        """Copy Secret to a namespace.

        Args:
            namespace: Target namespace
        """
        try:
            # Get the Secret from the source namespace
            secret = self.core_v1.read_namespaced_secret("moatless-tools-secrets", self.namespace)

            # Create a new Secret in the target namespace
            new_secret = client.V1Secret(
                metadata=client.V1ObjectMeta(name="moatless-tools-secrets", namespace=namespace),
                data=secret.data,
                type=secret.type,
            )

            self.core_v1.create_namespaced_secret(namespace, new_secret)
            self.logger.info(f"Copied Secret moatless-tools-secrets to namespace {namespace}")
        except ApiException as e:
            self.logger.exception(f"Error copying Secret to namespace {namespace}: {e}")

    async def _create_resource_quota(self, namespace: str) -> None:
        """Create a resource quota for a namespace.

        Args:
            namespace: Namespace name
        """
        # Create a resource quota that limits the number of active pods
        resource_quota = client.V1ResourceQuota(
            metadata=client.V1ObjectMeta(name="project-job-limit", namespace=namespace),
            spec=client.V1ResourceQuotaSpec(
                hard={
                    "count/pods": str(self.max_jobs_per_project),
                }
            ),
        )

        try:
            self.core_v1.create_namespaced_resource_quota(namespace=namespace, body=resource_quota)
            self.logger.info(
                f"Created resource quota in namespace {namespace} with limit of {self.max_jobs_per_project} active pods"
            )
        except ApiException as e:
            self.logger.exception(f"Error creating resource quota in namespace {namespace}: {e}")

    @tracer.start_as_current_span("KubernetesRunner.start_job")
    async def start_job(self, project_id: str, trajectory_id: str, job_func: Callable | str) -> bool:
        """Start a job as a Kubernetes Job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run or a string with the fully qualified function name

        Returns:
            True if the job was scheduled successfully, False otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)

        # Get or create namespace for the project
        namespace = await self._get_or_create_project_namespace(project_id)

        # Check if job exists and get its status
        if await self.job_exists(project_id, trajectory_id):
            job_status = await self.get_job_status(project_id, trajectory_id)

            # If job exists but failed, delete it first, then create a new one
            if job_status == JobStatus.FAILED:
                logger.info(f"Found failed job {job_id}, deleting it before creating a new one")
                try:
                    await self.cancel_job(project_id, trajectory_id)
                    # Short wait to ensure job is deleted
                    import asyncio

                    await asyncio.sleep(1)
                except Exception as e:
                    logger.warning(f"Error deleting failed job {job_id}: {e}")
            else:
                logger.info(f"Job {job_id} already exists with status {job_status}, skipping")
                return False

        try:
            # Determine fully qualified function name
            if isinstance(job_func, str):
                fully_qualified_name = job_func
            else:
                # If it's a callable, get the fully qualified name
                func_module = job_func.__module__
                func_name = job_func.__name__

                # Check if the module name ends with the function name
                if func_module.endswith(f".{func_name}"):
                    # Use the module path directly as the function is the module's name
                    fully_qualified_name = func_module
                else:
                    # Normal case: append the function name to the module path
                    fully_qualified_name = f"{func_module}.{func_name}"

            self.logger.info(f"Creating Kubernetes job for function: {fully_qualified_name}")

            # Extract OpenTelemetry context for propagation
            otel_context = extract_trace_context()

            # Create job spec
            job = self._create_job_object(
                job_id=job_id,
                project_id=project_id,
                trajectory_id=trajectory_id,
                func_name=fully_qualified_name,
                otel_context=otel_context,
            )

            # Create the job in the project namespace
            api_response = self.batch_v1.create_namespaced_job(
                namespace=namespace,
                body=job,
            )
            self.logger.info(f"Created job {api_response.metadata.name} in namespace {namespace}")
            return True

        except ApiException as e:
            if e.status == 403 and "exceeded quota" in str(e):
                self.logger.warning(f"Quota exceeded for project {project_id} in namespace {namespace}")
                return False
            self.logger.exception(f"Exception when creating job {job_id}: {e}")
            return False
        except Exception as exc:
            self.logger.exception(f"Error starting job {project_id}-{trajectory_id}: {exc}")
            raise exc

    async def get_jobs(self, project_id: str | None = None) -> list[JobInfo]:
        """Get all jobs for a project.

        Args:
            project_id: The project ID

        Returns:
            List of JobInfo objects with job status information
        """
        try:
            result = []
            namespace = self.namespace

            if project_id:
                # For a specific project, look in its namespace
                if self.use_project_namespaces:
                    namespace = await self._get_or_create_project_namespace(project_id)
                jobs = await self._list_jobs_in_namespace(namespace, project_id)

                # Process jobs from this namespace
                for job in jobs.items:
                    job_info = await self._process_job_info(job, project_id)
                    if job_info:
                        result.append(job_info)
            else:
                # For all projects, check all relevant namespaces
                result = await self._get_all_jobs()

            return result

        except ApiException as e:
            self.logger.exception(f"Exception when listing jobs: {e}")
            return []
        except Exception:
            self.logger.exception(f"Error getting jobs for project {project_id}")
            return []

    async def _get_all_jobs(self) -> list[JobInfo]:
        """Get all jobs from all project namespaces.

        Returns:
            List of JobInfo objects
        """
        result = []

        if self.use_project_namespaces:
            # Get all namespaces with moatless-project label
            namespaces = self.core_v1.list_namespace(label_selector="moatless-project")

            # Iterate through each namespace
            for ns in namespaces.items:
                namespace_name = ns.metadata.name

                # Get all moatless jobs in this namespace
                jobs = self.batch_v1.list_namespaced_job(
                    namespace=namespace_name,
                    label_selector="app=moatless-worker",
                )

                # Process jobs from this namespace
                for job in jobs.items:
                    job_info = await self._process_job_info(job)
                    if job_info:
                        result.append(job_info)
        else:
            # Just check the default namespace
            jobs = self.batch_v1.list_namespaced_job(
                namespace=self.namespace,
                label_selector="app=moatless-worker",
            )

            # Process jobs
            for job in jobs.items:
                job_info = await self._process_job_info(job)
                if job_info:
                    result.append(job_info)

        return result

    async def _list_jobs_in_namespace(self, namespace: str, project_id: str = None) -> Any:
        """List jobs in a namespace with optional project filtering.

        Args:
            namespace: Kubernetes namespace
            project_id: Optional project ID for filtering

        Returns:
            List of jobs
        """
        label_selector = "app=moatless-worker"

        if project_id:
            project_label = self._get_project_label(project_id)
            label_selector += f",project_id={project_label}"

        return self.batch_v1.list_namespaced_job(
            namespace=namespace,
            label_selector=label_selector,
        )

    async def get_job_status_summary(self, project_id: str) -> JobsStatusSummary:
        """Get a summary of job statuses for a project.

        Args:
            project_id: The project ID

        Returns:
            JobsStatusSummary with counts and IDs of jobs in different states
        """
        try:
            summary = JobsStatusSummary(project_id=project_id)
            namespace = await self._get_or_create_project_namespace(project_id)

            # Get all jobs for this project
            jobs = await self._list_jobs_in_namespace(namespace, project_id)

            # Count jobs by status
            for job in jobs.items:
                job_id = job.metadata.name
                status = self._get_job_status_from_k8s_job(job)

                await self._update_summary_for_job(summary, job_id, status)

            # Get quota information
            await self._add_quota_info_to_summary(summary, namespace)

            return summary

        except ApiException as e:
            self.logger.exception(
                f"Exception when getting job summary for project {project_id} in namespace {namespace}: {e}"
            )
            return JobsStatusSummary(project_id=project_id)
        except Exception as exc:
            self.logger.exception(f"Error getting job summary for project {project_id}: {exc}")
            return JobsStatusSummary(project_id=project_id)

    async def get_job_details(self, project_id: str, trajectory_id: str) -> Optional[JobDetails]:
        """Get detailed information about a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobDetails object containing detailed information about the job
        """
        job_id = self._job_id(project_id, trajectory_id)
        namespace = await self._get_or_create_project_namespace(project_id)

        try:
            # Check if job exists
            if not await self.job_exists(project_id, trajectory_id):
                return None

            # Get the Kubernetes Job resource
            job = self.batch_v1.read_namespaced_job(name=job_id, namespace=namespace)

            # Get job status
            job_status = self._get_job_status_from_k8s_job(job)

            # Get associated pod(s)
            pod_list = self.core_v1.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={job_id}")

            # Get the first pod (most relevant)
            pod = None
            if pod_list.items:
                pod = pod_list.items[0]

            # Get pod logs if available
            logs = ""
            if pod:
                try:
                    logs = self.core_v1.read_namespaced_pod_log(name=pod.metadata.name, namespace=namespace)
                except ApiException as e:
                    self.logger.warning(f"Error getting logs for pod {pod.metadata.name}: {e}")

            # Create JobDetails object with structured sections
            job_details = JobDetails(
                id=job_id,
                status=job_status,
                project_id=project_id,
                trajectory_id=trajectory_id,
                sections=[],
                raw_data=job.to_dict(),
            )

            # Extract timestamps from job
            if job.status.start_time:
                job_details.started_at = job.status.start_time.replace(tzinfo=timezone.utc)

            # For completion time
            if job.status.completion_time:
                job_details.ended_at = job.status.completion_time.replace(tzinfo=timezone.utc)

            # For creation time
            if job.metadata.creation_timestamp:
                job_details.enqueued_at = job.metadata.creation_timestamp.replace(tzinfo=timezone.utc)

            # Add Overview section
            job_labels = job.metadata.labels or {}
            overview_data = {
                "job_name": job.metadata.name,
                "namespace": namespace,
                "project_id": project_id,
                "trajectory_id": trajectory_id,
                "creation_time": job.metadata.creation_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
                if job.metadata.creation_timestamp
                else "Unknown",
                "start_time": job.status.start_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                if job.status.start_time
                else "Not started",
                "completion_time": job.status.completion_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                if job.status.completion_time
                else "Not completed",
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0,
            }

            overview_section = JobDetailSection(
                name="overview",
                display_name="Overview",
                data=overview_data,
            )
            job_details.sections.append(overview_section)

            # Add Job Spec section
            spec_data = {
                "parallelism": job.spec.parallelism,
                "completions": job.spec.completions,
                "active_deadline_seconds": job.spec.active_deadline_seconds,
                "backoff_limit": job.spec.backoff_limit,
                "ttl_seconds_after_finished": job.spec.ttl_seconds_after_finished,
            }

            spec_section = JobDetailSection(
                name="spec",
                display_name="Job Specification",
                data=spec_data,
            )
            job_details.sections.append(spec_section)

            # Add Pod Details section if pod exists
            if pod:
                pod_data = {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "host_ip": pod.status.host_ip,
                    "pod_ip": pod.status.pod_ip,
                    "start_time": pod.status.start_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                    if pod.status.start_time
                    else "Not started",
                }

                pod_section = JobDetailSection(
                    name="pod",
                    display_name="Pod Details",
                    data=pod_data,
                )
                job_details.sections.append(pod_section)

                # Add container details if available
                if pod.spec.containers:
                    containers_data = []
                    for container in pod.spec.containers:
                        container_data = {
                            "name": container.name,
                            "image": container.image,
                            "command": container.command,
                            "args": container.args,
                            "resources": container.resources.to_dict() if container.resources else {},
                        }
                        containers_data.append(container_data)

                    containers_section = JobDetailSection(
                        name="containers",
                        display_name="Containers",
                        items=containers_data,
                    )
                    job_details.sections.append(containers_section)

                # Add container status if available
                if pod.status.container_statuses:
                    statuses_data = []
                    for status in pod.status.container_statuses:
                        status_data = {
                            "name": status.name,
                            "ready": status.ready,
                            "restart_count": status.restart_count,
                            "image": status.image,
                            "image_id": status.image_id,
                            "container_id": status.container_id,
                        }

                        # Add detailed state information
                        if status.state.running:
                            status_data["state"] = "running"
                            status_data["started_at"] = (
                                status.state.running.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                                if status.state.running.started_at
                                else ""
                            )
                        elif status.state.terminated:
                            status_data["state"] = "terminated"
                            status_data["exit_code"] = status.state.terminated.exit_code
                            status_data["reason"] = status.state.terminated.reason
                            status_data["message"] = status.state.terminated.message
                            status_data["started_at"] = (
                                status.state.terminated.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                                if status.state.terminated.started_at
                                else ""
                            )
                            status_data["finished_at"] = (
                                status.state.terminated.finished_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                                if status.state.terminated.finished_at
                                else ""
                            )
                        elif status.state.waiting:
                            status_data["state"] = "waiting"
                            status_data["reason"] = status.state.waiting.reason
                            status_data["message"] = status.state.waiting.message

                        statuses_data.append(status_data)

                    status_section = JobDetailSection(
                        name="container_status",
                        display_name="Container Status",
                        items=statuses_data,
                    )
                    job_details.sections.append(status_section)

            # Add Labels section
            labels_section = JobDetailSection(
                name="labels",
                display_name="Labels",
                data=job_labels,
            )
            job_details.sections.append(labels_section)

            # Add Environment section (from container env if available)
            if pod and pod.spec.containers:
                env_data = {}
                for container in pod.spec.containers:
                    if container.env:
                        for env_var in container.env:
                            # Filter out sensitive data
                            if env_var.name and (
                                "API_KEY" in env_var.name
                                or "PASSWORD" in env_var.name
                                or "SECRET" in env_var.name
                                or "TOKEN" in env_var.name
                            ):
                                env_data[env_var.name] = "********"
                            else:
                                env_data[env_var.name] = env_var.value if env_var.value else "[from ConfigMap/Secret]"

                if env_data:
                    env_section = JobDetailSection(
                        name="environment",
                        display_name="Environment",
                        data=env_data,
                    )
                    job_details.sections.append(env_section)

            # Add Volumes section if available
            if pod and pod.spec.volumes:
                volumes_data = []
                for volume in pod.spec.volumes:
                    volume_data = {
                        "name": volume.name,
                    }

                    # Add volume source details
                    if volume.config_map:
                        volume_data["type"] = "configMap"
                        volume_data["config_map_name"] = volume.config_map.name
                    elif volume.secret:
                        volume_data["type"] = "secret"
                        volume_data["secret_name"] = volume.secret.secret_name
                    elif volume.persistent_volume_claim:
                        volume_data["type"] = "persistentVolumeClaim"
                        volume_data["claim_name"] = volume.persistent_volume_claim.claim_name
                    elif volume.empty_dir:
                        volume_data["type"] = "emptyDir"
                    else:
                        volume_data["type"] = "other"

                    volumes_data.append(volume_data)

                volumes_section = JobDetailSection(
                    name="volumes",
                    display_name="Volumes",
                    items=volumes_data,
                )
                job_details.sections.append(volumes_section)

            # Add Logs section if available
            if logs:
                logs_section = JobDetailSection(
                    name="logs",
                    display_name="Logs",
                    data={"logs": logs},
                )
                job_details.sections.append(logs_section)

            # Add error information if job failed
            if job_status == JobStatus.FAILED:
                error_message = "Job failed"

                # Check for specific error conditions
                if job.status.conditions:
                    for condition in job.status.conditions:
                        if condition.type == "Failed" and condition.message:
                            error_message = condition.message
                            break

                # Check pod for more detailed error
                if pod and pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        if container_status.state.terminated and container_status.state.terminated.exit_code != 0:
                            exit_code = container_status.state.terminated.exit_code
                            reason = container_status.state.terminated.reason or ""
                            message = container_status.state.terminated.message or ""

                            error_message = f"Container exited with code {exit_code}"
                            if reason:
                                error_message += f": {reason}"
                            if message:
                                error_message += f" - {message}"
                            break

                job_details.error = error_message

                error_data = {
                    "message": error_message,
                    "pod_phase": pod.status.phase if pod else "Unknown",
                }

                error_section = JobDetailSection(
                    name="error",
                    display_name="Error",
                    data=error_data,
                )
                job_details.sections.append(error_section)

            return job_details

        except ApiException as e:
            self.logger.exception(f"Exception when getting job details for {job_id} in namespace {namespace}: {e}")
            return None
        except Exception as exc:
            self.logger.exception(f"Error getting job details for {job_id}: {exc}")
            return None

    async def _update_summary_for_job(self, summary: JobsStatusSummary, job_id: str, status: JobStatus) -> None:
        """Update job status summary with a job.

        Args:
            summary: JobsStatusSummary to update
            job_id: Job ID
            status: Job status
        """
        summary.total_jobs += 1

        if status == JobStatus.PENDING:
            summary.pending_jobs += 1
            summary.job_ids["pending"].append(job_id)
        elif status == JobStatus.INITIALIZING:
            summary.initializing_jobs += 1
            summary.job_ids["initializing"].append(job_id)
        elif status == JobStatus.RUNNING:
            summary.running_jobs += 1
            summary.job_ids["running"].append(job_id)
        elif status == JobStatus.COMPLETED:
            summary.completed_jobs += 1
            summary.job_ids["completed"].append(job_id)
        elif status == JobStatus.FAILED:
            summary.failed_jobs += 1
            summary.job_ids["failed"].append(job_id)
        elif status == JobStatus.CANCELED:
            summary.canceled_jobs += 1
            summary.job_ids["canceled"].append(job_id)

    async def _add_quota_info_to_summary(self, summary: JobsStatusSummary, namespace: str) -> None:
        """Add quota information to job status summary.

        Args:
            summary: JobsStatusSummary to update
            namespace: Kubernetes namespace
        """
        if not self.use_project_namespaces:
            return

        try:
            quota = self.core_v1.read_namespaced_resource_quota(name="project-job-limit", namespace=namespace)
            if quota.status and quota.status.hard and quota.status.used:
                # Store quota data as a separate dictionary without trying to add it directly to the pydantic model
                quota_info = {
                    "max_active_pods": quota.status.hard.get("count/pods", self.max_jobs_per_project),
                    "used_active_pods": quota.status.used.get("count/pods", "0"),
                    "max_jobs": quota.status.hard.get("count/jobs.batch", self.max_jobs_per_project * 2),
                    "used_jobs": quota.status.used.get("count/jobs.batch", "0"),
                }

                # Only log the quota information instead of trying to add it to the model
                self.logger.debug(f"Quota info for namespace {namespace}: {quota_info}")
        except ApiException:
            pass

    async def _process_job_info(self, job, project_id: str = None) -> JobInfo | None:
        """Process a Kubernetes job and convert it to JobInfo.

        Args:
            job: Kubernetes job object
            project_id: Optional project ID for filtering

        Returns:
            JobInfo object or None if job should be skipped
        """
        job_id = job.metadata.name
        job_status = self._get_job_status_from_k8s_job(job)

        job_project_id = None
        job_trajectory_id = None

        # Extract project and trajectory IDs from annotations
        if job.metadata.annotations:
            if "moatless.ai/project-id" in job.metadata.annotations:
                job_project_id = job.metadata.annotations["moatless.ai/project-id"]
            if "moatless.ai/trajectory-id" in job.metadata.annotations:
                job_trajectory_id = job.metadata.annotations["moatless.ai/trajectory-id"]
        # Fallback to labels if annotations not available
        elif job.metadata.labels and "project_id" in job.metadata.labels:
            job_project_id = job.metadata.labels.get("project_id")
            # Try to extract trajectory_id from job name pattern "run-{project_id}-{trajectory_id}"
            if job_id.startswith(f"run-{job_project_id}"):
                # Extract trajectory_id from job name
                try:
                    parts = job_id.split("-")
                    if len(parts) >= 3:
                        job_trajectory_id = parts[2]
                except IndexError:
                    self.logger.warning(f"Could not extract trajectory_id from job name {job_id}")

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

    @tracer.start_as_current_span("KubernetesRunner.cancel_job")
    async def cancel_job(self, project_id: str, trajectory_id: str | None = None):
        """Cancel a job or all jobs for a project.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID. If None, cancels all jobs for the project.

        Returns:
            None
        """
        try:
            # Get namespace for the project
            namespace = await self._get_or_create_project_namespace(project_id)

            project_label = self._get_project_label(project_id)

            if trajectory_id is None:
                # Cancel all jobs for the project
                self.logger.info(f"Canceling all jobs for project {project_id} in namespace {namespace}")

                label_selector = f"app=moatless-worker,project_id={project_label}"

                # List all jobs for this project
                jobs = self.batch_v1.list_namespaced_job(
                    namespace=namespace,
                    label_selector=label_selector,
                )

                # Delete each job
                for job in jobs.items:
                    self.logger.info(f"Deleting job {job.metadata.name} in namespace {namespace}")
                    self.batch_v1.delete_namespaced_job(
                        name=job.metadata.name,
                        namespace=namespace,
                        body=client.V1DeleteOptions(
                            propagation_policy="Background",
                        ),
                    )
            else:
                # Cancel a specific job
                job_id = self._job_id(project_id, trajectory_id)
                self.logger.info(f"Canceling job {job_id} in namespace {namespace}")

                try:
                    self.batch_v1.delete_namespaced_job(
                        name=job_id,
                        namespace=namespace,
                        body=client.V1DeleteOptions(
                            propagation_policy="Background",
                        ),
                    )
                except ApiException as e:
                    if e.status != 404:  # Not Found
                        raise
                    self.logger.warning(
                        f"Job {job_id} not found in namespace {namespace}, it may have already been deleted"
                    )

        except ApiException as e:
            self.logger.exception(f"Exception when deleting jobs: {e}")
        except Exception as exc:
            self.logger.exception(f"Error canceling jobs for project {project_id}: {exc}")

    def _job_id(self, project_id: str, trajectory_id: str) -> str:
        """Format the job name for Kubernetes.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            The job name compliant with Kubernetes naming conventions
        """
        # Kubernetes has specific restrictions on resource names
        # The name must be lowercase, start with an alphanumeric character,
        # and only contain lowercase alphanumeric characters, '-', or '.'
        # Also, we add the 'run-' prefix to distinguish jobs
        job_id = f"run-{project_id}-{trajectory_id}"

        # Make compliant with Kubernetes naming restrictions
        job_id = job_id.lower()
        job_id = "".join(c if c.isalnum() or c == "-" else "-" for c in job_id)
        job_id = job_id[:63]  # Maximum length for Kubernetes resource names

        return job_id

    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job exists, False otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)
        namespace = await self._get_or_create_project_namespace(project_id)

        try:
            self.batch_v1.read_namespaced_job(name=job_id, namespace=namespace)
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            self.logger.exception(f"Error checking if job {job_id} exists: {e}")
            return False

    def _get_job_status_from_k8s_job(self, job) -> JobStatus:
        """Get the job status from a Kubernetes job.

        Args:
            job: The Kubernetes job object

        Returns:
            The job status
        """
        # Check if job is still active
        if job.status.active:
            return JobStatus.RUNNING

        # Check if job completed successfully
        if job.status.succeeded:
            return JobStatus.COMPLETED

        # Check if job failed
        if job.status.failed:
            return JobStatus.FAILED

        # Job is created but not started yet
        return JobStatus.PENDING
