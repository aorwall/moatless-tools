import logging
import os
from collections.abc import Callable
from typing import Any, Optional

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
)
from moatless.telemetry import extract_trace_context

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

            # Create the job
            api_response = self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=job,
            )
            self.logger.info(f"Created job {api_response.metadata.name}")
            return True

        except ApiException as e:
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
            label_selector = "app=moatless-worker"

            if project_id:
                project_label = self._get_project_label(project_id)
                label_selector += f",project_id={project_label}"

            api_response = self.batch_v1.list_namespaced_job(
                namespace=self.namespace,
                label_selector=label_selector,
            )

            for job in api_response.items:
                job_id = job.metadata.name

                # Extract status information
                job_status = self._get_job_status_from_k8s_job(job)

                job_project_id = None
                job_trajectory_id = None

                if job.metadata.annotations:
                    if "moatless.ai/project-id" in job.metadata.annotations:
                        job_project_id = job.metadata.annotations["moatless.ai/project-id"]
                    if "moatless.ai/trajectory-id" in job.metadata.annotations:
                        job_trajectory_id = job.metadata.annotations["moatless.ai/trajectory-id"]

                pod_metadata = None
                if job_status in [JobStatus.RUNNING, JobStatus.FAILED]:
                    pod_metadata = await self._get_pod_metadata(job_project_id, job_trajectory_id)

                job_info = JobInfo(
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

                result.append(job_info)

            return result

        except ApiException as e:
            self.logger.exception(f"Exception when listing jobs: {e}")
            return []
        except Exception:
            self.logger.exception(f"Error getting jobs for project {project_id}")
            return []

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
            project_label = self._get_project_label(project_id)

            if trajectory_id is None:
                # Cancel all jobs for the project
                self.logger.info(f"Canceling all jobs for project {project_id}")

                label_selector = f"app=moatless-worker,project_id={project_label}"

                # List all jobs for this project
                jobs = self.batch_v1.list_namespaced_job(
                    namespace=self.namespace,
                    label_selector=label_selector,
                )

                # Delete each job
                for job in jobs.items:
                    self.logger.info(f"Deleting job {job.metadata.name}")
                    self.batch_v1.delete_namespaced_job(
                        name=job.metadata.name,
                        namespace=self.namespace,
                        body=client.V1DeleteOptions(
                            propagation_policy="Background",
                        ),
                    )
            else:
                # Cancel a specific job
                job_id = self._job_id(project_id, trajectory_id)
                self.logger.info(f"Canceling job {job_id}")

                try:
                    self.batch_v1.delete_namespaced_job(
                        name=job_id,
                        namespace=self.namespace,
                        body=client.V1DeleteOptions(
                            propagation_policy="Background",
                        ),
                    )
                except ApiException as e:
                    if e.status != 404:  # Not Found
                        raise
                    self.logger.warning(f"Job {job_id} not found, it may have already been deleted")

        except ApiException as e:
            self.logger.exception(f"Exception when deleting jobs: {e}")
        except Exception as exc:
            self.logger.exception(f"Error canceling jobs for project {project_id}: {exc}")

    @tracer.start_as_current_span("KubernetesRunner.retry_job")
    async def retry_job(self, project_id: str, trajectory_id: str) -> bool:
        """Retry a failed job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job was restarted successfully, False otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)

        try:
            # Get the job to check if it exists and is failed
            job = self.batch_v1.read_namespaced_job(name=job_id, namespace=self.namespace)

            # Check if job is in failed state
            if job.status and job.status.failed and job.status.failed > 0:
                self.logger.info(f"Retrying failed job {job_id}")

                # Get the job's labels and env vars to recreate it
                labels = job.metadata.labels

                # Extract the function name from the job's env vars
                func_name = None
                for container in job.spec.template.spec.containers:
                    for env in container.env:
                        if env.name == "JOB_FUNC":
                            func_name = env.value
                            break

                if not func_name:
                    self.logger.error(f"Could not find function name in job {job_id}")
                    return False

                # Delete the old job
                self.batch_v1.delete_namespaced_job(
                    name=job_id,
                    namespace=self.namespace,
                    body=client.V1DeleteOptions(
                        propagation_policy="Background",
                    ),
                )

                # Create a new job with the same parameters
                new_job = self._create_job_object(
                    job_id=job_id,
                    project_id=project_id,
                    trajectory_id=trajectory_id,
                    func_name=func_name,
                )

                # Create the job
                self.batch_v1.create_namespaced_job(
                    namespace=self.namespace,
                    body=new_job,
                )

                return True

            else:
                self.logger.warning(f"Job {job_id} is not in failed state, cannot retry")
                return False

        except ApiException as e:
            if e.status == 404:
                self.logger.warning(f"Job {job_id} not found")
            else:
                self.logger.exception(f"Exception when retrying job {job_id}: {e}")
            return False
        except Exception as exc:
            self.logger.exception(f"Error retrying job {job_id}: {exc}")
            return False

    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists in Kubernetes.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job exists, False otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)

        try:
            self.batch_v1.read_namespaced_job(name=job_id, namespace=self.namespace)
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            else:
                self.logger.exception(f"Exception when checking if job {job_id} exists: {e}")
                raise
        except Exception as exc:
            self.logger.exception(f"Error checking if job {job_id} exists: {exc}")
            return False

    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobStatus enum value representing the job status
        """
        job_id = self._job_id(project_id, trajectory_id)

        try:
            job = self.batch_v1.read_namespaced_job(name=job_id, namespace=self.namespace)
            return self._get_job_status_from_k8s_job(job)
        except ApiException as e:
            if e.status == 404:
                return JobStatus.NOT_FOUND
            else:
                self.logger.exception(f"Exception when getting status for job {job_id}: {e}")
                return JobStatus.NOT_FOUND
        except Exception as exc:
            self.logger.exception(f"Error getting status for job {job_id}: {exc}")
            return JobStatus.NOT_FOUND

    async def get_runner_info(self) -> RunnerInfo:
        """Get information about the Kubernetes runner.

        Returns:
            RunnerInfo object with runner status information
        """
        try:
            # Check if the Kubernetes API is accessible
            version = self.batch_v1.get_api_resources()

            # Count the available nodes
            nodes = self.core_v1.list_node()
            ready_nodes = sum(
                1
                for node in nodes.items
                if any(
                    condition.type == "Ready" and condition.status == "True"
                    for condition in node.status.conditions or []
                )
            )

            # Build runner info
            return RunnerInfo(
                runner_type="kubernetes",
                status=RunnerStatus.RUNNING if ready_nodes > 0 else RunnerStatus.STOPPED,
                data={
                    "nodes": len(nodes.items),
                    "ready_nodes": ready_nodes,
                    "api_version": version.group_version,
                },
            )
        except Exception as exc:
            self.logger.exception(f"Error checking if runner is up: {exc}")
            return RunnerInfo(
                runner_type="kubernetes",
                status=RunnerStatus.ERROR,
                data={"error": str(exc)},
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

            # Get all jobs for this project
            project_label = self._get_project_label(project_id)
            label_selector = f"app=moatless-worker,project_id={project_label}"
            jobs = self.batch_v1.list_namespaced_job(
                namespace=self.namespace,
                label_selector=label_selector,
            )

            # Count jobs by status
            for job in jobs.items:
                job_id = job.metadata.name
                status = self._get_job_status_from_k8s_job(job)

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

            return summary

        except ApiException as e:
            self.logger.exception(f"Exception when getting job summary for project {project_id}: {e}")
            return JobsStatusSummary(project_id=project_id)
        except Exception as exc:
            self.logger.exception(f"Error getting job summary for project {project_id}: {exc}")
            return JobsStatusSummary(project_id=project_id)

    async def get_pod_logs(self, project_id: str, trajectory_id: str, tail_lines: int = None) -> str | None:
        """Get logs from the pod associated with a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            tail_lines: Number of lines to return from the end of the logs, None for all logs

        Returns:
            String containing the pod logs, or None if no logs are available
        """
        job_id = self._job_id(project_id, trajectory_id)

        try:
            # Get the pod for this job
            label_selector = f"job-name={job_id}"
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector,
            )

            if not pods.items:
                self.logger.warning(f"No pods found for job {job_id}")
                return None

            # Get the most recent pod
            pod = pods.items[-1]

            # Try to get current logs first
            try:
                logs = self.core_v1.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=self.namespace,
                    container="worker",
                    tail_lines=tail_lines,
                    timestamps=True,
                )
            except ApiException as e:
                logs = ""
                if e.status != 404:
                    self.logger.warning(f"Error getting current logs: {e}")

            # Try to get previous logs if container has restarted or terminated
            try:
                previous_logs = self.core_v1.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=self.namespace,
                    container="worker",
                    tail_lines=tail_lines,
                    timestamps=True,
                    previous=True,
                )
                if previous_logs:
                    logs = f"=== Previous container logs ===\n{previous_logs}\n=== Current container logs ===\n{logs}"
            except ApiException as e:
                if e.status != 404:
                    self.logger.warning(f"Error getting previous logs: {e}")

            return logs if logs else None

        except ApiException as e:
            if e.status != 404:  # Ignore 404 errors as they're expected when pod is not ready
                self.logger.exception(f"Exception when getting logs for job {job_id}: {e}")
            return None
        except Exception as exc:
            self.logger.exception(f"Error getting logs for job {job_id}: {exc}")
            return None

    async def get_job_logs(self, project_id: str, trajectory_id: str) -> str | None:
        """Get logs for a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            String containing the logs if available, None otherwise
        """
        # Get all logs (no tail_lines limit)
        return await self.get_pod_logs(project_id, trajectory_id, tail_lines=None)

    async def describe_pod(self, project_id: str, trajectory_id: str) -> dict | None:
        """Get detailed status and events for the pod associated with a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            Dictionary containing pod status and events, or None if no pod is found
        """
        job_id = self._job_id(project_id, trajectory_id)

        try:
            # Get the pod for this job
            label_selector = f"job-name={job_id}"
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector,
            )

            if not pods.items:
                self.logger.warning(f"No pods found for job {job_id}")
                return None

            # Get the most recent pod
            pod = pods.items[-1]
            pod_name = pod.metadata.name

            # Get pod status
            pod_status = {
                "name": pod_name,
                "phase": pod.status.phase,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "message": condition.message,
                    }
                    for condition in (pod.status.conditions or [])
                ],
                "container_statuses": [
                    {
                        "name": status.name,
                        "ready": status.ready,
                        "state": self._format_container_state(status.state),
                        "last_state": self._format_container_state(status.last_state),
                        "restart_count": status.restart_count,
                    }
                    for status in (pod.status.container_statuses or [])
                ],
            }

            # Get pod events
            field_selector = f"involvedObject.name={pod_name}"
            events = self.core_v1.list_namespaced_event(
                namespace=self.namespace,
                field_selector=field_selector,
            )

            pod_events = [
                {
                    "type": event.type,
                    "reason": event.reason,
                    "message": event.message,
                    "count": event.count,
                    "first_timestamp": event.first_timestamp,
                    "last_timestamp": event.last_timestamp,
                }
                for event in events.items
            ]

            # Get the job for this pod to fetch annotations
            job_metadata = {}
            try:
                job = self.batch_v1.read_namespaced_job(
                    name=job_id,
                    namespace=self.namespace,
                )

                if job.metadata.annotations:
                    for key, value in job.metadata.annotations.items():
                        if key.startswith("moatless.ai/"):
                            job_metadata[key.replace("moatless.ai/", "")] = value
            except ApiException:
                # Ignore if job is not found
                pass

            return {
                "status": pod_status,
                "events": pod_events,
                "env_vars": {},
                "job_metadata": job_metadata,
                "node": pod.spec.node_name if pod.spec and pod.spec.node_name else None,
                "ip": pod.status.pod_ip if pod.status and pod.status.pod_ip else None,
                "creation_timestamp": pod.metadata.creation_timestamp,
            }

        except ApiException as e:
            if e.status != 404:  # Ignore 404 errors as they're expected when pod is not ready
                self.logger.exception(f"Exception when describing pod for job {job_id}: {e}")
            return None
        except Exception as exc:
            self.logger.exception(f"Error describing pod for job {job_id}: {exc}")
            return None

    def _format_container_state(self, state) -> dict | None:
        """Format container state into a dictionary.

        Args:
            state: Kubernetes container state object

        Returns:
            Dictionary containing state information, or None if state is None
        """
        if not state:
            return None

        state_info = {}
        if state.running:
            state_info["status"] = "running"
            state_info["started_at"] = state.running.started_at
        elif state.waiting:
            state_info["status"] = "waiting"
            state_info["reason"] = state.waiting.reason
            state_info["message"] = state.waiting.message
        elif state.terminated:
            state_info["status"] = "terminated"
            state_info["exit_code"] = state.terminated.exit_code
            state_info["reason"] = state.terminated.reason
            state_info["message"] = state.terminated.message
            state_info["started_at"] = state.terminated.started_at
            state_info["finished_at"] = state.terminated.finished_at

        return state_info

    def _job_id(self, project_id: str, trajectory_id: str) -> str:
        """Create a job ID from project ID and trajectory ID.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            Job ID string
        """
        # Create a valid Kubernetes job name by replacing invalid characters
        # Kubernetes resource names must consist of lower case alphanumeric characters,
        # '-' or '.', and must start and end with an alphanumeric character
        base_id = f"run-{project_id}-{trajectory_id}"

        # Replace invalid characters with dashes
        job_id = "".join(c if c.isalnum() or c == "-" else "-" for c in base_id.lower())

        # Ensure it's not too long (Kubernetes has a 63 character limit for names)
        if len(job_id) > 63:
            # Truncate but keep the start and end parts to maintain uniqueness
            prefix = job_id[:31]
            suffix = job_id[-31:]
            job_id = f"{prefix}-{suffix}"

        return job_id

    def _get_job_status_from_k8s_job(self, job) -> JobStatus:
        """Map Kubernetes job status to JobStatus enum.

        Args:
            job: Kubernetes job object

        Returns:
            Corresponding JobStatus enum value
        """
        if not job.status:
            return JobStatus.PENDING

        # Check if job is being deleted
        if job.metadata.deletion_timestamp:
            return JobStatus.CANCELED

        # Check failed condition
        if job.status.failed and job.status.failed > 0:
            return JobStatus.FAILED

        # Check completed condition
        if job.status.succeeded and job.status.succeeded > 0:
            return JobStatus.COMPLETED

        # Check active condition
        if job.status.active and job.status.active > 0:
            # Job is active, but need to check pod status to determine if running or initializing
            try:
                job_id = job.metadata.name
                label_selector = f"job-name={job_id}"
                pods = self.core_v1.list_namespaced_pod(
                    namespace=self.namespace,
                    label_selector=label_selector,
                )

                if pods.items:
                    pod = pods.items[0]  # Get the first pod

                    # Check if pod is in Pending phase (initializing)
                    if pod.status.phase == "Pending":
                        return JobStatus.INITIALIZING

                    # If pod is in Running phase but all containers aren't ready, it's still initializing
                    elif pod.status.phase == "Running":
                        if pod.status.container_statuses:
                            all_ready = all(status.ready for status in pod.status.container_statuses)
                            if not all_ready:
                                return JobStatus.INITIALIZING

                        # All containers are ready and running
                        return JobStatus.RUNNING

                # If no pods found but job is active, it's pending
                return JobStatus.PENDING

            except Exception as e:
                # If there's an error getting pod status, fallback to basic job status
                self.logger.warning(f"Error checking pod status for job {job.metadata.name}: {e}")

                # If we can't check pod status but job has started, assume it's running
                if job.status.start_time:
                    return JobStatus.RUNNING

                # Otherwise it's still pending
                return JobStatus.PENDING

        # Default to pending if no conditions match
        return JobStatus.PENDING

    def _get_job_error(self, job) -> Optional[str]:
        """Extract error message from failed job.

        Args:
            job: Kubernetes job object

        Returns:
            Error message if available, None otherwise
        """
        if not job.status or not job.status.failed or job.status.failed == 0:
            return None

        try:
            # Get pods for this job
            label_selector = f"job-name={job.metadata.name}"
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector,
            )

            for pod in pods.items:
                # Check container statuses for error messages
                if pod.status and pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        if container_status.state and container_status.state.terminated:
                            terminated = container_status.state.terminated
                            if terminated.reason in ["Error", "OOMKilled", "ContainerCannotRun"]:
                                return f"{terminated.reason}: {terminated.message or 'Unknown error'}"

                # Check pod status for error messages
                if pod.status and pod.status.phase == "Failed":
                    return f"Pod failed: {pod.status.message or 'Unknown error'}"

            return "Job failed but no specific error was found"

        except Exception as e:
            self.logger.exception(f"Error getting error for job {job.metadata.name}: {e}")
            return "Error retrieving job failure details"

    def _sanitize_kubernetes_label(self, value: str) -> str:
        """Sanitize a string to be a valid Kubernetes label.

        Args:
            value: The string to sanitize

        Returns:
            A valid Kubernetes label string (max 63 chars, alphanumeric, '-', '_', or '.')
        """
        # Replace invalid characters with dashes
        clean_id = "".join(c if c.isalnum() or c in ["-", "_", "."] else "-" for c in value)

        # Ensure it starts and ends with alphanumeric character
        if clean_id and not clean_id[0].isalnum():
            clean_id = "x" + clean_id[1:]
        if clean_id and not clean_id[-1].isalnum():
            clean_id = clean_id[:-1] + "x"

        # Ensure it's not too long (Kubernetes has a 63 character limit for labels)
        return clean_id[:63]

    def _get_project_label(self, project_id: str) -> str:
        """Get a valid Kubernetes label for a project ID.

        Args:
            project_id: The project ID

        Returns:
            A valid Kubernetes label string (max 63 chars, alphanumeric, '-', '_', or '.')
        """
        return self._sanitize_kubernetes_label(project_id)

    def _get_trajectory_label(self, trajectory_id: str) -> str:
        """Get a valid Kubernetes label for a trajectory ID.

        Args:
            trajectory_id: The trajectory ID

        Returns:
            A valid Kubernetes label string (max 63 chars, alphanumeric, '-', '_', or '.')
        """
        return self._sanitize_kubernetes_label(trajectory_id)

    def _create_job_object(
        self,
        job_id: str,
        project_id: str,
        trajectory_id: str,
        func_name: str,
        otel_context: dict = None,
    ) -> client.V1Job:
        """Create a Kubernetes Job object.

        Args:
            job_id: The job ID
            project_id: The project ID
            trajectory_id: The trajectory ID
            func_name: Fully qualified function name to execute
            otel_context: OpenTelemetry context for tracing

        Returns:
            Kubernetes Job object
        """
        # Prepare labels for the job
        project_label = self._get_project_label(project_id)
        trajectory_label = self._get_trajectory_label(trajectory_id)
        labels = {
            "app": "moatless-worker",
            "project_id": project_label,
            "trajectory_id": trajectory_label,
        }

        # Store original values in annotations
        annotations = {
            "moatless.ai/project-id": project_id,
            "moatless.ai/trajectory-id": trajectory_id,
            "moatless.ai/function": func_name,
            "cluster-autoscaler.kubernetes.io/safe-to-evict": "false",  # Prevent autoscaler from evicting this pod
        }

        # Prepare environment variables
        env_vars = [
            client.V1EnvVar(name="PROJECT_ID", value=project_id),
            client.V1EnvVar(name="TRAJECTORY_ID", value=trajectory_id),
            client.V1EnvVar(name="JOB_FUNC", value=func_name),
        ]

        # Add API key environment variables
        for key, value in os.environ.items():
            if key.startswith("MOATLESS_"):
                env_vars.append(client.V1EnvVar(name=key, value=value))

        # Add OpenTelemetry context if available
        if otel_context:
            for key, value in otel_context.items():
                env_vars.append(client.V1EnvVar(name=f"OTEL_{key}", value=str(value)))

        env_vars.extend(
            [
                client.V1EnvVar(name="MOATLESS_DIR", value="/data/moatless"),
                client.V1EnvVar(name="MOATLESS_COMPONENTS_PATH", value="/opt/components"),
                client.V1EnvVar(name="NLTK_DATA", value="/data/nltk_data"),
                client.V1EnvVar(name="INDEX_STORE_DIR", value="/data/index_store"),
                client.V1EnvVar(name="REPO_DIR", value="/testbed"),
                client.V1EnvVar(name="INSTANCE_PATH", value="/data/instance.json"),
            ]
        )

        if os.environ.get("REDIS_URL"):
            env_vars.append(
                client.V1EnvVar(
                    name="REDIS_URL",
                    value=os.environ.get("REDIS_URL"),
                ),
            )

        volumes = [
            client.V1Volume(name="nltk-data", empty_dir=client.V1EmptyDirVolumeSource()),
            client.V1Volume(name="moatless-components", empty_dir=client.V1EmptyDirVolumeSource()),  # FIXME
        ]

        # Create volume mounts
        volume_mounts = [
            client.V1VolumeMount(name="moatless-components", mount_path="/opt/components"),
            client.V1VolumeMount(name="nltk-data", mount_path="/data/nltk_data"),
        ]

        # Create tolerations for spot instances etc
        tolerations = [
            # Add default toleration for testbeds node-purpose
            client.V1Toleration(
                key="node-purpose",
                operator="Equal",
                value="testbeds",
                effect="NoSchedule",
            )
        ]

        # Add provider-specific tolerations
        if self.kubernetes_provider == "azure":
            tolerations.append(
                client.V1Toleration(
                    key="kubernetes.azure.com/scalesetpriority",
                    operator="Equal",
                    value="spot",
                    effect="NoSchedule",
                )
            )

        # Create the pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels, annotations=annotations),
            spec=client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name="worker",
                        image=self._get_image_name(trajectory_id),
                        image_pull_policy="Always",
                        command=["/usr/bin/python"],
                        args=[
                            "-c",
                            f"from {func_name.rsplit('.', 1)[0]} import {func_name.rsplit('.', 1)[1]}; import sys; {func_name.rsplit('.', 1)[1]}('{project_id}', '{trajectory_id}')",
                        ],
                        env=env_vars,
                        env_from=[
                            client.V1EnvFromSource(
                                config_map_ref=client.V1ConfigMapEnvSource(name="moatless-tools-env", optional=False)
                            ),
                            client.V1EnvFromSource(
                                secret_ref=client.V1SecretEnvSource(name="moatless-tools-secrets", optional=False)
                            ),
                        ],
                        volume_mounts=volume_mounts,
                        resources=client.V1ResourceRequirements(
                            requests={"cpu": "100m", "memory": "1Gi"},
                            limits={"cpu": "1", "memory": "2Gi"},
                        ),
                        working_dir="/opt/moatless",
                    )
                ],
                volumes=volumes,
                restart_policy="Never",
                service_account_name=self.service_account,
                tolerations=tolerations,
                node_selector=self.node_selector,
            ),
        )

        # Create the job spec
        job_spec = client.V1JobSpec(
            template=pod_template,
            backoff_limit=0,  # No retries, we'll handle retries ourselves
            ttl_seconds_after_finished=self.job_ttl_seconds,
            active_deadline_seconds=self.timeout_seconds,
        )

        # Create the job
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_id,
                labels=labels,
                annotations=annotations,
            ),
            spec=job_spec,
        )

        return job

    def _get_image_name(self, trajectory_id: str) -> str:
        instance_id_split = trajectory_id.split("__")
        repo_name = instance_id_split[0]
        instance_id = instance_id_split[1]
        return f"aorwall/sweb.eval.x86_64.{repo_name}_moatless_{instance_id}"

    async def _get_pod_metadata(self, project_id: str, trajectory_id: str) -> dict | None:
        """Get detailed pod metadata for a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            Dictionary containing pod metadata or None if no pod is found
        """
        if not project_id or not trajectory_id:
            return None

        job_id = self._job_id(project_id, trajectory_id)

        try:
            # Get the pod for this job
            label_selector = f"job-name={job_id}"
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector,
            )

            if not pods.items:
                return None

            # Get the most recent pod
            pod = pods.items[-1]
            pod_name = pod.metadata.name

            # Extract environment variables from pod
            env_vars = {}
            if pod.spec and pod.spec.containers:
                for container in pod.spec.containers:
                    if container.env:
                        for env in container.env:
                            if env.name in ["PROJECT_ID", "TRAJECTORY_ID", "JOB_FUNC"]:
                                env_vars[env.name] = env.value

            # Get pod status
            pod_status = {
                "name": pod_name,
                "phase": pod.status.phase,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "message": condition.message if condition.message else None,
                    }
                    for condition in (pod.status.conditions or [])
                ],
                "container_statuses": [
                    {
                        "name": status.name,
                        "ready": status.ready,
                        "state": self._format_container_state(status.state),
                        "last_state": self._format_container_state(status.last_state),
                        "restart_count": status.restart_count,
                    }
                    for status in (pod.status.container_statuses or [])
                ],
            }

            # Get pod events
            field_selector = f"involvedObject.name={pod_name}"
            events = self.core_v1.list_namespaced_event(
                namespace=self.namespace,
                field_selector=field_selector,
            )

            pod_events = [
                {
                    "type": event.type,
                    "reason": event.reason,
                    "message": event.message,
                    "count": event.count,
                    "first_timestamp": event.first_timestamp,
                    "last_timestamp": event.last_timestamp,
                }
                for event in events.items
            ]

            # Get the job for this pod to fetch annotations
            job_metadata = {}
            try:
                job = self.batch_v1.read_namespaced_job(
                    name=job_id,
                    namespace=self.namespace,
                )

                if job.metadata.annotations:
                    for key, value in job.metadata.annotations.items():
                        if key.startswith("moatless.ai/"):
                            job_metadata[key.replace("moatless.ai/", "")] = value
            except ApiException:
                # Ignore if job is not found
                pass

            return {
                "status": pod_status,
                "events": pod_events,
                "env_vars": env_vars,
                "job_metadata": job_metadata,
                "node": pod.spec.node_name if pod.spec and pod.spec.node_name else None,
                "ip": pod.status.pod_ip if pod.status and pod.status.pod_ip else None,
                "creation_timestamp": pod.metadata.creation_timestamp,
            }

        except ApiException as e:
            if e.status != 404:  # Ignore 404 errors as they're expected when pod is not ready
                self.logger.exception(f"Exception when getting pod metadata for job {job_id}: {e}")
            return None
        except Exception as exc:
            self.logger.exception(f"Error getting pod metadata for job {job_id}: {exc}")
            return None
