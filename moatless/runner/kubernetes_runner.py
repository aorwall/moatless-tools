import logging
import os
from collections.abc import Callable
from datetime import timezone
from typing import Any, Optional

from kubernetes import client, config
from kubernetes.client import ApiException
from opentelemetry import trace

from moatless.runner.label_utils import (
    create_labels,
    create_annotations,
    get_project_label,
    get_trajectory_label,
    create_resource_id,
    create_job_args,
    sanitize_label,
)
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
        
        if node_selector:
            self.node_selector = node_selector
        else:
            node_selector_label = os.getenv("KUBERNETES_RUNNER_NODE_SELECTOR_LABEL", "testbeds")
            self.node_selector = {"node-purpose": node_selector_label}
            
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
    async def start_job(
        self, project_id: str, trajectory_id: str, job_func: Callable | str, node_id: int | None = None
    ) -> bool:
        """Start a job as a Kubernetes Job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run or a string with the fully qualified function name
            node_id: Optional node ID for routing jobs to specific nodes

        Returns:
            True if the job was scheduled successfully, False otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)
        job_status = await self.get_job_status(project_id, trajectory_id)
        logger.info(f"Job {job_id} status: {job_status}")

        # Check if job exists and get its status
        if job_status == JobStatus.RUNNING:
            logger.info(f"Job {job_id} already exists with status {job_status}, skipping")
            return True
        elif job_status in [JobStatus.PENDING, JobStatus.COMPLETED]:
            logger.info(f"Found job {job_id} with status {job_status}, deleting it before creating a new one")
            try:
                await self.cancel_job(project_id, trajectory_id)
                # Short wait to ensure job is deleted
                import asyncio
                await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Error deleting job {job_id}: {e}")
                return False
            
        try:
            self.logger.info(f"Creating Kubernetes job for function: {job_func}")

            # Extract OpenTelemetry context for propagation
            otel_context = extract_trace_context()

            job = self._create_job_object(
                job_id=job_id,
                project_id=project_id,
                trajectory_id=trajectory_id,
                job_func=job_func,
                otel_context=otel_context,
                node_id=node_id,
            )

            # Create the job in the namespace
            api_response = self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=job,
            )
            self.logger.info(f"Created job {api_response.metadata.name} in namespace {self.namespace}")
            return True

        except ApiException as e:
            if e.status == 403 and "exceeded quota" in str(e):
                self.logger.warning(f"Quota exceeded for project {project_id} in namespace {self.namespace}")
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
            
            # List jobs in the namespace
            jobs = await self._list_jobs_in_namespace(self.namespace, project_id)

            # Process jobs from this namespace
            for job in jobs.items:
                job_info = await self._process_job_info(job, project_id)
                if job_info:
                    result.append(job_info)

            return result

        except ApiException as e:
            self.logger.exception(f"Exception when listing jobs: {e}")
            return []
        except Exception:
            self.logger.exception(f"Error getting jobs for project {project_id}")
            return []

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

            # Get all jobs for this project
            jobs = await self._list_jobs_in_namespace(self.namespace, project_id)

            # Count jobs by status
            for job in jobs.items:
                job_id = job.metadata.name
                status = self._get_job_status_from_k8s_job(job)

                await self._update_summary_for_job(summary, job_id, status)

            return summary

        except ApiException as e:
            self.logger.exception(
                f"Exception when getting job summary for project {project_id} in namespace {self.namespace}: {e}"
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
        
        try:
            # Check if job exists
            if not await self.job_exists(project_id, trajectory_id):
                return None

            # Get the Kubernetes Job resource
            job = self.batch_v1.read_namespaced_job(name=job_id, namespace=self.namespace)

            # Get job status
            job_status = self._get_job_status_from_k8s_job(job)

            # Get associated pod(s)
            pod_list = self.core_v1.list_namespaced_pod(namespace=self.namespace, label_selector=f"job-name={job_id}")

            # Get the first pod (most relevant)
            pod = None
            if pod_list.items:
                pod = pod_list.items[0]

            # Get pod logs if available
            logs = ""
            if pod:
                try:
                    logs = self.core_v1.read_namespaced_pod_log(name=pod.metadata.name, namespace=self.namespace)
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
                "namespace": self.namespace,
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
            self.logger.exception(f"Exception when getting job details for {job_id} in namespace {self.namespace}: {e}")
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
        # Removed since we no longer manage quotas per project namespace
        pass

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
                self.logger.info(f"Canceling all jobs for project {project_id} in namespace {self.namespace}")

                label_selector = f"app=moatless-worker,project_id={project_label}"

                # List all jobs for this project
                jobs = self.batch_v1.list_namespaced_job(
                    namespace=self.namespace,
                    label_selector=label_selector,
                )

                # Delete each job
                for job in jobs.items:
                    self.logger.info(f"Deleting job {job.metadata.name} in namespace {self.namespace}")
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
                self.logger.info(f"Canceling job {job_id} in namespace {self.namespace}")

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
                    self.logger.warning(
                        f"Job {job_id} not found in namespace {self.namespace}, it may have already been deleted"
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
        return create_resource_id(project_id, trajectory_id, prefix="run")

    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job exists, False otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)
        self.logger.info(f"Checking if job {job_id} exists in namespace {self.namespace}")
        
        # Try to get the job directly first
        try:
            self.batch_v1.read_namespaced_job(name=job_id, namespace=self.namespace)
            self.logger.info(f"Job {job_id} found directly in namespace {self.namespace}")
            return True
        except ApiException as e:
            if e.status == 404:
                # Job not found by direct API call, try checking for associated pods
                try:
                    # Check if there are any pods with this job-name label
                    pod_list = self.core_v1.list_namespaced_pod(
                        namespace=self.namespace, 
                        label_selector=f"job-name={job_id}",
                        limit=1  # We only need to know if any exist
                    )
                    
                    # If we found any pods, the job still logically exists
                    if pod_list.items:
                        pod = pod_list.items[0]
                        self.logger.info(
                            f"Job {job_id} not found directly but found associated pod {pod.metadata.name} "
                            f"in phase {pod.status.phase}"
                        )
                        return True
                    
                    # No pods found either, job truly doesn't exist
                    return False
                    
                except ApiException as pod_e:
                    self.logger.warning(f"Error checking for pods for job {job_id}: {pod_e}")
                    # If we can't check pods either, fall back to "not found"
                    return False
            
            # For other API errors, log but don't assume job doesn't exist
            self.logger.warning(f"Error checking if job {job_id} exists: {e}, status code: {e.status}")
            if e.status >= 500:
                # For server errors, assume job might exist to prevent unnecessary job termination
                self.logger.warning(f"Server error checking job {job_id}, assuming it might exist")
                return True
            
            return False
        except Exception as exc:
            # For unexpected errors, log and assume job might exist to be safe
            self.logger.exception(f"Unexpected error checking if job {job_id} exists: {exc}")
            return True

    def _get_job_status_from_k8s_job(self, job) -> JobStatus:
        """Get the job status from a Kubernetes job.

        Args:
            job: The Kubernetes job object

        Returns:
            The job status
        """
        # Check if job is still active
        self.logger.info(f"Job {job.metadata.name} status: {job.status}")
        
        # Check if job completed successfully
        if job.status.succeeded:
            return JobStatus.COMPLETED

        # Check if job failed
        if job.status.failed:
            return JobStatus.FAILED

        # Consider the job has running if a job deployment exists and is not completed
        return JobStatus.RUNNING

    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            The job status
        """
        job_id = self._job_id(project_id, trajectory_id)
        
        try:
            # Get the Kubernetes Job resource
            job = self.batch_v1.read_namespaced_job(name=job_id, namespace=self.namespace)
            return self._get_job_status_from_k8s_job(job)
        except ApiException as e:
            if e.status == 404:
                self.logger.info(f"Job {job_id} not found in namespace {self.namespace}")
                return JobStatus.NOT_STARTED
            self.logger.exception(f"Error getting job status for {job_id}: {e}")
            return JobStatus.NOT_STARTED
        except Exception as exc:
            self.logger.exception(f"Unexpected error getting job status for {job_id}: {exc}")
            return JobStatus.NOT_STARTED

    async def get_runner_info(self) -> RunnerInfo:
        """Get information about the runner.

        Returns:
            RunnerInfo with details about the Kubernetes cluster
        """
        try:
            # Get information about nodes in the cluster
            nodes = self.core_v1.list_node()
            ready_nodes = 0
            total_nodes = 0

            for node in nodes.items:
                total_nodes += 1
                # Check if node is ready
                for condition in node.status.conditions:
                    if condition.type == "Ready" and condition.status == "True":
                        ready_nodes += 1
                        break

            # Get API resources info
            api_resources = self.batch_v1.get_api_resources()

            # Build runner info
            info = RunnerInfo(
                runner_type="kubernetes",
                status=RunnerStatus.RUNNING,
                data={
                    "nodes": total_nodes,
                    "ready_nodes": ready_nodes,
                    "api_version": api_resources.group_version,
                    "provider": self.kubernetes_provider,
                    "namespace": self.namespace,
                },
            )
            return info
        except Exception as exc:
            self.logger.exception(f"Error getting runner info: {exc}")
            return RunnerInfo(
                runner_type="kubernetes",
                status=RunnerStatus.ERROR,
                data={"error": str(exc)},
            )

    async def get_job_logs(self, project_id: str, trajectory_id: str) -> Optional[str]:
        """Get logs for a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            String containing the logs if available, None otherwise
        """
        job_id = self._job_id(project_id, trajectory_id)

        try:
            # Get pods associated with the job
            pod_list = self.core_v1.list_namespaced_pod(namespace=self.namespace, label_selector=f"job-name={job_id}")

            # If no pods found, return None
            if not pod_list.items:
                return None

            # Get logs from the first pod
            pod = pod_list.items[0]
            logs = self.core_v1.read_namespaced_pod_log(name=pod.metadata.name, namespace=self.namespace)
            return logs
        except ApiException as e:
            self.logger.warning(f"Error getting logs for job {job_id}: {e}")
            return None
        except Exception as exc:
            self.logger.exception(f"Unexpected error getting logs for job {job_id}: {exc}")
            return None

    async def _get_pod_metadata(self, project_id: str, trajectory_id: str) -> dict | None:
        """Get metadata from the pod associated with a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            Dictionary with pod metadata or None if not found
        """
        if not project_id or not trajectory_id:
            return None

        job_id = self._job_id(project_id, trajectory_id)

        try:
            # Get pods associated with the job
            pod_list = self.core_v1.list_namespaced_pod(namespace=self.namespace, label_selector=f"job-name={job_id}")

            # If no pods found, return None
            if not pod_list.items:
                return None

            # Get metadata from the first pod
            pod = pod_list.items[0]

            metadata = {
                "name": pod.metadata.name,
                "phase": pod.status.phase,
                "start_time": pod.status.start_time.isoformat() if pod.status.start_time else None,
            }

            # Add container status information if available
            if pod.status.container_statuses:
                container_status = pod.status.container_statuses[0]

                # Add state information
                if container_status.state.running:
                    metadata["container_state"] = "running"
                elif container_status.state.terminated:
                    metadata["container_state"] = "terminated"
                    metadata["exit_code"] = container_status.state.terminated.exit_code
                    metadata["reason"] = container_status.state.terminated.reason
                elif container_status.state.waiting:
                    metadata["container_state"] = "waiting"
                    metadata["reason"] = container_status.state.waiting.reason

                # Add readiness and restart count
                metadata["ready"] = container_status.ready
                metadata["restart_count"] = container_status.restart_count

            return metadata
        except Exception as exc:
            self.logger.warning(f"Error getting pod metadata for job {job_id}: {exc}")
            return None

    def _get_job_error(self, job) -> str | None:
        """Extract error information from a failed job.

        Args:
            job: Kubernetes job object

        Returns:
            Error message if available, None otherwise
        """
        if not job.status.failed:
            return None

        error_message = "Job failed"

        # Check for error conditions in job status
        if job.status.conditions:
            for condition in job.status.conditions:
                if condition.type == "Failed" and condition.message:
                    error_message = condition.message
                    break

        try:
            # Get more detailed error from the pod
            job_id = job.metadata.name
            namespace = job.metadata.namespace or self.namespace

            # Get pods associated with the job
            pod_list = self.core_v1.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={job_id}")

            # If no pods found, return basic error message
            if not pod_list.items:
                return error_message

            # Get error information from the pod
            pod = pod_list.items[0]

            # Add pod phase information
            if pod.status.phase in ["Failed", "Error"]:
                error_message = f"{error_message} (Pod {pod.status.phase})"

            # If pod has a status message, add it
            if pod.status.message:
                error_message = f"{error_message}: {pod.status.message}"

            # Check container status for more detailed error
            if pod.status.container_statuses:
                for container_status in pod.status.container_statuses:
                    if container_status.state.terminated and container_status.state.terminated.exit_code != 0:
                        exit_code = container_status.state.terminated.exit_code
                        reason = container_status.state.terminated.reason or ""
                        message = container_status.state.terminated.message or ""

                        # Create detailed error message
                        container_error = f"Container exited with code {exit_code}"
                        if reason:
                            container_error += f": {reason}"
                        if message:
                            container_error += f" - {message}"

                        # Only replace the error message if we have useful details
                        if reason or message:
                            error_message = container_error
                        break

            return error_message
        except Exception as exc:
            self.logger.warning(f"Error getting detailed job error for {job.metadata.name}: {exc}")
            return error_message

    def _create_env_vars(self, project_id: str, trajectory_id: str, func_name: str, otel_context: dict = None) -> list:
        """Create environment variables for the pod.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            func_name: Function name to execute
            otel_context: OpenTelemetry context

        Returns:
            List of environment variables
        """
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
                client.V1EnvVar(name="REDIS_URL", value=os.environ.get("REDIS_URL")),
                client.V1EnvVar(name="LITELLM_LOCAL_MODEL_COST_MAP", value="True"),
            ]
        )

        return env_vars

    def _create_volumes_and_mounts(self) -> tuple:
        """Create volumes and volume mounts for the pod.

        Returns:
            Tuple of (volumes, volume_mounts)
        """
        volumes = [
            client.V1Volume(name="nltk-data", empty_dir=client.V1EmptyDirVolumeSource()),
            client.V1Volume(name="moatless-components", empty_dir=client.V1EmptyDirVolumeSource()),
        ]

        # Create volume mounts
        volume_mounts = [
            client.V1VolumeMount(name="moatless-components", mount_path="/opt/components"),
            client.V1VolumeMount(name="nltk-data", mount_path="/data/nltk_data"),
        ]

        return volumes, volume_mounts

    def _create_tolerations(self) -> list:
        """Create tolerations for the pod.

        Returns:
            List of tolerations
        """

        tolerations = []

        if self.kubernetes_provider == "azure":
            tolerations.append(
                client.V1Toleration(
                    key="kubernetes.azure.com/scalesetpriority",
                    operator="Equal",
                    value="spot",
                    effect="NoSchedule",
                )
            )

        return tolerations

    def _create_job_object(
        self,
        job_id: str,
        project_id: str,
        trajectory_id: str,
        job_func: Callable,
        otel_context: dict = None,
        node_id: int = None,
    ) -> client.V1Job:
        """Create a Kubernetes Job object.

        Args:
            job_id: Unique ID for the job
            project_id: Project ID
            trajectory_id: Trajectory ID
            job_func: Fully qualified function name to execute
            otel_context: OpenTelemetry context for distributed tracing
            node_id: Optional node ID for job placement

        Returns:
            Kubernetes Job object
        """
        func_name = job_func.__name__
        env_vars = self._create_env_vars(project_id, trajectory_id, func_name, otel_context)
        volumes, volume_mounts = self._create_volumes_and_mounts()
        tolerations = self._create_tolerations()

        args = create_job_args(project_id, trajectory_id, job_func, node_id)

        container = client.V1Container(
            name="worker",
            image=self._get_image_name(trajectory_id),
            image_pull_policy="IfNotPresent",
            command=["bash"],
            args=[
                "-c",
                f"echo 'MOATLESS: Starting job for {project_id}/{trajectory_id}' && export PYTHONUNBUFFERED=1 && uv run --no-sync -  <<EOF 2>&1\n{args}\nEOF",
            ],
            env=env_vars,
            env_from=[
                client.V1EnvFromSource(
                    config_map_ref=client.V1ConfigMapEnvSource(
                        name="moatless-tools-env",
                    )
                ),
                client.V1EnvFromSource(
                    secret_ref=client.V1SecretEnvSource(
                        name="moatless-tools-secrets",
                    )
                ),
            ],
            volume_mounts=volume_mounts,
            resources=client.V1ResourceRequirements(
                requests={"cpu": "500m", "memory": "512Mi", "ephemeral-storage": "1Gi"},
                limits={"cpu": "1000m", "memory": "2Gi", "ephemeral-storage": "5Gi"},
            ),
            working_dir="/opt/moatless",
        )

        # Add OpenTelemetry context if available
        if otel_context:
            for key, value in otel_context.items():
                container.env.append(client.V1EnvVar(name=key, value=value))

        labels = create_labels(project_id, trajectory_id, func_name)
        annotations = create_annotations(project_id, trajectory_id, func_name)

        # Add annotations to prevent eviction by Karpenter
        annotations["karpenter.sh/do-not-evict"] = "true"
        annotations["karpenter.sh/do-not-disrupt"] = "true"

        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels=labels,
                annotations=annotations,
            ),
            spec=client.V1PodSpec(
                containers=[container],
                restart_policy="Never",
                volumes=volumes,
                service_account_name=self.service_account,
                node_selector=self._get_node_selector(node_id),
                tolerations=tolerations,
                priority_class_name="system-node-critical",  # Add high priority to prevent eviction
            ),
        )

        job_spec = client.V1JobSpec(
            template=pod_template,
            backoff_limit=2,
            ttl_seconds_after_finished=self.job_ttl_seconds,
            active_deadline_seconds=self.timeout_seconds,
        )

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

    def _get_node_selector(self, node_id: int = None) -> dict:
        """Get node selector to use for job placement.

        Args:
            node_id: Optional node ID for job placement

        Returns:
            Node selector dictionary
        """
        # Start with the base node selector
        node_selector = dict(self.node_selector) if self.node_selector else {}

        # If node_id is specified, add a selector for it
        if node_id is not None:
            node_selector["moatless.ai/node-id"] = str(node_id)

        return node_selector

    def _get_project_label(self, project_id: str) -> str:
        """Get a sanitized project ID for use in Kubernetes labels.

        Args:
            project_id: The project ID

        Returns:
            Sanitized project ID suitable for use as a Kubernetes label value
        """
        return get_project_label(project_id)

    def _get_trajectory_label(self, trajectory_id: str) -> str:
        """Get a sanitized trajectory ID for use in Kubernetes labels.

        Args:
            trajectory_id: The trajectory ID

        Returns:
            Sanitized trajectory ID suitable for use as a Kubernetes label value
        """
        return get_trajectory_label(trajectory_id)

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
