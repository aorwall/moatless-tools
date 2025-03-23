import logging
import os
import asyncio
import subprocess
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Dict, List

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


class DockerRunner(BaseRunner):
    """Runner for managing jobs with Docker."""

    def __init__(
        self,
        job_ttl_seconds: int = 3600,
        timeout_seconds: int = 3600,
        volume_mappings: List[str] = None,
        moatless_source_dir: str = None,
    ):
        """Initialize the runner with Docker configuration.

        Args:
            job_ttl_seconds: Time-to-live for completed jobs
            timeout_seconds: Timeout for jobs
            volume_mappings: List of volume mappings for the Docker container
                             in the format "host_path:container_path"
            moatless_source_dir: Path to the moatless source code directory to mount
                                at /opt/moatless in the container. If None, no source
                                directory will be mounted.
        """
        self.job_ttl_seconds = job_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)
        self.running_containers: Dict[str, Dict[str, Any]] = {}
        self.volume_mappings = volume_mappings or []
        self.moatless_source_dir = moatless_source_dir

    async def start_job(self, project_id: str, trajectory_id: str, job_func: Callable | str) -> bool:
        """Start a job as a Docker container.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run or a string with the fully qualified function name

        Returns:
            True if the job was scheduled successfully, False otherwise
        """
        try:
            # Determine fully qualified function name
            if isinstance(job_func, str):
                fully_qualified_name = job_func
                func_module = job_func.rsplit(".", 1)[0]
                func_name = job_func.rsplit(".", 1)[1]
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

            self.logger.info(f"Creating Docker container for function: {fully_qualified_name}")

            # Create container name from project and trajectory IDs
            container_name = self._container_name(project_id, trajectory_id)

            # Check if container already exists
            if await self._container_exists(container_name):
                self.logger.info(f"Container {container_name} already exists, skipping")
                return False

            # Extract OpenTelemetry context for propagation
            otel_context = extract_trace_context()

            # Build the Docker image name
            image_name = self._get_image_name(trajectory_id)

            # Prepare environment variables
            env_vars = [
                f"PROJECT_ID={project_id}",
                f"TRAJECTORY_ID={trajectory_id}",
                f"JOB_FUNC={fully_qualified_name}",
                "MOATLESS_DIR=/data/moatless",
                "MOATLESS_COMPONENTS_PATH=/opt/components",
                "NLTK_DATA=/data/nltk_data",
                "INDEX_STORE_DIR=/data/index_store",
                "REPO_DIR=/testbed",
                "INSTANCE_PATH=/data/instance.json",
            ]

            # Add API key environment variables from current environment
            for key, value in os.environ.items():
                if key.endswith("API_KEY") or key.startswith("AWS_"):
                    env_vars.append(f"{key}={value}")

            # Add OpenTelemetry context if available
            if otel_context:
                for key, value in otel_context.items():
                    env_vars.append(f"OTEL_{key}={value}")

            # Create command to run Docker container
            cmd = ["docker", "run", "--name", container_name, "-d"]

            # Add environment variables
            for env_var in env_vars:
                cmd.extend(["-e", env_var])

            # Add volume mappings
            for volume in self.volume_mappings:
                cmd.extend(["-v", volume])

            # Add moatless source directory mapping if provided
            if self.moatless_source_dir:
                cmd.extend(["-v", f"{self.moatless_source_dir}:/opt/moatless"])
                # Update Python path to include mounted source
                cmd.extend(["-e", "PYTHONPATH=/opt/moatless:$PYTHONPATH"])

            # Create Python command to import and call the function with project_id and trajectory_id
            python_command = (
                f"from {func_module} import {func_name}; import sys; {func_name}('{project_id}', '{trajectory_id}')"
            )

            # Run the command
            cmd.extend([image_name, "/usr/bin/python", "-c", python_command])

            # Run the container
            self.logger.info(f"Running Docker container with command: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                self.logger.error(f"Failed to start Docker container: {stderr.decode()}")
                return False

            container_id = stdout.decode().strip()
            self.logger.info(f"Started Docker container with ID: {container_id}")

            # Store container info
            self.running_containers[container_name] = {
                "id": container_id,
                "project_id": project_id,
                "trajectory_id": trajectory_id,
                "started_at": datetime.now(),
                "status": JobStatus.RUNNING,
            }

            return True

        except Exception as exc:
            self.logger.exception(f"Error starting job {project_id}-{trajectory_id}: {exc}")
            return False

    async def get_jobs(self, project_id: str | None = None) -> list[JobInfo]:
        """Get all jobs for a project.

        Args:
            project_id: The project ID

        Returns:
            List of JobInfo objects with job status information
        """
        result = []
        for container_name, container_info in self.running_containers.items():
            if project_id is None or container_info.get("project_id") == project_id:
                # Get container status
                status = await self._get_container_status(container_name)

                result.append(
                    JobInfo(
                        id=container_name,
                        status=status,
                        project_id=container_info.get("project_id"),
                        trajectory_id=container_info.get("trajectory_id"),
                        enqueued_at=container_info.get("started_at"),
                        started_at=container_info.get("started_at"),
                        ended_at=None,  # We don't track this yet
                        metadata={},
                    )
                )

        return result

    async def cancel_job(self, project_id: str, trajectory_id: str | None = None) -> None:
        """Cancel a job or all jobs for a project.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID. If None, cancels all jobs for the project.

        Returns:
            None
        """
        if trajectory_id is None:
            # Cancel all jobs for the project
            for container_name, container_info in list(self.running_containers.items()):
                if container_info.get("project_id") == project_id:
                    await self._stop_container(container_name)
        else:
            # Cancel a specific job
            container_name = self._container_name(project_id, trajectory_id)
            await self._stop_container(container_name)

    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists as a Docker container.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job exists, False otherwise
        """
        container_name = self._container_name(project_id, trajectory_id)
        return await self._container_exists(container_name)

    async def retry_job(self, project_id: str, trajectory_id: str) -> bool:
        """Retry a failed job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job was restarted successfully, False otherwise
        """
        # For now, we'll just return False as retry is not implemented
        return False

    async def get_job_status(self, project_id: str, trajectory_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobStatus enum value representing the job status
        """
        container_name = self._container_name(project_id, trajectory_id)
        return await self._get_container_status(container_name)

    async def get_runner_info(self) -> RunnerInfo:
        """Get information about the Docker runner.

        Returns:
            RunnerInfo object with runner status information
        """
        try:
            # Check if Docker is accessible
            process = await asyncio.create_subprocess_exec(
                "docker", "info", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return RunnerInfo(
                    runner_type="docker", status=RunnerStatus.ERROR, data={"error": "Docker is not accessible"}
                )

            # Count running containers
            running_containers = len(self.running_containers)

            return RunnerInfo(
                runner_type="docker", status=RunnerStatus.RUNNING, data={"running_containers": running_containers}
            )
        except Exception as exc:
            self.logger.exception(f"Error checking if runner is up: {exc}")
            return RunnerInfo(runner_type="docker", status=RunnerStatus.ERROR, data={"error": str(exc)})

    async def get_job_status_summary(self, project_id: str) -> JobsStatusSummary:
        """Get a summary of job statuses for a project.

        Args:
            project_id: The project ID

        Returns:
            JobsStatusSummary with counts and IDs of jobs in different states
        """
        summary = JobsStatusSummary(project_id=project_id)

        for container_name, container_info in self.running_containers.items():
            if container_info.get("project_id") == project_id:
                summary.total_jobs += 1
                status = await self._get_container_status(container_name)

                if status == JobStatus.PENDING:
                    summary.pending_jobs += 1
                    summary.job_ids["pending"].append(container_name)
                elif status == JobStatus.INITIALIZING:
                    summary.initializing_jobs += 1
                    summary.job_ids["initializing"].append(container_name)
                elif status == JobStatus.RUNNING:
                    summary.running_jobs += 1
                    summary.job_ids["running"].append(container_name)
                elif status == JobStatus.COMPLETED:
                    summary.completed_jobs += 1
                    summary.job_ids["completed"].append(container_name)
                elif status == JobStatus.FAILED:
                    summary.failed_jobs += 1
                    summary.job_ids["failed"].append(container_name)
                elif status == JobStatus.CANCELED:
                    summary.canceled_jobs += 1
                    summary.job_ids["canceled"].append(container_name)

        return summary

    async def get_job_logs(self, project_id: str, trajectory_id: str) -> str | None:
        """Get logs from the container associated with a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            String containing the container logs, or None if no logs are available
        """
        container_name = self._container_name(project_id, trajectory_id)

        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "logs", container_name, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                self.logger.warning(f"Error getting logs for container {container_name}: {stderr.decode()}")
                return None

            logs = stdout.decode()
            return logs if logs else None

        except Exception as exc:
            self.logger.exception(f"Error getting logs for container {container_name}: {exc}")
            return None

    def _container_name(self, project_id: str, trajectory_id: str) -> str:
        """Create a container name from project ID and trajectory ID.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            Container name string
        """
        # Create a valid Docker container name by replacing invalid characters
        base_id = f"moatless-{project_id}-{trajectory_id}"

        # Replace invalid characters with dashes
        container_name = "".join(c if c.isalnum() or c == "-" else "-" for c in base_id.lower())

        # Ensure it's not too long (Docker has a character limit for names)
        if len(container_name) > 63:
            # Truncate but keep the start and end parts to maintain uniqueness
            prefix = container_name[:31]
            suffix = container_name[-31:]
            container_name = f"{prefix}-{suffix}"

        return container_name

    def _get_image_name(self, trajectory_id: str) -> str:
        """Get the Docker image name based on the trajectory ID.

        Args:
            trajectory_id: The trajectory ID

        Returns:
            Docker image name
        """
        instance_id_split = trajectory_id.split("__")
        repo_name = instance_id_split[0]
        instance_id = instance_id_split[1]
        return f"aorwall/sweb.eval.x86_64.{repo_name}_moatless_{instance_id}"

    async def _container_exists(self, container_name: str) -> bool:
        """Check if a container exists.

        Args:
            container_name: The container name

        Returns:
            True if the container exists, False otherwise
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "inspect", container_name, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            return process.returncode == 0

        except Exception:
            return False

    async def _get_container_status(self, container_name: str) -> JobStatus:
        """Get the status of a container.

        Args:
            container_name: The container name

        Returns:
            JobStatus enum value representing the container status
        """
        if not await self._container_exists(container_name):
            return JobStatus.NOT_FOUND

        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "inspect",
                "--format",
                "{{.State.Status}}",
                container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return JobStatus.NOT_FOUND

            container_status = stdout.decode().strip()

            if container_status == "running":
                return JobStatus.RUNNING
            elif container_status == "created":
                return JobStatus.PENDING
            elif container_status == "exited":
                # Check exit code to determine if completed or failed
                exit_code_process = await asyncio.create_subprocess_exec(
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.ExitCode}}",
                    container_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                exit_code_stdout, _ = await exit_code_process.communicate()
                exit_code = int(exit_code_stdout.decode().strip())

                return JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
            elif container_status == "removing":
                return JobStatus.CANCELED
            else:
                return JobStatus.INITIALIZING

        except Exception as exc:
            self.logger.exception(f"Error getting status for container {container_name}: {exc}")
            return JobStatus.NOT_FOUND

    async def _stop_container(self, container_name: str) -> bool:
        """Stop and remove a container.

        Args:
            container_name: The container name

        Returns:
            True if the container was stopped successfully, False otherwise
        """
        if not await self._container_exists(container_name):
            return True

        try:
            # Stop the container
            stop_process = await asyncio.create_subprocess_exec(
                "docker", "stop", container_name, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await stop_process.communicate()

            if stop_process.returncode != 0:
                self.logger.warning(f"Error stopping container {container_name}: {stderr.decode()}")
                return False

            # Remove the container
            rm_process = await asyncio.create_subprocess_exec(
                "docker", "rm", container_name, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await rm_process.communicate()

            if rm_process.returncode != 0:
                self.logger.warning(f"Error removing container {container_name}: {stderr.decode()}")
                return False

            # Remove from tracking dict
            if container_name in self.running_containers:
                del self.running_containers[container_name]

            return True

        except Exception as exc:
            self.logger.exception(f"Error stopping container {container_name}: {exc}")
            return False
