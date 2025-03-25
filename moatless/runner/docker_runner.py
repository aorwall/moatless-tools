import logging
import os
import asyncio
import subprocess
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Dict, List, Tuple

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
from moatless.runner.label_utils import create_job_args, sanitize_label, create_docker_label_args

logger = logging.getLogger(__name__)


class DockerRunner(BaseRunner):
    """Runner for managing jobs with Docker."""

    def __init__(
        self,
        job_ttl_seconds: int = 3600,
        timeout_seconds: int = 3600,
        moatless_source_dir: str = None,
    ):
        """Initialize the runner with Docker configuration.

        Args:
            job_ttl_seconds: Time-to-live for completed jobs
            timeout_seconds: Timeout for jobs
            moatless_source_dir: Path to the moatless source code directory to mount
                                at /opt/moatless in the container. If None, no source
                                directory will be mounted.
        """
        self.job_ttl_seconds = job_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)
        self.running_containers: Dict[str, Dict[str, Any]] = {}
        self.moatless_source_dir = moatless_source_dir or os.environ.get("MOATLESS_RUNNER_MOUNT_SOURCE_DIR")
        logger.info(f"Mounting moatless source dir: {self.moatless_source_dir}")

    async def start_job(
        self, project_id: str, trajectory_id: str, job_func: Callable, node_id: int | None = None
    ) -> bool:
        """Start a job as a Docker container.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run
            node_id: Optional node ID to pass to the job function

        Returns:
            True if the job was scheduled successfully, False otherwise
        """
        try:
            # Get function info
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
                # Check if the container is completed or failed
                container_status = await self._get_container_status(container_name)
                if container_status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    # Remove the existing container first
                    self.logger.info(
                        f"Container {container_name} exists with status {container_status}, removing and restarting"
                    )
                    await self._stop_container(container_name)
                else:
                    self.logger.info(
                        f"Container {container_name} already exists with status {container_status}, skipping"
                    )
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

            # Add Docker labels for easier container identification and querying
            job_labels = create_job_labels(project_id, trajectory_id)
            cmd.extend(create_docker_label_args(job_labels))

            # Add environment variables
            for env_var in env_vars:
                cmd.extend(["-e", env_var])

            if os.environ.get("MOATLESS_DIR"):
                cmd.extend(["-v", f"{os.environ['MOATLESS_DIR']}:/data/moatless"])

            if os.environ.get("MOATLESS_COMPONENTS_PATH"):
                cmd.extend(["-v", f"{os.environ['MOATLESS_COMPONENTS_PATH']}:/opt/components"])

            if self.moatless_source_dir:
                cmd.extend(["-v", f"{self.moatless_source_dir}:/opt/moatless"])
                cmd.extend(["-e", "PYTHONPATH=/opt/moatless:$PYTHONPATH"])

            args = create_job_args(project_id, trajectory_id, job_func, node_id)
            logger.info(f"Running command: {args}")
            cmd.extend([image_name, "/usr/bin/python", "-c", args])

            # Run the command
            stdout, returncode = await self._run_docker_command(*cmd)

            if returncode != 0:
                self.logger.error(f"Failed to start Docker container: {stdout}")
                return False

            container_id = stdout.strip()
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

        try:
            # Build the docker command to list containers with labels
            cmd = [
                "docker",
                "ps",
                "-a",
                "--format",
                '{{.Names}}|{{.Label "moatless.project_id"}}|{{.Label "moatless.trajectory_id"}}|{{.State}}|{{.Label "moatless.started_at"}}',
                "--filter",
                "label=moatless.managed=true",
            ]

            # Add project filter if specified
            if project_id is not None:
                cmd.extend(["--filter", f"label=moatless.project_id={project_id}"])

            # Run docker command
            stdout, return_code = await self._run_docker_command(*cmd)

            if return_code != 0:
                self.logger.error(f"Failed to list containers: {stdout}")
                return result

            # Parse the output and build job info objects
            lines = stdout.strip().split("\n")
            for line in lines:
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) < 4:
                    continue

                container_name = parts[0]
                container_project_id = parts[1]
                container_trajectory_id = parts[2]
                container_state = parts[3]
                started_at_str = parts[4] if len(parts) > 4 else None

                # Convert status string to JobStatus enum
                status = JobStatus.RUNNING
                if container_state == "running":
                    status = JobStatus.RUNNING
                elif container_state == "created":
                    status = JobStatus.PENDING
                elif container_state == "exited":
                    # Check exit code
                    exit_code = await self._get_container_exit_code(container_name)
                    status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
                elif container_state == "removing":
                    status = JobStatus.CANCELED
                else:
                    status = JobStatus.INITIALIZING

                # Parse started_at datetime if available
                started_at = None
                if started_at_str:
                    try:
                        started_at = datetime.fromisoformat(started_at_str)
                    except ValueError:
                        pass

                result.append(
                    JobInfo(
                        id=container_name,
                        status=status,
                        project_id=container_project_id,
                        trajectory_id=container_trajectory_id,
                        enqueued_at=started_at,
                        started_at=started_at,
                        ended_at=None,
                        metadata={},
                    )
                )

        except Exception as exc:
            self.logger.exception(f"Error listing jobs: {exc}")

        return result

    async def _get_container_exit_code(self, container_name: str) -> int:
        """Get the exit code of a container.

        Args:
            container_name: The container name

        Returns:
            Exit code as an integer, or -1 if an error occurs
        """
        try:
            output, return_code = await self._run_docker_command(
                "docker", "inspect", "--format", "{{.State.ExitCode}}", container_name
            )

            if return_code != 0:
                return -1

            return int(output.strip())
        except Exception:
            return -1

    async def cancel_job(self, project_id: str, trajectory_id: str | None = None) -> None:
        """Cancel a job or all jobs for a project.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID. If None, cancels all jobs for the project.

        Returns:
            None
        """
        try:
            if trajectory_id is None:
                # Cancel all jobs for the project
                cmd = [
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    f"label=moatless.project_id={project_id}",
                    "--filter",
                    "label=moatless.managed=true",
                ]

                stdout, return_code = await self._run_docker_command(*cmd)

                if return_code != 0 or not stdout:
                    self.logger.warning(f"No containers found for project {project_id}")
                    return

                container_ids = stdout.strip().split("\n")
                if container_ids:
                    # Stop all containers in one command
                    stop_cmd = ["docker", "stop"] + container_ids
                    await self._run_docker_command(*stop_cmd)

                    # Remove all containers in one command
                    rm_cmd = ["docker", "rm"] + container_ids
                    await self._run_docker_command(*rm_cmd)
            else:
                # Cancel a specific job
                container_name = self._container_name(project_id, trajectory_id)
                await self._stop_container(container_name)

        except Exception as exc:
            self.logger.exception(f"Error canceling job(s): {exc}")

    async def job_exists(self, project_id: str, trajectory_id: str) -> bool:
        """Check if a job exists as a Docker container.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            True if the job exists, False otherwise
        """
        container_name = self._container_name(project_id, trajectory_id)

        try:
            # Use Docker to check if the container exists
            cmd = ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return False

            output = stdout.decode().strip()
            return output != ""

        except Exception as exc:
            self.logger.exception(f"Error checking if job exists: {exc}")
            return False

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
            _, return_code = await self._run_docker_command("docker", "info")

            if return_code != 0:
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

        try:
            # Get all jobs for the project
            jobs = await self.get_jobs(project_id)

            summary.total_jobs = len(jobs)

            # Count jobs by status
            for job in jobs:
                if job.status == JobStatus.PENDING:
                    summary.pending_jobs += 1
                    summary.job_ids["pending"].append(job.id)
                elif job.status == JobStatus.INITIALIZING:
                    summary.initializing_jobs += 1
                    summary.job_ids["initializing"].append(job.id)
                elif job.status == JobStatus.RUNNING:
                    summary.running_jobs += 1
                    summary.job_ids["running"].append(job.id)
                elif job.status == JobStatus.COMPLETED:
                    summary.completed_jobs += 1
                    summary.job_ids["completed"].append(job.id)
                elif job.status == JobStatus.FAILED:
                    summary.failed_jobs += 1
                    summary.job_ids["failed"].append(job.id)
                elif job.status == JobStatus.CANCELED:
                    summary.canceled_jobs += 1
                    summary.job_ids["canceled"].append(job.id)

        except Exception as exc:
            self.logger.exception(f"Error getting job status summary: {exc}")

        return summary

    async def _run_docker_command(self, *args) -> Tuple[str, int]:
        """Run a Docker command and return stdout and return code.

        Args:
            *args: Command arguments to pass to Docker

        Returns:
            Tuple containing (stdout_text, return_code)
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )

            stdout, _ = await process.communicate()
            output = stdout.decode()

            return output, 0 if process.returncode is None else process.returncode
        except Exception as exc:
            self.logger.exception(f"Error executing Docker command: {exc}")
            return str(exc), 1

    async def get_job_logs(self, project_id: str, trajectory_id: str) -> str | None:
        """Get logs from the container associated with a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            String containing the container logs, or None if no logs are available
        """
        container_name = self._container_name(project_id, trajectory_id)
        logger.info(f"Getting logs for container: {container_name}")

        try:
            logs, return_code = await self._run_docker_command("docker", "logs", container_name)

            if return_code != 0:
                self.logger.warning(f"Error getting logs for container {container_name}")
                return None

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
        sanitized_project_id = sanitize_label(project_id.lower())
        sanitized_trajectory_id = sanitize_label(trajectory_id.lower())

        # Generate a hash of the combination of project_id and trajectory_id to ensure uniqueness
        import hashlib

        unique_hash = hashlib.md5(f"{project_id}:{trajectory_id}".encode()).hexdigest()[:6]

        # Reserve 8 chars for prefix ("moatless-"), 6 chars for hash, and 2 chars for separators
        reserved_length = 8 + 6 + 2
        max_total_id_length = 63 - reserved_length

        # Prioritize trajectory_id, then use remaining space for project_id
        if len(sanitized_trajectory_id) <= max_total_id_length:
            # Trajectory ID fits completely, use remaining space for project_id
            remaining_space = max_total_id_length - len(sanitized_trajectory_id)
            project_id_part = sanitized_project_id[:remaining_space] if remaining_space > 0 else ""
            trajectory_id_part = sanitized_trajectory_id
        else:
            # Trajectory ID is too long, truncate it
            trajectory_id_part = sanitized_trajectory_id[:max_total_id_length]
            project_id_part = ""

        # Format: moatless-{trajectory_id_part}-{project_id_part}-{hash}
        if project_id_part:
            container_name = f"moatless-{trajectory_id_part}-{project_id_part}-{unique_hash}"
        else:
            container_name = f"moatless-{trajectory_id_part}-{unique_hash}"

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
            _, return_code = await self._run_docker_command("docker", "inspect", container_name)
            return return_code == 0
        except Exception:
            return False

    async def _get_container_status(self, container_name: str) -> JobStatus:
        """Get the status of a container.

        Args:
            container_name: The container name

        Returns:
            JobStatus enum value representing the container status
        """
        logger.info(f"Getting status for container: {container_name}")
        if not await self._container_exists(container_name):
            return JobStatus.NOT_FOUND

        try:
            container_status_output, return_code = await self._run_docker_command(
                "docker", "inspect", "--format", "{{.State.Status}}", container_name
            )

            if return_code != 0:
                return JobStatus.NOT_FOUND

            container_status = container_status_output.strip()

            if container_status == "running":
                return JobStatus.RUNNING
            elif container_status == "created":
                return JobStatus.PENDING
            elif container_status == "exited":
                # Check exit code to determine if completed or failed
                exit_code_output, _ = await self._run_docker_command(
                    "docker", "inspect", "--format", "{{.State.ExitCode}}", container_name
                )
                exit_code = int(exit_code_output.strip())

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
            _, stop_return_code = await self._run_docker_command("docker", "stop", container_name)

            if stop_return_code != 0:
                self.logger.warning(f"Error stopping container {container_name}")
                return False

            # Remove the container
            _, rm_return_code = await self._run_docker_command("docker", "rm", container_name)

            if rm_return_code != 0:
                self.logger.warning(f"Error removing container {container_name}")
                return False

            # Remove from tracking dict
            if container_name in self.running_containers:
                del self.running_containers[container_name]

            return True

        except Exception as exc:
            self.logger.exception(f"Error stopping container {container_name}: {exc}")
            return False

    async def get_job_details(self, project_id: str, trajectory_id: str) -> Optional[JobDetails]:
        """Get detailed information about a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobDetails object containing detailed information about the job
        """
        container_name = self._container_name(project_id, trajectory_id)

        try:
            # Check if container exists
            if not await self._container_exists(container_name):
                return None

            # Get basic container info
            container_info, return_code = await self._run_docker_command("docker", "inspect", container_name)

            if return_code != 0:
                self.logger.warning(f"Error getting container info for {container_name}")
                return None

            # Parse container info as JSON
            import json

            container_data = json.loads(container_info)[0]

            # Get container status
            job_status = await self._get_container_status(container_name)

            # Get container logs
            logs = await self.get_job_logs(project_id, trajectory_id)

            # Create JobDetails object with structured sections
            job_details = JobDetails(
                id=container_name,
                status=job_status,
                project_id=project_id,
                trajectory_id=trajectory_id,
                sections=[],
                raw_data=container_data,
            )

            # Extract timestamps
            state_data = container_data.get("State", {})
            created_at = container_data.get("Created")
            started_at = state_data.get("StartedAt")
            finished_at = state_data.get("FinishedAt")

            # Parse ISO timestamps if available
            if created_at and created_at != "0001-01-01T00:00:00Z":
                job_details.enqueued_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if started_at and started_at != "0001-01-01T00:00:00Z":
                job_details.started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if (
                finished_at
                and finished_at != "0001-01-01T00:00:00Z"
                and finished_at != "0001-01-01T00:00:00.000000000Z"
            ):
                job_details.ended_at = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))

            # Extract container metadata from Labels
            labels = container_data.get("Config", {}).get("Labels", {})

            # Add Overview section
            overview_section = JobDetailSection(
                name="overview",
                display_name="Overview",
                data={
                    "container_id": container_data.get("Id", "")[:12],
                    "image": container_data.get("Config", {}).get("Image", ""),
                    "command": container_data.get("Config", {}).get("Cmd", []),
                    "created_at": created_at,
                    "status": state_data.get("Status", ""),
                    "platform": container_data.get("Platform", ""),
                    "driver": container_data.get("Driver", ""),
                },
            )
            job_details.sections.append(overview_section)

            # Add State section
            state_section = JobDetailSection(name="state", display_name="Container State", data=state_data)
            job_details.sections.append(state_section)

            # Add Environment section
            env_vars = container_data.get("Config", {}).get("Env", [])
            env_data = {}

            for env_var in env_vars:
                if "=" in env_var:
                    key, value = env_var.split("=", 1)
                    # Filter out sensitive data
                    if "API_KEY" in key or "PASSWORD" in key or "SECRET" in key:
                        value = "********"
                    env_data[key] = value

            env_section = JobDetailSection(name="environment", display_name="Environment", data=env_data)
            job_details.sections.append(env_section)

            # Add Labels section
            labels_section = JobDetailSection(name="labels", display_name="Labels", data=labels)
            job_details.sections.append(labels_section)

            # Add Networking section
            networks = container_data.get("NetworkSettings", {}).get("Networks", {})
            networking_data = {
                "ip_address": container_data.get("NetworkSettings", {}).get("IPAddress", ""),
                "gateway": container_data.get("NetworkSettings", {}).get("Gateway", ""),
                "ports": container_data.get("NetworkSettings", {}).get("Ports", {}),
                "networks": networks,
            }

            networking_section = JobDetailSection(name="networking", display_name="Networking", data=networking_data)
            job_details.sections.append(networking_section)

            # Add Mounts section if available
            mounts = container_data.get("Mounts", [])
            if mounts:
                mounts_section = JobDetailSection(name="mounts", display_name="Mounts", items=mounts)
                job_details.sections.append(mounts_section)

            # Add Logs section if available
            if logs:
                logs_section = JobDetailSection(name="logs", display_name="Logs", data={"logs": logs})
                job_details.sections.append(logs_section)

            # Add error information if job failed
            if job_status == JobStatus.FAILED:
                exit_code = state_data.get("ExitCode", 0)
                error_data = {
                    "exit_code": exit_code,
                    "error": state_data.get("Error", ""),
                    "oom_killed": state_data.get("OOMKilled", False),
                }

                job_details.error = f"Container exited with code {exit_code}" + (
                    f": {state_data.get('Error')}" if state_data.get("Error") else ""
                )

                error_section = JobDetailSection(name="error", display_name="Error", data=error_data)
                job_details.sections.append(error_section)

            return job_details

        except Exception as exc:
            self.logger.exception(f"Error getting job details for {container_name}: {exc}")
            return None
