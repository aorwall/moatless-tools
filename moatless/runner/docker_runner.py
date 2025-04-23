import asyncio
import logging
import os
import platform
import subprocess
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Dict, Tuple, List, Deque, NamedTuple

from moatless.runner.label_utils import (
    create_job_args,
    sanitize_label,
    create_docker_label_args,
    create_labels,
    create_resource_id,
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


class DockerRunner(BaseRunner):
    """Runner for managing jobs with Docker."""

    def __init__(
        self,
        job_ttl_seconds: int = 3600,
        timeout_seconds: int = 3600,
        moatless_source_dir: Optional[str] = None,
        default_image_name: Optional[str] = None,
        update_on_start: bool = False,
        update_branch: str = "docker",
    ):
        """Initialize the runner with Docker configuration.

        Args:
            job_ttl_seconds: Time-to-live for completed jobs
            timeout_seconds: Timeout for jobs
            moatless_source_dir: Path to the moatless source code directory to mount
                                at /opt/moatless in the container. If None, no source
                                directory will be mounted.
            default_image_name: Default Docker image name to use for all jobs. If provided,
                              this overrides the standard image name generation.
            update_on_start: Whether to run the update-moatless.sh script when starting containers.
            update_branch: Git branch to use when running update-moatless.sh.
        """
        self.job_ttl_seconds = job_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)
        self.default_image_name = default_image_name
        self.update_on_start = update_on_start
        self.update_branch = update_branch

        # Get source directory - prefer explicitly passed parameter over environment variable
        self.moatless_source_dir = (
            moatless_source_dir
            or os.environ.get("MOATLESS_HOST_RUNNER_SOURCE_DIR")
            or os.environ.get("MOATLESS_RUNNER_MOUNT_SOURCE_DIR")
        )

        # Get components directory - prefer host path if running in container
        self.components_path = os.environ.get("MOATLESS_HOST_COMPONENTS_PATH") or os.environ.get(
            "MOATLESS_COMPONENTS_PATH"
        )

        # Get moatless data directory - prefer host path if running in container
        self.moatless_dir = os.environ.get("MOATLESS_HOST_DIR") or os.environ.get("MOATLESS_DIR")

        # Determine if running on ARM64 architecture
        self.is_arm64 = platform.machine().lower() in ["arm64", "aarch64"]

        logger.info(f"Docker runner initialized with:")
        logger.info(f"  - Source dir: {self.moatless_source_dir}")
        logger.info(f"  - Components path: {self.components_path}")
        logger.info(f"  - Moatless dir: {self.moatless_dir}")
        logger.info(f"  - Architecture: {'ARM64' if self.is_arm64 else 'AMD64'}")
        if self.update_on_start:
            logger.info(f"  - Update on start: True (branch: {self.update_branch})")

    async def start_job(
        self,
        project_id: str,
        trajectory_id: str,
        job_func: Callable,
        node_id: int | None = None,
        image_name: Optional[str] = None,
        update_on_start: Optional[bool] = None,
        update_branch: Optional[str] = None,
    ) -> bool:
        """Start a job in a Docker container.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run
            node_id: Optional node ID (ignored in DockerRunner)
            image_name: Optional image name, overrides default
            update_on_start: Whether to run update-moatless.sh when starting the container.
                             If None, uses the runner's default setting.
            update_branch: Git branch to use with update-moatless.sh script.
                           If None, uses the runner's default branch.

        Returns:
            True if the job was started successfully, False otherwise
        """
        self.logger.info(f"Starting job for {project_id}/{trajectory_id}")

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

        # Create container ID/name
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
                await self._cleanup_container(container_name)

            else:
                self.logger.info(f"Container {container_name} already exists with status {container_status}, skipping")
                return False

            self.logger.info(f"Container {container_name} already exists, not starting a new one")
            return False

        # Use the specified image or default
        if not image_name:
            image_name = self._get_image_name(trajectory_id)

        try:
            # Extract OpenTelemetry context for propagation
            otel_context = extract_trace_context()

            # Build the Docker image name
            if image_name:
                # Use the provided image name for this job
                image_name_to_use = image_name
            else:
                # Get image name from the _get_image_name method (which may use default_image_name)
                image_name_to_use = self._get_image_name(trajectory_id)

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
                "SKIP_CONDA_ACTIVATE=true",
                "INSTANCE_PATH=/data/instance.json",
                "VIRTUAL_ENV=",  # Empty value tells UV to ignore virtual environments
                "UV_NO_VENV=1",  # Tell UV explicitly not to use venv
            ]

            # Add API key environment variables from current environment
            for key, value in os.environ.items():
                if (
                    key.endswith("API_KEY")
                    or key.startswith("AWS_")
                    or key.startswith("GCP_")
                    or key.startswith("AZURE_")
                ):
                    env_vars.append(f"{key}={value}")

            # Add OpenTelemetry context if available
            if otel_context:
                for key, value in otel_context.items():
                    env_vars.append(f"OTEL_{key}={value}")

            # Create command to run Docker container
            cmd = ["docker", "run", "--name", container_name, "-d"]

            # Add platform flag if running on ARM64 architecture
            if self.is_arm64:
                cmd.extend(["--platform=linux/amd64"])

            # Add Docker labels for easier container identification and querying
            job_labels = create_labels(project_id, trajectory_id)
            cmd.extend(create_docker_label_args(job_labels))

            # Add environment variables
            for env_var in env_vars:
                cmd.extend(["-e", env_var])

            # Add volume mounts for data directory only if it exists
            if self.moatless_dir:
                cmd.extend(["-v", f"{self.moatless_dir}:/data/moatless"])

            # Add volume mounts for components
            if self.components_path:
                cmd.extend(["-v", f"{self.components_path}:/opt/components"])
                cmd.extend(["-e", "MOATLESS_COMPONENTS_PATH=/opt/components"])

            # Add volume mounts for source code
            if self.moatless_source_dir:
                logger.info(f"Mount {self.moatless_source_dir}:/opt/moatless")
                cmd.extend(["-v", f"{self.moatless_source_dir}:/opt/moatless"])
                cmd.extend(["-e", "PYTHONPATH=/opt/moatless:$PYTHONPATH"])

            args = create_job_args(project_id, trajectory_id, job_func, node_id)
            logger.info(f"Running command: {args}")

            # Determine the command to run in the container
            run_command = ""

            # Add update script if enabled
            should_update = self.update_on_start if update_on_start is None else update_on_start
            branch_to_use = self.update_branch if update_branch is None else update_branch

            if should_update:
                self.logger.info(f"Will run update-moatless.sh with branch {branch_to_use}")
                run_command += f"/opt/moatless/docker/update-moatless.sh --branch {branch_to_use} && "

            # Add the main job command
            run_command += f"uv run - <<EOF\n{args}\nEOF"

            cmd.extend([image_name_to_use, "sh", "-c", run_command])

            logger.info(f"Running Docker command: {' '.join(cmd)}")
            # Run the command
            stdout, returncode = await self._run_docker_command(*cmd)

            if returncode != 0:
                self.logger.error(f"Failed to start Docker container: {stdout}")
                return False

            container_id = stdout.strip()
            self.logger.info(f"Started Docker container with ID: {container_id}")

            return True

        except Exception as exc:
            self.logger.exception(f"Error starting job {project_id}-{trajectory_id}: {exc}")
            raise exc

    async def get_jobs(self, project_id: str | None = None) -> list[JobInfo]:
        """Get a list of jobs.

        Args:
            project_id: Optional project ID to filter jobs

        Returns:
            List of JobInfo objects
        """
        result = []

        try:
            # Build command to list containers
            cmd = [
                "docker",
                "ps",
                "-a",
                "--format",
                '{{.Names}}|{{.Label "project_id"}}|{{.Label "trajectory_id"}}|{{.State}}|{{.CreatedAt}}',
            ]

            # Add filter by project if specified
            if project_id:
                sanitized_project_id = sanitize_label(project_id)
                cmd.extend(["--filter", f"label=project_id={sanitized_project_id}"])

            # Add filter for Moatless-managed containers
            cmd.extend(["--filter", "label=moatless.managed=true"])

            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                self.logger.error(f"Error listing containers: {stderr.decode()}")
                return []

            # Parse output
            output = stdout.decode().strip()
            if not output:
                return []

            for line in output.split("\n"):
                parts = line.strip().split("|")
                if len(parts) < 5:
                    self.logger.warning(f"Invalid container info format: {line}")
                    continue

                container_name, container_project_id, container_trajectory_id, container_state, created_at = parts[:5]

                # Skip containers without proper project or trajectory ID
                if not container_project_id or not container_trajectory_id:
                    continue

                status = await self._get_container_status(container_name)

                # Skip containers with unknown status
                if status is None:
                    continue

                # Parse timestamp
                try:
                    created_datetime = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S %z")
                except ValueError:
                    # fallback to alternative format if the first one doesn't work
                    try:
                        created_datetime = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S %Z")
                    except ValueError:
                        self.logger.warning(f"Could not parse timestamp: {created_at}")
                        created_datetime = datetime.now()

                # TODO: Get trajectory id and project id from env vars

                # Create JobInfo object
                job_info = JobInfo(
                    id=f"{container_project_id}:{container_trajectory_id}",
                    project_id=container_project_id,
                    trajectory_id=container_trajectory_id,
                    status=status,
                    enqueued_at=created_datetime,
                    started_at=created_datetime,
                    ended_at=datetime.now()
                    if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]
                    else None,
                )

                result.append(job_info)

            return result

        except Exception as exc:
            self.logger.exception(f"Error getting jobs: {exc}")
            return []

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
                sanitized_project_id = sanitize_label(project_id)
                cmd = [
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    f"label=project_id={sanitized_project_id}",
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
                await self._cleanup_container(container_name)

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
        try:
            job_status = await self.get_job_status(project_id, trajectory_id)
            return job_status is not None

        except Exception as exc:
            self.logger.exception(f"Error checking if job exists: {exc}")
            return False

    async def get_job_status(self, project_id: str, trajectory_id: str) -> Optional[JobStatus]:
        """Get the status of a job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            JobStatus enum value representing the job status, or None if job doesn't exist
        """
        try:
            container_name = self._container_name(project_id, trajectory_id)
            return await self._get_container_status(container_name)

        except Exception as exc:
            self.logger.exception(f"Error getting job status: {exc}")
            return None

    async def _get_container_status(self, container_name: str) -> Optional[JobStatus]:
        try:
            inspect_cmd = [
                "docker",
                "inspect",
                "--format",
                '{{.State.Status}},{{.State.Running}},{{.State.ExitCode}},{{index .Config.Labels "project_id"}},{{index .Config.Labels "trajectory_id"}}',
                container_name,
            ]
            process = await asyncio.create_subprocess_exec(
                *inspect_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                container_info = stdout.decode().strip().split(",")
                if len(container_info) < 3:
                    self.logger.warning(
                        f"Unexpected output format from docker inspect command {' '.join(inspect_cmd)}: {stdout.decode()}"
                    )
                    return None

                status, running, exit_code = container_info[:3]

                return self._parse_container_status(status, running, exit_code)
            else:
                if "No such object" in stderr.decode():
                    # Container doesn't exist, return None
                    return None
                else:
                    self.logger.warning(
                        f"Error getting job status. Stderr: {stderr.decode()}. Stdout: {stdout.decode()}"
                    )
                    return None

        except Exception as exc:
            self.logger.exception(f"Error getting job status: {exc}")
            return None

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

            return RunnerInfo(runner_type="docker", status=RunnerStatus.RUNNING)

        except Exception as exc:
            self.logger.exception(f"Error checking if runner is up: {exc}")
            return RunnerInfo(runner_type="docker", status=RunnerStatus.ERROR, data={"error": str(exc)})

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
        return create_resource_id(project_id, trajectory_id, prefix="moatless")

    def _get_image_name(self, trajectory_id: str) -> str:
        """Get the Docker image name based on the trajectory ID.

        Args:
            trajectory_id: The trajectory ID

        Returns:
            Docker image name
        """
        # Use default image if specified
        if self.default_image_name:
            return self.default_image_name

        # Otherwise, use the standard image name generation logic
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

    def _parse_container_status(self, status: str, running: str, exit_code: str) -> JobStatus:
        """Parse Docker container status into JobStatus enum.

        Args:
            status: Container status string
            running: Whether container is running ("true" or "false")
            exit_code: Exit code as string

        Returns:
            JobStatus enum value
        """
        # Map Docker status to JobStatus
        if running.lower() == "true":
            return JobStatus.RUNNING
        elif status == "created":
            # Container is created but not yet started - consider it RUNNING
            return JobStatus.RUNNING
        elif status == "exited":
            # Container has exited
            if exit_code == "0":
                return JobStatus.COMPLETED
            else:
                return JobStatus.FAILED
        else:
            # Other statuses (paused, restarting, etc.) - map to appropriate JobStatus
            return JobStatus.RUNNING

    async def _cleanup_container(self, container_name: str) -> bool:
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

            return True

        except Exception as exc:
            self.logger.exception(f"Error stopping container {container_name}: {exc}")
            return False

    async def cleanup_job(self, project_id: str, trajectory_id: str):
        """Stop and remove the container associated with a specific job.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
        """
        container_name = self._container_name(project_id, trajectory_id)
        self.logger.info(f"Cleaning up container {container_name} for job {project_id}:{trajectory_id}")
        await self._cleanup_container(container_name)

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

            # If job status is None, return None
            if job_status is None:
                return None

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

    async def _get_running_container_count(self) -> int:
        """Get the actual count of running containers from Docker.

        Returns:
            Number of running containers managed by moatless
        """
        try:
            cmd = ["docker", "ps", "-q", "--filter", "label=moatless.managed=true"]
            stdout, return_code = await self._run_docker_command(*cmd)

            if return_code != 0:
                self.logger.error(f"Failed to get running container count: {stdout}")
                return 0

            container_ids = [id for id in stdout.strip().split("\n") if id]
            return len(container_ids)
        except Exception as exc:
            self.logger.exception(f"Error getting running container count: {exc}")
            return 0
