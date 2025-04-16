import asyncio
import logging
import os
import platform
import subprocess
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Dict, Tuple, List, Deque, NamedTuple
from collections import deque

from moatless.runner.label_utils import create_job_args, sanitize_label, create_docker_label_args, create_labels, create_resource_id
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
        max_concurrent_containers: Optional[int] = None,
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
            max_concurrent_containers: Maximum number of containers that can run concurrently.
                                     If None, there is no limit.
        """
        self.job_ttl_seconds = job_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)
        self.running_containers: Dict[str, Dict[str, Any]] = {}
        self.default_image_name = default_image_name
        self.max_concurrent_containers = max_concurrent_containers

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

    async def start_job(
        self, project_id: str, trajectory_id: str, job_func: Callable, node_id: int | None = None, 
        image_name: Optional[str] = None
    ) -> bool:
        """Start a job in a Docker container.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            job_func: The function to run
            node_id: Optional node ID (ignored in DockerRunner)
            image_name: Optional image name, overrides default

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
        
        # Check if we already have a container running for this job
        exists = await self._container_exists(container_name)
        if exists:
            self.logger.info(f"Container {container_name} already exists, not starting a new one")
            return False
            
        # Use the specified image or default
        if not image_name:
            image_name = self._get_image_name(trajectory_id)
            
        # Record the job in our tracking dict
        job_key = f"{project_id}:{trajectory_id}"
        self.running_containers[job_key] = {
            "project_id": project_id,
            "trajectory_id": trajectory_id,
            "status": JobStatus.RUNNING,
            "started_at": datetime.now()
        }
        
        try:
            # Create the helper module for running functions in the container
            # ... existing code ...

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
                if key.endswith("API_KEY") or key.startswith("AWS_") or key.startswith("GCP_") or key.startswith("AZURE_"):
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
            cmd.extend([image_name_to_use, "sh", "-c", f"uv run - <<EOF\n{args}\nEOF"])

            logger.info(f"Running Docker command: {' '.join(cmd)}")
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
            cmd = ["docker", "ps", "-a", "--format", "{{.Names}}|{{.Label \"project_id\"}}|{{.Label \"trajectory_id\"}}|{{.State}}|{{.CreatedAt}}"]
            
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
                    
                # Determine status
                if container_state == "running":
                    status = JobStatus.RUNNING
                elif container_state == "created":
                    status = JobStatus.PENDING
                elif container_state == "exited":
                    # Get exit code to determine if completed or failed
                    try:
                        inspect_cmd = ["docker", "inspect", "--format", "{{.State.ExitCode}}", container_name]
                        inspect_process = await asyncio.create_subprocess_exec(
                            *inspect_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                        )
                        inspect_stdout, _ = await inspect_process.communicate()
                        
                        if inspect_process.returncode == 0:
                            exit_code = int(inspect_stdout.decode().strip())
                            
                            if exit_code == 0:
                                status = JobStatus.COMPLETED
                                # Clean up completed containers
                                await self._cleanup_container(container_name)
                            else:
                                status = JobStatus.FAILED
                                # Also clean up failed containers
                                await self._cleanup_container(container_name)
                        else:
                            status = JobStatus.FAILED
                    except Exception as e:
                        self.logger.error(f"Error getting exit code for {container_name}: {e}")
                        status = JobStatus.FAILED
                elif container_state == "removing":
                    status = JobStatus.CANCELED
                else:
                    status = JobStatus.RUNNING
                
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
                
                # Create JobInfo object
                job_info = JobInfo(
                    id=f"{container_project_id}:{container_trajectory_id}",
                    project_id=container_project_id,
                    trajectory_id=container_trajectory_id,
                    status=status,
                    enqueued_at=created_datetime,
                    started_at=created_datetime,
                    ended_at=datetime.now() if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED] else None
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
            # Sanitize the project and trajectory IDs for use as Docker labels
            sanitized_project_id = sanitize_label(project_id)
            sanitized_trajectory_id = sanitize_label(trajectory_id)
            
            # Use Docker to check if the container exists by project_id and trajectory_id labels
            # This is more reliable than filtering by container name
            cmd = [
                "docker", "ps", "-a", 
                "--filter", f"label=project_id={sanitized_project_id}", 
                "--filter", f"label=trajectory_id={sanitized_trajectory_id}",
                "--filter", "label=moatless.managed=true",
                "--format", "{{.Names}}"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                self.logger.warning(f"Docker command error: {stderr.decode()}")
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
        try:
            # First, try the direct approach by checking if the container exists by name
            container_name = self._container_name(project_id, trajectory_id)
            
            # Check if the container exists by name first
            inspect_cmd = ["docker", "inspect", "--format", "{{.State.Status}},{{.State.ExitCode}}", container_name]
            process = await asyncio.create_subprocess_exec(
                *inspect_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Container exists, get its status
                container_info = stdout.decode().strip().split(",")
                if len(container_info) >= 2:
                    state, exit_code_str = container_info[:2]
                    
                    # Map Docker status to JobStatus
                    if state == "running":
                        return JobStatus.RUNNING
                    elif state == "created":
                        return JobStatus.PENDING
                    elif state == "exited":
                        try:
                            exit_code = int(exit_code_str)
                            if exit_code == 0:
                                return JobStatus.COMPLETED
                            else:
                                return JobStatus.FAILED
                        except (ValueError, TypeError):
                            self.logger.warning(f"Invalid exit code: {exit_code_str}")
                            return JobStatus.FAILED
                    elif state == "removing":
                        return JobStatus.CANCELED
                    else:
                        # Other states (paused, restarting, etc.)
                        return JobStatus.RUNNING
            
            # If not found by name, try using the labels as a backup
            # Sanitize IDs for use in labels
            sanitized_project_id = sanitize_label(project_id)
            sanitized_trajectory_id = sanitize_label(trajectory_id)
            
            # Check if any container exists with these labels
            cmd = [
                "docker", "ps", "-a",
                "--filter", f"label=project_id={sanitized_project_id}",
                "--filter", f"label=trajectory_id={sanitized_trajectory_id}",
                "--filter", "label=moatless.managed=true",
                "--format", "{{.Names}}"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0 or not stdout:
                self.logger.debug(f"No containers found for {project_id}/{trajectory_id}")
                return JobStatus.NOT_STARTED
            
            # Get the container name from the output
            found_container_name = stdout.decode().strip().split("\n")[0]
            
            # Now get the status of this specific container
            status_cmd = ["docker", "inspect", "--format", "{{.State.Status}},{{.State.ExitCode}}", found_container_name]
            status_process = await asyncio.create_subprocess_exec(
                *status_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            status_stdout, status_stderr = await status_process.communicate()
            
            if status_process.returncode != 0:
                if b"No such object" in status_stderr:
                    # Container was removed
                    self.logger.debug(f"Container {found_container_name} was removed")
                    return JobStatus.NOT_STARTED
                self.logger.warning(f"Error getting container status: {status_stderr.decode()}")
                return JobStatus.NOT_STARTED
            
            # Parse the status output
            status_output = status_stdout.decode().strip()
            status_parts = status_output.split(",")
            
            if len(status_parts) < 2:
                self.logger.warning(f"Unexpected status format: {status_output}")
                return JobStatus.NOT_STARTED
            
            state, exit_code_str = status_parts
            
            # Map Docker status to JobStatus
            if state == "running":
                return JobStatus.RUNNING
            elif state == "created":
                return JobStatus.PENDING
            elif state == "exited":
                try:
                    exit_code = int(exit_code_str)
                    if exit_code == 0:
                        return JobStatus.COMPLETED
                    else:
                        return JobStatus.FAILED
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid exit code: {exit_code_str}")
                    return JobStatus.FAILED
            elif state == "removing":
                return JobStatus.CANCELED
            else:
                # Other states (paused, restarting, etc.)
                return JobStatus.RUNNING
        
        except Exception as exc:
            self.logger.exception(f"Error getting job status: {exc}")
            return JobStatus.NOT_STARTED

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
                runner_type="docker", 
                status=RunnerStatus.RUNNING, 
                data={
                    "running_containers": running_containers,
                }
            )
        except Exception as exc:
            self.logger.exception(f"Error checking if runner is up: {exc}")
            return RunnerInfo(runner_type="docker", status=RunnerStatus.ERROR, data={"error": str(exc)})

    async def get_queue_size(self) -> int:
        """Get the current size of the job queue.
        
        Since we no longer have job queueing, this always returns 0.
        
        Returns:
            Always 0 since we no longer queue jobs
        """
        return 0

    def get_available_slots(self) -> int:
        """Get the number of available container slots.
        
        Since we no longer limit the number of containers, this returns a large number.
        
        Returns:
            Always returns a large number since we don't limit containers anymore
        """
        return 1000  # Return a large number as we don't limit container count

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

    async def _get_container_status(self, container_name: str) -> JobStatus:
        """Get the status of a Docker container.

        Args:
            container_name: Name of the container

        Returns:
            JobStatus enum value representing the container status
        """
        try:
            # Extract project_id and trajectory_id from container labels
            cmd = [
                "docker", "inspect",
                "--format", "{{.State.Status}},{{.State.Running}},{{.State.ExitCode}},{{index .Config.Labels \"project_id\"}},{{index .Config.Labels \"trajectory_id\"}}",
                container_name
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            # If container not found by name, try searching by labels
            if process.returncode != 0:
                if b"No such container" in stderr:
                    # Try to find container by project_id and trajectory_id from the container name
                    # Get project_id and trajectory_id from container_name if possible
                    parts = container_name.split("-")
                    if len(parts) >= 3:
                        # Format is often moatless-trajectory_id-project_id-hash or moatless-trajectory_id-hash
                        try:
                            # Get alternative containers with matching project_id and trajectory_id from labels
                            find_cmd = [
                                "docker", "ps", "-a", "--format", 
                                "{{.Names}},{{.State.Status}},{{.State.Running}},{{.State.ExitCode}}",
                                "--filter", "label=moatless.managed=true"
                            ]
                            
                            # If we have the original container name in our tracking dict, use its info
                            for key, container in self.running_containers.items():
                                project_parts = key.split(":")
                                if len(project_parts) == 2:
                                    proj_id, traj_id = project_parts
                                    # Sanitize the IDs for use as Docker labels
                                    sanitized_proj_id = sanitize_label(proj_id)
                                    sanitized_traj_id = sanitize_label(traj_id)
                                    find_cmd.extend([
                                        "--filter", f"label=project_id={sanitized_proj_id}", 
                                        "--filter", f"label=trajectory_id={sanitized_traj_id}"
                                    ])
                                    break
                                    
                            find_process = await asyncio.create_subprocess_exec(
                                *find_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                            )
                            find_stdout, _ = await find_process.communicate()
                            find_output = find_stdout.decode().strip()
                            
                            if find_output:
                                # Use the first matching container's status
                                container_info = find_output.split("\n")[0].split(",")
                                if len(container_info) >= 4:
                                    _, status, running, exit_code = container_info[:4]
                                    # Update our container name for future reference
                                    self.logger.info(f"Found alternative container for {container_name}: {container_info[0]}")
                                    
                                    # Continue processing with this info
                                    return self._parse_container_status(status, running, exit_code)
                        except Exception as e:
                            self.logger.warning(f"Error searching for alternative container: {e}")
                            
                self.logger.warning(f"Error getting container status: {stderr.decode()}")
                return JobStatus.NOT_STARTED
                
            # Parse the output
            container_info = stdout.decode().strip().split(",")
            if len(container_info) < 3:
                self.logger.warning(f"Unexpected output format from docker inspect: {stdout.decode()}")
                return JobStatus.NOT_STARTED
                
            status, running, exit_code = container_info[:3]
            
            return self._parse_container_status(status, running, exit_code)
            
        except Exception as exc:
            self.logger.exception(f"Error getting container status: {exc}")
            return JobStatus.FAILED
        
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

            # Remove from tracking dict
            if container_name in self.running_containers:
                del self.running_containers[container_name]

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
