import asyncio
import logging
import os
from typing import Optional

from moatless.environment.base import BaseEnvironment, EnvironmentExecutionError

logger = logging.getLogger(__name__)


class DockerEnvironment(BaseEnvironment):
    """
    A concrete implementation of BaseEnvironment that executes commands
    in a Docker container.
    """

    def __init__(
        self,
        image: str,
        container_name: Optional[str] = None,
        working_dir: str = "/tmp",
        env: Optional[dict[str, str]] = None,
        auto_remove: bool = True,
        volumes: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the DockerEnvironment.

        Args:
            image: Docker image to use for the container
            container_name: Name for the container (if None, Docker will generate one)
            working_dir: Working directory inside the container
            env: Environment variables for the container
            auto_remove: Whether to automatically remove the container when it exits
            volumes: Host to container volume mappings
        """
        self.image = image
        self.container_name = container_name
        self.working_dir = working_dir
        self.env = env or {}
        self.auto_remove = auto_remove
        self.volumes = volumes or {}
        self._container_id: Optional[str] = None
        self._container_running = False

    async def _ensure_container_running(self) -> str:
        """
        Ensure the container is running and return its ID.
        
        Returns:
            The container ID
        """
        if self._container_running and self._container_id:
            # Check if container is still running
            try:
                result = await self._run_docker_command(f"docker ps -q --filter id={self._container_id}")
                if result.strip():
                    return self._container_id
                else:
                    self._container_running = False
                    self._container_id = None
            except Exception:
                self._container_running = False
                self._container_id = None

        # Start new container
        await self._start_container()
        return self._container_id

    async def _start_container(self) -> None:
        """Start a new Docker container."""
        docker_cmd = ["docker", "run", "-d"]
        
        # Add working directory
        docker_cmd.extend(["-w", self.working_dir])
        
        # Add environment variables
        for key, value in self.env.items():
            docker_cmd.extend(["-e", f"{key}={value}"])
        
        # Add volumes
        for host_path, container_path in self.volumes.items():
            docker_cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        # Add container name if specified
        if self.container_name:
            docker_cmd.extend(["--name", self.container_name])
        
        # Add auto-remove flag
        if self.auto_remove:
            docker_cmd.append("--rm")
        
        # Add image and keep container alive
        docker_cmd.extend([self.image, "sleep", "infinity"])
        
        # Execute docker run command
        command = " ".join(docker_cmd)
        logger.info(f"Starting container: {command}")
        
        result = await self._run_docker_command(command)
        self._container_id = result.strip()
        self._container_running = True
        
        logger.info(f"Container started with ID: {self._container_id}")

    async def _run_docker_command(self, command: str) -> str:
        """
        Run a Docker command on the host.
        
        Args:
            command: The Docker command to run
            
        Returns:
            The output of the command
            
        Raises:
            EnvironmentExecutionError: If the command fails
        """
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            return_code = process.returncode or 0
            
            stdout_str = stdout.decode()
            stderr_str = stderr.decode()
            
            if return_code != 0:
                logger.error(f"Docker command failed: {command}")
                logger.error(f"Return code: {return_code}")
                logger.error(f"Stderr: {stderr_str}")
                raise EnvironmentExecutionError(
                    f"Docker command failed: {command}", return_code, stderr_str
                )
            
            return stdout_str
            
        except Exception as e:
            if isinstance(e, EnvironmentExecutionError):
                raise
            logger.error(f"Failed to execute Docker command: {command}, error: {str(e)}")
            raise EnvironmentExecutionError(f"Failed to execute Docker command: {command}", -1, str(e))

    async def execute(self, command: str, fail_on_error: bool = False, patch: str | None = None) -> str:
        """
        Execute a command in the Docker container.

        Args:
            command: The command to execute
            fail_on_error: Whether to raise an exception if the command returns a non-zero exit code
            patch: Unused in this implementation

        Returns:
            The standard output of the command as a string

        Raises:
            EnvironmentExecutionError: If the command execution fails
        """
        container_id = await self._ensure_container_running()
        
        # Escape the command for docker exec
        escaped_command = command.replace('"', '\\"')
        docker_exec_cmd = f'docker exec {container_id} bash -c "{escaped_command}"'
        
        try:
            logger.info(f"Executing in container {container_id[:12]}: {command}")
            
            process = await asyncio.create_subprocess_shell(
                docker_exec_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            return_code = process.returncode or 0
            
            stdout_str = stdout.decode()
            stderr_str = stderr.decode()
            
            # Combine stdout and stderr for output (similar to local implementation)
            output = stdout_str
            if stderr_str:
                output += "\n" + stderr_str
            
            if output.strip():
                logger.info("Command output:")
                logger.info(output[:200])
            
            if return_code != 0:
                logger.warning(f"Command return code {return_code}")
                if fail_on_error:
                    raise EnvironmentExecutionError(
                        f"Command failed with return code {return_code}: {command}", return_code, stderr_str
                    )
            
            return output
            
        except asyncio.CancelledError:
            raise
        except EnvironmentExecutionError:
            raise
        except Exception as e:
            logger.error(f"Failed to execute command in container: {command}, error: {str(e)}")
            if fail_on_error:
                raise EnvironmentExecutionError(f"Failed to execute command: {command}", -1, str(e))
            return str(e)

    async def read_file(self, path: str) -> str:
        """
        Read a file from the Docker container.

        Args:
            path: The path to the file

        Returns:
            The contents of the file as a string

        Raises:
            FileNotFoundError: If the file does not exist
            EnvironmentExecutionError: If there's an error reading the file
        """
        container_id = await self._ensure_container_running()
        
        # Use docker exec to cat the file
        command = f"cat {path}"
        
        try:
            output = await self.execute(command, fail_on_error=True)
            return output
        except EnvironmentExecutionError as e:
            if "No such file or directory" in e.stderr:
                raise FileNotFoundError(f"File not found: {path}")
            raise EnvironmentExecutionError(f"Failed to read file: {path}", e.return_code, e.stderr)

    async def write_file(self, path: str, content: str) -> None:
        """
        Write content to a file in the Docker container.

        Args:
            path: The path to the file
            content: The content to write to the file

        Raises:
            EnvironmentExecutionError: If there's an error writing to the file
        """
        await self._ensure_container_running()
        
        try:
            # Ensure the directory exists in the container
            dir_path = os.path.dirname(path)
            if dir_path:
                await self.execute(f"mkdir -p {dir_path}")
            
            # Use base64 encoding to safely write content with special characters
            import base64
            encoded_content = base64.b64encode(content.encode()).decode()
            command = f"echo '{encoded_content}' | base64 -d > {path}"
            
            await self.execute(command, fail_on_error=True)
            
        except Exception as e:
            logger.error(f"Error writing to file {path}: {str(e)}")
            raise EnvironmentExecutionError(f"Failed to write to file: {path}", -1, str(e))

    async def cleanup(self) -> None:
        """
        Clean up the Docker container.
        """
        if self._container_id and self._container_running:
            try:
                logger.info(f"Stopping container {self._container_id}")
                await self._run_docker_command(f"docker stop {self._container_id}")
                self._container_running = False
            except Exception as e:
                logger.warning(f"Failed to stop container {self._container_id}: {str(e)}")
            
            self._container_id = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()