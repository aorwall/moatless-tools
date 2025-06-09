import asyncio
import logging
import os
import subprocess
import aiofiles
from typing import Dict, Optional

from moatless.environment.base import BaseEnvironment, EnvironmentExecutionError

logger = logging.getLogger(__name__)


class LocalBashEnvironment(BaseEnvironment):
    """
    A concrete implementation of BaseEnvironment that executes commands
    on the local machine using bash.
    """

    def __init__(self, cwd: str | None = None, env: dict[str, str] | None = None, shell: bool = True):
        """
        Initialize the LocalBashEnvironment.

        Args:
            cwd: Current working directory for the command
            env: Environment variables for the command
            shell: Whether to run the command in a shell
        """
        self.cwd = cwd
        self.env = env
        self.shell = shell

    async def execute(self, command: str, fail_on_error: bool = False, patch: str | None = None) -> str:
        """
        Execute a command on the local machine.

        Args:
            command: The command to execute
            fail_on_error: Whether to raise an exception if the command returns a non-zero exit code

        Returns:
            The standard output of the command as a string

        Raises:
            EnvironmentExecutionError: If the command execution fails
        """
        try:
            logger.info(f"$ {command}")
            # Prepare environment variables
            process_env = os.environ.copy()
            if self.env:
                process_env.update(self.env)

            if self.shell:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self.cwd,
                    env=process_env,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *command.split(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self.cwd,
                    env=process_env,
                )

            stdout, _ = await process.communicate()  # TODO: Combine stdout and stderr
            output = stdout.decode()
            return_code = process.returncode or 0  # Ensure return_code is never None

            if output.strip():
                logger.info("Command output:")
                logger.info(output[200:])

            if return_code != 0:
                logger.warning(f"Command return code {return_code}")
                if fail_on_error:
                    raise EnvironmentExecutionError(
                        f"Command failed with return code {return_code}: {command}", return_code, output
                    )

            return output

        except asyncio.CancelledError:
            # Handle cancellation
            raise
        except EnvironmentExecutionError:
            # Re-raise EnvironmentExecutionError
            raise
        except Exception as e:
            # Fallback to synchronous execution if async fails
            logger.warning(f"Async execution failed, falling back to sync: {str(e)}")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=self.env,
                shell=self.shell,
                universal_newlines=True,
            )

            stdout, stderr = process.communicate()
            return_code = process.returncode

            if return_code != 0 and fail_on_error:
                raise EnvironmentExecutionError(
                    f"Command failed with return code {return_code}: {command}", return_code, stderr
                )

            return stdout

    async def read_file(self, path: str) -> str:
        """
        Read a file from the local filesystem.

        Args:
            path: The path to the file

        Returns:
            The contents of the file as a string

        Raises:
            FileNotFoundError: If the file does not exist
            EnvironmentExecutionError: If there's an error reading the file
        """
        try:
            # Build full path if cwd is set
            full_path = os.path.join(self.cwd, path) if self.cwd else path

            logger.info(f"Reading file: {full_path}")

            # Use aiofiles to read the file asynchronously
            async with aiofiles.open(full_path, mode="r") as file:
                content = await file.read()

            return content
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            raise EnvironmentExecutionError(f"Failed to read file: {path}", -1, str(e))

    async def write_file(self, path: str, content: str) -> None:
        """
        Write content to a file in the local filesystem.

        Args:
            path: The path to the file
            content: The content to write to the file

        Raises:
            EnvironmentExecutionError: If there's an error writing to the file
        """
        try:
            # Build full path if cwd is set
            full_path = os.path.join(self.cwd, path) if self.cwd else path

            logger.info(f"Writing to file: {full_path}")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Use aiofiles to write to the file asynchronously
            async with aiofiles.open(full_path, mode="w") as file:
                await file.write(content)

        except Exception as e:
            logger.error(f"Error writing to file {path}: {str(e)}")
            raise EnvironmentExecutionError(f"Failed to write to file: {path}", -1, str(e))
