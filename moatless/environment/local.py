import asyncio
import logging
import os
import subprocess
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

    async def execute(self, command: str) -> str:
        """
        Execute a command on the local machine.

        Args:
            command: The command to execute

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
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.cwd,
                    env=process_env,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *command.split(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.cwd,
                    env=process_env,
                )

            stdout, stderr = await process.communicate()
            output = stdout.decode()
            error = stderr.decode()
            return_code = process.returncode

            if output.strip():
                logger.info("Command output:")
                logger.info(output[:200])
            if error.strip():
                logger.warning("Command stderr:")
                logger.warning(error[:200])

            if return_code != 0:
                logger.info(f"Command return code {return_code}")

            return output

        except asyncio.CancelledError:
            # Handle cancellation
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

            if return_code != 0:
                raise EnvironmentExecutionError(
                    f"Command failed with return code {return_code}: {command}", return_code, stderr
                )

            return stdout
