import abc
from typing import Dict, Optional, Type


class EnvironmentExecutionError(Exception):
    """
    Exception raised when a command execution fails in an environment.
    """

    def __init__(self, message: str, return_code: int, stderr: str):
        self.return_code = return_code
        self.stderr = stderr
        super().__init__(message)


class BaseEnvironment(abc.ABC):
    """
    Abstract base class representing an execution environment.

    This class serves as an interface for different types of execution environments
    (local bash, remote SSH, container, etc.). It defines the common operations
    that all environments should support.
    """

    @abc.abstractmethod
    async def execute(self, command: str) -> str:
        """
        Execute a command in the environment.

        Args:
            command: The command to execute
            cwd: Current working directory for the command
            env: Environment variables for the command
            shell: Whether to run the command in a shell

        Returns:
            The standard output of the command as a string

        Raises:
            EnvironmentExecutionError: If the command execution fails
        """
        pass

    @staticmethod
    def get_default_environment() -> "BaseEnvironment":
        """
        Get the default environment implementation.

        This is useful for components that need an environment but don't want to
        force the caller to provide a specific implementation.

        Returns:
            The default BaseEnvironment implementation (LocalBashEnvironment)
        """
        # Import here to avoid circular imports
        from moatless.environment.local import LocalBashEnvironment

        return LocalBashEnvironment()
