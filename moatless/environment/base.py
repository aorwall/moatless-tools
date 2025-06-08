import abc


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
    async def execute(self, command: str, fail_on_error: bool = False, patch: str | None = None) -> str:
        """
        Execute a command in the environment.

        Args:
            command: The command to execute
            fail_on_error: If True, raise an exception if the command fails

        Returns:
            The output of the command as a string

        Raises:
            EnvironmentExecutionError: If the command execution fails and fail_on_error is True
        """
        pass

    @abc.abstractmethod
    async def read_file(self, path: str) -> str:
        """
        Read a file from the environment.

        Args:
            path: The path to the file

        Returns:
            The contents of the file as a string

        Raises:
            FileNotFoundError: If the file does not exist
            EnvironmentExecutionError: If there's an error reading the file
        """
        pass

    @abc.abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """
        Write content to a file in the environment.

        Args:
            path: The path to the file
            content: The content to write to the file

        Raises:
            EnvironmentExecutionError: If there's an error writing to the file
        """
        pass

    async def execute_python_code(self, code: str, cleanup: bool = True) -> str:
        """
        Execute Python code in the environment.

        Args:
            code: The Python code to execute
            cleanup: If True, clean up any temporary resources after execution

        Returns:
            The output of the code execution

        Raises:
            EnvironmentExecutionError: If there's an error during execution
        """
        import os

        # Generate a unique temporary filename in the current working directory
        # This is important so the code has access to the repository context
        temp_file = f"moatless_temp_{os.getpid()}_{id(code)}.py"

        try:
            # Write the code to the temporary file
            await self.write_file(temp_file, code)

            # Execute the temporary file
            output = await self.execute(f"python {temp_file}")

            return output
        finally:
            if cleanup:
                try:
                    # Clean up the temporary file
                    await self.execute(f"rm -f {temp_file}")
                except Exception:
                    # Ignore cleanup errors
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
