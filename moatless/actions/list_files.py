from pydantic import ConfigDict, Field
import logging

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
)
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from moatless.environment.local import LocalBashEnvironment

logger = logging.getLogger(__name__)


class ListFilesArgs(ActionArguments):
    """List files and directories in a specified directory."""

    directory: str = Field(
        default="",
        description="The directory path to list. Empty string means root directory.",
    )
    recursive: bool = Field(
        default=False,
        description="Whether to list files recursively (including subdirectories).",
    )

    model_config = ConfigDict(title="ListFiles")

    def to_prompt(self):
        recursive_str = " recursively" if self.recursive else ""
        return f"List contents of directory{recursive_str}: {self.directory or '(root)'}"

    def short_summary(self) -> str:
        param_str = f"directory={self.directory}"
        if self.recursive:
            param_str += f", recursive={self.recursive}"
        return f"{self.name}({param_str})"

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Show me what files are in the tests directory",
                action=ListFilesArgs(
                    thoughts="I'll list the contents of the tests directory to see what test files are available.",
                    directory="tests",
                    recursive=False,
                ),
            ),
            FewShotExample.create(
                user_input="What files are in the root directory?",
                action=ListFilesArgs(
                    thoughts="I'll list the contents of the root directory to see the project structure.",
                    directory="",
                    recursive=False,
                ),
            ),
            FewShotExample.create(
                user_input="Show me all files in the src directory including subdirectories",
                action=ListFilesArgs(
                    thoughts="I'll list all files recursively in the src directory to get a complete overview.",
                    directory="src",
                    recursive=True,
                ),
            ),
        ]


class ListFiles(Action):
    args_schema = ListFilesArgs

    async def execute(
        self,
        args: ListFilesArgs,
        file_context: FileContext | None = None,
    ) -> Observation:
        if not isinstance(args, ListFilesArgs):
            raise ValueError(f"Expected ListFilesArgs, got {type(args)}")
        list_files_args = args

        if file_context is None:
            raise ValueError("File context must be provided to execute the list files action.")

        if not self._workspace:
            raise RuntimeError("No workspace set")

        # Use the environment from the workspace if available, or create a new one
        if self.workspace.environment and isinstance(self.workspace.environment, LocalBashEnvironment):
            local_env = self.workspace.environment
        else:
            # Create a new LocalBashEnvironment with the repository path (if available)
            repo_path = getattr(self._repository, "repo_path", None)
            local_env = LocalBashEnvironment(cwd=repo_path)

        try:
            # Generate directory path for commands
            dir_path = list_files_args.directory.rstrip("/")
            target_dir = f"./{dir_path}" if dir_path else "."

            # Build the appropriate find command based on recursion setting
            if list_files_args.recursive:
                # Get all files and directories recursively using find
                # -xdev: don't cross filesystem boundaries
                dirs_command = f"find {target_dir} -xdev -type d | sort"
                files_command = f"find {target_dir} -xdev -type f | sort"
            else:
                # List only immediate subdirectories and files using find with maxdepth
                # -xdev: don't cross filesystem boundaries
                maxdepth = "2" if dir_path else "1"  # Use 2 for specified directory, 1 for root
                dirs_command = f"find {target_dir} -xdev -maxdepth 1 -type d | grep -v '^{target_dir}$' | sort"
                files_command = f"find {target_dir} -xdev -maxdepth 1 -type f | sort"

            try:
                # Execute commands to get directories and files
                try:
                    dirs_output = await local_env.execute(dirs_command)
                    files_output = await local_env.execute(files_command)

                except Exception as e:
                    # Check if it's a "no such file or directory" error
                    if "No such file or directory" in str(e):
                        return Observation.create(
                            message=f"Error listing directory: No such directory '{list_files_args.directory}'",
                            properties={"fail_reason": "directory_not_found"},
                        )
                    raise  # Re-raise if it's a different error

                # Process directory results
                directories = []
                # Check for common error patterns in command output
                no_such_file_error = False
                for cmd_output in [dirs_output, files_output]:
                    if cmd_output and "No such file or directory" in cmd_output:
                        no_such_file_error = True
                        break

                if no_such_file_error:
                    return Observation.create(
                        message=f"Error listing directory: No such directory '{list_files_args.directory}'",
                        properties={"fail_reason": "directory_not_found"},
                    )

                for line in dirs_output.strip().split("\n"):
                    if line.strip():
                        # Convert path to relative format
                        if line.startswith("./"):
                            rel_path = line[2:]
                        else:
                            rel_path = line

                        # For recursive listing, filter out the target directory itself
                        if rel_path and rel_path != dir_path:
                            if not list_files_args.recursive:
                                # For non-recursive, show only the directory name
                                if dir_path:
                                    # Strip the parent directory part to get just the name
                                    dir_name = rel_path.replace(f"{dir_path}/", "")
                                    if dir_name:  # Skip if empty after replacement
                                        directories.append(dir_name)
                                else:
                                    directories.append(rel_path)
                            else:
                                # For recursive, show full relative paths
                                directories.append(rel_path)

                # Process file results
                files = []
                for line in files_output.strip().split("\n"):
                    if line.strip():
                        # Convert path to relative format
                        if line.startswith("./"):
                            rel_path = line[2:]
                        else:
                            rel_path = line

                        if rel_path:
                            if not list_files_args.recursive:
                                # For non-recursive, show only the file name
                                if dir_path:
                                    # Strip the parent directory part to get just the name
                                    file_name = rel_path.replace(f"{dir_path}/", "")
                                    if file_name:  # Skip if empty after replacement
                                        files.append(file_name)
                                else:
                                    files.append(rel_path)
                            else:
                                # For recursive, show full relative paths
                                files.append(rel_path)

                # Create a result object
                result = {
                    "directories": sorted(directories),
                    "files": sorted(files),
                }

            except Exception as e:
                logger.error(f"Error using environment commands: {str(e)}")
                # Fall back to using repository list_directory method if available
                try:
                    if file_context._repo and hasattr(file_context._repo, "list_directory"):
                        result = file_context._repo.list_directory(list_files_args.directory)
                    else:
                        return Observation.create(
                            message=f"Error listing directory: {str(e)}",
                            properties={"fail_reason": "error_listing_directory"},
                        )
                except ValueError as e:
                    return Observation.create(
                        message=f"Error listing directory: {str(e)}",
                        properties={"fail_reason": "error_listing_directory"},
                    )

            recursive_str = " (recursive)" if list_files_args.recursive else ""
            message = f"Contents of directory '{list_files_args.directory or '(root)'}'{recursive_str}\n\n"

            if result["directories"]:
                message += "Directories:\n"
                for directory in result["directories"]:
                    message += f"📁 {directory}\n"
                message += "\n"

            if result["files"]:
                message += "Files:\n"
                for file in result["files"]:
                    message += f"📄 {file}\n"

            if not result["directories"] and not result["files"]:
                message += "Directory is empty."

            return Observation.create(
                message=message,
                summary=message,
                properties=result,
            )

        except Exception as e:
            return Observation.create(
                message=f"Error listing directory: {str(e)}",
                properties={"fail_reason": "error_listing_directory"},
            )

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> list[str]:
        return [
            "Directory Path Validity: Ensure the requested directory path exists and is valid.",
            "Usefulness: Assess if listing the directory contents is helpful for the current task.",
            "Efficiency: Evaluate if the action is being used at an appropriate time in the workflow.",
            "Recursion Option: Check if the recursive option is used appropriately for the task context.",
        ]
