from pydantic import ConfigDict, Field
import logging
import subprocess
from typing import List

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
)
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from moatless.environment.local import LocalBashEnvironment

logger = logging.getLogger(__name__)

# Default directories to always ignore
DEFAULT_IGNORED_DIRS = [".git", ".cursor", ".mvn", ".venv"]


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
    max_results: int = Field(
        default=100,
        description="Maximum number of results (files and directories) to return.",
    )

    model_config = ConfigDict(title="ListFiles")

    def to_prompt(self):
        recursive_str = " recursively" if self.recursive else ""
        return f"List contents of directory{recursive_str}: {self.directory or '(root)'}"

    def short_summary(self) -> str:
        param_str = f"directory={self.directory}"
        if self.recursive:
            param_str += f", recursive={self.recursive}"
        if self.max_results != 100:
            param_str += f", max_results={self.max_results}"
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
            FewShotExample.create(
                user_input="Show me the first 10 files in the project directory",
                action=ListFilesArgs(
                    thoughts="I'll list a limited number of files in the project directory.",
                    directory="",
                    recursive=False,
                    max_results=10,
                ),
            ),
        ]


class ListFiles(Action):
    args_schema = ListFilesArgs

    ignored_dirs: List[str] = Field(
        default_factory=lambda: DEFAULT_IGNORED_DIRS.copy(),
        description="Directories to ignore. Defaults to common dirs like .git, .cursor, etc.",
    )

    async def execute(
        self,
        args: ListFilesArgs,
        file_context: FileContext | None = None,
    ) -> Observation:
        list_files_args = args

        if file_context is None:
            raise ValueError("File context must be provided to execute the list files action.")

        if not self._workspace:
            raise RuntimeError("No workspace set")

        # Use the environment from the workspace if available, or create a new one
        if not self.workspace.environment:
            raise RuntimeError("No environment set")
        else:
            local_env = self.workspace.environment

        try:
            # Generate directory path for commands
            dir_path = list_files_args.directory.rstrip("/")
            target_dir = f"./{dir_path}" if dir_path else "."

            # Check if git is available in the workspace
            git_available = False
            try:
                git_check = await local_env.execute("git rev-parse --is-inside-work-tree 2>/dev/null || echo 'false'")
                git_available = git_check.strip() == "true"
            except Exception:
                # If the command fails, assume git is not available
                git_available = False

            # Create the ignore_pattern for find commands
            ignore_pattern = ""
            if self.ignored_dirs:
                # Create a pattern to exclude specified directories
                ignore_dirs = "|".join([f"^{d}$" for d in self.ignored_dirs])
                ignore_pattern = f" | grep -v -E '{ignore_dirs}'"

            # Build the appropriate find command based on recursion setting and git availability
            if git_available:
                # Use git commands when git is available to respect .gitignore
                if list_files_args.recursive:
                    # For recursive mode with gitignore
                    if dir_path:
                        files_command = f"cd {target_dir} && git ls-files | sort"
                    else:
                        files_command = "git ls-files | sort"
                else:
                    # For non-recursive mode with gitignore
                    if dir_path:
                        files_command = f"cd {target_dir} && git ls-files --directory | grep -v '/' | sort"
                    else:
                        files_command = "git ls-files --directory | grep -v '/' | sort"

                # Command for directories (git doesn't track directories, so we use find)
                if list_files_args.recursive:
                    dirs_command = f"find {target_dir} -xdev -type d | sort | xargs -I{{}} bash -c 'git check-ignore {{}} > /dev/null || echo {{}}'"
                    if ignore_pattern:
                        dirs_command += ignore_pattern
                else:
                    dirs_command = f"find {target_dir} -xdev -maxdepth 1 -type d | grep -v '^{target_dir}$' | sort | xargs -I{{}} bash -c 'git check-ignore {{}} > /dev/null || echo {{}}'"
                    if ignore_pattern:
                        dirs_command += ignore_pattern
            else:
                # Use regular find when git is not available
                if list_files_args.recursive:
                    # Get all files and directories recursively using find
                    # -xdev: don't cross filesystem boundaries
                    dirs_command = f"find {target_dir} -xdev -type d | sort"
                    if ignore_pattern:
                        dirs_command += ignore_pattern
                    files_command = f"find {target_dir} -xdev -type f | sort"
                else:
                    # List only immediate files and directories using find with maxdepth
                    # -xdev: don't cross filesystem boundaries
                    dirs_command = f"find {target_dir} -xdev -maxdepth 1 -type d | grep -v '^{target_dir}$' | sort"
                    if ignore_pattern:
                        dirs_command += ignore_pattern
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

                        # Skip if the directory should be ignored (relative path)
                        if any(
                            rel_path == ignored_dir or rel_path.startswith(f"{ignored_dir}/")
                            for ignored_dir in self.ignored_dirs
                        ):
                            continue

                        # For recursive listing, filter out the target directory itself
                        if rel_path and rel_path != dir_path:
                            if not list_files_args.recursive:
                                # For non-recursive, show only the directory name
                                if dir_path:
                                    # Strip the parent directory part to get just the name
                                    dir_name = rel_path.replace(f"{dir_path}/", "")
                                    if dir_name:  # Skip if empty after replacement
                                        # Skip if the directory name should be ignored
                                        if any(
                                            dir_name == ignored_dir or dir_name.startswith(f"{ignored_dir}/")
                                            for ignored_dir in self.ignored_dirs
                                        ):
                                            continue
                                        directories.append(dir_name)
                                else:
                                    # Skip if the directory name should be ignored
                                    if any(
                                        rel_path == ignored_dir or rel_path.startswith(f"{ignored_dir}/")
                                        for ignored_dir in self.ignored_dirs
                                    ):
                                        continue
                                    directories.append(rel_path)
                            else:
                                # For recursive, show full relative paths
                                # Skip if the directory path should be ignored
                                if any(
                                    rel_path == ignored_dir or rel_path.startswith(f"{ignored_dir}/")
                                    for ignored_dir in self.ignored_dirs
                                ):
                                    continue
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

                        # Skip if the file is in an ignored directory
                        if any(f"/{ignored_dir}/" in f"/{rel_path}/" for ignored_dir in self.ignored_dirs):
                            continue

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

                # Apply max_results limit, divided between directories and files
                total_results = list_files_args.max_results
                if len(directories) + len(files) > total_results:
                    # Distribute the results proportionally
                    total_items = len(directories) + len(files)
                    dir_ratio = len(directories) / total_items
                    file_ratio = len(files) / total_items

                    # Calculate how many results to allocate to each type
                    max_dirs = min(len(directories), max(1, int(total_results * dir_ratio)))
                    max_files = min(len(files), total_results - max_dirs)

                    directories = directories[:max_dirs]
                    files = files[:max_files]

                # Create a result object
                result = {
                    "directories": sorted(directories),
                    "files": sorted(files),
                    "total_dirs": len(directories),
                    "total_files": len(files),
                    "max_results": list_files_args.max_results,
                    "results_limited": len(directories) + len(files) == list_files_args.max_results,
                    "git_available": git_available,
                    "ignored_dirs": self.ignored_dirs,
                }

            except Exception as e:
                logger.error(f"Error using environment commands: {str(e)}")
                # Fall back to using repository list_directory method if available
                try:
                    if file_context._repo and hasattr(file_context._repo, "list_directory"):
                        raw_result = file_context._repo.list_directory(list_files_args.directory)

                        # Filter out ignored directories
                        filtered_dirs = [
                            d
                            for d in raw_result.get("directories", [])
                            if not any(
                                d == ignored_dir or d.startswith(f"{ignored_dir}/") for ignored_dir in self.ignored_dirs
                            )
                        ]

                        # Filter out files in ignored directories
                        filtered_files = [
                            f
                            for f in raw_result.get("files", [])
                            if not any(f"/{ignored_dir}/" in f"/{f}/" for ignored_dir in self.ignored_dirs)
                        ]

                        result = {"directories": filtered_dirs, "files": filtered_files}
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
            gitignore_str = " (respecting .gitignore)" if git_available else ""

            message = (
                f"Contents of directory '{list_files_args.directory or '(root)'}'{recursive_str}{gitignore_str}\n\n"
            )

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

            # Add a note about limited results if applicable
            if result.get("results_limited", False):
                total_dirs = result.get("total_dirs", 0)
                total_files = result.get("total_files", 0)
                if isinstance(total_dirs, list):
                    total_dirs = len(total_dirs)
                if isinstance(total_files, list):
                    total_files = len(total_files)
                total_found = total_dirs + total_files
                message += (
                    f"\n\nNote: Results limited to {list_files_args.max_results} items. Total found: {total_found}."
                )

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
            "Result Limiting: Verify if max_results parameter is used correctly to avoid overwhelming output.",
            "Git Integration: Confirm if .gitignore patterns are respected when git is available in the workspace.",
            "Directory Filtering: Check if specified directories are properly ignored in the output.",
        ]
