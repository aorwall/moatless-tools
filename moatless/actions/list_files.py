from pydantic import ConfigDict, Field
import logging
from typing import List
import shlex

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
)
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)

# Default directories to always ignore
DEFAULT_IGNORED_DIRS = [".git", ".venv"]


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
    show_hidden: bool = Field(
        default=False,
        description="Whether to show hidden files and directories (starting with '.'). Always excludes .git and .venv.",
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
        if self.show_hidden:
            param_str += f", show_hidden={self.show_hidden}"
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
            
        if file_context.shadow_mode:
            patch = file_context.generate_git_patch()
        else:
            patch = None

        try:
            # Generate directory path for commands
            dir_path = list_files_args.directory.rstrip("/")
            target_dir = f"./{dir_path}" if dir_path else "."

            # Build the commands using filesystem only
            dirs_command, files_command = build_commands(dir_path, target_dir, list_files_args.recursive, self.ignored_dirs)

            try:
                # Execute commands to get directories and files
                try:
                    dirs_output = await local_env.execute(dirs_command, patch=patch)
                    files_output = await local_env.execute(files_command, patch=patch)

                except Exception as e:
                    # Check if it's a "no such file or directory" error
                    if "No such file or directory" in str(e):
                        return Observation.create(
                            message=f"Error: Directory {list_files_args.directory} does not exist",
                            properties={"fail_reason": "directory_not_found"},
                        )
                    raise  # Re-raise if it's a different error

                # Check for directory not found
                if (dirs_output and "No such file or directory" in dirs_output) or (
                    files_output and "No such file or directory" in files_output
                ):
                    return Observation.create(
                        message=f"Error: Directory {list_files_args.directory} does not exist",
                        properties={"fail_reason": "directory_not_found"},
                    )

                # Process directory results
                directories: list[str] = []
                for line in dirs_output.strip().split("\n"):
                    rel_path = normalize_rel(line.strip())
                    if not rel_path or rel_path == dir_path:
                        continue
                    if not list_files_args.recursive:
                        dir_name = rel_path.replace(f"{dir_path}/", "") if dir_path else rel_path
                        if not dir_name or should_skip_dir(dir_name, list_files_args.show_hidden):
                            continue
                        directories.append(dir_name)
                    else:
                        if should_skip_dir(rel_path, list_files_args.show_hidden):
                            continue
                        directories.append(rel_path)

                # Process file results
                files: list[str] = []
                for line in files_output.strip().split("\n"):
                    rel_path = normalize_rel(line.strip())
                    if not rel_path or should_skip_file(rel_path, list_files_args.show_hidden):
                        continue

                    if not list_files_args.recursive:
                        if dir_path:
                            remainder = rel_path.replace(f"{dir_path}/", "", 1)
                            if remainder and "/" not in remainder:
                                files.append(remainder)
                        else:
                            if "/" not in rel_path:
                                files.append(rel_path)
                    else:
                        if dir_path:
                            if rel_path.startswith(f"{dir_path}/"):
                                files.append(rel_path[len(dir_path) + 1 :])
                        else:
                            files.append(rel_path)

                # Apply max_results limit, prioritizing directories over files
                directories, files, _ = apply_limits(directories, files, list_files_args.max_results)

                # Create a result object
                result = {
                    "directories": sort_breadth_first(directories),
                    "files": sorted(files),
                    "total_dirs": len(directories),
                    "total_files": len(files),
                    "max_results": list_files_args.max_results,
                    "results_limited": len(directories) + len(files) == list_files_args.max_results,
                    "show_hidden": list_files_args.show_hidden,
                }

            except Exception as e:
                logger.error(f"Error using environment commands: {str(e)}")
                # Fall back to using repository list_directory method if available
                try:
                    if file_context._repo and hasattr(file_context._repo, "list_directory"):
                        raw_result = file_context._repo.list_directory(list_files_args.directory)

                        # Filter out ignored directories and directories starting with '.'
                        filtered_dirs = [
                            d
                            for d in raw_result.get("directories", [])
                            if not d.startswith('.') and not any(
                                d == ignored_dir or d.startswith(f"{ignored_dir}/") for ignored_dir in self.ignored_dirs
                            )
                        ]

                        # Filter out files in ignored directories
                        filtered_files = [
                            f
                            for f in raw_result.get("files", [])
                            if not any(f"/{ignored_dir}/" in f"/{f}/" for ignored_dir in self.ignored_dirs)
                        ]

                        result = {
                            "directories": sort_breadth_first(filtered_dirs), 
                            "files": sorted(filtered_files)
                        }
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
            hidden_str = " (including hidden)" if list_files_args.show_hidden else ""

            message = (
                f"Contents of directory '{list_files_args.directory or '(root)'}'{recursive_str}{hidden_str}\n\n"
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


def normalize_rel(path: str) -> str:
    """Convert './a/b' -> 'a/b' and normalize simple cases."""
    if path.startswith("./"):
        return path[2:]
    return path


def is_hidden_segment(segments: list[str]) -> bool:
    return any(seg.startswith(".") for seg in segments)


def should_skip_dir(rel_path: str, show_hidden: bool) -> bool:
    if not rel_path:
        return True
    # Always skip .git and .venv
    if rel_path in [".git", ".venv"] or any(rel_path.startswith(f"{d}/") for d in [".git", ".venv"]):
        return True
    # Skip hidden files/dirs unless show_hidden is True
    if not show_hidden:
        if rel_path.startswith("."):
            return True
        parts = rel_path.split("/")
        if is_hidden_segment(parts):
            return True
    return False


def should_skip_file(rel_path: str, show_hidden: bool) -> bool:
    if not rel_path:
        return True
    parts = rel_path.split("/")
    base = parts[-1]
    # Skip files in .git and .venv directories
    if any(f"/{ignored_dir}/" in f"/{rel_path}/" for ignored_dir in [".git", ".venv"]):
        return True
    # Skip hidden files/dirs unless show_hidden is True
    if not show_hidden:
        if is_hidden_segment(parts[:-1]) or base.startswith("."):
            return True
    return False


def apply_limits(directories: list[str], files: list[str], max_results: int) -> tuple[list[str], list[str], dict]:
    """Prioritize directories when enforcing max_results."""
    total_dirs = len(directories)
    total_files = len(files)
    if total_dirs + total_files <= max_results:
        return directories, files, {
            "total_dirs": total_dirs,
            "total_files": total_files,
            "results_limited": False,
        }

    if total_dirs >= max_results:
        return directories[: max_results], [], {
            "total_dirs": total_dirs,
            "total_files": total_files,
            "results_limited": True,
        }

    remaining = max_results - total_dirs
    return directories, files[:remaining], {
        "total_dirs": total_dirs,
        "total_files": total_files,
        "results_limited": True,
    }




def build_commands(dir_path: str, target_dir: str, recursive: bool, ignored_dirs: list[str]) -> tuple[str, str]:
    """Build simple find commands for directories and files."""
    td = shlex.quote(target_dir)
    
    # Always exclude .git and .venv directories using prune
    prune_expr = "\\( -name .git -o -name .venv \\) -prune -o"
    
    if recursive:
        # All directories and files recursively
        dirs_cmd = f"find {td} -xdev {prune_expr} -mindepth 1 -type d -print | sort"
        files_cmd = f"find {td} -xdev {prune_expr} -type f -print | sort"
    else:
        # Immediate children only
        dirs_cmd = f"find {td} -xdev -maxdepth 1 {prune_expr} -mindepth 1 -type d -print | sort"
        files_cmd = f"find {td} -xdev -maxdepth 1 {prune_expr} -type f -print | sort"
    
    return dirs_cmd, files_cmd


def sort_breadth_first(paths):
    """Sort paths breadth-first: by depth (number of slashes), then alphabetically."""
    return sorted(paths, key=lambda path: (path.count('/'), path))
