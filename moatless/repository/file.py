import asyncio
import difflib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
from pydantic import BaseModel, Field, PrivateAttr

from moatless.codeblocks import get_parser_by_path
from moatless.codeblocks.module import Module
from moatless.repository.repository import Repository
from moatless.telemetry import instrument

logger = logging.getLogger(__name__)


# TODO: Remove this
class CodeFile(BaseModel):
    file_path: str = Field(..., description="The path to the file")

    _content: str = PrivateAttr("")
    _repo_path: Optional[str] = PrivateAttr(None)
    _module: Module | None = PrivateAttr(None)
    _dirty: bool = PrivateAttr(False)
    _last_modified: datetime | None = PrivateAttr(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._content = kwargs.get("_content", "")
        self._repo_path = kwargs.get("repo_path", None)
        self._module = kwargs.get("_module", None)
        self._last_modified = kwargs.get("_last_modified", None)

    @classmethod
    def from_file(cls, repo_path: str, file_path: str):
        return cls(file_path=file_path, repo_path=repo_path)

    @classmethod
    def from_content(cls, file_path: str, content: str):
        return cls(file_path=file_path, _content=content)

    def get_file_content(self, file_path: str) -> Optional[str]:
        return

    def has_been_modified(self) -> bool:
        if not self._repo_path:
            raise ValueError("CodeFile must be initialized with a repo path")

        try:
            full_file_path = os.path.join(self._repo_path, self.file_path)
            current_mod_time = datetime.fromtimestamp(os.path.getmtime(full_file_path))
            is_modified = self._last_modified is None or current_mod_time > self._last_modified
            if is_modified and self._last_modified:
                logger.debug(f"File {self.file_path} has been modified: {self._last_modified} -> {current_mod_time}")

            return is_modified
        except FileNotFoundError:
            logger.warning(f"File {self.file_path} not found")
            return False

    def save(self, updated_content: str):
        full_file_path = os.path.join(self._repo_path, self.file_path)
        with open(full_file_path, "w") as f:
            f.write(updated_content)
            self._content = updated_content
            self._last_modified = datetime.fromtimestamp(os.path.getmtime(f.name))
            self._module = None

    @property
    def content(self):
        if self.has_been_modified():
            with open(os.path.join(self._repo_path, self.file_path)) as f:
                self._content = f.read()
                self._last_modified = datetime.fromtimestamp(os.path.getmtime(f.name))
                self._module = None

        return self._content

    @property
    def module(self) -> Module | None:
        if self._module is None or self.has_been_modified() and self.content.strip():
            parser = get_parser_by_path(self.file_path)
            if parser:
                self._module = parser.parse(self.content)
            else:
                return None

        return self._module


class FileRepository(Repository):
    repo_path: str = Field(..., description="The path to the repository")

    @property
    def repo_dir(self):
        return self.repo_path

    def model_dump(self) -> dict:
        return {"type": "file", "repo_path": self.repo_path}

    def get_full_path(self, file_path: str) -> str:
        """
        Generates the full file path by combining repo_path and file_path.
        All paths are treated as relative to repo_path, even if they start with '/'.

        Args:
            file_path: The file path to process (e.g., 'file.py' or '/src/file.py')

        Returns:
            str: The full path relative to repo_path
        """
        # Strip leading slash if present
        file_path = file_path.lstrip("/")

        # If file_path starts with repo_dir, make it relative
        if file_path.startswith(self.repo_dir):
            file_path = file_path.replace(self.repo_dir, "").lstrip("/")

        # Claude sets /repo/, remove it
        if file_path.startswith("/repo/"):
            file_path = file_path.replace("/repo/", "")
        elif file_path.startswith("repo/"):
            file_path = file_path.replace("repo/", "")

        return os.path.join(self.repo_path, file_path)

    def get_relative_path(self, file_path: str) -> str:
        """
        Generates the relative path by removing repo_path from the full path.

        Args:
            file_path: The file path to process

        Returns:
            str: The relative path
        """

        full_path = self.get_full_path(file_path)
        return full_path.replace(self.repo_path, "").lstrip("/")

    def get_file_content(self, file_path: str) -> str | None:
        full_path = self.get_full_path(file_path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                return f.read()
        logger.warning(f"File {file_path} does not exist, cannot get content")
        return None

    def snapshot(self) -> dict:
        return {}

    def restore_from_snapshot(self, snapshot: dict):
        pass

    @property
    def path(self):
        return self.repo_path

    def is_directory(self, path: str):
        return os.path.isdir(self.get_full_path(path))

    def get_file(self, file_path: str):
        if file_path.startswith(self.repo_dir):
            file_path = file_path.replace(self.repo_dir, "")
            if file_path.startswith("/"):
                file_path = file_path[1:]

        full_file_path = self.get_full_path(file_path)
        if not os.path.exists(full_file_path):
            logger.debug(f"File not found: {full_file_path}")
            return None

        if not os.path.isfile(full_file_path):
            logger.warning(f"{full_file_path} is not a file")
            return None

        file = CodeFile.from_file(file_path=file_path, repo_path=self.repo_path)
        return file

    def file_exists(self, file_path: str):
        full_path = Path(self.get_full_path(file_path))
        return full_path.exists()

    def create_empty_file(self, file_path: str):
        full_file_path = self.get_full_path(file_path)
        if not os.path.exists(os.path.dirname(full_file_path)):
            logger.info(f"Creating directory for {full_file_path}")
            os.makedirs(os.path.dirname(full_file_path))

        with open(full_file_path, "w") as f:
            f.write("")

    def save_file(self, file_path: str, updated_content: str):
        assert updated_content, "Updated content must be provided"

        if not self.file_exists(file_path):
            file = self.create_empty_file(file_path)

        with open(self.get_full_path(file_path), "w") as f:
            f.write(updated_content)

    @instrument()
    async def matching_files(self, file_pattern: str) -> list[str]:
        """
        Returns a list of files matching the given pattern within the repository.

        Parameters:
            file_pattern (str): The glob pattern to match files.

        Returns:
            List[str]: A list of relative file paths matching the pattern.
        """

        try:
            # If absolute path, log warning and remove first slash
            if file_pattern.startswith("/"):
                logger.warning(f"Converting absolute path {file_pattern} to relative path")
                file_pattern = file_pattern[1:]

            # Split pattern into directory and filename parts
            pattern_parts = file_pattern.split("/")
            filename = pattern_parts[-1]

            # Fix invalid ** patterns in filename (e.g. **.py -> **/*.py)
            if "**." in filename:
                filename = filename.replace("**.", "**/*.")
                pattern_parts[-1] = filename

            # If filename doesn't contain wildcards, it should be an exact match
            has_wildcards = any(c in filename for c in "*?[]")
            if not has_wildcards:
                # Prepend **/ only to the directory part if it exists
                if len(pattern_parts) > 1:
                    dir_pattern = "/".join(pattern_parts[:-1])
                    if not dir_pattern.startswith(("/", "\\", "**/")) and "**/" not in dir_pattern:
                        file_pattern = f"**/{dir_pattern}/{filename}"
                    else:
                        file_pattern = f"{dir_pattern}/{filename}"
                else:
                    file_pattern = f"**/{filename}"
            else:
                # Original behavior for patterns with wildcards
                if not file_pattern.startswith(("/", "\\", "**/")) and "**/" not in file_pattern:
                    file_pattern = f"**/{file_pattern}"

            # Reconstruct pattern if it was modified
            if pattern_parts[-1] != filename:
                file_pattern = "/".join(pattern_parts)

            repo_path = anyio.Path(self.repo_path)
            matched_files = []

            async for path in repo_path.glob(file_pattern):
                if await path.is_file():
                    # For exact filename matches, verify the filename matches exactly
                    if not has_wildcards and path.name != filename:
                        continue
                    relative_path = str(path.relative_to(self.repo_path)).replace(os.sep, "/")
                    matched_files.append(relative_path)
        except Exception:
            logger.exception(f"Error finding files for pattern {file_pattern}:")
            return []

        return matched_files

    @instrument()
    async def find_by_pattern(self, patterns: list[str]) -> list[str]:
        """
        Returns a list of files matching the given patterns within the repository.
        Uses native async file operations via anyio.Path.
        """

        matched_files = []
        for pattern in patterns:
            repo_path = anyio.Path(self.repo_path)
            async for path in repo_path.glob(f"**/{pattern}"):
                if await path.is_file():
                    relative_path = str(path.relative_to(self.repo_path))
                    matched_files.append(relative_path)

        return matched_files

    @classmethod
    def model_validate(cls, obj: dict):
        repo = cls(repo_path=obj["path"])
        return repo

    async def find_exact_matches(self, search_text: str, file_pattern: Optional[str] = None) -> list[tuple[str, int]]:
        """
        Uses grep to search for exact text matches in files asynchronously.
        """
        matches = []
        if not file_pattern:
            file_pattern = "."

        try:
            # Remove '**' and everything after it
            grep_pattern = file_pattern
            if "**" in grep_pattern:
                grep_pattern = grep_pattern.split("**")[0]

            if not grep_pattern:
                grep_pattern = "."

            # Always escape special regex characters to handle them literally
            escaped_search_text = (
                search_text.replace("[", "\\[")
                .replace("]", "\\]")
                .replace(".", "\\.")
                .replace("+", "\\+")
                .replace("*", "\\*")
                .replace("?", "\\?")
                .replace("|", "\\|")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("$", "\\$")
                .replace("^", "\\^")
            )

            cmd = ["grep", "-n", "-r", escaped_search_text, grep_pattern]
            logger.info(f"Executing grep command: {' '.join(cmd)}")
            logger.info(f"Search directory: {self.repo_path}")

            process = await asyncio.create_subprocess_exec(
                *cmd, cwd=self.repo_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, text=True
            )
            stdout, stderr = await process.communicate()

            if process.returncode not in (0, 1):  # grep returns 1 if no matches found
                logger.info(f"Grep returned non-standard exit code: {process.returncode}")
                if stderr:
                    logger.warning(f"Grep error output: {stderr}")
                return []

            logger.info(f"Found {len(stdout.splitlines())} potential matches")

            for line in stdout.splitlines():
                try:
                    parts = line.split(":", 2)
                    if len(parts) < 2:
                        logger.info(f"Skipping malformed line: {line}")
                        continue

                    if os.path.isfile(os.path.join(self.repo_path, file_pattern)) and "/" not in parts[0]:
                        # Format: "5:def test_partitions():"
                        line_num = int(parts[0])
                        content = parts[1]
                        file_path = file_pattern
                    else:
                        # Format: "path/to/file:5:def test_partitions():"
                        file_path = parts[0]
                        if file_path.startswith("./"):
                            file_path = file_path[2:]
                        line_num = int(parts[1])
                        content = parts[2]

                    matches.append((file_path, int(line_num)))
                except (ValueError, IndexError) as e:
                    logger.info(f"Error parsing line '{line}': {e}")
                    continue

        except Exception as e:
            logger.info(f"Grep command failed: {e}")
            return []

        logger.info(f"Returning {len(matches)} matches")
        return matches

    def list_directory(self, directory_path: str = "") -> dict[str, list[str]]:
        """
        Lists files and directories in the specified directory.
        Returns a dictionary with 'files' and 'directories' lists.
        """
        full_path = self.get_full_path(directory_path)

        if not os.path.exists(full_path):
            return {"files": [], "directories": []}

        if not os.path.isdir(full_path):
            return {"files": [], "directories": []}

        files = []
        directories = []

        for entry in os.listdir(full_path):
            entry_path = os.path.join(full_path, entry)
            relative_path = os.path.relpath(entry_path, self.repo_path).replace(os.sep, "/")

            if os.path.isfile(entry_path):
                files.append(relative_path)
            elif os.path.isdir(entry_path):
                directories.append(relative_path)

        return {"files": sorted(files), "directories": sorted(directories)}

    async def find_regex_matches(
        self, regex_pattern: str, include_pattern: Optional[str] = None, max_results: int = 100
    ) -> list[dict[str, Any]]:
        """
        Uses grep to search for regex pattern matches in files asynchronously.
        Returns files sorted by modification time (most recent first).

        Parameters:
            regex_pattern (str): The regex pattern to search for
            include_pattern (str, optional): Glob pattern for files to include (e.g. '*.js', '*.{ts,tsx}')
            max_results (int, optional): Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing file path, line number,
                                 line content, and modification time
        """
        matches = []
        search_path = "."
        include_arg = []

        try:
            # Apply include pattern if provided
            if include_pattern:
                include_arg = ["--include", include_pattern]

            # Build the grep command with proper regex support
            cmd = ["grep", "-n", "-r", "--color=never"] + include_arg + ["-E", regex_pattern, search_path]
            logger.info(f"Executing grep command: {' '.join(cmd)}")
            logger.info(f"Search directory: {self.repo_path}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                universal_newlines=False,
            )
            stdout, stderr = await process.communicate()

            # Decode the bytes to strings
            stdout_str = stdout.decode("utf-8") if stdout else ""
            stderr_str = stderr.decode("utf-8") if stderr else ""

            if process.returncode not in (0, 1):  # grep returns 1 if no matches found
                logger.info(f"Grep returned non-standard exit code: {process.returncode}")
                if stderr_str:
                    logger.warning(f"Grep error output: {stderr_str}")
                return []

            logger.info(f"Found {len(stdout_str.splitlines())} potential matches")

            # Process and organize the results
            file_matches = {}
            for line in stdout_str.splitlines():
                try:
                    # Parse line format: "path/to/file:line_num:line_content"
                    parts = line.split(":", 2)
                    if len(parts) < 3:
                        logger.info(f"Skipping malformed line: {line}")
                        continue

                    file_path = parts[0]
                    if file_path.startswith("./"):
                        file_path = file_path[2:]
                    line_num = int(parts[1])
                    content = parts[2]

                    # Get file modification time
                    full_file_path = os.path.join(self.repo_path, file_path)
                    mod_time = os.path.getmtime(full_file_path)

                    if file_path not in file_matches:
                        file_matches[file_path] = {"file_path": file_path, "mod_time": mod_time, "matches": []}

                    file_matches[file_path]["matches"].append({"line_num": line_num, "content": content})

                except (ValueError, IndexError) as e:
                    logger.info(f"Error parsing line '{line}': {e}")
                    continue

            # Sort files by modification time (newest first) and create results list
            sorted_matches = sorted(file_matches.values(), key=lambda x: x["mod_time"], reverse=True)

            # Format the final results
            for file_match in sorted_matches[:max_results]:
                file_path = file_match["file_path"]
                for match in file_match["matches"]:
                    matches.append(
                        {
                            "file_path": file_path,
                            "line_num": match["line_num"],
                            "content": match["content"],
                            "mod_time": file_match["mod_time"],
                        }
                    )

        except Exception as e:
            logger.info(f"Grep command failed: {e}")
            return []

        logger.info(f"Returning {len(matches)} matches")
        return matches


def remove_duplicate_lines(replacement_lines, original_lines):
    """
    Removes overlapping lines at the end of replacement_lines that match the beginning of original_lines.
    """
    if not replacement_lines or not original_lines:
        return replacement_lines

    max_overlap = min(len(replacement_lines), len(original_lines))

    for overlap in range(max_overlap, 0, -1):
        if replacement_lines[-overlap:] == original_lines[:overlap]:
            return replacement_lines[:-overlap]

    return replacement_lines


def do_diff(file_path: str, original_content: str, updated_content: str) -> Optional[str]:
    return "".join(
        difflib.unified_diff(
            original_content.strip().splitlines(True),
            updated_content.strip().splitlines(True),
            fromfile=file_path,
            tofile=file_path,
            lineterm="\n",
        )
    )
