import asyncio
import difflib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
from moatless.codeblocks import get_parser_by_path
from moatless.codeblocks.module import Module
from moatless.repository.repository import Repository
from opentelemetry import trace
from pydantic import BaseModel, Field, PrivateAttr

tracer = trace.get_tracer(__name__)
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
        if self._repo_path is None:
            raise ValueError("Repository path is not set")
        full_file_path = os.path.join(self._repo_path, self.file_path)
        with open(full_file_path, "w") as f:
            f.write(updated_content)
            self._content = updated_content
            self._last_modified = datetime.fromtimestamp(os.path.getmtime(f.name))
            self._module = None

    @property
    def content(self):
        if self.has_been_modified():
            if self._repo_path is None:
                raise ValueError("Repository path is not set")
            try:
                with open(os.path.join(self._repo_path, self.file_path)) as f:
                    self._content = f.read()
                    self._last_modified = datetime.fromtimestamp(os.path.getmtime(f.name))
                    self._module = None
            except UnicodeDecodeError as e:
                logger.warning(f"Failed to decode {self.file_path} as UTF-8: {e}")
                # Try to read file as binary, then decode with errors='replace'
                try:
                    with open(os.path.join(self._repo_path, self.file_path), "rb") as f:
                        binary_content = f.read()
                        self._content = binary_content.decode("utf-8", errors="replace")
                        self._last_modified = datetime.fromtimestamp(os.path.getmtime(f.name))
                        self._module = None
                        logger.info(f"Successfully read {self.file_path} with replacement characters")
                except Exception as e2:
                    logger.error(f"Failed to read {self.file_path} even with error handling: {e2}")
                    self._content = f"[Error reading file: {str(e)}]"

        return self._content

    @property
    def module(self) -> Module | None:
        if self._module is None or self.has_been_modified() and self.content.strip():
            parser = get_parser_by_path(self.file_path)
            if parser:
                try:
                    self._module = parser.parse(self.content)
                except Exception as e:
                    logger.warning(f"Failed to parse {self.file_path}: {e}")
                    return None
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

    @tracer.start_as_current_span("matching_files")
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

    @tracer.start_as_current_span("find_by_pattern")
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
        search_path = "."
        include_arg = []

        try:
            # Handle file pattern 
            if file_pattern:
                # Directory search - treat differently
                if file_pattern.endswith('/') or os.path.isdir(os.path.join(self.repo_path, file_pattern)):
                    search_path = file_pattern
                    # When searching in a directory, include all common code files by default
                    include_arg = ["--include", "*.py"]
                elif "**" in file_pattern:
                    # ** isn't directly supported by grep, so handle it differently
                    # First check if it's just a filename pattern or a path pattern
                    if "/" in file_pattern:
                        # For paths with **, we need to use find first to get matching files
                        matches_files = await self.matching_files(file_pattern)
                        if not matches_files:
                            return []
                        
                        # Process each matching file separately
                        all_matches = []
                        for file_path in matches_files:
                            file_matches = await self.find_exact_matches(search_text, file_path)
                            all_matches.extend(file_matches)
                        return all_matches
                    else:
                        # For simple filename patterns like "**/*.py", convert to shell glob
                        include_arg = ["--include", file_pattern.replace("**", "*")]
                elif "/" in file_pattern and ("*" in file_pattern or "?" in file_pattern):
                    # For path patterns with wildcards, use matching_files to expand
                    matches_files = await self.matching_files(file_pattern)
                    if not matches_files:
                        return []
                    
                    # Process each matching file separately
                    all_matches = []
                    for file_path in matches_files:
                        file_matches = await self.find_exact_matches(search_text, file_path)
                        all_matches.extend(file_matches)
                    return all_matches
                elif "*" in file_pattern:
                    # Handle normal glob patterns
                    include_arg = ["--include", file_pattern]
                elif "/" in file_pattern:
                    # For patterns with paths but no wildcards, extract directory and filename
                    dir_path = os.path.dirname(file_pattern)
                    if dir_path:
                        search_path = dir_path
                    filename = os.path.basename(file_pattern)
                    if filename != "*":
                        include_arg = ["--include", filename]
                elif file_pattern != ".":
                    # If it's a specific file, use it as the search path
                    search_path = file_pattern

            # Use -F flag for fixed-string matching instead of regex
            # This works on both BSD grep (macOS) and GNU grep (Linux)
            cmd = ["grep", "-n", "-r", "-F"] + include_arg + [search_text, search_path]
            logger.info(f"Executing grep command: {' '.join(cmd)} in {self.repo_path}")

            process = await asyncio.create_subprocess_exec(
                *cmd, cwd=self.repo_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout_bytes, stderr_bytes = await process.communicate()
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            if process.returncode not in (0, 1):  # grep returns 1 if no matches found
                logger.info(f"Grep returned non-standard exit code: {process.returncode}")
                if stderr:
                    logger.warning(f"Grep error output: {stderr}")
                return []

            logger.info(f"Found {len(stdout.splitlines())} potential matches")

            for line in stdout.splitlines():
                try:
                    parts = line.split(":", 2)  # type: ignore
                    if len(parts) < 2:
                        logger.info(f"Skipping malformed line: {line}")
                        continue

                    if (
                        os.path.isfile(os.path.join(self.repo_path, search_path))
                        and search_path != "."
                        and "/" not in parts[0]
                    ):  # type: ignore
                        # Format: "5:def test_partitions():"
                        line_num = int(parts[0])
                        content = parts[1]
                        file_path = search_path
                    else:
                        # Format: "path/to/file:5:def test_partitions():"
                        file_path = parts[0]
                        if file_path.startswith("./"):  # type: ignore
                            file_path = file_path[2:]
                        # Normalize path to avoid double slashes
                        file_path = file_path.replace("//", "/")
                        line_num = int(parts[1])
                        content = parts[2]

                    matches.append((file_path, int(line_num)))
                except (ValueError, IndexError) as e:
                    logger.info(f"Error parsing line '{line}': {e}")
                    continue

        except Exception as e:
            logger.exception(f"Grep command failed: {cmd}")
            raise e

        logger.info(f"Returning {len(matches)} matches")
        return matches

    def list_directory(self, directory_path: str = "") -> dict[str, list[str]]:
        """
        Lists files and directories in the specified directory.
        Returns a dictionary with 'files' and 'directories' lists.
        """
        full_path = self.get_full_path(directory_path)

        if not os.path.exists(full_path):
            logger.warning(f"Directory {full_path} does not exist")
            raise ValueError(f"Directory {directory_path} does not exist")

        if not os.path.isdir(full_path):
            logger.warning(f"Directory {full_path} is not a directory")
            raise ValueError(f"Directory {directory_path} is not a directory")

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
        try:
            # Apply include pattern if provided
            if include_pattern and "**" in include_pattern:
                # Use matching_files which properly handles '**' patterns
                matching_files = await self.matching_files(include_pattern)
                if not matching_files:
                    logger.info(f"No files matched pattern: {include_pattern}")
                    return []
                
                logger.info(f"Found {len(matching_files)} files matching pattern: {include_pattern}")
                
                # Process files in batches rather than one by one for better efficiency
                # This is much faster than running grep for each file individually
                batch_size = 20  # Adjust based on typical file counts
                all_matches = []
                
                for i in range(0, len(matching_files), batch_size):
                    batch = matching_files[i:i+batch_size]
                    # Search all files in this batch
                    batch_results = await self._run_grep_batch(regex_pattern, batch, max_results)
                    all_matches.extend(batch_results)
                    
                    # If we have enough results, stop processing batches
                    if len(all_matches) >= max_results:
                        logger.info(f"Found enough matches ({len(all_matches)}), stopping batch processing")
                        break
                
                # Sort by modification time and limit results
                all_matches.sort(key=lambda x: x.get("mod_time", 0), reverse=True)
                return all_matches[:max_results]
            elif include_pattern:
                # For simple include patterns like *.py, let grep handle it directly
                include_arg = []
                search_path = "."
                
                # Directory search - treat differently
                if include_pattern.endswith('/') or os.path.isdir(os.path.join(self.repo_path, include_pattern)):
                    search_path = include_pattern
                    # When searching in a directory, include all common code files by default
                    include_arg = ["--include", "*.py"]
                # If include pattern has a path with /, we need to handle it differently
                elif "/" in include_pattern and "*" in include_pattern:
                    # For patterns with wildcards in paths, fallback to matching_files approach
                    matching_files = await self.matching_files(include_pattern)
                    if not matching_files:
                        return []
                    
                    return await self._run_grep_batch(regex_pattern, matching_files, max_results)
                elif "/" in include_pattern:
                    # For patterns with exact paths but no wildcards
                    dir_path = os.path.dirname(include_pattern)
                    if dir_path:
                        search_path = dir_path
                    filename = os.path.basename(include_pattern)
                    include_arg = ["--include", filename]
                else:
                    # Simple pattern like "*.py"
                    include_arg = ["--include", include_pattern]
                
                # Run grep with appropriate include pattern
                return await self._run_grep_command(regex_pattern, search_path, max_results, include_arg)
            else:
                # No include pattern, search everything
                return await self._run_grep_command(regex_pattern, ".", max_results, [])
        except Exception as e:
            logger.exception(f"Grep command failed: {e}")
            return []

    async def _run_grep_batch(
        self, regex_pattern: str, file_paths: list[str], max_results: int
    ) -> list[dict[str, Any]]:
        """Search a batch of files with a single grep command"""
        if not file_paths:
            return []
            
        matches = []
        
        # Build the grep command - don't use recursive since we're providing specific files
        cmd = ["grep", "-n", "--color=never", "-E", regex_pattern] + file_paths
        logger.info(f"Executing batch grep command with {len(file_paths)} files")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            universal_newlines=False,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode not in (0, 1):  # grep returns 1 if no matches found
            logger.info(f"Grep returned non-standard exit code: {process.returncode}")
            if stderr:
                logger.warning(f"Grep error output: {stderr}")
            return []

        logger.info(f"Found {len(stdout.splitlines())} potential matches")

        # Process and organize the results
        file_matches = {}
        for line in stdout.splitlines():
            try:
                # Parse line format for multiple files: "path/to/file:line_num:line_content"
                parts = line.split(":", 2)
                if len(parts) < 3:
                    logger.info(f"Skipping malformed line: {line}")
                    continue

                file_path = parts[0]
                if file_path.startswith("./"):
                    file_path = file_path[2:]
                    
                try:
                    line_num = int(parts[1])
                except ValueError:
                    logger.info(f"Invalid line number in '{line}': {parts[1]}")
                    continue
                    
                content = parts[2]

                # Get file modification time
                try:
                    full_file_path = os.path.join(self.repo_path, file_path)
                    mod_time = os.path.getmtime(full_file_path)
                except (FileNotFoundError, OSError) as e:
                    logger.warning(f"Error getting file stats for {file_path}: {e}")
                    mod_time = 0

                if file_path not in file_matches:
                    file_matches[file_path] = {"file_path": file_path, "mod_time": mod_time, "matches": []}

                file_matches[file_path]["matches"].append({"line_num": line_num, "content": content})

            except (ValueError, IndexError) as e:
                logger.info(f"Error parsing line '{line}': {e}")
                continue

        # Sort files by modification time (newest first) and create results list
        sorted_matches = sorted(file_matches.values(), key=lambda x: x["mod_time"], reverse=True)

        # Format the final results
        for file_match in sorted_matches:
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
                
                if len(matches) >= max_results:
                    logger.info(f"Reached max results ({max_results}), returning early")
                    return matches
                
        return matches

    async def _run_grep_command(
        self, regex_pattern: str, search_path: str, max_results: int, include_arg: list = []
    ) -> list[dict[str, Any]]:
        """Helper method to run grep command and parse results"""
        if include_arg is None:
            include_arg = []
            
        matches = []
        
        # Determine if we're searching a directory or single file
        is_dir_search = os.path.isdir(os.path.join(self.repo_path, search_path)) or search_path == "."
        
        # Build the grep command with proper regex support
        # Only use recursive flag (-r) when searching directories
        cmd_flags = ["-n"]
        if is_dir_search:
            cmd_flags.append("-r")
        cmd_flags.extend(["--color=never", "-E"])
        
        cmd = ["grep"] + cmd_flags + include_arg + [regex_pattern, search_path]
        logger.info(f"Executing grep command: {' '.join(cmd)}")
        logger.info(f"Search directory: {self.repo_path}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            universal_newlines=False,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode not in (0, 1):  # grep returns 1 if no matches found
            logger.info(f"Grep returned non-standard exit code: {process.returncode}")
            if stderr:
                logger.warning(f"Grep error output: {stderr}")
            return []

        logger.info(f"Found {len(stdout.splitlines())} potential matches")

        # Process and organize the results
        file_matches = {}
        for line in stdout.splitlines():
            try:
                if is_dir_search:
                    # Parse line format for directory search: "path/to/file:line_num:line_content"
                    parts = line.split(":", 2)
                    if len(parts) < 3:
                        logger.info(f"Skipping malformed line: {line}")
                        continue

                    file_path = parts[0]
                    if file_path.startswith("./"):
                        file_path = file_path[2:]
                    line_num = int(parts[1])
                    content = parts[2]
                else:
                    # Parse line format for single file search: "line_num:line_content"
                    parts = line.split(":", 1)
                    if len(parts) < 2:
                        logger.info(f"Skipping malformed line: {line}")
                        continue
                    
                    file_path = search_path
                    line_num = int(parts[0])
                    content = parts[1]

                # Get file modification time
                try:
                    full_file_path = os.path.join(self.repo_path, file_path)
                    mod_time = os.path.getmtime(full_file_path)
                except (FileNotFoundError, OSError) as e:
                    logger.warning(f"Error getting file stats for {file_path}: {e}")
                    mod_time = 0

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
                
                if len(matches) >= max_results:
                    return matches
                
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
