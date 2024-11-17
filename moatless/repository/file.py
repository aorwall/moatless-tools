import difflib
import glob
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from pydantic import BaseModel, Field, PrivateAttr

from moatless.codeblocks import get_parser_by_path
from moatless.codeblocks.module import Module
from moatless.repository.repository import Repository

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
            is_modified = (
                self._last_modified is None or current_mod_time > self._last_modified
            )
            if is_modified and self._last_modified:
                logger.debug(
                    f"File {self.file_path} has been modified: {self._last_modified} -> {current_mod_time}"
                )

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
    def supports_codeblocks(self):
        return self.module is not None

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

    def model_dump(self) -> Dict:
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

    def matching_files(self, file_pattern: str):
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
                logger.warning(
                    f"Converting absolute path {file_pattern} to relative path"
                )
                file_pattern = file_pattern[1:]

            # Split pattern into directory and filename parts
            pattern_parts = file_pattern.split("/")
            filename = pattern_parts[-1]

            # If filename doesn't contain wildcards, it should be an exact match
            has_wildcards = any(c in filename for c in "*?[]")
            if not has_wildcards:
                # Prepend **/ only to the directory part if it exists
                if len(pattern_parts) > 1:
                    dir_pattern = "/".join(pattern_parts[:-1])
                    if (
                        not dir_pattern.startswith(("/", "\\", "**/"))
                        and "**/" not in dir_pattern
                    ):
                        file_pattern = f"**/{dir_pattern}/{filename}"
                    else:
                        file_pattern = f"{dir_pattern}/{filename}"
                else:
                    file_pattern = f"**/{filename}"
            else:
                # Original behavior for patterns with wildcards
                if (
                    not file_pattern.startswith(("/", "\\", "**/"))
                    and "**/" not in file_pattern
                ):
                    file_pattern = f"**/{file_pattern}"

            repo_path = Path(self.repo_path)
            matched_files = []
            for path in repo_path.glob(file_pattern):
                if path.is_file():
                    # For exact filename matches, verify the filename matches exactly
                    if not has_wildcards and path.name != filename:
                        continue
                    relative_path = str(path.relative_to(self.repo_path)).replace(
                        os.sep, "/"
                    )
                    matched_files.append(relative_path)
        except Exception as e:
            logger.exception(f"Error finding files for pattern {file_pattern}:")
            return []

        return matched_files

    def find_files(self, file_patterns: list[str]) -> set[str]:
        found_files = set()
        for file_pattern in file_patterns:
            matched_files = self.matching_files(file_pattern)
            found_files.update(matched_files)

        return found_files

    def has_matching_files(self, file_pattern: str):
        for _matched_file in glob.iglob(
            file_pattern, root_dir=self.repo_path, recursive=True
        ):
            return True
        return False

    def file_match(self, file_pattern: str, file_path: str):
        match = False
        for matched_file in glob.iglob(
            file_pattern, root_dir=self.repo_path, recursive=True
        ):
            if matched_file == file_path:
                match = True
                break
        return match

    def find_by_pattern(self, patterns: list[str]) -> List[str]:
        matched_files = []
        for pattern in patterns:
            matched_files.extend(
                glob.iglob(f"**/{pattern}", root_dir=self.repo_path, recursive=True)
            )
        return matched_files

    def model_dump(self) -> Dict:
        return {
            "type": "file",
            "path": self.repo_path,
        }

    @classmethod
    def model_validate(cls, obj: Dict):
        repo = cls(repo_path=obj["path"])
        return repo

    def find_exact_matches(
        self, search_text: str, file_pattern: Optional[str] = None
    ) -> List[tuple[str, int]]:
        """
        Uses grep to search for exact text matches in files.
        """
        matches = []
        if not file_pattern:
            file_pattern = "."

        try:
            cmd = ["grep", "-n", "-r", search_text, file_pattern]

            logger.info(f"Executing grep command: {' '.join(cmd)}")
            logger.info(f"Search directory: {self.repo_path}")

            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True
            )

            if result.returncode not in (0, 1):  # grep returns 1 if no matches found
                logger.info(
                    f"Grep returned non-standard exit code: {result.returncode}"
                )
                if result.stderr:
                    logger.info(f"Grep error output: {result.stderr}")
                return []

            logger.info(f"Found {len(result.stdout.splitlines())} potential matches")

            for line in result.stdout.splitlines():
                try:
                    parts = line.split(":", 2)
                    if len(parts) < 2:
                        logger.info(f"Skipping malformed line: {line}")
                        continue

                    if (
                        os.path.isfile(os.path.join(self.repo_path, file_pattern))
                        and "/" not in parts[0]
                    ):
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

        except subprocess.SubprocessError as e:
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


def do_diff(
    file_path: str, original_content: str, updated_content: str
) -> Optional[str]:
    return "".join(
        difflib.unified_diff(
            original_content.strip().splitlines(True),
            updated_content.strip().splitlines(True),
            fromfile=file_path,
            tofile=file_path,
            lineterm="\n",
        )
    )
