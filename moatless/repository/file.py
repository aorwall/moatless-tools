import difflib
import glob
import logging
import os
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr

from moatless.codeblocks import get_parser_by_path
from moatless.codeblocks.codeblocks import CodeBlockType, CodeBlockTypeGroup
from moatless.codeblocks.module import Module

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    file_path: str
    updated: bool
    diff: Optional[str] = None
    error: Optional[str] = None
    new_span_ids: set[str] | None = None


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
        self._repo_path = kwargs.get("_repo_path", None)
        self._module = kwargs.get("_module", None)
        self._last_modified = kwargs.get("_last_modified", None)

    @classmethod
    def from_file(cls, repo_path: str, file_path: str):
        return cls(file_path=file_path, _repo_path=repo_path)

    @classmethod
    def from_content(cls, file_path: str, content: str):
        return cls(file_path=file_path, _content=content)

    def has_been_modified(self) -> bool:
        if not self._repo_path:
            raise ValueError("CodeFile must be initialized with a repo path")

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
                if len(self._module.children) == 0:
                    raise ValueError(
                        f"No code blocks found in module for {self.file_path}"
                    )
            else:
                return None

        return self._module


class FileRepository:
    def __init__(self, repo_path: str):
        self._repo_path = repo_path
        self._files = {}

    @property
    def repo_dir(self):
        return self._repo_path

    def dict(self):
        return {"type": "file", "path": self._repo_path}

    def snapshot(self) -> dict:
        return {}

    def restore_from_snapshot(self, snapshot: dict):
        pass

    @property
    def path(self):
        return self._repo_path

    def get_file(self, file_path: str):
        if file_path in self._files:
            return self._files[file_path]

        if file_path.startswith(self.repo_dir):
            file_path = file_path.replace(self.repo_dir, "")
            if file_path.startswith("/"):
                file_path = file_path[1:]

        full_file_path = os.path.join(self._repo_path, file_path)
        if not os.path.exists(full_file_path):
            logger.debug(f"File not found: {full_file_path}")
            return None

        if not os.path.isfile(full_file_path):
            logger.warning(f"{full_file_path} is not a file")
            return None

        file = CodeFile.from_file(file_path=file_path, repo_path=self._repo_path)
        self._files[file_path] = file

        return file

    def save_file(self, file_path: str, updated_content: str) -> CodeFile:
        assert updated_content, "Updated content must be provided"
        file = self.get_file(file_path)
        file.save(updated_content)
        return file

    def matching_files(self, file_pattern: str):
        matched_files = []
        for matched_file in glob.iglob(
            file_pattern, root_dir=self._repo_path, recursive=True
        ):
            matched_files.append(matched_file)

        if not matched_files and not file_pattern.startswith("*"):
            return self.matching_files(f"**/{file_pattern}")

        return matched_files

    def find_files(self, file_patterns: list[str]) -> set[str]:
        found_files = set()
        for file_pattern in file_patterns:
            matched_files = self.matching_files(file_pattern)
            found_files.update(matched_files)

        return found_files

    def has_matching_files(self, file_pattern: str):
        for _matched_file in glob.iglob(
            file_pattern, root_dir=self._repo_path, recursive=True
        ):
            return True
        return False

    def file_match(self, file_pattern: str, file_path: str):
        match = False
        for matched_file in glob.iglob(
            file_pattern, root_dir=self._repo_path, recursive=True
        ):
            if matched_file == file_path:
                match = True
                break
        return match

    def find_by_pattern(self, patterns: list[str]) -> List[str]:
        matched_files = []
        for pattern in patterns:
            matched_files.extend(
                glob.iglob(f"**/{pattern}", root_dir=self._repo_path, recursive=True)
            )
        return matched_files


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
