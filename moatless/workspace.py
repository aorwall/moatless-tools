import logging
from typing import Optional

from moatless.codeblocks.parser.python import PythonParser
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.repository import FileRepository
from moatless.types import FileWithSpans

_parser = PythonParser()

logger = logging.getLogger(__name__)


class Workspace:

    def __init__(
        self,
        file_repo: FileRepository,
        code_index: Optional[CodeIndex] = None,
    ):
        self.code_index = code_index
        self.file_repo = file_repo
        self._file_context = self.create_file_context()

    @classmethod
    def from_dirs(
        cls,
        repo_dir: str,
        index_dir: str,
    ):
        file_repo = FileRepository(repo_dir)
        code_index = CodeIndex.from_persist_dir(index_dir, file_repo=file_repo)
        workspace = cls(
            file_repo=file_repo,
            code_index=code_index,
        )
        return workspace

    def create_file_context(
        self, files_with_spans: Optional[list[FileWithSpans]] = None
    ):
        file_context = FileContext(self.file_repo)
        if files_with_spans:
            file_context.add_files_with_spans(files_with_spans)
        return file_context

    @property
    def file_context(self):
        return self._file_context

    def get_file(self, file_path, refresh: bool = False, from_origin: bool = False):
        return self.file_repo.get_file(file_path, refresh=refresh, from_origin=from_origin)

    def save_file(self, file_path: str, updated_content: Optional[str] = None):
        self.file_repo.save_file(file_path, updated_content)

    def save(self):
        self.file_repo.save()
