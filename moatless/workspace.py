import logging
from typing import Optional

from moatless.codeblocks.parser.python import PythonParser
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.repository import FileRepository, CodeFile
from moatless.types import FileWithSpans
from moatless.verify.lint import run_pylint
from moatless.verify.maven import run_maven_and_parse_errors
from moatless.verify.types import VerificationError

_parser = PythonParser()

logger = logging.getLogger(__name__)


class Workspace:

    def __init__(
        self,
        file_repo: FileRepository,
        code_index: Optional[CodeIndex] = None,
        max_file_context_tokens: int = 4000,
    ):
        self.code_index = code_index
        self.file_repo = file_repo
        self._file_context = self.create_file_context(
            max_tokens=max_file_context_tokens
        )

    @classmethod
    def from_dirs(
        cls,
        repo_dir: str,
        index_dir: Optional[str] = None,
        max_results: int = 25,
        max_file_context_tokens=4000,
    ):
        file_repo = FileRepository(repo_dir)
        if index_dir:
            code_index = CodeIndex.from_persist_dir(
                index_dir, file_repo=file_repo, max_results=max_results
            )
        else:
            code_index = None
        workspace = cls(
            file_repo=file_repo,
            code_index=code_index,
            max_file_context_tokens=max_file_context_tokens,
        )
        return workspace

    def create_file_context(
        self,
        files_with_spans: Optional[list[FileWithSpans]] = None,
        max_tokens: int = 4000,
    ):
        file_context = FileContext(self.file_repo, max_tokens=max_tokens)
        if files_with_spans:
            file_context.add_files_with_spans(files_with_spans)
        return file_context

    @property
    def file_context(self):
        return self._file_context

    def get_file(self, file_path, refresh: bool = False, from_origin: bool = False):
        return self.file_repo.get_file(
            file_path, refresh=refresh, from_origin=from_origin
        )

    def save_file(self, file_path: str, updated_content: Optional[str] = None):
        self.file_repo.save_file(file_path, updated_content)

    def save(self):
        self.file_repo.save()

    def verify(self, file: CodeFile) -> list[VerificationError]:
        if file.file_path.endswith(".java"):
            return run_maven_and_parse_errors(self.file_repo.path)
        elif file.file_path.endswith(".py"):
            return run_pylint(self.file_repo.path, file.file_path)
        else:
            logger.warning(f"Verification not supported for {file.file_path}")
            return []
