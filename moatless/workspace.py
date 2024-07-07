import logging

from moatless.codeblocks.parser.python import PythonParser
from moatless.file_context import FileContext
from moatless.index import IndexSettings
from moatless.index.code_index import CodeIndex
from moatless.repository import CodeFile, FileRepository
from moatless.types import FileWithSpans, VerificationError
from moatless.verify.lint import PylintVerifier
from moatless.verify.maven import MavenVerifier

_parser = PythonParser()

logger = logging.getLogger(__name__)


class Workspace:

    def __init__(
        self,
        file_repo: FileRepository,
        verification_job: str | None = "pylint",
        code_index: CodeIndex | None = None,
        max_file_context_tokens: int = 4000,
    ):
        self.code_index = code_index
        self.file_repo = file_repo

        if verification_job == "maven":
            self.verifier = MavenVerifier(self.file_repo.path)
        elif verification_job == "pylint":
            self.verifier = PylintVerifier(self.file_repo.path)
        else:
            self.verifier = None

        self._file_context = self.create_file_context(
            max_tokens=max_file_context_tokens
        )


    @classmethod
    def from_dirs(
        cls,
        repo_dir: str,
        index_dir: str | None = None,
        index_settings: IndexSettings | None = None,
        max_results: int = 25,
        max_file_context_tokens=4000,
        **kwargs
    ):
        file_repo = FileRepository(repo_dir)
        if index_dir:
            try:
                code_index = CodeIndex.from_persist_dir(
                    index_dir, file_repo=file_repo, max_results=max_results
                )
            except FileNotFoundError:
                logger.info("No index found. Creating a new index.")
                code_index = CodeIndex(
                    file_repo=file_repo,
                    settings=index_settings,
                    max_results=max_results
                )
                code_index.run_ingestion()
                code_index.persist(index_dir)
        else:
            code_index = None

        workspace = cls(
            file_repo=file_repo,
            code_index=code_index,
            max_file_context_tokens=max_file_context_tokens,
            **kwargs
        )
        return workspace

    def create_file_context(
        self,
        files_with_spans: list[FileWithSpans] | None = None,
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

    def save_file(self, file_path: str, updated_content: str | None = None):
        self.file_repo.save_file(file_path, updated_content)

    def save(self):
        self.file_repo.save()

    def verify(self, file: CodeFile | None = None) -> list[VerificationError]:
        if self.verifier:
            return self.verifier.verify(file)

        logger.info("No verifier configured.")
        return []
