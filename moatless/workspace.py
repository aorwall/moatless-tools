import logging
from typing import Optional

from moatless.codeblocks.parser.python import PythonParser
from moatless.file_context import FileContext
from moatless.index import IndexSettings
from moatless.index.code_index import CodeIndex
from moatless.repository import CodeFile, FileRepository, GitRepository
from moatless.types import FileWithSpans, VerificationError
from moatless.verify.lint import PylintVerifier
from moatless.verify.maven import MavenVerifier

_parser = PythonParser()

logger = logging.getLogger(__name__)


class Workspace:
    def __init__(
        self,
        file_repo: FileRepository,
        index_dir: Optional[str] = None,
        index_settings: IndexSettings | None = None,
        max_results: int = 25,
        code_index: CodeIndex | None = None,
        verification_job: Optional[str] = "pylint",
        max_file_context_tokens: int = 4000,
        file_context: FileContext | None = None,
    ):
        self.file_repo = file_repo

        if code_index:
            self.code_index = code_index
        elif index_dir:
            try:
                self.code_index = CodeIndex.from_persist_dir(
                    index_dir, file_repo=file_repo, max_results=max_results
                )
            except FileNotFoundError:
                logger.info("No index found. Creating a new index.")
                code_index = CodeIndex(
                    file_repo=file_repo,
                    settings=index_settings,
                    max_results=max_results,
                )
                code_index.run_ingestion()
                code_index.persist(index_dir)
                self.code_index = code_index
        else:
            self.code_index = None

        if verification_job == "maven":
            self.verifier = MavenVerifier(self.file_repo.path)
        elif verification_job == "pylint":
            self.verifier = PylintVerifier(self.file_repo.path)
        else:
            self.verifier = None

        if file_context:
            self._file_context = file_context
        else:
            self._file_context = self.create_file_context(
                max_tokens=max_file_context_tokens
            )

    @classmethod
    def from_dirs(
        cls,
        git_repo_url: Optional[str] = None,
        commit: Optional[str] = None,
        repo_path: Optional[str] = None,
        max_file_context_tokens: int = 4000,
        **kwargs,
    ):
        if git_repo_url:
            file_repo = GitRepository.from_repo(
                git_repo_url=git_repo_url, repo_path=repo_path, commit=commit
            )
        elif repo_path:
            file_repo = FileRepository(repo_path)
        else:
            raise ValueError("Either git_repo_url or repo_dir must be provided.")

        return cls(
            file_repo=file_repo,
            max_file_context_tokens=max_file_context_tokens,
            **kwargs,
        )

    @classmethod
    def from_dict(cls,
                  data: dict,
                  **kwargs):
        if "repository" not in data:
            raise ValueError("Missing repository key")

        if data["repository"].get("git_repo_url"):
            file_repo = GitRepository.from_repo(
                git_repo_url=data["repository"].get("git_repo_url"),
                repo_path=data["repository"].get("repo_path"),
                commit=data["repository"].get("commit"),
            )
        elif data["repository"].get("repo_path"):
            file_repo = FileRepository(data["repository"].get("repo_path"))
        else:
            raise ValueError("Either git_repo_url or repo_dir must be provided.")

        file_context = FileContext(
            repo=file_repo, max_tokens=data["file_context"].get("max_tokens")
        )
        file_context.load_files_from_dict(data["file_context"].get("files", []))

        if data.get("code_index", {}).get("index_name"):
            code_index = CodeIndex.from_index_name(data["code_index"].get("index_name"), file_repo=file_repo)
        else:
            code_index = None

        return cls(
            file_repo=file_repo,
            file_context=file_context,
            code_index=code_index,
            **kwargs,
        )

    def restore_from_snapshot(self, snapshot: dict):
        self.file_repo.restore_from_snapshot(snapshot["repository"])
        self._file_context.restore_from_snapshot(snapshot["file_context"])

    def dict(self):
        return {
            "repository": self.file_repo.dict(),
            "file_context": self.file_context.model_dump(
                exclude_none=True, exclude_unset=True
            ),
            "code_index": self.code_index.dict() if self.code_index else None,
        }

    def snapshot(self):
        return {
            "repository": self.file_repo.snapshot(),
            "file_context": self.file_context.snapshot(),
        }

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

    def save(self):
        self.file_repo.save()

    def verify(self, file: CodeFile | None = None) -> list[VerificationError]:
        if self.verifier:
            return self.verifier.verify(file)

        logger.info("No verifier configured.")
        return []
