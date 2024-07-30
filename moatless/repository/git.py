import logging

from git import Repo

from moatless.repository.file import FileRepository
from moatless.utils.repo import maybe_clone, checkout_commit

logger = logging.getLogger(__name__)


class GitRepository(FileRepository):
    def __init__(self, repo_path: str, repo_url: str | None, commit: str | None = None):
        super().__init__(repo_path)
        self._repo_path = repo_path
        self._repo_url = repo_url
        self._repo = Repo(path=repo_path)
        if not self._repo.heads:
            raise Exception(
                "Git repository has no heads, you need to do an initial commit."
            )

        # TODO: Check if current branch is mainline

        # TODO: Check if repo is dirty

        self._current_branch = self._repo.active_branch.name
        self._current_commit = self._repo.head.commit.hexsha

    @classmethod
    def from_repo(cls, repo_url: str, repo_path: str, commit: str | None = None):
        logger.info(
            f"Clone GitRepository from {repo_url} with commit {commit} to {repo_path} "
        )

        maybe_clone(repo_url, repo_path)

        if commit:
            checkout_commit(repo_path, commit)

        return cls(repo_path=repo_path, repo_url=repo_url, commit=commit)

    def dict(self):
        return {
            "type": "git",
            "path": self._repo_path,
            "repo_url": self._repo_url,
            "branch": self._current_branch,
            "commit": self._current_commit,
        }

    def snapshot(self) -> dict:
        return {
            "branch": self._current_branch,
            "commit": self._current_commit,
        }

    def save(self):
        super().save()
        commit_message = self.commit_message()
        self._repo.index.add("*")
        self._repo.index.commit(commit_message)
        self._current_commit = self._repo.head.commit.hexsha

    def commit_message(self):
        # TODO: Generate commit message from git diff
        return "Commit"
