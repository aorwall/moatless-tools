import logging
from typing import Optional

import litellm
from git import Repo

from moatless.repository.file import FileRepository
from moatless.settings import Settings
from moatless.utils.repo import (
    maybe_clone,
    checkout_commit,
    create_and_checkout_new_branch,
)

logger = logging.getLogger(__name__)


class GitRepository(FileRepository):
    def __init__(
        self, repo_path: str, git_repo_url: Optional[str], commit: Optional[str] = None
    ):
        super().__init__(repo_path)
        self._repo_path = repo_path
        self._repo_url = git_repo_url
        self._repo = Repo(path=repo_path)
        if not self._repo.heads:
            raise Exception(
                "Git repository has no heads, you need to do an initial commit."
            )

        # TODO: Check if current branch is mainline

        # TODO: Check if repo is dirty

        try:
            self._current_branch = self._repo.active_branch.name
        except TypeError as e:
            logger.warning(f"Could not determine current branch: {e}")
            self._current_branch = None

        self._current_commit = self._repo.head.commit.hexsha
        self._initial_commit = self._current_commit

    @classmethod
    def from_repo(
        cls,
        git_repo_url: str,
        repo_path: str,
        commit: Optional[str] = None,
        new_branch: Optional[str] = None,
    ):
        logger.info(
            f"Clone GitRepository from {git_repo_url} with commit {commit} to {repo_path} "
        )

        maybe_clone(git_repo_url, repo_path)

        if commit:
            checkout_commit(repo_path, commit)

        if new_branch:
            create_and_checkout_new_branch(repo_path, new_branch)

        return cls(repo_path=repo_path, git_repo_url=git_repo_url, commit=commit)

    @classmethod
    def from_dict(cls, data: dict):
        return cls.from_repo(
            git_repo_url=data["repo_url"],
            repo_path=data["path"],
            commit=data["commit"],
        )

    def restore_from_snapshot(self, snapshot: dict):
        self._current_branch = snapshot["branch"]
        self._current_commit = snapshot["commit"]
        self._repo.git.checkout(self._current_commit)

    def dict(self):
        return {
            "type": "git",
            "repo_path": self._repo_path,
            "git_repo_url": self._repo_url,
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

    def diff(self):
        return self._repo.git.diff(self._initial_commit, self._current_commit)

    def commit_message(self) -> str:
        diff = self._repo.git.diff(None)
        if not diff:
            return "No changes."

        prompt = f"Generate a concise commit message for the following git diff:\n\n{diff}\n\nCommit message:"

        try:
            response = litellm.completion(
                model=Settings.cheap_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating commit message: {e}")
            return "Automated commit"
