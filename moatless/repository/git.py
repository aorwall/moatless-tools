import logging
from typing import Optional, List

import litellm
from git import Repo

from moatless.repository.file import FileRepository, CodeFile
from moatless.settings import Settings
from moatless.utils.repo import maybe_clone, checkout_commit, clone_and_checkout

logger = logging.getLogger(__name__)


class GitRepository(FileRepository):
    def __init__(
        self, repo_path: str, git_repo_url: Optional[str], commit: Optional[str] = None, generate_commit_message: bool = False
    ):
        super().__init__(repo_path)
        self._repo_path = repo_path
        self._repo_url = git_repo_url
        self._repo = Repo(path=repo_path)
        self._generate_commit_message = generate_commit_message

        if not self._repo.heads:
            logger.error(f"Repo at {repo_path} has no branches")
            # TODO: Fail?

        # TODO: Add support for branches
        # self._current_branch = self._repo.active_branch.name

        # TODO: Check if current branch is mainline

        # TODO: Check if repo is dirty

        if commit:
            checkout_commit(repo_path, commit)

        self._current_commit = self._repo.head.commit.hexsha
        self._initial_commit = self._current_commit

    @classmethod
    def from_repo(cls, git_repo_url: str, repo_path: str, commit: Optional[str] = None):
        logger.info(
            f"Create GitRepository for {git_repo_url} with commit {commit} on path {repo_path} "
        )

        if commit:
            clone_and_checkout(git_repo_url, repo_path, commit)
        else:
            maybe_clone(git_repo_url, repo_path)

        return cls(repo_path=repo_path, git_repo_url=git_repo_url, commit=commit)

    @classmethod
    def from_dict(cls, data: dict):
        return cls.from_repo(
            git_repo_url=data["repo_url"],
            repo_path=data["path"],
            commit=data["commit"],
        )

    def restore_from_snapshot(self, snapshot: dict):
        self._current_commit = snapshot["commit"]

        self._repo.git.checkout(self._current_commit)

        # TODO: Check diff and only reset changed files

    def dict(self):
        return {
            "type": "git",
            "repo_path": self._repo_path,
            "git_repo_url": self._repo_url,
            "commit": self._initial_commit,
        }

    def snapshot(self) -> dict:
        return {
            "commit": self._current_commit,
        }

    def save_file(self, file_path: str, updated_content: Optional[str] = None) -> CodeFile:
        file = super().save_file(file_path, updated_content)
        self.commit(file_path)
        return file

    def commit(self, file_path: str | None = None):
        commit_message = self.commit_message(file_path)

        if file_path:
            self._repo.index.add(file_path)
        else:
            self._repo.index.add("*")
        self._repo.index.commit(commit_message)
        self._current_commit = self._repo.head.commit.hexsha

        logger.info(
            f"Committed changes to git with message '{commit_message}' and commit hash '{self._current_commit}'"
        )

    def commit_message(self, file_path: str | None = None) -> str:
        if file_path:
            diff = self._repo.git.diff("HEAD", file_path)
        else:
            diff = self._repo.git.diff("HEAD")

        if not diff:
            return "No changes."

        if Settings.cheap_model and self._generate_commit_message:
            prompt = f"Generate a concise commit message for the following git diff"
            if file_path:
                prompt += f" of file {file_path}"
            prompt += f":\n\n{diff}\n\nCommit message:"

            try:
                response = litellm.completion(
                    model=Settings.cheap_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"Error generating commit message: {e}")

        return "Automated commit by Moatless Tools"

    def diff(self, ignore_paths: Optional[List[str]] = None):
        logger.info(
            f"Get diff between {self._initial_commit} and {self._current_commit}"
        )
        if ignore_paths:
            exclude_patterns = [f':(exclude){path}' for path in ignore_paths]
            diff_command = [self._initial_commit, self._current_commit, '--'] + exclude_patterns
            return self._repo.git.diff(*diff_command)
        else:
            return self._repo.git.diff(self._initial_commit, self._current_commit)
