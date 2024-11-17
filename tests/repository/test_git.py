import os
import pytest
from git import Repo
from moatless.repository.git import GitRepository


@pytest.fixture
def temp_git_repo(tmp_path):
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    repo = Repo.init(repo_dir)

    (repo_dir / "file1.txt").write_text("Initial content for file1")
    (repo_dir / "file2.txt").write_text("Initial content for file2")
    (repo_dir / "ignore_me.txt").write_text("This file should be ignored")

    repo.index.add(["file1.txt", "file2.txt", "ignore_me.txt"])
    repo.index.commit("Initial commit")

    yield repo_dir


@pytest.fixture
def git_repository(temp_git_repo):
    return GitRepository(repo_path=str(temp_git_repo))


def test_diff_with_ignore_paths(temp_git_repo, git_repository):
    git_repository.save_file(
        "file1.txt", "Initial content for file1\nAdded content to file1"
    )
    git_repository.save_file(
        "file2.txt", "Initial content for file2\nAdded content to file2"
    )
    git_repository.save_file(
        "ignore_me.txt", "This file should be ignored\nAdded content to ignore_me"
    )

    full_diff = git_repository.diff()
    assert "file1.txt" in full_diff
    assert "file2.txt" in full_diff
    assert "ignore_me.txt" in full_diff

    partial_diff = git_repository.diff(ignore_paths=["ignore_me.txt"])
    assert "file1.txt" in partial_diff
    assert "file2.txt" in partial_diff
    assert "ignore_me.txt" not in partial_diff


def test_save_new_file_and_diff(temp_git_repo, git_repository):
    git_repository.save_file("new_file.txt", "Content for new file")

    diff = git_repository.diff()
    assert "new_file.txt" in diff
    assert "Content for new file" in diff


def test_snapshot_and_restore(temp_git_repo, git_repository):
    git_repository.save_file("file1.txt", "Initial content for file1\nNew content")

    snapshot = git_repository.snapshot()

    git_repository.save_file(
        "file1.txt", "Initial content for file1\nNew content\nEven newer content"
    )

    git_repository.restore_from_snapshot(snapshot)

    assert (
        temp_git_repo / "file1.txt"
    ).read_text() == "Initial content for file1\nNew content"
