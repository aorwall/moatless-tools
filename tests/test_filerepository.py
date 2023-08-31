import os
import tempfile
from git import Repo
from ghostcoder.filerepository import FileRepository


def test_update_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repo.init(tmpdir)
        file_path = 'file1.txt'
        full_path = os.path.join(tmpdir, file_path)
        with open(full_path, 'w') as f:
            f.write('test\n')
        repo.git.add('-A')
        repo.git.commit('-m', 'Initial commit')
        file_repository = FileRepository(repo_path=tmpdir, repo=repo)
        diff = file_repository.update_file(file_path, 'updated content\n')
        assert "\n-test\n+updated content" in diff

        file_path_new = 'file_new.txt'
        diff_new = file_repository.update_file(file_path_new, 'new content\n')
        assert diff_new == "--- file_new.txt\n+++ file_new.txt\n@@ -0,0 +1 @@\n+new content\n"

        diff_new = file_repository.update_file(file_path_new, 'new content\nnew row\n')
        print("diff_new ", diff_new)
        assert diff_new == "--- file_new.txt\n+++ file_new.txt\n@@ -1 +1,2 @@\n new content\n+new row\n"


def test_generate_file_tree():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repo.init(tmpdir)

        # Create some files in the repository
        file_paths = ['file1.txt', 'dir1/file2.txt', 'dir2/dir3/file3.txt']
        for file_path in file_paths:
            full_path = os.path.join(tmpdir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write('test')

        # Stage and commit the files
        repo.git.add('-A')
        repo.git.commit('-m', 'Initial commit')

        # Create a FileRepository for the git repository
        file_repository = FileRepository(repo_path=tmpdir, repo=repo)

        # Generate the file tree
        file_tree = file_repository.file_tree()

        # Assert that the file tree matches our expectations
        assert file_tree.name == os.path.basename(tmpdir)
        assert file_tree.path == ''
        assert len(file_tree.children) == 3
        assert sorted(child.name for child in file_tree.children) == ['dir1', 'dir2', 'file1.txt']


def test_get_file_by_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repo.init(tmpdir)
        file_path = 'file1.txt'
        full_path = os.path.join(tmpdir, file_path)
        with open(full_path, 'w') as f:
            f.write('test')
        repo.git.add('-A')
        repo.git.commit('-m', 'Initial commit')
        file_repository = FileRepository(repo_path=tmpdir, repo=repo)
        file_content = file_repository.get_file_content(file_path)
        assert file_content == 'test'


def ignore_test_save_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repo.init(tmpdir)
        file_path = 'file1.txt'
        full_path = os.path.join(tmpdir, file_path)
        with open(full_path, 'w') as f:
            f.write('test')
        repo.git.add('-A')
        repo.git.commit('-m', 'Initial commit')
        file_repository = FileRepository(repo_path=tmpdir, repo=repo)
        diff = file_repository.update_file(file_path, 'updated content')
        print(diff)
        file_repository.save_file(file_path)
        updated_content = file_repository.get_file_content(file_path)

        assert any(diff.a_path == file_path for diff in repo.index.diff(None))


def test_discard_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repo.init(tmpdir)
        file_path = 'file1.txt'
        full_path = os.path.join(tmpdir, file_path)
        with open(full_path, 'w') as f:
            f.write('test')
        repo.git.add('-A')
        repo.git.commit('-m', 'Initial commit')
        file_repository = FileRepository(repo_path=tmpdir, repo=repo)
        file_repository.update_file(file_path, 'updated content')
        file_repository.discard_file(file_path)
        content_after_discard = file_repository.get_file_content(file_path)
        assert content_after_discard == 'test'


def test_filetree_file_stages():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repo.init(tmpdir)
        file_path = 'file1.txt'
        full_path = os.path.join(tmpdir, file_path)
        with open(full_path, 'w') as f:
            f.write('committed\n')
        repo.git.add('-A')
        repo.git.commit('-m', 'Initial commit')

        file_repository = FileRepository(repo_path=tmpdir, repo=repo)
        file_tree = file_repository.file_tree()
        assert not file_tree.children[0].staged
        assert not file_tree.children[0].untracked

        # new untracked file
        file_path = 'file2.txt'
        full_path = os.path.join(tmpdir, file_path)
        with open(full_path, 'w') as f:
            f.write('new\n')
        file_tree = file_repository.file_tree()
        assert not file_tree.children[1].staged
        assert file_tree.children[1].untracked

        # stage new file
        repo.git.add('-A')
        file_tree = file_repository.file_tree()
        assert file_tree.children[1].staged
        assert not file_tree.children[1].untracked
        assert "+++ b/file2.txt\n@@ -0,0 +1 @@\n+new" in file_repository.get_diff()

        # committed file
        repo.git.commit('-m', 'Second commit')
        file_tree = file_repository.file_tree()
        assert not any(file.staged for file in file_tree.children)
        assert not any(file.untracked for file in file_tree.children)

        with open(full_path, 'w') as f:
            f.write('update\n')
        file_tree = file_repository.file_tree()
        assert not file_tree.children[1].staged
        assert file_tree.children[1].untracked

        repo.git.add('-A')
        file_tree = file_repository.file_tree()
        assert file_tree.children[1].staged
        assert not file_tree.children[1].untracked
        assert "+++ b/file2.txt\n@@ -1 +1 @@\n-new\n+update" in file_repository.get_diff()

        repo.git.commit('-m', 'Third commit')
        file_tree = file_repository.file_tree()
        assert not any(file.staged for file in file_tree.children)
        assert not any(file.untracked for file in file_tree.children)

        os.remove(full_path)
        file_tree = file_repository.file_tree()
        assert len(file_tree.children) == 1

        repo.git.add('-A')
        file_tree = file_repository.file_tree()
        assert len(file_tree.children) == 1
        assert "@@ -1 +0,0 @@\n-update" in file_repository.get_diff()