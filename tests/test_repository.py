import pytest
from moatless.repository.repository import InMemRepository
from moatless.repository.file import FileRepository
import tempfile
import os


def test_inmem_repository_dump_and_validate():
    # Create an InMemRepository with some files
    files = {"file1.txt": "Content of file 1", "file2.py": "print('Hello, World!')"}
    repo = InMemRepository(files=files)

    # Dump the repository
    dumped = repo.model_dump()

    # Validate the dumped data
    loaded_repo = InMemRepository.model_validate(dumped)

    # Check if the loaded repository has the same files
    assert loaded_repo.files == repo.files


def test_file_repository_dump_and_validate():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        file1_path = os.path.join(temp_dir, "file1.txt")
        file2_path = os.path.join(temp_dir, "file2.py")

        with open(file1_path, "w") as f:
            f.write("Content of file 1")
        with open(file2_path, "w") as f:
            f.write("print('Hello, World!')")

        # Create a FileRepository
        repo = FileRepository(repo_path=temp_dir)

        # Access the files to populate the _files dictionary
        repo.get_file("file1.txt")
        repo.get_file("file2.py")

        # Dump the repository
        dumped = repo.model_dump()

        # Validate the dumped data
        loaded_repo = FileRepository.model_validate(dumped)

        # Check if the loaded repository has the same files
        assert loaded_repo.path == repo.path
        assert loaded_repo.get_file_content("file1.txt") == "Content of file 1"
        assert loaded_repo.get_file_content("file2.py") == "print('Hello, World!')"
