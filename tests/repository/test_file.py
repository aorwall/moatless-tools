from pathlib import Path

import pytest

from moatless.repository.file import FileRepository


@pytest.fixture
def temp_repo(tmp_path):
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    # Create a more complex directory structure
    (repo_dir / "tests").mkdir()
    (repo_dir / "tests" / "unit").mkdir()
    (repo_dir / "tests" / "integration").mkdir()
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "utils").mkdir()
    (repo_dir / "docs").mkdir()

    # Create test files
    (repo_dir / "test_main.py").touch()
    (repo_dir / "tests" / "test_utils.py").touch()
    (repo_dir / "tests" / "unit" / "test_core.py").touch()
    (repo_dir / "tests" / "unit" / "test_helpers.py").touch()
    (repo_dir / "tests" / "integration" / "test_api.py").touch()
    (repo_dir / "src" / "main.py").touch()
    (repo_dir / "src" / "utils" / "helpers.py").touch()
    (repo_dir / "docs" / "README.md").touch()
    (repo_dir / ".gitignore").touch()

    return FileRepository(repo_path=str(repo_dir))


def test_matching_files_basic(temp_repo):
    assert set(temp_repo.matching_files("*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
        "src/main.py",
        "src/utils/helpers.py",
    }


def test_matching_files_subdirectory(temp_repo):
    assert set(temp_repo.matching_files("tests/*.py")) == {"tests/test_utils.py"}
    assert set(temp_repo.matching_files("tests/**/*.py")) == {
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }
    assert set(temp_repo.matching_files("tests/unit/*.py")) == {
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
    }


def test_matching_files_complex_patterns(temp_repo):
    assert set(temp_repo.matching_files("**/*test*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }
    assert set(temp_repo.matching_files("src/**/*.py")) == {
        "src/main.py",
        "src/utils/helpers.py",
    }


def test_matching_files_non_py(temp_repo):
    assert set(temp_repo.matching_files("**/*.md")) == {"docs/README.md"}
    assert set(temp_repo.matching_files(".*")) == {".gitignore"}


def test_matching_files_case_sensitivity(temp_repo):
    assert set(temp_repo.matching_files("**/*TEST*.py")) == set()
    assert set(temp_repo.matching_files("**/*[Tt]est*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }


def test_matching_files_empty_result(temp_repo):
    assert temp_repo.matching_files("nonexistent*.py") == []


def test_matching_files_specific_subdirectory(temp_repo):
    assert set(temp_repo.matching_files("unit/*.py")) == {
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
    }
    assert set(temp_repo.matching_files("src/utils/*.py")) == {"src/utils/helpers.py"}


def test_matching_files(temp_repo):
    # Test various patterns
    assert set(temp_repo.matching_files("*test*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }

    assert set(temp_repo.matching_files("*test_*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }

    assert set(temp_repo.matching_files("tests/**/*.py")) == {
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }

    assert set(temp_repo.matching_files("*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
        "src/main.py",
        "src/utils/helpers.py",
    }

    assert set(temp_repo.matching_files("src/*.py")) == {"src/main.py"}

    assert temp_repo.matching_files("nonexistent*.py") == []


def test_matching_files_exact_filename(temp_repo):
    assert set(temp_repo.matching_files("*/helpers.py")) == {"src/utils/helpers.py"}
    assert set(temp_repo.matching_files("**/helpers.py")) == {"src/utils/helpers.py"}
    # Should not match test_helpers.py
    assert "tests/unit/test_helpers.py" not in temp_repo.matching_files("*/helpers.py")


def test_find_exact_matches(temp_repo):
    # Create a test file with special regex characters
    test_file = Path(temp_repo.repo_path) / "tests" / "test_functions.py"
    test_file.parent.mkdir(exist_ok=True)
    test_file.write_text("""
def some_other_function():
    pass

def test_partitions():
    pass

# Test special regex characters
value = "[test].data"
pattern = "*.+?|{test}$^"
text = "This is a (test) string"

def another_function():
    pass
""")

    # Test searching with special regex characters
    special_chars = [
        "[test].data",
        "*.+?|{test}$^",
        "This is a (test) string"
    ]

    for search_text in special_chars:
        matches = temp_repo.find_exact_matches(search_text, "tests/test_functions.py")
        assert len(matches) == 1
        assert matches[0][0] == "tests/test_functions.py"

    # Test original function search still works
    matches = temp_repo.find_exact_matches("def test_partitions():", "tests/test_functions.py")
    assert len(matches) == 1
    assert matches[0] == ("tests/test_functions.py", 5)

    # Test searching in a directory
    matches = temp_repo.find_exact_matches("def test_partitions():", "tests/")
    assert len(matches) == 1
    assert matches[0] == ("tests/test_functions.py", 5)
