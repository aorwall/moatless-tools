import fnmatch
from pathlib import Path


def is_test(file_path: str) -> bool:
    path = Path(file_path)

    # All files in test directories are considered test files
    if any(part in ["testing"] for part in path.parts):
        return True

    test_file_patterns = [
        "unittest_*.py",
        "test_*.py",
        "tests_*.py",
        "*_test.py",
        "test.py",
        "tests.py",
    ]
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in test_file_patterns)
