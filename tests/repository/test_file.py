from pathlib import Path
from typing import List

import pytest
import pytest_asyncio
from moatless.repository.file import FileRepository


@pytest_asyncio.fixture
async def temp_repo(tmp_path):
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


@pytest.mark.asyncio
async def test_matching_files_basic(temp_repo):
    assert set(await temp_repo.matching_files("*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
        "src/main.py",
        "src/utils/helpers.py",
    }


@pytest.mark.asyncio
async def test_matching_files_subdirectory(temp_repo):
    assert set(await temp_repo.matching_files("tests/*.py")) == {"tests/test_utils.py"}
    assert set(await temp_repo.matching_files("tests/**/*.py")) == {
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }
    assert set(await temp_repo.matching_files("tests/unit/*.py")) == {
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
    }


@pytest.mark.asyncio
async def test_matching_files_complex_patterns(temp_repo):
    assert set(await temp_repo.matching_files("**/*test*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }
    assert set(await temp_repo.matching_files("src/**/*.py")) == {
        "src/main.py",
        "src/utils/helpers.py",
    }


@pytest.mark.asyncio
async def test_matching_files_non_py(temp_repo):
    assert set(await temp_repo.matching_files("**/*.md")) == {"docs/README.md"}
    assert set(await temp_repo.matching_files(".*")) == {".gitignore"}


@pytest.mark.asyncio
async def test_matching_files_case_sensitivity(temp_repo):
    assert set(await temp_repo.matching_files("**/*TEST*.py")) == set()
    assert set(await temp_repo.matching_files("**/*[Tt]est*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }


@pytest.mark.asyncio
async def test_matching_files_empty_result(temp_repo):
    assert await temp_repo.matching_files("nonexistent*.py") == []


@pytest.mark.asyncio
async def test_matching_files_specific_subdirectory(temp_repo):
    assert set(await temp_repo.matching_files("unit/*.py")) == {
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
    }
    assert set(await temp_repo.matching_files("src/utils/*.py")) == {"src/utils/helpers.py"}


@pytest.mark.asyncio
async def test_matching_files(temp_repo):
    # Test various patterns
    assert set(await temp_repo.matching_files("*test*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }

    assert set(await temp_repo.matching_files("*test_*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }

    assert set(await temp_repo.matching_files("tests/**/*.py")) == {
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
    }

    assert set(await temp_repo.matching_files("*.py")) == {
        "test_main.py",
        "tests/test_utils.py",
        "tests/unit/test_core.py",
        "tests/unit/test_helpers.py",
        "tests/integration/test_api.py",
        "src/main.py",
        "src/utils/helpers.py",
    }

    assert set(await temp_repo.matching_files("src/*.py")) == {"src/main.py"}

    assert await temp_repo.matching_files("nonexistent*.py") == []


@pytest.mark.asyncio
async def test_matching_files_exact_filename(temp_repo):
    assert set(await temp_repo.matching_files("*/helpers.py")) == {"src/utils/helpers.py"}
    assert set(await temp_repo.matching_files("**/helpers.py")) == {"src/utils/helpers.py"}
    # Should not match test_helpers.py
    assert "tests/unit/test_helpers.py" not in await temp_repo.matching_files("*/helpers.py")


@pytest.mark.asyncio
async def test_find_exact_matches(temp_repo):
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

    # Create another test file in a nested directory
    nested_dir = Path(temp_repo.repo_path) / "src" / "backends"
    nested_dir.mkdir(exist_ok=True)
    backend_file = nested_dir / "backend_ps.py"
    backend_file.write_text("""
def some_function():
    pass

def process_stream(stream):
    for ps_name, xs_names in stream:
        print(f"Processing {ps_name}")
        
    return stream
""")

    # Test searching with special regex characters
    special_chars = ["[test].data", "*.+?|{test}$^", "This is a (test) string"]

    for search_text in special_chars:
        matches = await temp_repo.find_exact_matches(search_text, "tests/test_functions.py")
        assert len(matches) == 1
        assert matches[0][0] == "tests/test_functions.py"

    # Test original function search still works
    matches = await temp_repo.find_exact_matches("def test_partitions():", "tests/test_functions.py")
    assert len(matches) == 1
    assert matches[0] == ("tests/test_functions.py", 5)

    # Test searching in a directory
    matches = await temp_repo.find_exact_matches("def test_partitions():", "tests/")
    assert len(matches) == 1
    assert matches[0] == ("tests/test_functions.py", 5)

    # Test searching with ** patterns
    matches = await temp_repo.find_exact_matches("for ps_name, xs_names in stream:", "**/*.py")
    assert len(matches) == 1
    assert matches[0][0] == "src/backends/backend_ps.py"
    assert matches[0][1] == 6

    # Test with more specific ** pattern
    matches = await temp_repo.find_exact_matches("for ps_name, xs_names in stream:", "**/backend_ps.py")
    assert len(matches) == 1
    assert matches[0][0] == "src/backends/backend_ps.py"

    # Test with path pattern
    matches = await temp_repo.find_exact_matches("for ps_name, xs_names in stream:", "src/**/backend_ps.py")
    assert len(matches) == 1
    assert matches[0][0] == "src/backends/backend_ps.py"


@pytest.mark.asyncio
async def test_find_with_path_patterns(temp_repo):
    # Create a more complex directory structure for testing
    nested_dir = Path(temp_repo.repo_path) / "sphinx" / "ext" / "napoleon"
    nested_dir.mkdir(parents=True)
    docstring_file = nested_dir / "docstring.py"
    docstring_file.write_text("""
class DocstringParser:
    def __init__(self):
        pass
        
    def parse_docstring(self, docstring):
        return self._parse_sections(docstring)
        
    def _parse_sections(self, content):
        sections = []
        for section in content.split('\\n\\n'):
            sections.append(section)
        return sections
    
    def _parse_other_parameters_section(self, section: str) -> List[str]:
        return self._parse_parameters_section(section)
    
    def _parse_parameters_section(self, section: str) -> List[str]:
        params = []
        for line in section.split('\\n'):
            if ':' in line:
                params.append(line)
        return params
""")

    # Test with exact path pattern
    matches = await temp_repo.find_exact_matches("_parse_parameters_section", "sphinx/ext/napoleon/docstring.py")
    assert len(matches) >= 1  # May find multiple matches
    assert all(match[0] == "sphinx/ext/napoleon/docstring.py" for match in matches)

    # Test with regex and exact path pattern
    regex_matches = await temp_repo.find_regex_matches(
        "def _parse_(other_parameters|parameters)_section", "sphinx/ext/napoleon/docstring.py"
    )
    assert len(regex_matches) == 2
    assert {m["file_path"] for m in regex_matches} == {"sphinx/ext/napoleon/docstring.py"}

    # Test with path pattern including wildcards
    matches = await temp_repo.find_exact_matches("_parse_parameters_section", "sphinx/ext/*/docstring.py")
    assert len(matches) >= 1  # May find multiple matches
    assert all(match[0] == "sphinx/ext/napoleon/docstring.py" for match in matches)

    # Test with directory search
    matches = await temp_repo.find_exact_matches("_parse_parameters_section", "sphinx/ext/napoleon/")
    assert len(matches) >= 1  # May find multiple matches
    assert all(match[0] == "sphinx/ext/napoleon/docstring.py" for match in matches)


@pytest.mark.asyncio
async def test_find_regex_with_nested_paths(temp_repo):
    """Test finding regex matches in nested directory paths like **/requests/**/*.py"""
    # Create a requests-like directory structure
    nested_dir = Path(temp_repo.repo_path) / "build" / "lib" / "requests" / "packages" / "urllib3"
    nested_dir.mkdir(parents=True)
    exceptions_file = nested_dir / "exceptions.py"
    exceptions_file.write_text("""
class HTTPError(Exception):
    \"\"\"Base exception used by this module.\"\"\"
    pass

class DecodeError(HTTPError):
    \"\"\"Raised when a response can't be decoded.\"\"\"
    pass

class TimeoutError(HTTPError):
    \"\"\"Raised when a socket timeout error occurs.

    Catching this error will catch both :exc:`ReadTimeoutErrors
    <ReadTimeoutError>` and :exc:`ConnectTimeoutErrors <ConnectTimeoutError>`.
    \"\"\"
    pass
    
class ReadTimeoutError(TimeoutError, RequestError):
    \"\"\"Raised when a socket timeout error occurs while receiving data from a server.\"\"\"
    pass

class ConnectTimeoutError(TimeoutError):
    \"\"\"Raised when a socket timeout error occurs while connecting to a server.\"\"\"
    pass
""")

    # Create another file with similar errors
    models_file = Path(temp_repo.repo_path) / "build" / "lib" / "requests" / "models.py"
    models_file.parent.mkdir(parents=True, exist_ok=True)
    models_file.write_text("""
from .packages.urllib3.exceptions import (
    DecodeError, ReadTimeoutError, ProtocolError, LocationParseError)

class Response:
    def json(self):
        try:
            return json.loads(self.text)
        except DecodeError as e:
            raise
        except ReadTimeoutError as e:
            raise
    
    def decode_content(self):
        try:
            return self.content.decode('utf-8')
        except UnicodeDecodeError:
            return self.content.decode('utf-8', errors='replace')
""")

    # Test using the exact path pattern mentioned in the user problem
    matches = await temp_repo.find_regex_matches("TimeoutError|DecodeError", "**/requests/**/*.py")
    assert len(matches) >= 3  # Should find at least 3 matches

    # Verify we found both files
    file_paths = {match["file_path"] for match in matches}
    assert "build/lib/requests/packages/urllib3/exceptions.py" in file_paths
    assert "build/lib/requests/models.py" in file_paths

    # Verify specific line matches
    timeout_error_matches = [m for m in matches if "TimeoutError" in m["content"]]
    decode_error_matches = [m for m in matches if "DecodeError" in m["content"]]

    assert len(timeout_error_matches) >= 1
    assert len(decode_error_matches) >= 1


@pytest.mark.asyncio
async def test_find_regex_with_root_level_directory(temp_repo):
    """Test finding regex matches when the target directory is at the root level"""
    # Create a 'requests' directory at the root level
    requests_dir = Path(temp_repo.repo_path) / "requests"
    requests_dir.mkdir()
    (requests_dir / "packages").mkdir()
    (requests_dir / "packages" / "urllib3").mkdir()

    # Create test files in the root-level requests directory
    exceptions_file = requests_dir / "packages" / "urllib3" / "exceptions.py"
    exceptions_file.write_text("""
class HTTPError(Exception):
    \"\"\"Base exception used by this module.\"\"\"
    pass

class DecodeError(HTTPError):
    \"\"\"Raised when a response can't be decoded.\"\"\"
    pass

class TimeoutError(HTTPError):
    \"\"\"Raised when a socket timeout error occurs.\"\"\"
    pass
""")

    models_file = requests_dir / "models.py"
    models_file.write_text("""
from .packages.urllib3.exceptions import (
    DecodeError, ReadTimeoutError, ProtocolError, LocationParseError)

def handle_errors():
    try:
        process_response()
    except DecodeError:
        pass
    except TimeoutError:
        pass
""")

    # Test using the exact path pattern mentioned by the user
    matches = await temp_repo.find_regex_matches("TimeoutError|DecodeError", "**/requests/**/*.py")
    assert len(matches) >= 3  # Should find at least 3 matches

    # Verify we found both files
    file_paths = {match["file_path"] for match in matches}
    assert "requests/packages/urllib3/exceptions.py" in file_paths
    assert "requests/models.py" in file_paths

    # Test using a pattern without the leading **/ to ensure it still works
    matches2 = await temp_repo.find_regex_matches("TimeoutError|DecodeError", "requests/**/*.py")
    assert len(matches2) >= 3

    # Files should match the same ones
    file_paths2 = {match["file_path"] for match in matches2}
    assert file_paths == file_paths2


@pytest.mark.asyncio
async def test_batch_grep_processing(temp_repo):
    """Test that the batch grep processing works efficiently with many files."""
    # Create a requests-like directory structure with many files
    batch_test_dir = Path(temp_repo.repo_path) / "batch_test"
    batch_test_dir.mkdir()

    # Create 30 Python files with error patterns
    expected_files = []
    for i in range(30):
        if i % 3 == 0:  # Put error pattern in every 3rd file
            file_path = batch_test_dir / f"module_{i}.py"
            file_path.write_text(f"""
def process_{i}():
    try:
        do_something()
    except TimeoutError:
        log_error("Timeout occurred")
    except DecodeError:
        log_error("Decoding failed")
""")
            expected_files.append(f"batch_test/module_{i}.py")
        else:
            file_path = batch_test_dir / f"module_{i}.py"
            file_path.write_text(f"""
def process_{i}():
    try:
        do_something()
    except ValueError:
        log_error("Value error")
""")

    # Test batch processing with the improved implementation
    matches = await temp_repo.find_regex_matches("TimeoutError|DecodeError", "**/batch_test/**/*.py")

    # We should find matches in 10 files (every 3rd out of 30)
    # and each file has 2 matches (TimeoutError and DecodeError)
    assert len(matches) == 20

    # Check that all expected files were found
    found_files = {match["file_path"] for match in matches}
    for expected_file in expected_files:
        assert expected_file in found_files

    # Verify we found both error types
    timeout_matches = [m for m in matches if "TimeoutError" in m["content"]]
    decode_matches = [m for m in matches if "DecodeError" in m["content"]]

    assert len(timeout_matches) == 10
    assert len(decode_matches) == 10
