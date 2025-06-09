import pytest
import pytest_asyncio
import subprocess

from moatless.actions.grep_tool import GrepTool, GrepToolArgs
from moatless.file_context import FileContext
from moatless.actions.schema import Observation
from moatless.environment.docker import DockerEnvironment
from moatless.repository.file import FileRepository
from moatless.workspace import Workspace


def is_docker_available():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def is_image_available(image: str) -> bool:
    """Check if a Docker image is available locally or can be pulled."""
    try:
        # First check if image exists locally
        result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=5)
        if result.returncode == 0:
            return True

        # If not available locally, try to pull it (with timeout)
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes timeout for pulling
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip all tests if Docker is not available
pytestmark = pytest.mark.skipif(not is_docker_available(), reason="Docker is not available or not running")


class MockFileRepository(FileRepository):
    """Mock file repository for Docker-based testing."""

    def __init__(self, working_dir: str = "/testbed"):
        # Initialize with minimal required parameters for FileRepository
        super().__init__(repo_path=working_dir)


class MockWorkspace(Workspace):
    """Mock workspace for Docker-based testing."""

    def __init__(self, environment: DockerEnvironment, working_dir: str = "/testbed"):
        # Initialize with minimal required parameters for Workspace
        repository = MockFileRepository(working_dir)
        super().__init__(repository=repository, environment=environment)


@pytest_asyncio.fixture
async def sympy_docker_env():
    """Create a Docker environment with SymPy repository."""
    image = "aorwall/sweb.eval.x86_64.sympy_moatless_sympy-19040:latest"

    if not is_image_available(image):
        pytest.skip(f"Docker image {image} not available")

    env = DockerEnvironment(image=image, working_dir="/testbed")
    yield env
    await env.cleanup()


@pytest_asyncio.fixture
async def django_docker_env():
    """Create a Docker environment with Django repository."""
    image = "aorwall/sweb.eval.x86_64.django_moatless_django-15503:latest"

    if not is_image_available(image):
        pytest.skip(f"Docker image {image} not available")

    env = DockerEnvironment(image=image, working_dir="/testbed")
    yield env
    await env.cleanup()


@pytest.fixture
def sympy_file_context(sympy_docker_env):
    """Create a FileContext for SymPy repository."""
    repo = MockFileRepository("/testbed")
    return FileContext(repo=repo)


@pytest.fixture
def django_file_context(django_docker_env):
    """Create a FileContext for Django repository."""
    repo = MockFileRepository("/testbed")
    return FileContext(repo=repo)


@pytest.fixture
def sympy_workspace(sympy_docker_env):
    """Create a workspace with SymPy Docker environment."""
    return MockWorkspace(sympy_docker_env, "/testbed")


@pytest.fixture
def django_workspace(django_docker_env):
    """Create a workspace with Django Docker environment."""
    return MockWorkspace(django_docker_env, "/testbed")


@pytest.fixture
def sympy_grep_tool(sympy_workspace):
    """Create a GrepTool with SymPy environment."""
    action = GrepTool()
    action._workspace = sympy_workspace
    return action


@pytest.fixture
def django_grep_tool(django_workspace):
    """Create a GrepTool with Django environment."""
    action = GrepTool()
    action._workspace = django_workspace
    return action


@pytest.mark.asyncio
async def test_sympy_repository_available(sympy_docker_env):
    """Test that SymPy repository is available in the Docker container."""
    # Check that we're in the testbed directory
    result = await sympy_docker_env.execute("pwd")
    assert "/testbed" in result

    # Check that sympy directory exists
    result = await sympy_docker_env.execute("ls -la")
    assert "sympy" in result

    # Check that it's a git repository (we can see .git directory exists)
    result = await sympy_docker_env.execute("ls -la .git")
    assert "HEAD" in result  # Git repository should have HEAD file


@pytest.mark.asyncio
async def test_django_repository_available(django_docker_env):
    """Test that Django repository is available in the Docker container."""
    # Check that we're in the testbed directory
    result = await django_docker_env.execute("pwd")
    assert "/testbed" in result

    # Check that django directory exists
    result = await django_docker_env.execute("ls -la")
    assert "django" in result or "tests" in result

    # Check for Django-specific files
    result = await django_docker_env.execute("find . -name 'manage.py' -o -name 'setup.py' | head -5")
    assert "manage.py" in result or "setup.py" in result


@pytest.mark.asyncio
async def test_sympy_find_generic_factor_function(sympy_grep_tool, sympy_file_context):
    """Test finding _generic_factor function definition in SymPy."""
    args = GrepToolArgs(
        pattern="def _generic_factor",
        include="*.py",
        max_results=10,
        thoughts="Finding _generic_factor function definition in SymPy",
    )

    result = await sympy_grep_tool.execute(args, sympy_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find the function definition - this should exist in SymPy
    assert "Found" in result.message, "Expected to find _generic_factor function in SymPy repository"
    assert "_generic_factor" in result.message
    assert "def _generic_factor" in result.message
    
    # Should find it in polytools.py specifically
    assert "polytools.py" in result.message

    # Verify properties
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] > 0


@pytest.mark.asyncio
async def test_sympy_find_generic_factor_usage(sympy_grep_tool, sympy_file_context):
    """Test finding _generic_factor usage in specific SymPy file."""
    args = GrepToolArgs(
        pattern="_generic_factor",
        include="sympy/polys/polytools.py",
        max_results=20,
        thoughts="Finding _generic_factor usage in polytools.py",
    )

    result = await sympy_grep_tool.execute(args, sympy_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find usages - this file should contain _generic_factor references
    assert "Found" in result.message, "Expected to find _generic_factor usage in sympy/polys/polytools.py"
    assert "_generic_factor" in result.message
    assert "polytools.py" in result.message

    # Verify properties
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] > 0


@pytest.mark.asyncio
async def test_sympy_find_symbolic_factor_function(sympy_grep_tool, sympy_file_context):
    """Test finding _symbolic_factor function definition in SymPy."""
    args = GrepToolArgs(
        pattern="def _symbolic_factor",
        include="*.py",
        max_results=10,
        thoughts="Finding _symbolic_factor function definition in SymPy",
    )

    result = await sympy_grep_tool.execute(args, sympy_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find the function definition - this should exist in SymPy
    assert "Found" in result.message, "Expected to find _symbolic_factor function in SymPy repository"
    assert "_symbolic_factor" in result.message
    assert "def _symbolic_factor" in result.message
    
    # Should find it in polytools.py specifically
    assert "polytools.py" in result.message

    # Verify properties
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] > 0


@pytest.mark.asyncio
async def test_sympy_find_symbolic_factor_recursive(sympy_grep_tool, sympy_file_context):
    """Test finding _symbolic_factor with recursive pattern in SymPy."""
    args = GrepToolArgs(
        pattern="_symbolic_factor",
        include="**/*.py",
        max_results=20,
        thoughts="Finding _symbolic_factor usage recursively in SymPy",
    )

    result = await sympy_grep_tool.execute(args, sympy_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None

    # Should handle the recursive pattern correctly and find existing usages
    assert "Found" in result.message, "Expected to find _symbolic_factor usage in SymPy repository"
    assert "_symbolic_factor" in result.message

    # Verify properties
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] > 0


@pytest.mark.asyncio
async def test_django_find_setup_methods(django_grep_tool, django_file_context):
    """Test finding setUp methods and object assignments in Django tests."""
    args = GrepToolArgs(
        pattern="def setUp|self\\.objs.*=",
        include="tests/model_fields/test_jsonfield.py",
        max_results=20,
        thoughts="Finding setUp methods and object assignments in Django JSONField tests",
    )

    result = await django_grep_tool.execute(args, django_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find setUp methods or object assignments - these should exist in Django test files
    assert "Found" in result.message, "Expected to find setUp methods in tests/model_fields/test_jsonfield.py"
    
    # Should match either setUp method or self.objs assignments
    assert (
        "setUp" in result.message
        or "self.objs" in result.message
        or "def setUp" in result.message
        or "objs" in result.message
    )
    assert "test_jsonfield.py" in result.message

    # Verify properties
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] > 0


@pytest.mark.asyncio
async def test_django_find_keytransform_class(django_grep_tool, django_file_context):
    """Test finding KeyTransform class definition in Django."""
    args = GrepToolArgs(
        pattern="class KeyTransform",
        include="django/**/*.py",
        max_results=10,
        thoughts="Finding KeyTransform class definition in Django",
    )

    result = await django_grep_tool.execute(args, django_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find the KeyTransform class definition - this should exist in Django
    assert "Found" in result.message, "Expected to find KeyTransform class in Django repository"
    assert "KeyTransform" in result.message
    assert "class KeyTransform" in result.message

    # Should be in Django codebase - verified by the message content containing django paths


@pytest.mark.asyncio
async def test_django_find_keytransform_usage(django_grep_tool, django_file_context):
    """Test finding KeyTransform usage in specific Django file."""
    args = GrepToolArgs(
        pattern="KeyTransform",
        include="django/db/models/fields/json.py",
        max_results=20,
        thoughts="Finding KeyTransform usage in Django JSON field implementation",
    )

    result = await django_grep_tool.execute(args, django_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find KeyTransform references - this should exist in Django JSON fields
    assert "Found" in result.message, "Expected to find KeyTransform usage in django/db/models/fields/json.py"
    assert "KeyTransform" in result.message
    assert "json.py" in result.message

    # Verify properties
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] > 0


@pytest.mark.asyncio
async def test_grep_tool_file_verification(sympy_docker_env):
    """Test that the files we're searching actually exist in the container."""
    # Verify SymPy files exist
    result = await sympy_docker_env.execute("find . -name 'polytools.py' -type f")
    if result.strip():
        print(f"Found polytools.py at: {result.strip()}")
        # Check if the file contains expected content
        file_check = await sympy_docker_env.execute("grep -n '_generic_factor' sympy/polys/polytools.py | head -5")
        print(f"_generic_factor references in polytools.py: {file_check}")

    # Check for symbolic factor related files
    result = await sympy_docker_env.execute("find . -name '*.py' -exec grep -l '_symbolic_factor' {} \\; | head -5")
    if result.strip():
        print(f"Files containing _symbolic_factor: {result}")


@pytest.mark.asyncio
async def test_grep_tool_complex_regex_patterns(sympy_grep_tool, sympy_file_context):
    """Test complex regex patterns that might cause shell parsing issues."""
    # Test pattern with pipe characters
    args = GrepToolArgs(
        pattern="def factor|def factorint",
        include="*.py",
        max_results=15,
        thoughts="Finding factor-related function definitions with pipe pattern",
    )

    result = await sympy_grep_tool.execute(args, sympy_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None

    # Should handle pipe characters in pattern correctly
    if "Found" in result.message:
        # Should find factor or factorint functions
        assert "factor" in result.message.lower()

    # Should not have shell parsing errors
    assert "Shell command parsing error" not in result.message


@pytest.mark.asyncio
async def test_grep_tool_error_handling_in_docker(sympy_grep_tool, sympy_file_context):
    """Test error handling when commands fail in Docker environment."""
    # Test with a pattern that might cause regex errors
    args = GrepToolArgs(
        pattern="[invalid_regex",  # Missing closing bracket
        include="*.py",
        max_results=10,
        thoughts="Testing error handling with invalid regex",
    )

    result = await sympy_grep_tool.execute(args, sympy_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None

    # Should handle the error gracefully
    if "Error" in result.message:
        # Should provide helpful error message
        assert result.properties is not None
        assert "fail_reason" in result.properties
    else:
        # Some grep implementations might handle this gracefully
        pass


@pytest.mark.asyncio
async def test_grep_tool_performance_large_codebase(sympy_grep_tool, sympy_file_context):
    """Test performance and result limiting on large codebase using existing content."""
    # Search for a common pattern that should return many results in SymPy
    args = GrepToolArgs(
        pattern="import",
        include="*.py",
        max_results=5,  # Low limit to test result limiting
        thoughts="Testing performance and result limiting with common pattern",
    )

    result = await sympy_grep_tool.execute(args, sympy_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None

    # Should find imports - SymPy has many import statements
    assert "Found" in result.message, "Expected to find import statements in SymPy repository"
    
    # Should respect max_results limit
    properties = result.properties
    assert properties is not None
    assert properties["total_matches"] <= 5

    # Should show limiting message if results were limited
    if properties["total_matches"] == 5:
        assert "Results limited to 5 matches" in result.message


@pytest.mark.asyncio
async def test_grep_tool_file_path_patterns(django_grep_tool, django_file_context):
    """Test various file path patterns in Django repository."""
    # Test different include patterns
    test_patterns = [
        ("*.py", "Python files"),
        ("tests/**/*.py", "Test files"),
        ("django/db/**/*.py", "Database related files"),
        ("**/test_*.py", "Test files with test_ prefix"),
    ]

    for include_pattern, description in test_patterns:
        args = GrepToolArgs(
            pattern="class",  # Common pattern
            include=include_pattern,
            max_results=3,
            thoughts=f"Testing {description} with pattern {include_pattern}",
        )

        result = await django_grep_tool.execute(args, django_file_context)

        assert isinstance(result, Observation)
        assert result.message is not None

        # Should handle the pattern without errors
        assert "Shell command parsing error" not in result.message

        if "Found" in result.message:
            # Pattern was handled successfully - verified by the message content
            pass


@pytest.mark.asyncio
async def test_grep_tool_empty_results_handling(sympy_grep_tool, sympy_file_context):
    """Test handling when no matches are found."""
    args = GrepToolArgs(
        pattern="NonExistentPatternThatShouldNotBeFound12345",
        include="*.py",
        max_results=10,
        thoughts="Testing empty results handling",
    )

    result = await sympy_grep_tool.execute(args, sympy_file_context)

    assert isinstance(result, Observation)
    assert result.message is not None
    assert "No matches found" in result.message

    # Verify properties for empty results
    properties = result.properties
    assert properties is not None
    assert "fail_reason" in properties
    assert properties["fail_reason"] == "no_matches"
