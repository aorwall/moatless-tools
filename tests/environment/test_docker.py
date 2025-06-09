import asyncio
import os
import pytest
import pytest_asyncio
import subprocess

from moatless.environment.docker import DockerEnvironment
from moatless.environment.base import EnvironmentExecutionError


def is_docker_available():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def is_test_image_available():
    """Check if the test image is available locally or can be pulled."""
    test_image = "aorwall/sweb.eval.x86_64.sympy_moatless_sympy-19040:latest"
    try:
        # First check if image exists locally
        result = subprocess.run(
            ["docker", "image", "inspect", test_image],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return True
        
        # If not available locally, try to pull it (with timeout)
        result = subprocess.run(
            ["docker", "pull", test_image],
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout for pulling
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip all tests if Docker is not available
pytestmark = pytest.mark.skipif(
    not is_docker_available(), 
    reason="Docker is not available or not running"
)


@pytest_asyncio.fixture
async def docker_env():
    """Create a DockerEnvironment instance for testing."""
    # Use a lightweight image for faster tests, fallback to test image
    test_images = [
        "python:3.11-slim",  # Lightweight image for basic tests
        "aorwall/sweb.eval.x86_64.sympy_moatless_sympy-19040:latest"  # Specific test image
    ]
    
    env = None
    for image in test_images:
        try:
            # Quick check if image is available
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                env = DockerEnvironment(image=image, working_dir="/testbed" if "sympy" in image else "/tmp")
                break
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    if env is None:
        pytest.skip("No suitable Docker image available for testing")
    
    yield env
    
    # Cleanup
    await env.cleanup()


@pytest.mark.asyncio
async def test_execute_basic_commands(docker_env):
    """Test basic command execution in Docker environment."""
    # Test echo command
    result = await docker_env.execute("echo 'Hello World'")
    assert "Hello World" in result
    
    # Test pwd command
    result = await docker_env.execute("pwd")
    assert result.strip() in ["/testbed", "/tmp"]  # Depending on which image is used
    
    # Test ls command
    result = await docker_env.execute("ls -la")
    assert "." in result and ".." in result


@pytest.mark.asyncio
async def test_execute_with_error_handling(docker_env):
    """Test command execution with error scenarios."""
    # Test command that fails but doesn't raise exception by default
    result = await docker_env.execute("ls /nonexistent")
    assert "No such file or directory" in result or "cannot access" in result
    
    # Test command that fails and should raise exception
    with pytest.raises(EnvironmentExecutionError):
        await docker_env.execute("ls /nonexistent", fail_on_error=True)


@pytest.mark.asyncio
async def test_file_operations(docker_env):
    """Test file read and write operations."""
    test_content = "Hello from Docker!\nThis is a test file."
    test_file = "test_docker_file.txt"
    
    # Write a file
    await docker_env.write_file(test_file, test_content)
    
    # Read the file back
    read_content = await docker_env.read_file(test_file)
    assert read_content.strip() == test_content
    
    # Test reading non-existent file
    with pytest.raises(FileNotFoundError):
        await docker_env.read_file("non_existent_file.txt")


@pytest.mark.asyncio
async def test_file_operations_with_special_characters(docker_env):
    """Test file operations with special characters and formatting."""
    # Test content with special characters, quotes, and formatting
    special_content = '''Test content with "quotes" and 'single quotes'
    Special chars: $HOME, `command`, $(echo test)
    Multiple lines
    Unicode: 你好 世界 🌍
    '''
    
    test_file = "special_test_file.txt"
    
    # Write file with special content
    await docker_env.write_file(test_file, special_content)
    
    # Read it back and verify
    read_content = await docker_env.read_file(test_file)
    assert read_content.strip() == special_content.strip()


@pytest.mark.asyncio
async def test_subdirectory_file_operations(docker_env):
    """Test file operations in subdirectories."""
    test_content = "Content in subdirectory"
    test_file = "subdir/nested/test_file.txt"
    
    # Write to subdirectory (should create directories)
    await docker_env.write_file(test_file, test_content)
    
    # Read from subdirectory
    read_content = await docker_env.read_file(test_file)
    assert read_content.strip() == test_content
    
    # Verify directory was created
    result = await docker_env.execute("ls -la subdir/nested/")
    assert "test_file.txt" in result


@pytest.mark.asyncio
async def test_python_code_execution(docker_env):
    """Test Python code execution."""
    python_code = """
import sys
import os
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print(f"Current directory: {os.getcwd()}")
print("Test completed successfully")
"""
    
    result = await docker_env.execute_python_code(python_code)
    assert "Python version:" in result
    assert "Current directory:" in result
    assert "Test completed successfully" in result


@pytest.mark.asyncio
async def test_python_code_with_imports_and_calculations(docker_env):
    """Test Python code execution with imports and calculations."""
    python_code = """
import math
import json

# Test calculations
result = math.sqrt(16) + math.pi
print(f"Math calculation result: {result}")

# Test data structures
data = {"test": True, "value": 42}
print(f"JSON data: {json.dumps(data)}")

# Test file operations
with open("python_test.txt", "w") as f:
    f.write("Created from Python")

with open("python_test.txt", "r") as f:
    content = f.read()
    print(f"File content: {content}")
"""
    
    result = await docker_env.execute_python_code(python_code)
    assert "Math calculation result:" in result
    assert "JSON data:" in result
    assert "File content: Created from Python" in result


@pytest.mark.asyncio
async def test_container_persistence(docker_env):
    """Test that the container persists between operations."""
    # Create a file in first operation
    await docker_env.write_file("persistence_test.txt", "persistent data")
    
    # Verify it exists in second operation
    content = await docker_env.read_file("persistence_test.txt")
    assert content.strip() == "persistent data"
    
    # Create another file using execute
    await docker_env.execute("echo 'command created' > command_file.txt")
    
    # Verify both files exist
    result = await docker_env.execute("ls -la")
    assert "persistence_test.txt" in result
    assert "command_file.txt" in result


@pytest.mark.asyncio 
async def test_environment_variables(docker_env):
    """Test environment variables in Docker environment."""
    # Create Docker environment with custom env vars
    custom_env = DockerEnvironment(
        image="python:3.11-slim",
        env={"TEST_VAR": "test_value", "CUSTOM_PATH": "/custom/path"}
    )
    
    try:
        # Test environment variables using printenv
        result = await custom_env.execute("printenv TEST_VAR")
        assert "test_value" in result
        
        result = await custom_env.execute("printenv CUSTOM_PATH")  
        assert "/custom/path" in result
        
    finally:
        await custom_env.cleanup()


@pytest.mark.skipif(
    not is_test_image_available(),
    reason="Test image aorwall/sweb.eval.x86_64.sympy_moatless_sympy-19040:latest not available"
)
@pytest.mark.asyncio
async def test_with_specific_test_image():
    """Test with the specific sympy test image."""
    env = DockerEnvironment(
        image="aorwall/sweb.eval.x86_64.sympy_moatless_sympy-19040:latest",
        working_dir="/testbed"
    )
    
    try:
        # Test that we're in the testbed directory
        result = await env.execute("pwd")
        assert "/testbed" in result
        
        # Test that sympy is available
        result = await env.execute("ls -la")
        assert "sympy" in result or ".git" in result
        
        # Test Python with sympy (if available)
        python_code = """
try:
    import sympy
    print(f"SymPy version available: {sympy.__version__}")
    x = sympy.Symbol('x')
    expr = x**2 + 2*x + 1
    print(f"Expression: {expr}")
    print("SymPy test successful")
except ImportError:
    print("SymPy not available in this environment")
"""
        result = await env.execute_python_code(python_code)
        # Either sympy works or we get the import error message
        assert "SymPy" in result or "not available" in result
        
    finally:
        await env.cleanup()


@pytest.mark.asyncio
async def test_cleanup_and_context_manager(docker_env):
    """Test cleanup functionality and context manager usage."""
    # Test that environment works
    await docker_env.execute("echo 'test'")
    
    # Test explicit cleanup
    await docker_env.cleanup()
    
    # Test context manager usage
    async with DockerEnvironment(image="python:3.11-slim") as env:
        result = await env.execute("echo 'context manager test'")
        assert "context manager test" in result
    
    # Container should be automatically cleaned up after context manager exit