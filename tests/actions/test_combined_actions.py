import pytest
import pytest_asyncio
import textwrap
from unittest.mock import AsyncMock, Mock, patch

from moatless.actions.create_file import CreateFile, CreateFileArgs
from moatless.actions.schema import Observation
from moatless.file_context import FileContext
from moatless.repository.repository import InMemRepository
from moatless.workspace import Workspace


def dedent(text):
    """Remove common leading whitespace from a multi-line string."""
    return textwrap.dedent(text).strip()


@pytest_asyncio.fixture
async def repository():
    repo = InMemRepository()
    return repo


@pytest_asyncio.fixture
async def workspace(repository):
    workspace = Mock(spec=Workspace)
    workspace.repository = repository
    return workspace


@pytest_asyncio.fixture
async def file_context(repository):
    context = FileContext(repo=repository, shadow_mode=False)
    return context


@pytest.mark.asyncio
async def test_create_file_action(file_context, repository, workspace):
    """Test the CreateFile action creates a file with the specified content."""
    # Create file action
    create_file_action = CreateFile()
    create_file_action.workspace = workspace
    
    # Create a Python file with some initial content
    file_content = dedent("""
    def greet(name):
        message = "Hello World"
        return f"{message}, {name}!"

    def main():
        print(greet("User"))
        
    if __name__ == "__main__":
        main()
    """)
    
    create_args = CreateFileArgs(
        path="hello.py",
        file_text=file_content
    )
    
    # Execute the create file action
    create_observation = await create_file_action.execute(create_args, file_context=file_context)
    
    # Verify the create file observation
    assert isinstance(create_observation, Observation)
    assert "File created successfully" in create_observation.message
    assert "File created successfully" in create_observation.summary
    assert not create_observation.error
    assert repository.file_exists("hello.py")
    
    # Verify the file content
    actual_content = repository.get_file_content("hello.py")
    assert 'message = "Hello World"' in actual_content
    assert 'return f"{message}, {name}!"' in actual_content

