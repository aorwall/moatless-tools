import pytest
import pytest_asyncio  # Added for async test support

from moatless.actions.create_file import CreateFile, CreateFileArgs
from moatless.file_context import FileContext
from moatless.repository.repository import InMemRepository


@pytest_asyncio.fixture
async def repository():
    repo = InMemRepository()
    return repo

@pytest_asyncio.fixture
async def file_context(repository):
    return FileContext(repo=repository)

@pytest.mark.asyncio
async def test_create_file_basic(repository, file_context):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        path="new_file.py",
        file_text='''def greet(name: str) -> str:
    return f"Hello, {name}!"
''',
        scratch_pad="Creating a new greeting function"
    )
    
    observation = await action.execute(args, file_context)
    
    content = file_context.get_file("new_file.py").content
    assert "def greet" in content

@pytest.mark.asyncio
async def test_create_file_already_exists(repository, file_context):
    # First create a file
    repository.save_file("existing.py", "# existing content")
    
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        path="existing.py",
        file_text="# new content",
        scratch_pad="Trying to create an existing file"
    )
    
    observation = await action.execute(args, file_context)
    
    assert observation.properties["fail_reason"] == "file_exists"

@pytest.mark.asyncio
async def test_create_file_with_path(repository, file_context):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        path="utils/helpers/string_utils.py",
        file_text='''def capitalize_words(text: str) -> str:
    return " ".join(word.capitalize() for word in text.split())
''',
        scratch_pad="Creating utility function in new directory"
    )
    
    observation = await action.execute(args, file_context)
    
    content = file_context.get_file("utils/helpers/string_utils.py").content
    assert "def capitalize_words" in content

@pytest.mark.asyncio
async def test_create_file_normalize_path(repository, file_context):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        path="/repo/test/normalize.py",  # Path starts with /repo/
        file_text="# normalized path test",
        scratch_pad="Testing path normalization"
    )
    
    observation = await action.execute(args, file_context)
    
    content = file_context.get_file("test/normalize.py").content
    assert content == "# normalized path test\n"

@pytest.mark.asyncio
async def test_create_file_empty_content(repository, file_context):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        path="empty.py",
        file_text="",
        scratch_pad="Creating empty file"
    )
    
    observation = await action.execute(args, file_context)
    
    content = file_context.get_file("empty.py").content
    assert content == ""

@pytest.mark.asyncio
async def test_create_file_with_indentation(repository, file_context):
    action = CreateFile(repository=repository)
    args = CreateFileArgs(
        path="indented.py",
        file_text='''class MyClass:
    def __init__(self):
        self.value = 0
        
    def increment(self):
        self.value += 1
        return self.value''',
        scratch_pad="Creating class with proper indentation"
    )
    
    observation = await action.execute(args, file_context)
    
    content = file_context.get_file("indented.py").content
    assert "class MyClass:" in content
    assert "    def __init__(self):" in content
    assert "        self.value = 0" in content
