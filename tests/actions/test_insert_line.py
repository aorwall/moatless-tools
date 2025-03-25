from unittest.mock import Mock

import pytest
from moatless.actions.insert_line import InsertLine, InsertLinesArgs
from moatless.file_context import FileContext
from moatless.repository.repository import InMemRepository


@pytest.fixture
def repository():
    repo = InMemRepository()
    repo.save_file("test.py", """def hello():
    message = "Hello World"
    print(message)
""")
    return repo

@pytest.fixture
def file_context(repository):
    context = FileContext(repo=repository)
    # Add file to context to ensure it's available
    context.add_file("test.py", show_all_spans=True)
    return context

@pytest.mark.asyncio
async def test_insert_line_basic(repository, file_context):
    action = InsertLine(repository=repository)
    args = InsertLinesArgs(
        path="test.py",
        insert_line=2,
        new_str='    logger.info(message)',
        thoughts="Adding logging statement"
    )
    
    observation = await action.execute(args, file_context)

    content = file_context.get_file("test.py").content
    assert 'logger.info(message)' in content
    assert "def hello():" in content  # Verify the rest of the file is intact
    assert "print(message)" in content
    assert "file test.py has been edited" in observation.message

@pytest.mark.asyncio
async def test_insert_line_at_start(repository, file_context):
    action = InsertLine(repository=repository)
    args = InsertLinesArgs(
        path="test.py",
        insert_line=0,
        new_str='import logging\n',
        thoughts="Adding import statement"
    )
    
    observation = await action.execute(args, file_context)
    
    content = file_context.get_file("test.py").content
    assert content.startswith('import logging\n')
    assert "file test.py has been edited" in observation.message

@pytest.mark.asyncio
async def test_insert_line_invalid_line(repository, file_context):
    action = InsertLine(repository=repository)
    args = InsertLinesArgs(
        path="test.py",
        insert_line=999,
        new_str='invalid line',
        thoughts="Trying to insert at invalid line number"
    )
    
    observation = await action.execute(args, file_context)
    
    assert observation.properties.get("fail_reason") == "invalid_line_number"

@pytest.mark.asyncio
async def test_insert_line_file_not_found(repository, file_context):
    action = InsertLine(repository=repository)
    args = InsertLinesArgs(
        path="nonexistent.py",
        insert_line=1,
        new_str='new line',
        thoughts="Trying to insert in non-existent file"
    )
    
    observation = await action.execute(args, file_context)
    
    assert observation.properties.get("fail_reason") == "file_not_found"

@pytest.mark.asyncio
async def test_insert_multiline(repository, file_context):
    action = InsertLine(repository=repository)
    args = InsertLinesArgs(
        path="test.py",
        insert_line=1,
        new_str='''def setup():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
''',
        thoughts="Adding setup function"
    )
    
    observation = await action.execute(args, file_context)
    
    content = file_context.get_file("test.py").content
    assert 'def setup():' in content
    assert 'logging.basicConfig' in content
    assert "file test.py has been edited" in observation.message

@pytest.mark.asyncio
async def test_insert_line_with_indentation(repository, file_context):
    # Create file with class
    repository.save_file("test2.py", """class Test:
    def method(self):
        pass
""")
    file_context.add_file("test2.py", show_all_spans=True)
    
    action = InsertLine(repository=repository)
    args = InsertLinesArgs(
        path="test2.py",
        insert_line=2,
        new_str='    def new_method(self):\n        return "test"',
        thoughts="Adding new method"
    )
    
    observation = await action.execute(args, file_context)
    
    content = file_context.get_file("test2.py").content
    print(content)
    assert '    def new_method(self):' in content
    assert '        return "test"' in content
    assert "class Test:" in content
    assert "file test2.py has been edited" in observation.message
