from moatless.actions.find_function import FindFunction, FindFunctionArgs
from moatless.benchmark.swebench import create_repository, create_index, create_index_async
from moatless.evaluation.utils import get_moatless_instance
from moatless.completion import BaseCompletionModel, LLMResponseFormat
from moatless.file_context import FileContext, ContextFile, ContextSpan
from moatless.workspace import Workspace
from moatless.index.types import SearchCodeResponse, SearchCodeHit, SpanHit
from moatless.index.code_index import CodeIndex
from moatless.repository.repository import Repository
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock


@pytest.mark.asyncio
async def test_find_function_init_method():
    instance_id = "django__django-13658"
    instance = get_moatless_instance(instance_id)
    repository = create_repository(instance)
    code_index = create_index(instance, repository)
    file_context = FileContext(repo=repository)
    # Mock the completion model
    completion_model = AsyncMock(spec=BaseCompletionModel)

    # Create and initialize workspace
    workspace = Workspace(repository=repository, code_index=code_index)
    
    action = FindFunction(
        repository=repository, code_index=code_index, completion_model=completion_model
    )
    # Initialize the action with the workspace
    await action.initialize(workspace)

    action_args = FindFunctionArgs(
        scratch_pad="",
        class_name="ManagementUtility",
        function_name="__init__",
    )

    message = await action.execute(action_args, file_context)
    print(message)
    assert len(file_context.files) == 1
    assert "ManagementUtility.__init__" in file_context.files[0].span_ids


@pytest.mark.asyncio
async def test_find_function():
    instance_id = "django__django-14855"
    instance = get_moatless_instance(instance_id)
    repository = create_repository(instance)
    code_index = await create_index_async(instance, repository)
    file_context = FileContext(repo=repository)
    # Mock the completion model
    completion_model = AsyncMock(spec=BaseCompletionModel)

    # Create and initialize workspace
    workspace = Workspace(repository=repository, code_index=code_index)
    
    action = FindFunction(
        repository=repository, code_index=code_index, completion_model=completion_model
    )
    # Initialize the action with the workspace
    await action.initialize(workspace)

    action_args = FindFunctionArgs(
        scratch_pad="",
        function_name="cached_eval",
        file_pattern="**/*.py",
    )

    message = await action.execute(action_args, file_context)


@pytest.mark.asyncio
async def test_find_function_with_mocks():
    """Test FindFunction with completely mocked dependencies."""
    # Setup - create mocks
    repository = MagicMock(spec=Repository)
    repository.file_exists.return_value = True  # Ensure file exists
    repository.get_file_content.return_value = "def test_function():\n    pass"
    
    code_index = AsyncMock(spec=CodeIndex)
    file_context = FileContext(repo=repository)
    completion_model = AsyncMock(spec=BaseCompletionModel)
    
    # Mock search response
    mock_span_hit = SpanHit(span_id="test_function")
    mock_search_hit = SearchCodeHit(
        file_path="test_file.py",
        spans=[mock_span_hit]
    )
    mock_search_response = SearchCodeResponse(hits=[mock_search_hit])
    
    # Configure the mock to return our predefined response
    code_index.find_function.return_value = mock_search_response
    
    # Create workspace and action
    workspace = Workspace(repository=repository, code_index=code_index)
    action = FindFunction(
        repository=repository, code_index=code_index, completion_model=completion_model
    )
    await action.initialize(workspace)
    
    # Execute action
    action_args = FindFunctionArgs(
        scratch_pad="",
        function_name="test_function",
        file_pattern="**/*.py",
    )
    
    # Since we can't easily mock the span system, add the span directly to the file_context
    context_file = file_context.add_file("test_file.py")
    context_file.spans.append(ContextSpan(span_id="test_function"))
    
    message = await action.execute(action_args, file_context)
    
    # Verify
    code_index.find_function.assert_awaited_once_with(
        "test_function", class_name=None, file_pattern="**/*.py"
    )
    assert len(file_context.files) == 1
    assert "test_function" in file_context.files[0].span_ids
