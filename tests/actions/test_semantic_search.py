import os
import pytest
import pytest_asyncio
from typing import Any, Dict, cast
from unittest.mock import AsyncMock, MagicMock, patch

from moatless.actions.schema import Observation
from moatless.actions.semantic_search import SemanticSearch, SemanticSearchArgs
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.index.types import SearchCodeHit, SearchCodeResponse, SpanHit
from moatless.repository.repository import InMemRepository, Repository
from moatless.workspace import Workspace


@pytest.fixture
def repo():
    """Create an in-memory repository with sample files for testing."""
    repo = InMemRepository()
    repo.save_file("file1.py", """def method1():
    return "sample method 1"
""")
    repo.save_file("file2.py", """def method2():
    return "sample method 2"
    
def method3():
    return "sample method 3"
""")
    return repo


@pytest_asyncio.fixture
async def workspace(repo):
    """Create a workspace using the in-memory repository with a mocked code_index."""
    workspace = Workspace(repository=repo)
    # Create a mock code_index
    mock_code_index = MagicMock(spec=CodeIndex)
    workspace._code_index = mock_code_index
    return workspace


@pytest.mark.asyncio
async def test_semantic_search_without_identifier(workspace, repo):
    """Test semantic search action with use_identifier=False."""
    # Create mock search response
    search_result = SearchCodeResponse()
    search_result.hits = [
        SearchCodeHit(
            file_path="file1.py",
            spans=[
                SpanHit(span_id="method1", rank=0, tokens=20)
            ]
        ),
        SearchCodeHit(
            file_path="file2.py",
            spans=[
                SpanHit(span_id="method2", rank=1, tokens=25),
                SpanHit(span_id="method3", rank=2, tokens=25)
            ]
        )
    ]
    
    # Create args for the search
    args = SemanticSearchArgs(
        thoughts="Searching for sample methods",
        query="sample methods",
        category="implementation"
    )
    
    # Create file context
    file_context = FileContext(repo=repo)
    
    # Create the SemanticSearch action with use_identifier=False
    search_action = SemanticSearch(
        use_identifier=False,
        max_search_tokens=2000,
        max_identify_tokens=8000,
        max_identify_prompt_tokens=16000,
        max_hits=10
    )
    
    # Set the workspace directly
    search_action._workspace = workspace
    
    # Mock the _search method to return our search result
    with patch.object(search_action, '_search', return_value=search_result):
        # Execute the search action
        observation = await search_action.execute(args, file_context)
        
        # Verify the output format when use_identifier is False
        message = cast(str, observation.message)
        
        # Test that the message contains expected information
        assert "Found 3 matches in 2 files" in message
        assert "File: file1.py" in message
        assert "File: file2.py" in message
        assert "Line:" in message
        
        # Check properties exist but don't assert specific content
        # since it depends on file_context integration
        properties = cast(Dict[str, Any], observation.properties)
        assert "search_tokens" in properties
        assert "search_hits" in properties  # This is the raw hits from the search result


@pytest.mark.asyncio
async def test_semantic_search_empty_results(workspace, repo):
    """Test semantic search action with use_identifier=False and empty results."""
    # Create empty search response
    search_result = SearchCodeResponse()
    
    # Create args for the search
    args = SemanticSearchArgs(
        thoughts="Searching for nonexistent methods",
        query="nonexistent methods",
        category="implementation"
    )
    
    # Create file context
    file_context = FileContext(repo=repo)
    
    # Create the SemanticSearch action with use_identifier=False
    search_action = SemanticSearch(
        use_identifier=False,
        max_search_tokens=2000,
        max_identify_tokens=8000,
        max_identify_prompt_tokens=16000,
        max_hits=10
    )
    
    # Set the workspace directly
    search_action._workspace = workspace
    
    # Mock the _search and _search_for_alternative_suggestion methods
    with patch.object(search_action, '_search', return_value=search_result), \
         patch.object(search_action, '_search_for_alternative_suggestion', return_value=search_result):
        # Execute the search action
        observation = await search_action.execute(args, file_context)
        
        # Verify the output format for empty results
        message = cast(str, observation.message)
        properties = cast(Dict[str, Any], observation.properties)
        
        assert "No search results found" in message
        assert "fail_reason" in properties
        assert properties["fail_reason"] == "no_search_hits"


@pytest.mark.asyncio
async def test_semantic_search_with_file_pattern(workspace, repo):
    """Test semantic search action with use_identifier=False and file pattern."""
    # Create mock search response
    search_result = SearchCodeResponse()
    search_result.hits = [
        SearchCodeHit(
            file_path="file2.py",
            spans=[
                SpanHit(span_id="method2", rank=1, tokens=25)
            ]
        )
    ]
    
    # Create args for the search with file pattern
    args = SemanticSearchArgs(
        thoughts="Searching for sample methods in specific files",
        query="sample methods",
        category="implementation",
        file_pattern="*2.py"
    )
    
    # Create file context
    file_context = FileContext(repo=repo)
    
    # Create the SemanticSearch action with use_identifier=False
    search_action = SemanticSearch(
        use_identifier=False,
        max_search_tokens=2000,
        max_identify_tokens=8000,
        max_identify_prompt_tokens=16000,
        max_hits=10
    )
    
    # Set the workspace directly
    search_action._workspace = workspace
    
    # Mock the _search method to return our mock search result
    with patch.object(search_action, '_search', return_value=search_result):
        # Execute the search action
        observation = await search_action.execute(args, file_context)
        
        # Verify the output format
        message = cast(str, observation.message)
        
        # Test that the message contains expected information
        assert "Found 1 match" in message
        assert "File: file2.py" in message
        assert "file1.py" not in message
        
        # Check properties exist but don't assert specific content
        # since it depends on file_context integration
        properties = cast(Dict[str, Any], observation.properties)
        assert "search_tokens" in properties
        assert "search_hits" in properties  # This is the raw hits from the search result 