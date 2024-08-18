import pytest
from moatless.find.search import SearchCode, Search, SearchRequest
from moatless.schema import StateOutcome
from moatless.workspace import Workspace
from unittest.mock import Mock, MagicMock
from pydantic import ValidationError

class TestSearchCode:
    @pytest.fixture
    def search_code(self):
        mock_file_repo = Mock()
        mock_workspace = Workspace(file_repo=mock_file_repo)
        mock_code_index = MagicMock()
        mock_workspace.code_index = mock_code_index
        
        return SearchCode(id=1, _workspace=mock_workspace, _initial_message="Test initial message")

    def test_action_type(self, search_code):
        assert search_code.action_type() == Search

    def test_execute_action_complete(self, search_code):
        action = Search(
            scratch_pad="Search complete",
            search_requests=[],
            complete=True
        )

        response = search_code._execute_action(action)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "finish"
        assert response.output["message"] == "Search complete"

    def test_validate_search_without_search_attributes(self):
        with pytest.raises(ValidationError) as excinfo:
            Search(
                scratch_pad="Invalid search",
                search_requests=[]
            )
        
        assert "At least one search request must exist." in str(excinfo.value)

    def test_execute_action_with_search_results(self, search_code):
        mock_code_index = MagicMock()
        mock_code_index.search.return_value.hits = [
            MagicMock(file_path="test.py", spans=[MagicMock(span_id="span1", rank=1, tokens=10)])
        ]
        search_code.workspace.code_index = mock_code_index

        action = Search(
            scratch_pad="Valid search",
            search_requests=[SearchRequest(query="test query")]
        )

        response = search_code._execute_action(action)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "did_search"
        assert "ranked_spans" in response.output
        assert len(response.output["ranked_spans"]) == 1

    def test_messages(self, search_code):
        messages = search_code.messages()

        assert len(messages) == 1
        assert "<issue>" in messages[0].content
        assert "Test initial message" in messages[0].content
        assert "<file_context>" in messages[0].content

    def test_handle_direct_search_attributes(self):
        # Test with direct search attributes
        search = Search(
            scratch_pad="Test search",
            file_pattern="*.py",
            query="test query",
            code_snippet="def test_function():",
            class_names=["TestClass"],
            function_names=["test_method"]
        )

        assert len(search.search_requests) == 1
        assert search.search_requests[0].file_pattern == "*.py"
        assert search.search_requests[0].query == "test query"
        assert search.search_requests[0].code_snippet == "def test_function():"
        assert search.search_requests[0].class_names == ["TestClass"]
        assert search.search_requests[0].function_names == ["test_method"]

        # Test with both direct attributes and search_requests
        search = Search(
            scratch_pad="Test search",
            file_pattern="*.py",
            query="test query",
            search_requests=[
                SearchRequest(
                    class_names=["AnotherClass"],
                    function_names=["another_method"]
                )
            ]
        )

        assert len(search.search_requests) == 2
        assert search.search_requests[0].class_names == ["AnotherClass"]
        assert search.search_requests[0].function_names == ["another_method"]
        assert search.search_requests[1].file_pattern == "*.py"
        assert search.search_requests[1].query == "test query"

        # Test with only search_requests
        search = Search(
            scratch_pad="Test search",
            search_requests=[
                SearchRequest(
                    file_pattern="*.js",
                    query="javascript query"
                )
            ]
        )

        assert len(search.search_requests) == 1
        assert search.search_requests[0].file_pattern == "*.js"
        assert search.search_requests[0].query == "javascript query"