import pytest

from moatless.benchmark.swebench import create_workspace
from moatless.benchmark.utils import get_moatless_instance, get_moatless_instances
from moatless.find.search import SearchCode, Search, SearchRequest
from moatless.state import StateOutcome
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


def test_find_impl_span():

    instances = get_moatless_instances(split="verified")

    # Filter and sort instances
    filtered_instances = {
        k: v for k, v in instances.items()
        if "django__django-" in k and "12273" <= k.split("-")[-1] <= "12419"
    }
    sorted_instances = dict(sorted(filtered_instances.items()))

    for instance_id, instance in sorted_instances.items():
        print(f"Instance: {instance_id}")
        workspace = create_workspace(instance)

        search_code = SearchCode(id=0, _workspace=workspace, initial_message="Test initial message")

        mocked_action = Search(
            scratch_pad="Applying change",
            search_requests=[
                SearchRequest(file_pattern="**/global_settings.py", query="SECURE_REFERRER_POLICY setting")
            ]
        )

        outcome = search_code.execute(mocked_action)
        print(outcome)

        workspace.file_context.add_ranked_spans(outcome.output["ranked_spans"])
        assert "SECURE_REFERRER_POLICY" in workspace.file_context.create_prompt()

def test_find():
    instance_id = "django__django-12419" #
    instance = get_moatless_instance(instance_id, split="verified")
    print(f"Instance: {instance_id}")
    workspace = create_workspace(instance)

    search_code = SearchCode(id=0, _workspace=workspace, initial_message="Test initial message")

    mocked_action = Search(
        scratch_pad="Applying change",
        search_requests=[
            SearchRequest(file_pattern="**/global_settings.py", query="SECURE_REFERRER_POLICY setting")
        ]
    )

    outcome = search_code.execute(mocked_action)

    for ranked_span in outcome.output["ranked_spans"]:
        print(ranked_span)

    workspace.file_context.add_ranked_spans(outcome.output["ranked_spans"])
    print(workspace.file_context.create_prompt(show_span_ids=True))
    assert "SECURE_REFERRER_POLICY = None" in workspace.file_context.create_prompt()


def test_find_2():
    instance_id = "django__django-15104"
    instance = get_moatless_instance(instance_id, split="verified")
    workspace = create_workspace(instance)

    search_code = SearchCode(id=0, _workspace=workspace, initial_message="Test initial message")

    print(instance["expected_spans"])

    mocked_action = Search(
        scratch_pad="Applying change",
        max_search_results=250,
        search_requests=[
            SearchRequest(file_pattern="**/migrations/*.py", query="MigrationAutodetector class with generate_renamed_models method")
        ]
    )

    outcome = search_code.execute(mocked_action)

    for span in outcome.output["ranked_spans"]:
        print(span)