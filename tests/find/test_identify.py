import pytest
from moatless.codeblocks.codeblocks import BlockSpan, SpanType
from moatless.find.identify import IdentifyCode, Identify, is_test_pattern
from moatless.file_context import RankedFileSpan
from moatless.repository.file import CodeFile
from moatless.schema import FileWithSpans
from moatless.state import StateOutcome
from moatless.workspace import Workspace
from unittest.mock import Mock, MagicMock

class TestIdentifyCode:
    @pytest.fixture
    def identify_code(self):
        mock_file_repo = Mock()
        mock_workspace = Workspace(file_repo=mock_file_repo)
        
        mock_module = Mock()
        mock_module.find_span_by_id.side_effect = lambda span_id: BlockSpan(
            span_type=SpanType.IMPLEMENTATION,
            span_id=span_id,
            start_line=0,
            end_line=10,
        )
        
        mock_code_file = MagicMock(spec=CodeFile)
        mock_code_file.content = "Mock file content"
        mock_code_file.file_path = "test.py"
        mock_code_file.module = mock_module
        
        mock_file_repo.get_file.side_effect = lambda path: mock_code_file
        
        return IdentifyCode(id=1, _workspace=mock_workspace, _initial_message="Test initial message")

    def test_action_type(self, identify_code):
        assert identify_code.action_type() == Identify

    def test_system_prompt(self, identify_code):
        assert isinstance(identify_code.system_prompt(), str)
        assert "You are an autonomous AI assistant" in identify_code.system_prompt()

    def test_execute_action_with_identified_spans(self, identify_code):
        action = Identify(
            scratch_pad="Test scratch pad",
            identified_spans=[
                FileWithSpans(file_path="test.py", span_ids=["span1", "span2"])
            ]
        )

        response = identify_code._execute_action(action)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "finish"

        # Verify that the file was added to the file context
        assert "test.py" in identify_code.file_context._file_context
        context_file = identify_code.file_context._file_context["test.py"]
        assert context_file.file_path == "test.py"
        assert set(context_file.span_ids) == {"span1", "span2"}

    def test_execute_action_without_identified_spans(self, identify_code):
        identify_code.ranked_spans = [RankedFileSpan(file_path="test.py", span_id="span1", rank=1)]
        action = Identify(scratch_pad="No relevant spans found")

        response = identify_code._execute_action(action)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "search"
        assert "The search returned 1 results" in response.output["message"]

    def test_messages(self, identify_code):
        messages = identify_code.messages()

        assert len(messages) == 1
        assert "<issue>" in messages[0].content
        assert "Test initial message" in messages[0].content
        assert "<file_context>" in messages[0].content
        assert "<search_results>" in messages[0].content

    def test_initial_message(self, identify_code):
        assert identify_code.initial_message == "Test initial message"

    def test_workspace_initialization(self, identify_code):
        assert identify_code._workspace is not None
        assert isinstance(identify_code._workspace, Workspace)

def test_is_test_pattern():
    assert is_test_pattern("test_file.py") == True
    assert is_test_pattern("file_test.py") == False
    assert is_test_pattern("/tests/some_file.py") == True
    assert is_test_pattern("src/main.py") == False
    assert is_test_pattern("test_utils/helper.py") == True