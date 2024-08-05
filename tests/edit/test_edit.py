import pytest
from unittest.mock import Mock, patch
from moatless.edit.edit import EditCode
from moatless.repository.file import UpdateResult
from moatless.types import ActionResponse, Content
from moatless.workspace import Workspace
from moatless.file_context import FileContext
from moatless.repository import CodeFile

class TestEditCode:
    
    @pytest.fixture
    def edit_code(self):
        mock_file_repo = Mock()
        mock_workspace = Workspace(file_repo=mock_file_repo)
        
        return EditCode(
            id=1,
            instructions="Update function",
            file_path="test.py",
            span_id="span1",
            verify=False,
            start_line=1,
            end_line=5,
            _workspace=mock_workspace,
            model="gpt-3.5-turbo"
        )

    def test_required_fields(self, edit_code: EditCode):
        assert edit_code.required_fields() == {"instructions", "file_path", "span_id", "start_line", "end_line"}

    @patch('moatless.edit.edit.EditCode.file_context')
    def test_init(self, mock_file_context, edit_code: EditCode):
        mock_file = Mock(spec=CodeFile)
        mock_file.content = "line1\nline2\nline3\nline4\nline5"
        mock_file_wrapper = Mock(file=mock_file)
        mock_file_context.get_file.return_value = mock_file_wrapper

        edit_code.init()

        assert edit_code._code_to_replace == "line1\nline2\nline3\nline4\nline5"

    @patch('moatless.edit.edit.EditCode.file_context')
    def test_execute_action_reject(self, mock_file_context, edit_code: EditCode):
        content = Content(content="<reject>Cannot complete the task</reject>")

        response = edit_code._execute_action(content)

        assert isinstance(response, ActionResponse)
        assert response.trigger == "reject"
        assert response.output["message"] == "Cannot complete the task"

    @patch('moatless.edit.edit.EditCode.file_context')
    def test_execute_action_edit_code(self, mock_file_context, edit_code: EditCode):
        update_result = UpdateResult(diff="diff", updated=True, file_path="test.py")
        
        mock_file = Mock(spec=CodeFile)
        mock_file.update_content_by_line_numbers.return_value = update_result
        
        mock_context_file = Mock()
        mock_context_file.file = mock_file
        mock_context_file.update_content_by_line_numbers.return_value = update_result
        
        mock_file_context.get_file.return_value = mock_context_file

        content = Content(content="<replace>updated code</replace>")

        response = edit_code._execute_action(content)

        assert isinstance(response, ActionResponse)
        assert response.trigger == "finish"
        assert "Applied the change to test.py." in response.output["message"]
        assert response.output["diff"] == "diff"
        
        mock_context_file.update_content_by_line_numbers.assert_called_once()

    @patch('moatless.edit.edit.EditCode.file_context')
    def test_execute_action_retry(self, mock_file_context, edit_code: EditCode):
        mock_file = Mock(spec=CodeFile)
        mock_file.update_content_by_line_numbers.return_value = Mock(diff=None, updated=False)
        mock_context_file = Mock()
        mock_context_file.file = mock_file
        mock_context_file.update_content_by_line_numbers.return_value = Mock(diff=None, updated=False)
        mock_file_context.get_file.return_value = mock_context_file

        content = Content(content="<replace>unchanged code</replace>")

        response = edit_code._execute_action(content)

        assert isinstance(response, ActionResponse)
        assert response.trigger == "retry"
        assert "The code in the replace tag is the same as in the search" in response.retry_message

    def test_system_prompt(self, edit_code: EditCode):
        system_prompt = edit_code.system_prompt()

        assert "You are autonomous AI assisistant with superior programming skills." in system_prompt

    @patch('moatless.edit.edit.EditCode.file_context')
    def test_messages(self, mock_file_context, edit_code: EditCode):
        mock_file_context.create_prompt.return_value = "Mock file context"
        edit_code._code_to_replace = "code to replace"

        messages = edit_code.messages()

        assert len(messages) == 1
        assert "<instructions>" in messages[0].content
        assert "Update function" in messages[0].content
        assert "<file_context>" in messages[0].content
        assert "Mock file context" in messages[0].content
        assert "<search>" in messages[0].content
        assert "code to replace" in messages[0].content

    def test_action_type(self, edit_code: EditCode):
        assert edit_code.action_type() is None

    def test_stop_words(self, edit_code: EditCode):
        assert edit_code.stop_words() == ["</replace>"]