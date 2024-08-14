import pytest
from unittest.mock import Mock, patch
from moatless.edit.plan import PlanToCode, ApplyChange
from moatless.schema import ActionResponse, ActionTransaction
from moatless.workspace import Workspace
from moatless.file_context import FileContext

class TestPlanToCode:
    @pytest.fixture
    def plan_to_code(self):
        mock_file_repo = Mock()
        mock_workspace = Workspace(file_repo=mock_file_repo)
        mock_file_context = Mock(spec=FileContext)
        
        return PlanToCode(
            id=1,
            _workspace=mock_workspace,
            _initial_message="Test initial message",
            file_context=mock_file_context
        )

    def test_action_type(self, plan_to_code):
        assert plan_to_code.action_type() == ApplyChange

    def test_execute_action_finish(self, plan_to_code):
        action = ApplyChange(
            scratch_pad="Finished",
            action="finish",
            finish="Task completed successfully"
        )

        response = plan_to_code._execute_action(action)

        assert isinstance(response, ActionResponse)
        assert response.trigger == "finish"
        assert response.output["message"] == "Task completed successfully"

    def test_execute_action_reject(self, plan_to_code):
        action = ApplyChange(
            scratch_pad="Rejected",
            action="reject",
            reject="Cannot complete the task"
        )

        response = plan_to_code._execute_action(action)

        assert isinstance(response, ActionResponse)
        assert response.trigger == "reject"
        assert response.output["message"] == "Cannot complete the task"

    def test_execute_action_review(self, plan_to_code):
        action = ApplyChange(
            scratch_pad="Review needed",
            action="review"
        )

        response = plan_to_code._execute_action(action)

        assert isinstance(response, ActionResponse)
        assert response.trigger == "retry"
        assert "Review isn't possible" in response.retry_message

    @patch('moatless.edit.plan.PlanToCode._request_for_change')
    def test_execute_action_apply_change(self, mock_request_for_change, plan_to_code):
        action = ApplyChange(
            scratch_pad="Applying change",
            action="modify",
            file_path="test.py",
            span_id="span1",
            instructions="Update function"
        )

        mock_request_for_change.return_value = ActionResponse(trigger="edit_code")

        response = plan_to_code._execute_action(action)

        assert isinstance(response, ActionResponse)
        assert response.trigger == "edit_code"
        mock_request_for_change.assert_called_once_with(action)

    @patch('moatless.file_context.FileContext.create_prompt')
    def test_messages(self, mock_create_prompt, plan_to_code):
        mock_create_prompt.return_value = "Mock file context"
        
        messages = plan_to_code.messages()

        assert len(messages) == 1
        assert "<issue>" in messages[0].content
        assert "Test initial message" in messages[0].content
        assert "<file_context>" in messages[0].content
        assert "Mock file context" in messages[0].content
        
        mock_create_prompt.assert_called_once()

    @patch('moatless.file_context.FileContext.get_file')
    @patch('moatless.file_context.FileContext.get_spans')
    def test_request_for_change_file_not_found(self, mock_get_spans, mock_get_file, plan_to_code):
        mock_get_file.return_value = None
        mock_get_spans.return_value = []

        action = ApplyChange(
            scratch_pad="Change request",
            action="modify",
            file_path="nonexistent.py",
            span_id="span1",
            instructions="Update function"
        )

        response = plan_to_code._request_for_change(action)

        assert isinstance(response, ActionResponse)
        assert response.trigger == "retry"
        assert "File nonexistent.py is not found in the file context" in response.retry_message

    # Add more tests for other scenarios in _request_for_change method