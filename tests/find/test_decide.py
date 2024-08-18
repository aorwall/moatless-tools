import pytest
from moatless.find.decide import DecideRelevance, Decision
from moatless.find.identify import Identify, IdentifyCode
from moatless.state import StateOutcome, ActionTransaction
from moatless.workspace import Workspace
from moatless.file_context import FileContext
from unittest.mock import Mock, MagicMock, patch

class TestDecideRelevance:
    @pytest.fixture
    def decide_relevance(self):
        mock_file_repo = Mock()
        mock_workspace = Workspace(file_repo=mock_file_repo)
        mock_file_context = Mock(spec=FileContext)
        
        return DecideRelevance(
            id=1,
            _workspace=mock_workspace,
            _initial_message="Test initial message",
            expand_context=False,
            file_context=mock_file_context
        )

    def test_action_type(self, decide_relevance):
        assert decide_relevance.action_type() == Decision

    def test_execute_action_complete_and_relevant(self, decide_relevance):
        action = Decision(
            scratch_pad="Complete and relevant",
            relevant=True,
            complete=True
        )

        response = decide_relevance._execute_action(action)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "finish"

    def test_execute_action_relevant_but_not_complete(self, decide_relevance):
        decide_relevance.finish_after_relevant_count = 1
        decide_relevance._relevant_count = Mock(return_value=1)
        action = Decision(
            scratch_pad="Relevant but not complete",
            relevant=True,
            complete=False
        )

        response = decide_relevance._execute_action(action)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "finish"

    def test_execute_action_not_relevant_not_complete(self, decide_relevance):
        action = Decision(
            scratch_pad="Not relevant, not complete",
            relevant=False,
            complete=False,
            search_suggestions="Try searching for X"
        )

        response = decide_relevance._execute_action(action)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "search"
        assert response.output["message"] == "Try searching for X"

    def test_relevant_count(self, decide_relevance: DecideRelevance):
        state3 = DecideRelevance(id=3, expand_context=False, file_context=Mock())
        state3._actions = [ActionTransaction(request=Decision(scratch_pad="Test", relevant=True), response=StateOutcome(trigger="finish"))]
        state2 = DecideRelevance(id=2, expand_context=False, file_context=Mock())
        state2._actions = [ActionTransaction(request=Decision(scratch_pad="Test", relevant=False), response=StateOutcome(trigger="finish"))]
        state2.previous_state = state3
        state1 = DecideRelevance(id=1, expand_context=False, file_context=Mock())
        state1._actions = [ActionTransaction(request=Decision(scratch_pad="Test", relevant=True), response=StateOutcome(trigger="finish"))]
        state1.previous_state = state2
        
        decide_relevance.previous_state = state1
        assert len(decide_relevance.get_previous_states(decide_relevance)) == 3
        assert decide_relevance._relevant_count() == 2

    @patch('moatless.file_context.FileContext.create_prompt')
    def test_messages(self, mock_create_prompt, decide_relevance):
        mock_create_prompt.return_value = "Mock file context"
        
        messages = decide_relevance.messages()

        assert len(messages) == 1
        assert "<issue>" in messages[0].content
        assert "Test initial message" in messages[0].content
        assert "<file_context>" in messages[0].content
        assert "Mock file context" in messages[0].content
        
        mock_create_prompt.assert_called_once()

    @patch('moatless.file_context.FileContext.create_prompt')
    def test_messages_with_last_scratch_pad(self, mock_create_prompt, decide_relevance):
        mock_create_prompt.return_value = "Mock file context"

        previous_state = IdentifyCode(id=3)
        previous_state._actions = [ActionTransaction(request=Identify(scratch_pad="Previous scratch pad", relevant=True))]
        decide_relevance.previous_state = previous_state
        
        messages = decide_relevance.messages()

        assert len(messages) == 1
        assert "<scratch_pad>" in messages[0].content
        assert "Previous scratch pad" in messages[0].content

    def test_system_prompt(self, decide_relevance):
        system_prompt = decide_relevance.system_prompt()
        
        assert "You will be provided a reported issue and the file context" in system_prompt
        assert "Analyze the Issue:" in system_prompt
        assert "Analyze File Context:" in system_prompt
        assert "Make a Decision:" in system_prompt