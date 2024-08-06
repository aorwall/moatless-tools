import pytest
from unittest.mock import MagicMock
from moatless.state import AgenticState, NoopState, Finished
from moatless.workspace import Workspace
from moatless.repository import FileRepository
from moatless.file_context import FileContext
from moatless.types import ActionRequest, ActionResponse, Content, FileWithSpans, Usage


class ConcreteAgenticState(AgenticState):
    def _execute_action(self, action: ActionRequest) -> ActionResponse:
        return ActionResponse(output={"content": "Test response"})


@pytest.fixture
def test_state():
    return ConcreteAgenticState(id=1)


def test_agentic_state_initialization(test_state):
    assert test_state.id == 1
    assert test_state.include_message_history == False
    assert test_state.model is None
    assert test_state.temperature == 0.0
    assert test_state.max_tokens == 1000
    assert test_state.max_iterations is None


def test_agentic_state_name(test_state):
    assert test_state.name == "ConcreteAgenticState"


def test_agentic_state_create_file_context(test_state):
    mock_workspace = MagicMock(spec=Workspace)
    test_state._workspace = mock_workspace

    files = [FileWithSpans(file_path="test.py", content="print('hello')", spans=[])]
    test_state.create_file_context(files)

    mock_workspace.create_file_context.assert_called_once_with(files)


def test_agentic_state_model_dump(test_state):
    dump = test_state.model_dump(exclude_none=True)
    assert dump == {'id': 1, 'include_message_history': False, 'max_tokens': 1000, 'temperature': 0.0}


def test_agentic_state_equality_same_state():
    state1 = ConcreteAgenticState(id=1, temperature=0.5, max_tokens=500)
    state2 = ConcreteAgenticState(id=1, temperature=0.5, max_tokens=500)
    assert state1 == state2

def test_agentic_state_equality_different_state():
    state1 = ConcreteAgenticState(id=1, temperature=0.5, max_tokens=500)
    state2 = ConcreteAgenticState(id=2, temperature=0.7, max_tokens=500)
    assert state1 != state2

def test_agentic_state_equality_different_types():
    state1 = ConcreteAgenticState(id=1)
    state2 = NoopState(id=2)
    assert state1 != state2

def test_handle_action(test_state):
    action = ActionRequest(content="Test action")
    usage = Usage(prompt_tokens=10, completion_tokens=20, completion_cost=0.2)
    response = test_state.handle_action(action, usage)
    
    assert isinstance(response, ActionResponse)
    assert response.output == {"content": "Test response"}
    assert len(test_state._actions) == 1
    assert test_state._actions[0].request == action
    assert test_state._actions[0].response == response
    assert test_state._actions[0].usage == usage

def test_handle_action_executed_state():
    state = ConcreteAgenticState(id=1)
    state._executed = True
    
    with pytest.raises(ValueError, match="State has already been executed"):
        state.handle_action(ActionRequest(content="Test"), None)

def test_last_action(test_state):
    assert test_state.last_action is None
    
    action = Content(content="Test action")
    test_state.handle_action(action, None)
    
    assert test_state.last_action is not None
    assert test_state.last_action.request == action

def test_response(test_state):
    assert test_state.response is None
    
    action = Content(content="Test action")
    response = test_state.handle_action(action, None)
    
    assert test_state.response == response

def test_retries(test_state):
    assert test_state.retries() == 0
    
    test_state.handle_action(Content(content="Test 1"), None)
    assert test_state.retries() == 0
    
    test_state.handle_action(Content(content="Test 2"), None)
    test_state._actions[-1].response.trigger = "retry"
    assert test_state.retries() == 1
    
    test_state.handle_action(Content(content="Test 3"), None)
    test_state._actions[-1].response.trigger = "retry"
    assert test_state.retries() == 2

def test_retry_messages(test_state):
    test_state.handle_action(Content(content="Test 1"), None)
    test_state._actions[-1].response.retry_message = "Retry 1"
    
    test_state.handle_action(Content(content="Test 2"), None)
    test_state._actions[-1].response.retry_message = "Retry 2"
    
    messages = test_state.retry_messages()
    assert len(messages) == 4
    assert messages[0].role == "assistant"
    assert messages[0].content == "Test 1"
    assert messages[1].role == "user"
    assert messages[1].content == "Retry 1"
    assert messages[2].role == "assistant"
    assert messages[2].content == "Test 2"
    assert messages[3].role == "user"
    assert messages[3].content == "Retry 2"

def test_clone(test_state):
    test_state.handle_action(Content(content="Test"), None)
    cloned_state = test_state.clone()
    
    assert cloned_state.id == test_state.id
    assert cloned_state.model_dump() == test_state.model_dump()
    assert len(cloned_state._actions) == 0  # Actions should not be cloned
    assert cloned_state._executed == False

def test_total_cost(test_state):
    usage1 = Usage(prompt_tokens=10, completion_tokens=20, completion_cost=0.2)
    usage2 = Usage(prompt_tokens=15, completion_tokens=25, completion_cost=0.25)
    
    test_state.handle_action(Content(content="Test 1"), usage1)
    test_state.handle_action(Content(content="Test 2"), usage2)
    
    assert test_state.total_cost() == pytest.approx(0.45)

def test_finished_state_creation_and_dump():
    message = "Task completed successfully"
    output = {"result": "success", "data": [1, 2, 3]}
    
    finished_state = Finished.model_validate({"id": 1, "message": message, "output": output})
    
    assert finished_state.id == 1
    assert finished_state.message == message
    assert finished_state.output == output
    
    dumped_state = finished_state.model_dump()
    
    assert dumped_state["id"] == 1
    assert dumped_state["message"] == message
    assert dumped_state["output"] == output