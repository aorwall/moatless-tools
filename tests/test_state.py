import pytest
from unittest.mock import MagicMock
from moatless.state import State, AgenticState, NoopState, Finished
from moatless.workspace import Workspace
from moatless.repository import FileRepository
from moatless.file_context import FileContext
from moatless.schema import ActionRequest, StateOutcome, Completion, Content, FileWithSpans, Usage

class ConcreteState(State):
    def clone(self):
        return ConcreteState(**self.model_dump())

class ConcreteAgenticState(AgenticState):
    def _execute_action(self, action: ActionRequest) -> StateOutcome:
        return StateOutcome(output={"content": "Test response"})

    def action_type(self):
        return ActionRequest

@pytest.fixture
def test_state():
    return ConcreteState(id=1)

@pytest.fixture
def test_agentic_state():
    return ConcreteAgenticState(id=1)

def test_state_initialization(test_state):
    assert test_state.id == 1
    assert test_state.previous_state is None
    assert test_state.next_states == []

def test_state_name(test_state):
    assert test_state.name == "ConcreteState"

def test_state_executed(test_state):
    assert test_state.executed == False

def test_state_create_file_context(test_state):
    mock_workspace = MagicMock(spec=Workspace)
    test_state._workspace = mock_workspace

    files = [FileWithSpans(file_path="test.py", content="print('hello')", spans=[])]
    test_state.create_file_context(files)

    mock_workspace.create_file_context.assert_called_once_with(files)

def test_state_get_previous_states(test_state):
    prev_state1 = ConcreteState(id=2)
    prev_state2 = ConcreteState(id=3)
    test_state.previous_state = prev_state1
    prev_state1.previous_state = prev_state2

    previous_states = test_state.get_previous_states()
    assert len(previous_states) == 2
    assert previous_states[0].id == 3
    assert previous_states[1].id == 2

def test_state_clone(test_state):
    cloned_state = test_state.clone()
    assert cloned_state.id == test_state.id
    assert cloned_state.model_dump() == test_state.model_dump()

def test_state_equality(test_state):
    same_state = ConcreteState(id=1)
    different_state = ConcreteState(id=2)
    
    assert test_state == same_state
    assert test_state != different_state

# Existing AgenticState tests (modified)

def test_agentic_state_initialization(test_agentic_state):
    assert test_agentic_state.id == 1
    assert test_agentic_state.include_message_history == False
    assert test_agentic_state.model is None
    assert test_agentic_state.temperature == 0.0
    assert test_agentic_state.max_tokens == 1000
    assert test_agentic_state.max_iterations is None

def test_agentic_state_name(test_agentic_state):
    assert test_agentic_state.name == "ConcreteAgenticState"

def test_agentic_state_model_dump(test_agentic_state):
    dump = test_agentic_state.model_dump(exclude_none=True)
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

def test_handle_action(test_agentic_state):
    action = ActionRequest(content="Test action")
    completion = Completion(model="gpt-4", input=[], output={}, usage=Usage(prompt_tokens=10, completion_tokens=20, completion_cost=0.2))
    response = test_agentic_state.handle_action(action, completion)
    
    assert isinstance(response, StateOutcome)
    assert response.output == {"content": "Test response"}
    assert len(test_agentic_state._actions) == 1
    assert test_agentic_state._actions[0].request == action
    assert test_agentic_state._actions[0].response == response
    assert test_agentic_state._actions[0].completion == completion

def test_handle_action_executed_state():
    state = ConcreteAgenticState(id=1)
    state._executed = True
    
    with pytest.raises(ValueError, match="State has already been executed"):
        state.handle_action(ActionRequest(content="Test"), None)

def test_last_action(test_agentic_state):
    assert test_agentic_state.last_action is None
    
    action = Content(content="Test action")
    test_agentic_state.handle_action(action, None)
    
    assert test_agentic_state.last_action is not None
    assert test_agentic_state.last_action.request == action

def test_response(test_agentic_state):
    assert test_agentic_state.outcome is None
    
    action = Content(content="Test action")
    response = test_agentic_state.handle_action(action, None)
    
    assert test_agentic_state.last_action.response == response

def test_retries(test_agentic_state):
    assert test_agentic_state.retries() == 0
    
    test_agentic_state.handle_action(Content(content="Test 1"), None)
    assert test_agentic_state.retries() == 0
    
    test_agentic_state.handle_action(Content(content="Test 2"), None)
    test_agentic_state._actions[-1].response.trigger = "retry"
    assert test_agentic_state.retries() == 1
    
    test_agentic_state.handle_action(Content(content="Test 3"), None)
    test_agentic_state._actions[-1].response.trigger = "retry"
    assert test_agentic_state.retries() == 2

def test_retry_messages(test_agentic_state):
    test_agentic_state.handle_action(Content(content="Test 1"), None)
    test_agentic_state._actions[-1].response.retry_message = "Retry 1"
    
    test_agentic_state.handle_action(Content(content="Test 2"), None)
    test_agentic_state._actions[-1].response.retry_message = "Retry 2"
    
    messages = test_agentic_state.retry_messages()
    assert len(messages) == 4
    assert messages[0].role == "assistant"
    assert messages[0].content == "Test 1"
    assert messages[1].role == "user"
    assert messages[1].content == "Retry 1"
    assert messages[2].role == "assistant"
    assert messages[2].content == "Test 2"
    assert messages[3].role == "user"
    assert messages[3].content == "Retry 2"

def test_agentic_state_clone(test_agentic_state):
    test_agentic_state.handle_action(Content(content="Test"), None)
    cloned_state = test_agentic_state.clone()
    
    assert cloned_state.id == test_agentic_state.id
    assert cloned_state.model_dump() == test_agentic_state.model_dump()
    assert len(cloned_state._actions) == 0  # Actions should not be cloned
    assert cloned_state._executed == False

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