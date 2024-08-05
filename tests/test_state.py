import pytest
from unittest.mock import MagicMock
from moatless.state import AgenticState, NoopState
from moatless.workspace import Workspace
from moatless.repository import FileRepository
from moatless.file_context import FileContext
from moatless.types import ActionRequest, ActionResponse, FileWithSpans


class ConcreteAgenticState(AgenticState):
    def _execute_action(self, action: ActionRequest) -> ActionResponse:
        return ActionResponse(content="Test response")


@pytest.fixture
def test_state():
    return ConcreteAgenticState()


def test_agentic_state_initialization(test_state):
    assert test_state.include_message_history == False
    assert test_state.model is None
    assert test_state.temperature == 0.0
    assert test_state.max_tokens == 1000
    assert test_state.max_iterations is None


def test_agentic_state_name(test_state):
    assert test_state.name == "ConcreteAgenticState"


def test_agentic_state_set_loop(test_state):
    mock_loop = MagicMock()
    test_state._set_loop(mock_loop)
    assert test_state.loop == mock_loop


def test_agentic_state_workspace_properties(test_state):
    mock_loop = MagicMock()
    mock_workspace = MagicMock(spec=Workspace)
    mock_file_repo = MagicMock(spec=FileRepository)
    mock_file_context = MagicMock(spec=FileContext)

    mock_loop.workspace = mock_workspace
    mock_workspace.file_repo = mock_file_repo
    mock_workspace.file_context = mock_file_context

    test_state._set_loop(mock_loop)

    assert test_state.workspace == mock_workspace
    assert test_state.file_repo == mock_file_repo
    assert test_state.file_context == mock_file_context


def test_agentic_state_create_file_context(test_state):
    mock_loop = MagicMock()
    mock_workspace = MagicMock(spec=Workspace)
    mock_loop.workspace = mock_workspace

    test_state._set_loop(mock_loop)

    files = [FileWithSpans(file_path="test.py", content="print('hello')", spans=[])]
    test_state.create_file_context(files)

    mock_workspace.create_file_context.assert_called_once_with(files)


def test_agentic_state_transition_to(test_state):
    mock_loop = MagicMock()
    test_state._set_loop(mock_loop)

    new_state = NoopState()
    test_state.transition_to(new_state)

    mock_loop.transition_to.assert_called_once_with(new_state)


def test_agentic_state_model_dump(test_state):
    dump = test_state.model_dump()
    assert "name" in dump
    assert dump["name"] == "ConcreteAgenticState"

def test_agentic_state_equality_same_state():
    state1 = ConcreteAgenticState(temperature=0.5, max_tokens=500)
    state2 = ConcreteAgenticState(temperature=0.5, max_tokens=500)
    assert state1 == state2

def test_agentic_state_equality_different_state():
    state1 = ConcreteAgenticState(temperature=0.5, max_tokens=500)
    state2 = ConcreteAgenticState(temperature=0.7, max_tokens=500)
    assert state1 != state2

def test_agentic_state_equality_different_types():
    state1 = ConcreteAgenticState()
    state2 = NoopState()
    assert state1 != state2

def test_agentic_state_equality_with_file_context():
    mock_loop1 = MagicMock()
    mock_workspace1 = MagicMock(spec=Workspace)
    mock_file_context1 = MagicMock(spec=FileContext)
    mock_loop1.workspace = mock_workspace1
    mock_workspace1.file_context = mock_file_context1
    mock_file_context1.model_dump.return_value = {"files": [{"file_path": "test.py", "spans": []}]}

    mock_loop2 = MagicMock()
    mock_workspace2 = MagicMock(spec=Workspace)
    mock_file_context2 = MagicMock(spec=FileContext)
    mock_loop2.workspace = mock_workspace2
    mock_workspace2.file_context = mock_file_context2
    mock_file_context2.model_dump.return_value = {"files": [{"file_path": "test.py", "spans": []}]}

    state1 = ConcreteAgenticState()
    state2 = ConcreteAgenticState()
    
    state1._set_loop(mock_loop1)
    state2._set_loop(mock_loop2)

    assert state1 == state2

def test_agentic_state_inequality_with_different_file_context():
    mock_loop1 = MagicMock()
    mock_workspace1 = MagicMock(spec=Workspace)
    mock_file_context1 = MagicMock(spec=FileContext)
    mock_loop1.workspace = mock_workspace1
    mock_workspace1.file_context = mock_file_context1
    mock_file_context1.model_dump.return_value = {"files": [{"file_path": "test1.py", "spans": ["foo"]}]}

    mock_loop2 = MagicMock()
    mock_workspace2 = MagicMock(spec=Workspace)
    mock_file_context2 = MagicMock(spec=FileContext)
    mock_loop2.workspace = mock_workspace2
    mock_workspace2.file_context = mock_file_context2
    mock_file_context2.model_dump.return_value = {"files": [{"file_path": "test1.py", "spans": ["bar"]}]}

    state1 = ConcreteAgenticState()
    state2 = ConcreteAgenticState()
    
    state1._set_loop(mock_loop1)
    state2._set_loop(mock_loop2)

    assert state1 != state2

def test_agentic_state_equality_without_loop():
    state1 = ConcreteAgenticState(temperature=0.5, max_tokens=500)
    state2 = ConcreteAgenticState(temperature=0.5, max_tokens=500)
    assert state1 == state2

def test_agentic_state_equality_one_with_loop():
    mock_loop = MagicMock()
    mock_workspace = MagicMock(spec=Workspace)
    mock_file_context = MagicMock(spec=FileContext)
    mock_loop.workspace = mock_workspace
    mock_workspace.file_context = mock_file_context
    mock_file_context.model_dump.return_value = {"files": [{"file_path": "test.py", "spans": []}]}

    state1 = ConcreteAgenticState(temperature=0.5, max_tokens=500)
    state2 = ConcreteAgenticState(temperature=0.5, max_tokens=500)
    
    state1._set_loop(mock_loop)

    assert state1 == state2