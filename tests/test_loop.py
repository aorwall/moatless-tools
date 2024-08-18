import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch
from moatless.loop import AgenticLoop
from moatless.state import AgenticState, Finished, Rejected, Pending
from moatless.transition_rules import TransitionRules, TransitionRule
from moatless.workspace import Workspace
from moatless.state import ActionRequest, StateOutcome, Content

from moatless.benchmark.swebench import create_workspace, load_instance
from moatless.repository import GitRepository
from moatless.settings import Settings
from moatless.trajectory import Trajectory

pytest.mark.api_keys_required = pytest.mark.skipif(
    "VOYAGE_API_KEY" not in os.environ or os.environ["VOYAGE_API_KEY"] == "",
    reason="VOYAGE_API_KEY environment variable is required"
)

class TestState(AgenticState):
    def _execute_action(self, action: ActionRequest) -> StateOutcome:
        if action.content == "reject":
            return StateOutcome(trigger="reject", output={"message": "Rejected"})
        elif action.content == "continue":
            return StateOutcome(trigger="continue", output={"message": "Continue"})
        return StateOutcome(trigger="finish", output={"message": "Finished"})

    def action_type(self):
        return Content

class TestState2(AgenticState):
    def _execute_action(self, action: ActionRequest) -> StateOutcome:
        if action.content == "reject":
            return StateOutcome(trigger="reject", output={"message": "Rejected"})
        elif action.content == "continue":
            return StateOutcome(trigger="continue", output={"message": "Continue"})
        return StateOutcome(trigger="finish", output={"message": "Finished"})

    def action_type(self):
        return Content

class TestTransitionRules(TransitionRules):
    def __init__(self, rules):
        super().__init__(transition_rules=rules)
    
    def get_next_rule(self, source, trigger, data):
        for rule in self.transition_rules:
            if rule.source == source.__class__ and rule.trigger == trigger:
                return rule
        return None

@pytest.fixture
def mock_workspace():
    workspace = MagicMock(spec=Workspace)
    workspace.snapshot.return_value = {}
    return workspace

@pytest.fixture
def test_transition_rules():
    rules = [
        TransitionRule(trigger="init", source=Pending, dest=TestState),
        TransitionRule(trigger="finish", source=TestState, dest=Finished),
        TransitionRule(trigger="continue", source=TestState, dest=TestState2),
        TransitionRule(trigger="continue", source=TestState2, dest=TestState),
        TransitionRule(trigger="reject", source=TestState, dest=Rejected),
    ]
    return TestTransitionRules(rules)

def test_loop_initialization(mock_workspace, test_transition_rules):
    loop = AgenticLoop(test_transition_rules, mock_workspace)
    assert loop.workspace == mock_workspace
    assert loop._transition_rules == test_transition_rules

def test_loop_run_until_finished(mock_workspace, test_transition_rules):
    loop = AgenticLoop(test_transition_rules, mock_workspace)
    
    with patch.object(AgenticLoop, '_next_action', return_value=(Content(content="test"), None)):
        response = loop.run("initial message")

    assert response.status == "finished"
    assert len(loop._trajectory.transitions) == 3, f"Expected 3 states, got {[state.state.name for state in loop._trajectory.transitions()]}"
    assert loop._initial_message == "initial message"
    assert isinstance(loop._trajectory.transitions[2].state, Finished)

def test_loop_run_until_rejected(mock_workspace, test_transition_rules):
    loop = AgenticLoop(test_transition_rules, mock_workspace)
    
    def mock_next_action()  :
        return Content(content="reject"), None

    with patch.object(AgenticLoop, '_next_action', side_effect=mock_next_action):
        response = loop.run("initial message")
    
    assert response.status == "rejected"
    assert len(loop._trajectory.transitions) == 3  # Pending -> TestState -> Rejected

def test_loop_max_transitions(mock_workspace, test_transition_rules):
    loop = AgenticLoop(test_transition_rules, mock_workspace, max_transitions=3)
    
    with patch.object(AgenticLoop, '_next_action', return_value=(Content(content="continue"), None)):
        response = loop.run("initial message")
    
    assert response.status == "rejected"
    assert response.message == "Max transitions exceeded."
    assert len(loop._trajectory.transitions) == 5, f"Expected 5 states, got {[t.state.name for t in loop._trajectory.transitions]}"

@pytest.mark.api_keys_required
def test_rerun_save_and_load_trajectory():
    trajectory = Trajectory.load("tests/trajectories/django__django_16379.json")
    Settings.cheap_model = None  # To not use an LLM when generating commit messages

    # Start by running the trajectory again with mocked action requests
    instance = load_instance("django__django-16379")
    workspace = create_workspace(instance)
    assert isinstance(workspace.file_repo, GitRepository)
    mocked_actions = trajectory.get_mocked_actions()
    expected_states = trajectory.get_expected_states()

    loop = AgenticLoop(
        trajectory.transition_rules, workspace=workspace, mocked_actions=mocked_actions, expected_states=expected_states
    )
    response = loop.run(message=trajectory.initial_message)

    assert workspace.file_context.has_span(
        "django/core/cache/backends/filebased.py", "FileBasedCache.has_key"
    )
    assert loop.workspace.file_repo._initial_commit != loop.workspace.file_repo._current_commit
    assert loop.workspace.file_repo.diff()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        loop.persist(tmp_file.name)

        # Verify that the loop can be iniatied from the saved trajectory
        saved_loop = AgenticLoop.from_trajectory_file(tmp_file.name)

        saved_response = saved_loop.run(message=trajectory.initial_message)
        assert saved_response == response
        assert saved_loop.workspace.file_repo._initial_commit == loop.workspace.file_repo._initial_commit
        assert saved_loop.workspace.file_repo._current_commit == loop.workspace.file_repo._current_commit
        assert saved_loop.workspace.file_repo.diff() == loop.workspace.file_repo.diff()