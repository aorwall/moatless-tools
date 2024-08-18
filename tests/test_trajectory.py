import os
import json
from datetime import datetime

from moatless.repository import GitRepository
from moatless.trajectory import Trajectory

def test_load_django_trajectory():
    file_path = "tests/trajectories/django__django_16379.json"
    trajectory = Trajectory.load(file_path)
    
    with open(file_path, 'r') as f:
        original_data = json.load(f)
    
    assert trajectory._name == original_data["name"]
    assert trajectory._initial_message == original_data["initial_message"]
    assert isinstance(trajectory._workspace.file_repo, GitRepository)
    assert trajectory._workspace.file_repo.path == original_data["workspace"]["repository"]["repo_path"]
    assert trajectory._workspace.file_context._max_tokens == original_data["workspace"]["file_context"]["max_tokens"]

    assert trajectory._current_transition_id == original_data["current_transition_id"]
    
    assert len(trajectory._transitions) == len(original_data["transitions"])
    
    for loaded_transition, original_transition in zip(trajectory.transitions, original_data["transitions"]):
        loaded_state = loaded_transition.state
        assert loaded_state.id == original_transition["id"]
        assert loaded_state.name == original_transition["name"]
        assert loaded_transition.snapshot == original_transition.get("snapshot")

        if "actions" in original_transition:
            assert len(loaded_state._actions) == len(original_transition["actions"])
            for loaded_action, original_action in zip(loaded_state._actions, original_transition["actions"]):
                assert loaded_action.request.__class__.__name__ == loaded_state.action_type().__name__ if loaded_state.action_type() else "Content"
                assert loaded_action.response
                assert loaded_action.response.trigger == original_action["response"]["trigger"]
                assert loaded_action.completion
                assert loaded_action.completion.usage
                assert loaded_action.completion.usage.completion_cost == original_action["completion"]["usage"]["completion_cost"]
                assert loaded_action.completion.usage.completion_tokens == original_action["completion"]["usage"]["completion_tokens"]
                assert loaded_action.completion.usage.prompt_tokens == original_action["completion"]["usage"]["prompt_tokens"]
    
    for loaded_transition, original_transition in zip(trajectory.transitions, original_data["transitions"]):
        if original_transition.get("previous_state_id") is not None:
            assert loaded_transition.state.previous_state.id == original_transition["previous_state_id"]
        else:
            assert loaded_transition.state.previous_state is None
    
    assert trajectory._info == original_data.get("info", {})
