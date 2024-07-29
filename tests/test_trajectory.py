import json
from datetime import datetime
import logging
from pydantic import Field

import pytest
from pydantic_core import to_jsonable_python

from moatless.state import AgenticState
from moatless.trajectory import Trajectory, TrajectoryAction, TrajectoryTransition
from moatless.types import ActionRequest, ActionResponse


class DummyState(AgenticState):
    def handle_action(self, action: ActionRequest) -> ActionResponse:
        return ActionResponse(content="Test response")

@pytest.fixture
def test_state():
    return DummyState()


class DummyAction(ActionRequest):
    dummy_field: str = Field(...)

def test_trajectory_serialization():
    # Create a dummy trajectory
    trajectory = Trajectory(
        name="Test Trajectory",
        initial_message="Initial message",
        persist_path="/tmp/test_trajectory.json",
        workspace={"key": "value"}
    )

    # Add a transition
    state = DummyState()
    trajectory.new_transition(state, snapshot={"snapshot_key": "snapshot_value"})

    # Add an action
    action = DummyAction(dummy_field="test")
    trajectory.save_action(
        action=action,
        output={"output_key": "output_value"},
        retry_message="Retry message",
        completion_cost=0.1,
        input_tokens=10,
        output_tokens=20
    )

    # Save some info
    trajectory.save_info({"info_key": "info_value"})

    # Serialize the trajectory
    serialized = json.dumps(
        trajectory.to_dict(exclude_none=True),
        indent=2,
        default=to_jsonable_python
    )

    # Deserialize and verify
    deserialized = json.loads(serialized)

    assert deserialized["name"] == "Test Trajectory"
    assert deserialized["initial_message"] == "Initial message"
    assert deserialized["workspace"] == {"key": "value"}
    assert deserialized["info"] == {"info_key": "info_value"}

    assert len(deserialized["transitions"]) == 1
    transition = deserialized["transitions"][0]
    assert "name" in transition["state"], "Name field is missing in the state data"
    assert transition["snapshot"] == {"snapshot_key": "snapshot_value"}
    assert isinstance(transition["timestamp"], str)  # Ensure timestamp is serialized

    assert len(transition["actions"]) == 1
    action = transition["actions"][0]
    logging.info(action)
    assert action["action"]["dummy_field"] == "test"
    assert action["output"] == {"output_key": "output_value"}
    assert action["retry_message"] == "Retry message"
    assert action["completion_cost"] == 0.1
    assert action["input_tokens"] == 10
    assert action["output_tokens"] == 20

# Run the test
if __name__ == "__main__":
    pytest.main([__file__])