import json
import os
import tempfile

import litellm
import pytest
from unittest.mock import MagicMock, patch

from moatless.benchmark.utils import trace_metadata
from moatless.edit import EditCode
from moatless.loop import AgenticLoop
from moatless.benchmark.swebench import create_workspace, load_instance
from moatless.repository import GitRepository
from moatless.settings import Settings
from moatless.state import AgenticState
from moatless.trajectory import Trajectory

# django__django-9296.json check if the LLM corrects on retry

@pytest.mark.skip(reason="Test is not ready")
def test_expect_failed_edit():
    instance_id = "django__django-9296"
    trajectory = Trajectory.load(f"tests/trajectories/{instance_id}/trajectory.json")
    Settings.cheap_model = None

    instance = load_instance(instance_id, dataset_name="princeton-nlp/SWE-bench_Verified")
    workspace = create_workspace(instance)
    assert isinstance(workspace.file_repo, GitRepository)
    mocked_actions = trajectory.get_mocked_actions()
    expected_states = trajectory.get_expected_states()

    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    def verify_state_func(state: AgenticState):
        if isinstance(state, EditCode):
            for message in state.messages():
                print(f"Role: {message.role}")
                print(message.content)

    trajectory_path = f"tests/trajectories/{instance_id}/trajectory-correction.json"
    loop = AgenticLoop(
        trajectory.transition_rules, workspace=workspace, mocked_actions=mocked_actions, verify_state_func=verify_state_func, continue_after_mocks=False, trajectory_path=trajectory_path, metadata=trace_metadata(instance_id=instance["instance_id"], session_id="test_rerun", trace_name="test_rerun")
    )
    response = loop.run(message=trajectory.initial_message)
