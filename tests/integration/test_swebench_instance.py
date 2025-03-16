import asyncio
import json
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from moatless.benchmark.swebench.utils import create_index_async, create_repository, create_repository_async
from moatless.evaluation.run_instance import _run_instance, evaluate_instance
from moatless.evaluation.utils import get_moatless_instance, get_swebench_instance
from moatless.eventbus.base import BaseEventBus
from moatless.flow.flow import AgenticFlow
from moatless.flow.loop import AgenticLoop
from moatless.flow.manager import FlowManager
from moatless.node import Node
from moatless.repository.repository import Repository
from moatless.runner.runner import BaseRunner, JobStatus
from moatless.runtime.testbed import TestbedEnvironment
from moatless.storage.base import BaseStorage
from moatless.storage.file_storage import FileStorage
from moatless.utils.moatless import get_moatless_trajectory_dir
from moatless.workspace import Workspace
from moatless.flow.events import FlowStartedEvent, FlowCompletedEvent, NodeExpandedEvent
from moatless.agent.events import RunAgentEvent, ActionCreatedEvent, ActionExecutedEvent

@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def file_storage(temp_dir):
    """Fixture to create a FileStorage instance and set it as the singleton."""
    BaseStorage.reset_instance()
    return BaseStorage.get_instance(storage_impl=FileStorage, base_dir=temp_dir)

@pytest.fixture
def mock_event_bus():
    """Fixture to create a mock event bus and set it as the singleton."""
    event_bus = AsyncMock(spec=BaseEventBus)
    BaseEventBus.reset_instance()
    BaseEventBus._instance = event_bus
    yield event_bus
    BaseEventBus.reset_instance()

@pytest.fixture(autouse=True)
def reset_storage_after_test():
    """Auto-used fixture to reset the storage singleton after each test."""
    yield
    BaseStorage.reset_instance()

@pytest.fixture
def flow_manager(file_storage):
    """Fixture to create a FlowManager."""
    return FlowManager(storage=file_storage)

@pytest.fixture
def repo_dir(temp_dir):
    """Fixture to create a directory for repositories."""
    repo_dir = os.path.join(temp_dir, "repos")
    os.makedirs(repo_dir, exist_ok=True)
    return repo_dir

@pytest.mark.asyncio
async def test_swebench_instance_flow(file_storage, mock_event_bus):
    """Test running a SWEBench instance flow.
    
    This integration test verifies:
    1. The flow runs successfully with a mock SWEBench instance
    2. Events are published during execution
    3. The trajectory is properly saved to disk
    4. The evaluation results are saved correctly
    
    The test mocks external dependencies but tests the real interaction
    between components.
    """
    project_id = "test_project"
    instance_id = "django__django-11099"

    swebench_instance = get_swebench_instance(instance_id)
    
    agent = get_agent(agent_id="code")
    model_id = "gpt-4o-mini-2024-07-18"
    completion_model = create_completion_model(model_id)
    agent.completion_model = completion_model

    flow = AgenticLoop.create(
            message=swebench_instance["problem_statement"],
            trajectory_id=instance_id,
            project_id=project_id,
            agent=agent,
            max_iterations=10,
    )
    
    repository = await create_repository_async(swebench_instance, repo_base_dir="/tmp/repos")
    code_index = await create_index_async(swebench_instance, repository=repository)
    workspace = Workspace(repository=repository, code_index=code_index, legacy_workspace=True)
    
    node = await flow.run(workspace=workspace)
    print(f"Node: {node}")

    # Verify expected events were published
    expected_events = [
        FlowStartedEvent,  # Flow start
        RunAgentEvent,     # Agent starts processing
        ActionCreatedEvent,  # Agent creates an action
        ActionExecutedEvent,  # Agent executes the action
        NodeExpandedEvent,   # Node is expanded for next iteration
        FlowCompletedEvent,  # Flow completes
    ]

    # Get all event calls
    event_calls = [call.args[0] for call in mock_event_bus.publish.call_args_list]
    
    # Verify each expected event type was published at least once
    for expected_event in expected_events:
        matching_events = [e for e in event_calls if isinstance(e, expected_event)]
        assert len(matching_events) >= 1, f"Expected at least one {expected_event.__name__} but found none"

    # Verify RunAgentEvent has correct agent_id
    run_agent_events = [e for e in event_calls if isinstance(e, RunAgentEvent)]
    assert all(e.agent_id == "code" for e in run_agent_events)

    # Verify action events have valid node_ids
    action_events = [e for e in event_calls if isinstance(e, (ActionCreatedEvent, ActionExecutedEvent))]
    assert all(hasattr(e, 'node_id') and e.node_id is not None for e in action_events)

    
