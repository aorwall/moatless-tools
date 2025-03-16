import pytest
import tempfile
import shutil
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

from moatless.flow.manager import FlowManager
from moatless.storage.base import BaseStorage
from moatless.storage.file_storage import FileStorage
from moatless.evaluation.manager import EvaluationManager
from moatless.evaluation.schema import Evaluation, EvaluationInstance, EvaluationStatus, InstanceStatus
from moatless.runner.runner import JobStatus, BaseRunner
from moatless.flow.loop import AgenticLoop
from moatless.flow.flow import AgenticFlow, Node
from moatless.agent.agent import ActionAgent
from moatless.eventbus.base import BaseEventBus


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_runner():
    """Fixture to create a mock Runner."""
    runner = AsyncMock(spec=BaseRunner)
    # Set up default return values for commonly called methods
    runner.start_job.return_value = True
    runner.job_exists.return_value = False
    runner.get_job_status.return_value = JobStatus.PENDING
    return runner


@pytest.fixture
def mock_event_bus():
    """Fixture to create a mock event bus."""
    mock_bus = AsyncMock(spec=BaseEventBus)
    mock_bus.subscribe = AsyncMock()
    mock_bus.publish = AsyncMock()
    mock_bus.read_events = AsyncMock(return_value=[])
    mock_bus._read_events_from_storage = AsyncMock(return_value=[])
    return mock_bus


@pytest.fixture
def file_storage(temp_dir):
    """Fixture to create a FileStorage instance and set it as the singleton."""
    # First, reset the singleton to ensure we're starting fresh
    BaseStorage.reset_instance()
    # Create and return the new instance as the singleton
    return BaseStorage.get_instance(storage_impl=FileStorage, base_dir=temp_dir)


@pytest.fixture(autouse=True)
def reset_storage_after_test():
    """Auto-used fixture to reset the storage singleton after each test."""
    yield
    BaseStorage.reset_instance()


@pytest.fixture
def manager(mock_runner, file_storage, mock_event_bus):
    """Fixture to create an EvaluationManager with mocked dependencies."""
    manager = EvaluationManager(runner=mock_runner, storage=file_storage, eventbus=mock_event_bus)
    return manager

@pytest.fixture
def flow_manager(mock_runner, file_storage, mock_event_bus):
    """Fixture to create a mock FlowManager."""
    return FlowManager(runner=mock_runner, storage=file_storage, eventbus=mock_event_bus)


@pytest.mark.asyncio
async def test_create_evaluation_with_dataset(manager, flow_manager, file_storage):
    """Test creating an evaluation with the 'easy' dataset."""
    # Create an evaluation using the easy dataset
    evaluation = await manager.create_evaluation(
        flow_id="simple_coding",
        model_id="gpt-4o-mini-2024-07-18",
        evaluation_name="eval_test_easy",
        dataset_name="easy"
    )
    
    assert evaluation.evaluation_name == "eval_test_easy"
    assert evaluation.dataset_name == "easy"
    assert evaluation.flow_id == "simple_coding"
    assert evaluation.model_id == "gpt-4o-mini-2024-07-18"
    assert evaluation.status in [EvaluationStatus.PENDING, EvaluationStatus.PAUSED]
    assert len(evaluation.instances) == 4
    
    for instance in evaluation.instances:
        assert instance.status == InstanceStatus.CREATED
        await file_storage.assert_exists_in_trajectory("trajectory", "eval_test_easy", instance.instance_id)
    
    saved_eval = await manager.get_evaluation("eval_test_easy")
    assert saved_eval is not None
    assert saved_eval.evaluation_name == "eval_test_easy"
    assert len(saved_eval.instances) == 4

    for instance in saved_eval.instances:
        trajectory = await flow_manager.get_trajectory("eval_test_easy", instance.instance_id)
        assert trajectory is not None
        assert trajectory.trajectory_id == instance.instance_id
        assert trajectory.project_id == "eval_test_easy"


    
    
    
