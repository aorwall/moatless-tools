import json
from datetime import datetime
from pathlib import Path

import pytest
import pytest
from moatless.completion.log_handler import LogHandler
from moatless.context_data import (
    current_node_id,
    current_action_step,
    current_project_id,
    current_trajectory_id,
)
from moatless.storage.file_storage import FileStorage
from pydantic import BaseModel


class TestModel(BaseModel):
    name: str
    value: int


@pytest.fixture
def temp_storage(tmp_path):
    storage = FileStorage(base_dir=tmp_path)
    return storage


@pytest.fixture
def log_handler(temp_storage):
    return LogHandler(storage=temp_storage)


@pytest.fixture(autouse=True)
def setup_context():
    """Auto-used fixture to set up and tear down project and trajectory context."""
    project_token = current_project_id.set("test_project")
    trajectory_token = current_trajectory_id.set("test_trajectory")
    try:
        yield
    finally:
        current_project_id.reset(project_token)
        current_trajectory_id.reset(trajectory_token)


@pytest.mark.asyncio
async def test_get_log_path_basic(log_handler):
    key = await log_handler._get_log_path()
    assert "completions" in key
    assert key.endswith("completion")


@pytest.mark.asyncio
async def test_get_log_path_with_node(log_handler):
    token = current_node_id.set("test_node")
    try:
        key = await log_handler._get_log_path()
        assert "node_test_node" in key
    finally:
        current_node_id.reset(token)


@pytest.mark.asyncio
async def test_get_log_path_with_node_and_action(log_handler):
    node_token = current_node_id.set("test_node")
    action_token = current_action_step.set(1)
    try:
        key = await log_handler._get_log_path()
        assert "node_test_node" in key
        assert "action_1" in key
    finally:
        current_action_step.reset(action_token)
        current_node_id.reset(node_token)


@pytest.mark.asyncio
async def test_write_to_file_async(log_handler):
    test_data = {"test": "data"}
    await log_handler._write_to_file_async(test_data)
    
    # Verify data was written
    trajectory_path = log_handler._storage.get_trajectory_path()
    files = await log_handler._storage.list_paths(f"{trajectory_path}/completions")
    assert len(files) == 1
    
    # Read back the data
    data = await log_handler._storage.read(files[0])
    assert data == test_data


@pytest.mark.asyncio
async def test_log_success_event(log_handler):
    start_time = datetime.now()
    end_time = datetime.now()
    
    test_response = {"choices": [{"message": {"content": "test"}}]}
    test_kwargs = {
        "response": json.dumps(test_response),
        "input": [{"role": "user", "content": "test message"}],
        "optional_params": {"temperature": 0.7}
    }
    
    await log_handler.async_log_success_event(
        kwargs=test_kwargs,
        response_obj={"response": "test"},
        start_time=start_time,
        end_time=end_time
    )
    
    # Verify log was written
    trajectory_path = log_handler._storage.get_trajectory_path()
    files = await log_handler._storage.list_paths(f"{trajectory_path}/completions")
    assert len(files) == 1
    
    # Read back the log
    data = await log_handler._storage.read(files[0])
    assert "start_time" in data
    assert "end_time" in data
    assert "original_response" in data
    assert "original_input" in data
    assert "litellm_response" in data


@pytest.mark.asyncio
async def test_handle_kwargs_item_with_base_model(log_handler):
    model = TestModel(name="test", value=42)
    result = log_handler._handle_kwargs_item(model)
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 42


@pytest.mark.asyncio
async def test_handle_kwargs_item_with_nested_structures(log_handler):
    test_data = {
        "model": TestModel(name="test", value=42),
        "list": [TestModel(name="item1", value=1), {"key": "value"}],
        "nested": {"model": TestModel(name="nested", value=100)}
    }
    
    result = log_handler._handle_kwargs_item(test_data)
    
    assert isinstance(result, dict)
    assert isinstance(result["model"], dict)
    assert result["model"]["name"] == "test"
    assert isinstance(result["list"], list)
    assert isinstance(result["list"][0], dict)
    assert result["list"][0]["name"] == "item1"
    assert result["nested"]["model"]["name"] == "nested"


@pytest.mark.asyncio
async def test_parse_response(log_handler):
    # Test JSON string
    json_str = '{"key": "value"}'
    result = log_handler.parse_response(json_str)
    assert isinstance(result, dict)
    assert result["key"] == "value"
    
    # Test dict
    dict_input = {"key": "value"}
    result = log_handler.parse_response(dict_input)
    assert isinstance(result, dict)
    assert result["key"] == "value"
    
    # Test None
    result = log_handler.parse_response(None)
    assert result is None
    
    # Test invalid JSON string
    result = log_handler.parse_response("invalid json")
    assert isinstance(result, str)
    assert result == "invalid json" 