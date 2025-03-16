"""Tests for the FlowManager.get_completions method."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from moatless.flow.manager import FlowManager
from moatless.flow.schema import CompletionDTO


@pytest.mark.asyncio
async def test_flow_manager_get_completions():
    """Test the FlowManager.get_completions method."""
    # Mock storage
    mock_storage = AsyncMock()
    
    # Sample raw completion data
    anthropic_completion = {
        "original_input": {
            "system": "You are a helpful assistant.",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather?"
                }
            ]
        },
        "original_response": {
            "content": [
                {
                    "type": "text",
                    "text": "I don't have access to real-time weather data."
                }
            ]
        }
    }
    
    # Set up the mock storage to return our test data
    mock_storage.get_trajectory_key.return_value = "project/trajectory"
    mock_storage.list_keys = AsyncMock(return_value=["project/trajectory/completions/node_1/1"])
    mock_storage.read = AsyncMock(return_value=anthropic_completion)
    
    # Create a FlowManager instance with our mocks
    manager = FlowManager(
        runner=AsyncMock(),
        storage=mock_storage,
        eventbus=MagicMock(),
        agent_manager=MagicMock(),
        model_manager=MagicMock()
    )
    
    # Call the method
    completions = await manager.get_completions("project", "trajectory", "1")
    
    # Verify the results
    assert len(completions) == 1
    
    completion = completions[0]
    assert isinstance(completion, CompletionDTO)
    assert completion.original_input == anthropic_completion["original_input"]
    assert completion.original_output == anthropic_completion["original_response"]
    assert completion.system_prompt == "You are a helpful assistant."
    assert completion.input is not None
    assert completion.input.content == "What's the weather?"
    assert completion.output is not None
    assert completion.output.content == "I don't have access to real-time weather data."


@pytest.mark.asyncio
async def test_flow_manager_get_completions_error_handling():
    """Test error handling in the FlowManager.get_completions method."""
    # Mock storage
    mock_storage = AsyncMock()
    
    # Malformed completion data
    malformed_completion = {
        "not_original_input": {},
        "not_original_response": {}
    }
    
    # Set up the mock storage to return our test data
    mock_storage.get_trajectory_key.return_value = "project/trajectory"
    mock_storage.list_keys = AsyncMock(return_value=["project/trajectory/completions/node_1/1"])
    mock_storage.read = AsyncMock(return_value=malformed_completion)
    
    # Create a FlowManager instance with our mocks
    manager = FlowManager(
        runner=AsyncMock(),
        storage=mock_storage,
        eventbus=MagicMock(),
        agent_manager=MagicMock(),
        model_manager=MagicMock()
    )
    
    # Call the method
    completions = await manager.get_completions("project", "trajectory", "1")
    
    # Verify the results
    assert len(completions) == 1
    
    # Even with malformed data, we should get a CompletionDTO
    completion = completions[0]
    assert isinstance(completion, CompletionDTO)
    # Basic fields should be None because the input was malformed
    assert completion.original_input is None
    assert completion.original_output is None


@pytest.mark.asyncio
async def test_flow_manager_get_completions_anthropic_format():
    """Test the FlowManager.get_completions method with the newer Anthropic format."""
    # Mock storage
    mock_storage = AsyncMock()
    
    # Sample raw completion data with the newer Anthropic format
    anthropic_new_format = {
        "original_input": {
            "model": "claude-3-7-sonnet-20250219",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather in San Francisco?"
                        }
                    ]
                }
            ],
            "system": [
                {
                    "type": "text",
                    "text": "You are an autonomous AI assistant with superior programming skills."
                }
            ]
        },
        "original_response": {
            "content": [
                {
                    "type": "text",
                    "text": "I don't have access to real-time weather data."
                },
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "search",
                    "input": {
                        "query": "weather in San Francisco"
                    }
                }
            ]
        }
    }
    
    # Set up the mock storage to return our test data
    mock_storage.get_trajectory_key.return_value = "project/trajectory"
    mock_storage.list_keys = AsyncMock(return_value=["project/trajectory/completions/node_1/1"])
    mock_storage.read = AsyncMock(return_value=anthropic_new_format)
    
    # Create a FlowManager instance with our mocks
    manager = FlowManager(
        runner=AsyncMock(),
        storage=mock_storage,
        eventbus=MagicMock(),
        agent_manager=MagicMock(),
        model_manager=MagicMock()
    )
    
    # Call the method
    completions = await manager.get_completions("project", "trajectory", "1")
    
    # Verify the results
    assert len(completions) == 1
    
    completion = completions[0]
    assert isinstance(completion, CompletionDTO)
    assert completion.original_input == anthropic_new_format["original_input"]
    assert completion.original_output == anthropic_new_format["original_response"]
    assert completion.system_prompt == "You are an autonomous AI assistant with superior programming skills."
    assert completion.input is not None
    assert completion.input.content == "What's the weather in San Francisco?"
    assert completion.output is not None
    assert completion.output.content == "I don't have access to real-time weather data."
    assert completion.output.tool_calls is not None
    assert len(completion.output.tool_calls) == 1
    assert completion.output.tool_calls[0].name == "search"
    assert completion.output.tool_calls[0].arguments == {"query": "weather in San Francisco"} 