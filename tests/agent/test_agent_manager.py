import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from moatless.agent.agent import ActionAgent
from moatless.agent.manager import AgentConfigManager
from moatless.actions.action import Action
from moatless.actions.finish import Finish
from moatless.actions.view_code import ViewCode
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.storage.base import BaseStorage
from moatless.storage.file_storage import FileStorage
from moatless.completion.tool_call import ToolCallCompletionModel


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def file_storage(temp_dir):
    """Real file storage for testing."""
    return FileStorage(temp_dir)


@pytest.fixture
def completion_model_factory():
    """Create a factory function for creating new ToolCallCompletionModel instances."""
    def create_model():
        return ToolCallCompletionModel(
            model="mock-model",
            api_key="mock-key",
            thoughts_in_action=True
        )
    return create_model


@pytest.fixture
def test_agent(completion_model_factory):
    """Create a test agent with real actions."""
    view_code_action = ViewCode()
    finish_action = Finish()
    
    agent = ActionAgent(
        agent_id="test-agent",
        description="Test agent for unit tests",
        system_prompt="You are a helpful AI assistant.",
        actions=[view_code_action, finish_action],
        memory=MessageHistoryGenerator(max_tokens=1000, include_file_context=True),
        completion_model=completion_model_factory()
    )
    
    return agent


@pytest.mark.asyncio
async def test_manager_init(file_storage):
    """Test initialization of the manager."""
    manager = AgentConfigManager(file_storage)
    await manager.initialize()
    
    await file_storage.exists("agents.json")


@pytest.mark.asyncio
async def test_create_get_agent(file_storage, test_agent):
    """Test creating and getting an agent."""
    manager = AgentConfigManager(file_storage)
    await manager.initialize()
    
    # Create the agent
    await manager.create_agent(test_agent)
    
    # Check that the agent was saved
    await file_storage.exists("agents.json")
    
    # Test get_agent
    agent = manager.get_agent("test-agent")
    assert agent.agent_id == "test-agent"
    assert agent.description == "Test agent for unit tests"
    assert agent.system_prompt == "You are a helpful AI assistant."
    assert len(agent.actions) == 2
    
    # Verify the actions are of the correct types
    action_types = [type(action) for action in agent.actions]
    assert ViewCode in action_types
    assert Finish in action_types


@pytest.mark.asyncio
async def test_get_all_agents(file_storage, test_agent, completion_model_factory):
    """Test getting all agents."""
    manager = AgentConfigManager(file_storage)
    await manager.initialize()
    
    # Create multiple agents
    await manager.create_agent(test_agent)
    
    test_agent2 = ActionAgent(
        agent_id="test-agent-2",
        description="Another test agent",
        system_prompt="You are another helpful AI assistant.",
        actions=[Finish()],
        memory=MessageHistoryGenerator(),
        completion_model=completion_model_factory()  # Create a new model instance
    )
    await manager.create_agent(test_agent2)
    
    # Test get_all_agents
    agents = manager.get_all_agents()
    assert len(agents) == 2
    
    # Agents should be sorted by agent_id
    assert agents[0].agent_id == "test-agent"
    assert agents[1].agent_id == "test-agent-2"


@pytest.mark.asyncio
async def test_update_agent(file_storage, test_agent):
    """Test updating an agent."""
    manager = AgentConfigManager(file_storage)
    await manager.initialize()
    
    # Create the agent
    await manager.create_agent(test_agent)
    
    # Update the agent
    test_agent.description = "Updated description"
    await manager.update_agent(test_agent)
    
    # Test get_agent returns updated agent
    agent = manager.get_agent("test-agent")
    assert agent.description == "Updated description"


@pytest.mark.asyncio
async def test_delete_agent(file_storage, test_agent):
    """Test deleting an agent."""
    manager = AgentConfigManager(file_storage)
    await manager.initialize()
    
    # Create the agent
    await manager.create_agent(test_agent)
    
    # Delete the agent
    await manager.delete_agent("test-agent")
    
    # Test that get_agent raises an error
    with pytest.raises(ValueError):
        manager.get_agent("test-agent")


@pytest.mark.asyncio
async def test_persistence_with_real_storage(file_storage, completion_model_factory):
    """Test that agents can be saved and loaded from disk."""
    # Create a test agent with real completion model
    view_code_action = ViewCode()
    finish_action = Finish()
    
    test_agent = ActionAgent(
        agent_id="test-agent",
        description="Test agent for unit tests",
        system_prompt="You are a helpful AI assistant.",
        actions=[view_code_action, finish_action],
        memory=MessageHistoryGenerator(max_tokens=1000, include_file_context=True),
        completion_model=completion_model_factory()
    )
    
    # Set up the first manager and create the agent
    manager = AgentConfigManager(file_storage)
    await manager.initialize()
    await manager.create_agent(test_agent)
    
    # Create a new manager to load from disk
    new_manager = AgentConfigManager(file_storage)
    await new_manager.initialize()
    
    # Test that the agent was loaded
    agent = new_manager.get_agent("test-agent")
    assert agent.agent_id == "test-agent"
    assert agent.description == "Test agent for unit tests"
    assert agent.system_prompt == "You are a helpful AI assistant."
    
    # Verify the actions are still present and of the correct types
    assert len(agent.actions) == 2
    action_types = [type(action) for action in agent.actions]
    assert ViewCode in action_types
    assert Finish in action_types
    
    # Verify memory settings were preserved
    assert isinstance(agent.memory, MessageHistoryGenerator)
    assert agent.memory.max_tokens == 1000
    assert agent.memory.include_file_context == True


@pytest.mark.asyncio
async def test_agent_with_complex_actions(file_storage, completion_model_factory):
    """Test creating and loading an agent with complex action configuration."""
    view_code = ViewCode(max_tokens=5000, show_code_blocks=True)
    finish = Finish()
    
    agent = ActionAgent(
        agent_id="complex-agent",
        description="Agent with complex action config",
        system_prompt="You are an AI assistant with complex actions.",
        actions=[view_code, finish],
        memory=MessageHistoryGenerator(max_tokens=2000, include_git_patch=False),
        completion_model=completion_model_factory()
    )
    
    manager = AgentConfigManager(file_storage)
    await manager.initialize()
    await manager.create_agent(agent)
    
    # Get the agent back and check complex settings
    loaded_agent = manager.get_agent("complex-agent")
    assert loaded_agent.agent_id == "complex-agent"
    
    # Check the view_code action was properly serialized and deserialized
    view_code_action = None
    for action in loaded_agent.actions:
        if isinstance(action, ViewCode):
            view_code_action = action
            break
    
    assert view_code_action is not None
    assert view_code_action.max_tokens == 5000
    assert view_code_action.show_code_blocks == True
    
    # Check memory settings
    assert loaded_agent.memory.max_tokens == 2000
    assert loaded_agent.memory.include_git_patch == False
