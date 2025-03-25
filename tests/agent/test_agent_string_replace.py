import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.agent.agent import ActionAgent
from moatless.completion import BaseCompletionModel, CompletionResponse
from moatless.file_context import FileContext
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.node import Node, ActionStep
from moatless.repository.repository import InMemRepository
from moatless.schema import MessageHistoryType
from moatless.workspace import Workspace


@pytest.fixture
def repository():
    repo = InMemRepository()
    repo.save_file("test_file.py", """
def hello_world():
    message = "Hello World"
    print(message)
    
    other_message = "Goodbye World"
    print(other_message)
""")
    return repo


@pytest.fixture
def workspace(repository):
    workspace = Mock(spec=Workspace)
    workspace.repository = repository
    return workspace


@pytest.fixture
def file_context(repository):
    context = FileContext(repo=repository)
    context.add_file("test_file.py", show_all_spans=True)
    return context


@pytest.fixture
def mock_completion_model():
    model = Mock(spec=BaseCompletionModel)
    model.create_completion = AsyncMock()
    model.clone.return_value = model
    model.initialize = Mock()
    model.model = "mock-model"
    model.message_history_type = MessageHistoryType.MESSAGES
    return model


@pytest.fixture
def agent(mock_completion_model, workspace):
    string_replace_action = StringReplace(auto_correct_indentation=True)
    agent = ActionAgent(
        agent_id="test-agent",
        system_prompt="You are a helpful assistant",
        actions=[string_replace_action],
        memory=MessageHistoryGenerator()
    )
    agent.workspace = workspace
    agent.completion_model = mock_completion_model
    return agent


@pytest.mark.asyncio
async def test_agent_multiple_string_replace(agent, file_context, repository):
    # Create a node with file context
    node = Node.create_root("Please update the file", shadow_mode=True)
    node.file_context = file_context
    
    # Create StringReplaceArgs for the first change
    replace_args1 = StringReplaceArgs(
        path="test_file.py",
        old_str='    message = "Hello World"',
        new_str='    message = "Hello Universe"',
        thoughts="Updating greeting message"
    )
    
    # Create StringReplaceArgs for the second change
    replace_args2 = StringReplaceArgs(
        path="test_file.py",
        old_str='    other_message = "Goodbye World"',
        new_str='    other_message = "Farewell Universe"',
        thoughts="Updating farewell message"
    )
    
    # Create action steps for the node
    step1 = ActionStep(action=replace_args1)
    step2 = ActionStep(action=replace_args2)
    node.action_steps = [step1, step2]

    await agent._execute(node, step1)
    await agent._execute(node, step2)
    
    # Verify that both changes were applied
    updated_content = file_context.get_file("test_file.py").content
    print(updated_content)
    assert 'message = "Hello Universe"' in updated_content
    assert 'other_message = "Farewell Universe"' in updated_content
    
    # Verify that the original structure is preserved
    assert "def hello_world():" in updated_content
    assert "print(message)" in updated_content
    assert "print(other_message)" in updated_content
