import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
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
    repo.save_file(
        "test_file.py",
        """
def hello_world():
    message = "Hello World"
    print(message)
    
    other_message = "Goodbye World"
    print(other_message)
""",
    )
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
def mock_litellm_response():
    """Mock LiteLLM response with ReAct format content"""

    def _create_mock(content="", usage=None):
        from litellm.types.utils import Message, Usage, ModelResponse

        # Create message
        message = Message(content=content, role="assistant")

        # Create usage
        if usage:
            usage_obj = Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            )
        else:
            usage_obj = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        # Create ModelResponse
        return ModelResponse(
            id="test_id",
            created=1234567890,
            model="test",
            choices=[{"message": message, "finish_reason": "stop", "index": 0}],
            usage=usage_obj,
        )

    return _create_mock


@pytest.fixture
def agent(mock_completion_model, workspace):
    string_replace_action = StringReplace(auto_correct_indentation=True)
    agent = ActionAgent(
        agent_id="test-agent",
        system_prompt="You are a helpful assistant",
        actions=[string_replace_action],
        memory=MessageHistoryGenerator(),
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
        thoughts="Updating greeting message",
    )

    # Create StringReplaceArgs for the second change
    replace_args2 = StringReplaceArgs(
        path="test_file.py",
        old_str='    other_message = "Goodbye World"',
        new_str='    other_message = "Farewell Universe"',
        thoughts="Updating farewell message",
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


@pytest.mark.asyncio
async def test_agent_run_with_react_model(repository, workspace, mock_litellm_response):
    """Test agent run method with ReActCompletionModel and ReactMessageHistoryGenerator"""
    from moatless.completion.react import ReActCompletionModel
    from moatless.message_history.react import ReactMessageHistoryGenerator
    from moatless.actions.string_replace import StringReplace

    # Create the agent with ReAct model and memory
    string_replace_action = StringReplace(auto_correct_indentation=True)
    model = ReActCompletionModel(
        model_id="test",
        model="test",
        disable_thoughts=False,
        response_format={"type": "text"},  # Required for JsonCompletionModel
    )
    agent = ActionAgent(
        agent_id="test-agent",
        system_prompt="You are a helpful assistant",
        actions=[string_replace_action],
        memory=ReactMessageHistoryGenerator(),
        completion_model=model,
    )
    await agent.initialize(workspace)  # Initialize agent with workspace

    # Create a root node and a child node for testing
    root_node = Node.create_root("Initial message", shadow_mode=True)
    node = Node(node_id=1, user_message="Please update the greeting message")
    node.set_parent(root_node)  # This will make it a non-root node
    node.file_context = FileContext(repo=repository)
    node.file_context.add_file("test_file.py", show_all_spans=True)

    # Mock the completion response with thoughts
    mock_response = """Thoughts: I will update the greeting message to be more welcoming

Action: StringReplace
<path>test_file.py</path>
<old_str>    message = "Hello World"</old_str>
<new_str>    message = "Welcome to our World!"</new_str>"""

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value = mock_litellm_response(
            mock_response, usage={"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40}
        )

        # Run the agent
        await agent.run(node)

        # Verify the node has thoughts set
        assert node.thoughts is not None
        assert node.thoughts.text == "I will update the greeting message to be more welcoming"

        # Verify action was executed
        assert len(node.action_steps) == 1
        assert isinstance(node.action_steps[0].action, StringReplaceArgs)
        assert node.action_steps[0].action.path == "test_file.py"
        assert node.action_steps[0].action.old_str == '    message = "Hello World"'
        assert node.action_steps[0].action.new_str == '    message = "Welcome to our World!"'

        # Verify file was updated
        updated_content = node.file_context.get_file("test_file.py").content
        assert 'message = "Welcome to our World!"' in updated_content

        # Verify original structure is preserved
        assert "def hello_world():" in updated_content
        assert "print(message)" in updated_content
