import json

import pytest
import pytest_asyncio
from moatless.actions.create_file import CreateFileArgs
from moatless.actions.finish import FinishArgs
from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.schema import Observation, ActionArguments
from moatless.actions.string_replace import StringReplaceArgs
from moatless.actions.view_code import CodeSpan, ViewCodeArgs
from moatless.completion.schema import ChatCompletionAssistantMessage, ChatCompletionUserMessage
from moatless.file_context import FileContext
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.node import Node, ActionStep
from moatless.repository.repository import InMemRepository
from moatless.testing.schema import TestResult, TestStatus
from moatless.utils.tokenizer import count_tokens
from moatless.workspace import Workspace


# pyright: reportCallIssue=false, reportAttributeAccessIssue=false


class TestActionArguments(ActionArguments):
    pass


@pytest.fixture
def repo():
    repo = InMemRepository()
    repo.save_file("file1.py", """def method1():
    return "original1"
""")
    repo.save_file("file2.py", """def method2():
    return "original2"
    
def method3():
    return "original3"
""")
    return repo


@pytest_asyncio.fixture
async def workspace(repo):
    workspace = Workspace(repository=repo)
    return workspace


@pytest.fixture
def test_tree(repo) -> tuple[Node, Node, Node, Node, Node, Node]:
    """Creates a test tree with various actions and file contexts"""
    root = Node(node_id=0, file_context=FileContext(repo=repo))  
    root.message = "Initial task"

    # Node1: View code action
    node1 = Node(node_id=1)
    action1 = TestActionArguments()
    node1.action_steps.append(ActionStep(action=action1, observation=Observation(message="Added method1 to context")))
    node1.file_context = FileContext(repo=repo)
    node1.file_context.add_span_to_context("file1.py", "method1")
    root.add_child(node1)

    # Node2: Another view action
    node2 = Node(node_id=2)
    action2 = TestActionArguments()
    node2.action_steps.append(ActionStep(action=action2, observation=Observation(message="Added method2 to context")))
    node2.file_context = node1.file_context.clone()
    node2.file_context.add_span_to_context("file2.py", "method2")
    node1.add_child(node2)

    # Node3: Apply change action
    node3 = Node(node_id=3)
    action3 = StringReplaceArgs(
        path="file1.py",
        old_str='return "original1"',
        new_str='return "modified1"',
        scratch_pad="Modifying method1 return value"
    )
    node3.action_steps.append(ActionStep(action=action3, observation=Observation(message="Modified method1")))
    node3.file_context = node2.file_context.clone()
    node3.file_context.add_file("file1.py").apply_changes("""def method1():
    return "modified1"
""")
    node2.add_child(node3)

    # Node4: View another method
    node4 = Node(node_id=4)
    action4 = TestActionArguments()
    node4.action_steps.append(ActionStep(action=action4, observation=Observation(message="Added method3 to context")))
    node4.file_context = node3.file_context.clone()
    node4.file_context.add_span_to_context("file2.py", "method3")
    node3.add_child(node4)

    # Node5: Finish action
    node5 = Node(node_id=5)
    action5 = FinishArgs(
        scratch_pad="All changes complete",
        finish_reason="Successfully modified the code"
    )
    node5.action_steps.append(ActionStep(action=action5, observation=Observation(message="Task completed successfully", terminal=True)))
    node5.terminal = True
    node4.add_child(node5)

    return root, node1, node2, node3, node4, node5


@pytest.mark.asyncio
async def test_messages_history(test_tree, workspace):
    """Test message history with different configurations"""
    _, _, node2, node3, node4, _ = test_tree
    
    # Basic message history
    generator = MessageHistoryGenerator(
        include_file_context=True
    )
    messages = await generator.generate_messages(node2, workspace)
    messages = list(messages)
    print(messages)
    
    # Verify initial message
    assert messages[0]["role"] == "user"
    assert "content" in messages[0]
    
    # Verify message content contains "Initial task"
    initial_msg_str = str(messages[0]["content"])
    assert "Initial task" in initial_msg_str
    
    # Verify action and observation messages
    assert messages[1]["role"] == "assistant"  # Action message
    assert "tool_calls" in messages[1]
    assert messages[2]["role"] == "tool"  # Observation message
    
    # Verify observation content contains the expected text
    observation_msg_str = str(messages[2]["content"])
    assert "Added method1 to context" in observation_msg_str
    
    # With file changes
    messages = await generator.generate_messages(node3, workspace)
    messages = list(messages)
    assert len(messages) >= 5
    
    # Debug output
    print("\nMessages for node3:")
    for i, msg in enumerate(messages):
        print(f"Message {i}: Role: {msg['role']} - Content: {str(msg.get('content'))}")
        if "tool_calls" in msg:
            print(f"  Tool calls: {msg['tool_calls']}")
    
    # Verify file modification is included in at least one message
    modification_found = False
    for msg in messages:
        message_str = str(msg)
        if "modified1" in message_str:
            modification_found = True
            break
    
    assert modification_found, "Modified content not found in messages"
    
    # With multiple file contexts
    messages = await generator.generate_messages(node4, workspace)
    messages = list(messages)
    assert len(messages) >= 7
    
    # Check for method3 in messages
    method3_found = False
    for msg in messages:
        message_str = str(msg)
        if "method3" in message_str:
            method3_found = True
            break
    
    assert method3_found, "Method3 not found in messages"


@pytest.mark.asyncio
async def test_token_limited_messages(test_tree, workspace):
    """Test message history with token limit configuration"""
    root, node1, node2, node3, node4, node5 = test_tree
    
    # Add more content to nodes to increase token count
    root.user_message = "Initial task with a lot more detailed content to increase token count. " * 5
    node1.assistant_message = "Processing your request with additional detailed explanation. " * 5
    node3.assistant_message = "Making changes to the code with extensive details about what's happening. " * 5
    
    # Get all messages without limit for comparison
    full_generator = MessageHistoryGenerator(include_file_context=True)
    full_messages = await full_generator.generate_messages(node5, workspace)
    full_messages = list(full_messages)
    
    # Count tokens in the full message history
    total_tokens = 0
    for message in full_messages:
        message_str = str(message)
        total_tokens += count_tokens(message_str)
    
    # Set a very small token limit that should include only the most recent nodes
    # This ensures we'll see truncation even with small test data
    tiny_limit = min(100, int(total_tokens * 0.2))  # Very small limit
    limited_generator = MessageHistoryGenerator(
        include_file_context=True,
        max_tokens=tiny_limit
    )
    limited_messages = await limited_generator.generate_messages(node5, workspace)
    limited_messages = list(limited_messages)
    
    # The limited messages should be fewer than the full messages
    assert len(limited_messages) < len(full_messages)
    
    # Count tokens in the limited messages
    limited_tokens = 0
    for message in limited_messages:
        message_str = str(message)
        limited_tokens += count_tokens(message_str)
    
    # Token count should be below or very close to the limit
    # Allow slight margin for individual message that can't be split
    assert limited_tokens <= tiny_limit * 1.5
    
    # Verify the most recent content is included
    # The finish action from node5 should be present
    finish_found = any("Task completed successfully" in str(m) for m in limited_messages)
    assert finish_found, "Most recent action (finish) not found in limited messages"
    
    # Check for content from early nodes - should not be present with tiny limit
    if len(limited_messages) < len(full_messages) // 2:
        initial_found = any("Initial task with a lot more" in str(m) for m in limited_messages)
        assert not initial_found, "Oldest messages should not be included with small token limit"
    
    # Test with a medium limit that should include more nodes
    medium_limit = int(total_tokens * 0.6)
    medium_generator = MessageHistoryGenerator(
        include_file_context=True,
        max_tokens=medium_limit
    )
    medium_messages = await medium_generator.generate_messages(node5, workspace)
    medium_messages = list(medium_messages)
    
    # Count tokens in the medium message set
    medium_tokens = 0
    for message in medium_messages:
        message_str = str(message)
        medium_tokens += count_tokens(message_str)
    
    # Token count should be below the limit
    assert medium_tokens <= medium_limit * 1.2
    
    # Debug output
    print("\nToken counts:")
    print(f"Full messages: {len(full_messages)} messages, {total_tokens} tokens")
    print(f"Medium messages: {len(medium_messages)} messages, {medium_tokens} tokens (limit: {medium_limit})")
    print(f"Limited messages: {len(limited_messages)} messages, {limited_tokens} tokens (limit: {tiny_limit})")


@pytest.mark.asyncio
async def test_terminal_node_history(test_tree, workspace):
    """Test history generation for terminal nodes"""
    _, _, _, _, _, node5 = test_tree
    
    generator = MessageHistoryGenerator()
    messages = await generator.generate_messages(node5, workspace)
    messages = list(messages)
    
    # Verify finish action content
    finish_action_found = any(
        "Finish" in str(m) for m in messages
    )
    
    # Verify observation content
    finish_observation_found = any(
        "Task completed successfully" in str(m) for m in messages
    )
    
    assert finish_action_found, "Finish action message not found"
    assert finish_observation_found, "Finish observation message not found"

def test_message_history_serialization():
    """Test MessageHistoryGenerator serialization"""
    generator = MessageHistoryGenerator(
        include_file_context=True,
        include_git_patch=False,
        thoughts_in_action=True
    )
    
    # Test serialization
    data = generator.model_dump()
    assert data["include_file_context"] is True
    assert data["include_git_patch"] is False
    assert data["thoughts_in_action"] is True


def test_message_history_dump_and_load():
    """Test MessageHistoryGenerator dump and load functionality"""
    # Create original generator
    original = MessageHistoryGenerator(
        include_file_context=True,
        include_git_patch=False
    )
    
    # Test JSON serialization
    json_str = original.model_dump_json()
    loaded_dict = json.loads(json_str)
    assert loaded_dict["include_file_context"] is True
    assert loaded_dict["include_git_patch"] is False
    
    # Test model reconstruction from JSON
    loaded = MessageHistoryGenerator.model_validate_json(json_str)
    assert loaded.include_file_context is True
    assert loaded.include_git_patch is False
    
    # Test dictionary serialization
    dict_data = original.model_dump()
    loaded_from_dict = MessageHistoryGenerator.model_validate(dict_data)
    assert loaded_from_dict.include_file_context is True
    assert loaded_from_dict.include_git_patch is False


@pytest.mark.asyncio
async def test_max_tokens_serialization():
    """Test max_tokens is properly serialized and loaded"""
    # Create generator with max_tokens
    generator = MessageHistoryGenerator(
        include_file_context=True,
        max_tokens=1000
    )
    
    # Verify max_tokens is included in serialization
    data = generator.model_dump()
    assert data["max_tokens"] == 1000
    
    # Test JSON serialization
    json_str = generator.model_dump_json()
    loaded_dict = json.loads(json_str)
    assert loaded_dict["max_tokens"] == 1000
    
    # Test model reconstruction
    loaded = MessageHistoryGenerator.model_validate_json(json_str)
    assert loaded.max_tokens == 1000
