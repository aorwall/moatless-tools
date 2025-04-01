import json
from typing import Any, cast, Dict

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
    node1.action_steps.append(ActionStep(action=action1, observation=Observation(message="First action completed")))
    node1.file_context = FileContext(repo=repo)
    node1.file_context.add_span_to_context("file1.py", "method1")
    root.add_child(node1)

    # Node2: Another view action
    node2 = Node(node_id=2)
    action2 = TestActionArguments()
    node2.action_steps.append(ActionStep(action=action2, observation=Observation(message="Second action completed")))
    node2.file_context = node1.file_context.clone()
    node2.file_context.add_span_to_context("file2.py", "method2")
    node1.add_child(node2)

    # Node3: Apply change action
    node3 = Node(node_id=3)
    action3 = TestActionArguments()
    node3.action_steps.append(ActionStep(action=action3, observation=Observation(message="Third action completed")))
    node3.file_context = node2.file_context.clone()
    node3.file_context.add_file("file1.py").apply_changes("""def method1():
    return "modified1"
""")
    node2.add_child(node3)

    # Node4: View another method
    node4 = Node(node_id=4)
    action4 = TestActionArguments()
    node4.action_steps.append(ActionStep(action=action4, observation=Observation(message="Fourth action completed")))
    node4.file_context = node3.file_context.clone()
    node4.file_context.add_span_to_context("file2.py", "method3")
    node3.add_child(node4)

    # Node5: Finish action
    node5 = Node(node_id=5)
    action5 = TestActionArguments()
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
    assert "First action completed" in observation_msg_str
    
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
    
    # Verify message order
    for i in range(len(messages) - 1):
        msg = messages[i]
        next_msg = messages[i + 1]
        
        # If current message is user, next should be assistant
        if msg["role"] == "user":
            assert next_msg["role"] == "assistant", f"Expected assistant after user at index {i}"
        
        # If current message is assistant with tool calls, next should be tool response
        elif msg["role"] == "assistant" and "tool_calls" in msg:
            assert next_msg["role"] == "tool", f"Expected tool response after assistant with tool calls at index {i}"
            assert next_msg["tool_call_id"] == msg["tool_calls"][0]["id"], "Tool response ID should match tool call ID"
    
    # With multiple file contexts
    messages = await generator.generate_messages(node4, workspace)
    messages = list(messages)
    assert len(messages) >= 7
    
    # Verify message order
    for i in range(len(messages) - 1):
        msg = messages[i]
        next_msg = messages[i + 1]
        
        # If current message is user, next should be assistant
        if msg["role"] == "user":
            assert next_msg["role"] == "assistant", f"Expected assistant after user at index {i}"
        
        # If current message is assistant with tool calls, next should be tool response
        elif msg["role"] == "assistant" and "tool_calls" in msg:
            assert next_msg["role"] == "tool", f"Expected tool response after assistant with tool calls at index {i}"
            assert next_msg["tool_call_id"] == msg["tool_calls"][0]["id"], "Tool response ID should match tool call ID"


@pytest.mark.asyncio
async def test_token_limited_messages(test_tree, workspace):
    """Test message history with token limit configuration"""
    root, node1, node2, node3, node4, node5 = test_tree
    
    # Add more content to nodes to increase token count
    root.user_message = "Initial task with a lot more detailed content to increase token count. " * 5
    node1.assistant_message = "Processing your request with additional detailed explanation. " * 5
    node3.assistant_message = "Making changes to the code with extensive details about what's happening. " * 5
    
    # Add tool calls to verify ordering
    action1 = TestActionArguments()
    action2 = TestActionArguments()
    action3 = TestActionArguments()
    
    node1.action_steps = [
        ActionStep(
            action=action1,
            observation=Observation(message="First action completed"),
            thoughts=["First action"]
        )
    ]
    
    node2.action_steps = [
        ActionStep(
            action=action2,
            observation=Observation(message="Second action completed"),
            thoughts=["Second action"]
        ),
        ActionStep(
            action=action3,
            observation=Observation(message="Third action completed"),
            thoughts=["Third action"]
        )
    ]
    
    # Get all messages without limit for comparison
    full_generator = MessageHistoryGenerator(include_file_context=True)
    full_messages = await full_generator.generate_messages(node5, workspace)
    full_messages = list(full_messages)
    
    # Verify message order in full messages
    current_idx = 0
    for node in [root, node1, node2, node3, node4, node5]:
        if node.user_message:
            assert full_messages[current_idx]["role"] == "user", f"Expected user message at index {current_idx}"
            current_idx += 1
        
        if node.action_steps:
            assert full_messages[current_idx]["role"] == "assistant", f"Expected assistant message at index {current_idx}"
            if "tool_calls" in full_messages[current_idx]:
                tool_calls = full_messages[current_idx]["tool_calls"]
                assert len(tool_calls) == len(node.action_steps), f"Expected {len(node.action_steps)} tool calls"
                current_idx += 1
                
                # Verify each tool response follows its tool call
                for i, step in enumerate(node.action_steps):
                    assert full_messages[current_idx]["role"] == "tool", f"Expected tool response at index {current_idx}"
                    assert full_messages[current_idx]["tool_call_id"] == tool_calls[i]["id"]
                    current_idx += 1
    
    print(json.dumps(full_messages, indent=2))
    
    # Count tokens for the first and the last 4 messages
    token_limit = 0
    for message in full_messages[:1] + full_messages[-4:]:
        message_str = str(message)
        token_limit += count_tokens(message_str)

    print(f"Limited tokens: {token_limit}")
    
    limited_generator = MessageHistoryGenerator(
        include_file_context=True,
        max_tokens=token_limit
    )
    limited_messages = await limited_generator.generate_messages(node5, workspace)
    
    print(json.dumps(limited_messages, indent=2))
    assert len(limited_messages) == 5, f"Expected 5 messages, first message and the last 4, got {len(limited_messages)}"
    
    # Verify order in limited messages
    for i in range(len(limited_messages) - 1):
        msg = cast(Dict[str, Any], limited_messages[i])
        next_msg = cast(Dict[str, Any], limited_messages[i + 1])
        
        # If current message is user, next should be assistant
        if msg["role"] == "user":
            assert next_msg["role"] == "assistant", f"Expected assistant after user at index {i}"
        
        # If current message is assistant with tool calls, next should be tool response
        elif msg["role"] == "assistant" and "tool_calls" in msg:
            assert next_msg["role"] == "tool", f"Expected tool response after assistant with tool calls at index {i}"
            assert next_msg["tool_call_id"] == msg["tool_calls"][0]["id"], "Tool response ID should match tool call ID"

@pytest.mark.asyncio
async def test_terminal_node_history(test_tree, workspace):
    """Test history generation for terminal nodes"""
    _, _, _, _, _, node5 = test_tree
    
    generator = MessageHistoryGenerator()
    messages = await generator.generate_messages(node5, workspace)
    messages = list(messages)
    
    # Verify terminal action content
    terminal_action_found = any(
        "Task completed successfully" in str(m) for m in messages
    )
    
    assert terminal_action_found, "Terminal action message not found"
    
    # Verify message order
    for i in range(len(messages) - 1):
        msg = messages[i]
        next_msg = messages[i + 1]
        
        # If current message is user, next should be assistant
        if msg["role"] == "user":
            assert next_msg["role"] == "assistant", f"Expected assistant after user at index {i}"
        
        # If current message is assistant with tool calls, next should be tool response
        elif msg["role"] == "assistant" and "tool_calls" in msg:
            assert next_msg["role"] == "tool", f"Expected tool response after assistant with tool calls at index {i}"
            assert next_msg["tool_call_id"] == msg["tool_calls"][0]["id"], "Tool response ID should match tool call ID"

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
