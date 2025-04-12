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
    """Test message history with token limit configuration for different message pairs"""
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
    
    # Test for multiple message pairs (1+2, 1+4, 1+6)
    message_pair_counts = [3, 5, 7]
    
    for pair_count in message_pair_counts:
        print(f"\nTesting with {pair_count} messages...")
        
        # Count tokens for the first message and the last (pair_count-1) messages
        token_limit = 0
        selected_messages = full_messages[:1] + full_messages[-(pair_count-1):]
        for message in selected_messages:
            message_str = str(message)
            token_limit += count_tokens(message_str)

        print(f"Limited tokens for {pair_count} messages: {token_limit}")
        
        limited_generator = MessageHistoryGenerator(
            include_file_context=True,
            max_tokens=token_limit
        )
        limited_messages = await limited_generator.generate_messages(node5, workspace)
        limited_messages = list(limited_messages)
        
        print(json.dumps(limited_messages, indent=2))
        assert len(limited_messages) == pair_count, f"Expected {pair_count} messages, first message and the last {pair_count-1}, got {len(limited_messages)}"
        
        # Verify the first message is included
        assert limited_messages[0]["role"] == "user"
        assert str(limited_messages[0]) == str(full_messages[0]), "First message should match the original first message"
        
        # Verify the last (pair_count-1) messages are included and in the correct order
        for i in range(1, pair_count):
            full_msg_idx = len(full_messages) - (pair_count - i)
            assert str(limited_messages[i]) == str(full_messages[full_msg_idx]), f"Message at position {i} should match full message at position {full_msg_idx}"
        
        # Verify message order
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


@pytest.mark.asyncio
async def test_observation_summary_usage(test_tree, workspace):
    """Test that observations use summaries when exceeding max_tokens_per_observation, except for the last node."""
    root, node1, node2, node3, node4, node5 = test_tree
    
    # Create a long message that will exceed the default max_tokens_per_observation (16000)
    long_message = "This is a very long message. " * 1000  # This will exceed the token limit
    short_message = "This is a short message."
    summary = "This is a summary of the long message."
    
    # Add observations with long messages and summaries to all nodes
    for node in [node1, node2, node3, node4, node5]:
        node.action_steps[0].observation.message = long_message
        node.action_steps[0].observation.summary = summary
    
    # Create generator with a lower max_tokens_per_observation for testing
    generator = MessageHistoryGenerator(
        include_file_context=True,
        max_tokens_per_observation=1000  # Set a low limit to trigger summary usage
    )
    
    # Test 1: Generate messages for node3 - node3 should be the most recent and show full message
    messages = await generator.generate_messages(node3, workspace)
    messages = list(messages)
    
    # Count tool responses
    tool_responses = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    assert len(tool_responses) == 3, "Expected 3 tool responses for node3 (one for each node in trajectory)"
    
    # Verify non-most-recent nodes (node1, node2) use summaries
    for i in range(len(tool_responses) - 1):
        content = str(tool_responses[i]["content"])
        assert summary in content, f"Expected summary in tool response {i}"
        assert long_message not in content, f"Expected long message to be replaced with summary in tool response {i}"
    
    # Verify most recent node (node3) uses full message
    last_content = str(tool_responses[-1]["content"])
    assert long_message in last_content, "Expected full message for most recent node"
    assert summary not in last_content, "Did not expect summary for most recent node"
    
    # Test 2: Generate messages for node5 (terminal) - node5 should be the most recent and show full message
    messages = await generator.generate_messages(node5, workspace)
    messages = list(messages)
    
    # Count tool responses
    tool_responses = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    assert len(tool_responses) == 5, "Expected 5 tool responses for node5 (one for each node in trajectory)"
    
    # Verify non-most-recent nodes use summaries
    for i in range(len(tool_responses) - 1):
        content = str(tool_responses[i]["content"])
        assert summary in content, f"Expected summary in tool response {i}"
        assert long_message not in content, f"Expected long message to be replaced with summary in tool response {i}"
    
    # Verify most recent node (node5) uses full message
    last_content = str(tool_responses[-1]["content"])
    assert long_message in last_content, "Expected full message for most recent node"
    assert summary not in last_content, "Did not expect summary for most recent node"
    
    # Test 3: Verify that when message is shorter than max_tokens_per_observation, full message is used
    # Update node2 to have a short message
    node2.action_steps[0].observation.message = short_message
    
    # Generate messages for node3
    messages = await generator.generate_messages(node3, workspace)
    messages = list(messages)
    
    # Find the tool response for node2
    tool_responses = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    assert len(tool_responses) >= 2, "Expected at least 2 tool responses"
    
    # Node2's message should be shown in full since it's short
    node2_response_content = str(tool_responses[1]["content"])  # Second tool response should be for node2
    assert short_message in node2_response_content, "Expected short message to be shown in full for node2"
    assert summary not in node2_response_content, "Did not expect summary for node2 with short message"
    
    # Test 4: Verify that multiple action steps in the most recent node all show full messages
    # Add a second action step to node4 with a long message
    second_action = TestActionArguments()
    node4.action_steps.append(
        ActionStep(
            action=second_action, 
            observation=Observation(message=long_message, summary=summary)
        )
    )
    
    # Generate messages for node4 (now with multiple action steps)
    messages = await generator.generate_messages(node4, workspace)
    messages = list(messages)
    
    # Get tool responses
    tool_responses = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    
    # The last two tool responses should be from node4 and both should show the full message
    node4_responses = tool_responses[-2:]  # Last two responses
    assert len(node4_responses) == 2, "Expected 2 tool responses for node4"
    
    for i, response in enumerate(node4_responses):
        content = str(response["content"])
        assert long_message in content, f"Expected full message in action step {i} of the most recent node"
        assert summary not in content, f"Did not expect summary in action step {i} of the most recent node"
    
    # Test 5: Verify mixed message lengths in the most recent node
    # Reset node4's action steps
    node4.action_steps = [node4.action_steps[0]]  # Keep only the first action step
    
    # Add two action steps to node4: one short, one long
    node4.action_steps[0].observation.message = short_message  # First action: short message
    node4.action_steps[0].observation.summary = summary
    
    # Add second action with long message
    node4.action_steps.append(
        ActionStep(
            action=TestActionArguments(), 
            observation=Observation(message=long_message, summary=summary)
        )
    )
    
    # Generate messages for node4 with mixed message lengths
    messages = await generator.generate_messages(node4, workspace)
    messages = list(messages)
    
    # Get tool responses
    tool_responses = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    
    # The last two tool responses should be from node4
    node4_responses = tool_responses[-2:]
    assert len(node4_responses) == 2, "Expected 2 tool responses for node4"
    
    # First response should contain the short message
    content1 = str(node4_responses[0]["content"])
    assert short_message in content1, "Expected short message in first action step of most recent node"
    assert summary not in content1, "Did not expect summary in first action step of most recent node"
    
    # Second response should contain the long message (not summary)
    content2 = str(node4_responses[1]["content"])
    assert long_message in content2, "Expected long message in second action step of most recent node"
    assert summary not in content2, "Did not expect summary in second action step of most recent node"
    
    # Test 6: Verify that the last executed node (not the last node in trajectory) shows full messages
    # Setup: Add actions to node4, but not to node5 (simulating that node5 hasn't executed yet)
    node4.action_steps = [
        ActionStep(
            action=TestActionArguments(), 
            observation=Observation(message=long_message, summary=summary)
        )
    ]
    node5.action_steps = []  # No action steps for the last node
    
    # Generate messages for node5 (which doesn't have action steps)
    messages = await generator.generate_messages(node5, workspace)
    messages = list(messages)
    
    # Get tool responses
    tool_responses = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    
    # node4 should be the last executed node, so it should show full messages
    # Get the last tool response (should be from node4)
    if tool_responses:
        last_executed_content = str(tool_responses[-1]["content"])
        assert long_message in last_executed_content, "Expected full message for last executed node (node4)"
        assert summary not in last_executed_content, "Did not expect summary for last executed node (node4)"
    
    # Test 7: Verify root node behavior
    # Create a simple trajectory with only root and node1
    root_only = Node(node_id=100, file_context=FileContext(repo=repo))
    root_only.message = "Root only task"
    
    # Add action steps to root
    root_action = TestActionArguments()
    root_only.action_steps = [
        ActionStep(
            action=root_action, 
            observation=Observation(message=long_message, summary=summary)
        )
    ]
    
    # Node1 without action steps (not executed yet)
    node1_only = Node(node_id=101)
    node1_only.file_context = FileContext(repo=repo)
    root_only.add_child(node1_only)
    
    # Generate messages for node1
    messages = await generator.generate_messages(node1_only, workspace)
    messages = list(messages)
    
    # Get tool responses
    tool_responses = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    
    # Root should be the last (and only) executed node, so it should show full messages
    if tool_responses:
        root_content = str(tool_responses[0]["content"])
        assert long_message in root_content, "Expected full message for root node as the only executed node"
        assert summary not in root_content, "Did not expect summary for root node as the only executed node"
