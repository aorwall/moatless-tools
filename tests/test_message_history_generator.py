import json
import pytest
import pytest_asyncio

from moatless.actions.create_file import CreateFileArgs
from moatless.actions.finish import FinishArgs
from moatless.actions.schema import Observation, ActionArguments
from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.actions.view_code import CodeSpan, ViewCodeArgs
from moatless.completion.schema import ChatCompletionAssistantMessage, ChatCompletionUserMessage
from moatless.file_context import FileContext
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.node import Node, ActionStep
from moatless.repository.repository import InMemRepository
from moatless.runtime.runtime import TestResult, TestStatus
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
    
    # Verify initial message
    assert messages[0]["content"] == "Initial task"
    # Verify action and observation messages
    assert messages[1]["role"] == "assistant"  # Action message
    assert messages[2]["role"] == "user"  # Observation message
    assert "Added method1 to context" in messages[2]["content"]
    
    # With file changes
    messages = await generator.generate_messages(node3, workspace)
    messages = list(messages)
    assert len(messages) >= 5
    
    # Debug output
    print("\nMessages for node3:")
    for i, msg in enumerate(messages):
        print(f"Message {i}: {type(msg).__name__} - Content: {msg['content']}")
        if hasattr(msg, 'tool_call'):
            print(f"Tool call: {msg.tool_call}")
    
    # Verify file modification is included
    modification_found = any(
        ("modified1" in (m["content"] or "")) or  # Check content
        (hasattr(m, 'tool_call') and  # Check tool call input
         "modified1" in str(m.tool_call.input))  # Convert input to string to search
        for m in messages
    )
    
    assert modification_found, "Modified content not found in messages"
    
    # With multiple file contexts
    messages = await generator.generate_messages(node4, workspace)
    messages = list(messages)
    assert len(messages) >= 7
    assert any("method3" in (m["content"] or "") for m in messages), "Method3 not found in messages"


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


@pytest.mark.asyncio
async def test_empty_history(workspace):
    """Test history generation for nodes without history"""
    root = Node(node_id=0)
    root.message = "Initial task"
    
    generator = MessageHistoryGenerator()
    messages = await generator.generate_messages(root, workspace)
    messages = list(messages)
    
    assert len(messages) == 0


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
async def test_message_history_max_tokens(test_tree, workspace):
    """Test that message history respects max token limit"""
    _, _, _, node3, node4, _ = test_tree
    
    # Set a very low token limit that should only allow a few messages
    generator = MessageHistoryGenerator(
        include_file_context=True,
        max_tokens=150  # Small limit to force truncation
    )
    
    # Get messages for node4 (which has the most history)
    messages = await generator.generate_messages(node4, workspace)
    messages = list(messages)
    
    print(f"\n=== {len(messages)} Messages with token limit ===")
    for i, msg in enumerate(messages):
        print(f"{i}. {'Assistant' if isinstance(msg, ChatCompletionAssistantMessage) else 'User'}: {msg['content']}")
    
    # Verify basics
    assert len(messages) > 0, "Should have at least some messages"
    assert isinstance(messages[0], ChatCompletionUserMessage), "Should start with initial task"
    
    # Verify token count is under limit
    total_content = "".join([m["content"] for m in messages if m.get("content") is not None])
    tokens = count_tokens(total_content)
    assert tokens <= 150, f"Token count {tokens} exceeds limit of 150"
    
    # Verify messages are properly paired
    assert len(messages) % 2 == 1, "Messages should be in pairs plus initial message"
    for i in range(1, len(messages), 2):
        assert isinstance(messages[i], ChatCompletionAssistantMessage), f"Message {i} should be Assistant"
        assert isinstance(messages[i + 1], ChatCompletionUserMessage), f"Message {i + 1} should be User"
    
    # Compare with unlimited history
    unlimited_generator = MessageHistoryGenerator(
        include_file_context=True,
        max_tokens=10000
    )
    unlimited_messages = await unlimited_generator.generate_messages(node4, workspace)
    unlimited_messages = list(unlimited_messages)
    
    print(f"\n=== {len(unlimited_messages)} Messages without token limit ===")
    for i, msg in enumerate(unlimited_messages):
        print(f"{i}. {'Assistant' if isinstance(msg, ChatCompletionAssistantMessage) else 'User'}: {msg['content']}")
    
    assert len(unlimited_messages) > len(messages), "Limited messages should be shorter than unlimited"
    
    # Verify that we get the most recent complete message pairs
    assert messages[0]["content"] == unlimited_messages[0]["content"], "Initial task should be preserved"
    assert messages[-2:][0]["content"] == unlimited_messages[-2:][0]["content"], "Last message pair should match"
    assert messages[-2:][1]["content"] == unlimited_messages[-2:][1]["content"], "Last message pair should match"


def test_get_node_messages_with_failed_viewcode(repo, workspace):
    """Test get_node_messages with a failed ViewCode action"""
    # Create root node
    root = Node(node_id=0, file_context=FileContext(repo=repo))
    root.message = "Initial task"
    
    # Create node with failed ViewCode action
    node1 = Node(node_id=1)
    action1 = ViewCodeArgs(
        scratch_pad="Let's look at the voting code",
        files=[CodeSpan(file_path="sklearn/ensemble/_voting.py")]
    )
    observation1 = Observation(
        message="The requested file sklearn/ensemble/_voting.py is not found in the file repository. "
        "Use the search functions to search for the code if you are unsure of the file path."
    )
    node1.action_steps.append(ActionStep(action=action1, observation=observation1))
    node1.file_context = FileContext(repo=repo)
    root.add_child(node1)
    
    # Create node2 as child of node1
    node2 = Node(node_id=2)
    node2.file_context = node1.file_context.clone()
    node1.add_child(node2)
    
    # Create generator and get messages
    generator = MessageHistoryGenerator(
        include_file_context=True
    )
    
    messages = generator.get_node_messages(node2)
    
    # Verify we got exactly one message pair
    assert len(messages) == 1, f"Expected one message pair, got {len(messages)}"
    
    # Verify the action and observation content
    action, observation = messages[0]
    assert isinstance(action, ViewCodeArgs)
    assert action.files[0].file_path == "sklearn/ensemble/_voting.py"
    assert observation == (
        "The requested file sklearn/ensemble/_voting.py is not found in the file repository. "
        "Use the search functions to search for the code if you are unsure of the file path."
    ) 