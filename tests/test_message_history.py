import json

import pytest

from moatless.actions.create_file import CreateFileArgs
from moatless.actions.finish import FinishArgs
from moatless.actions.model import Observation, ActionArguments
from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.actions.view_code import CodeSpan, ViewCodeArgs
from moatless.completion.model import UserMessage, AssistantMessage
from moatless.file_context import FileContext
from moatless.message_history import MessageHistoryGenerator, MessageHistoryType
from moatless.node import Node
from moatless.repository.repository import InMemRepository
from moatless.runtime.runtime import TestResult, TestStatus
from moatless.utils.tokenizer import count_tokens


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


@pytest.fixture
def test_tree(repo) -> tuple[Node, Node, Node, Node, Node, Node]:
    """Creates a test tree with various actions and file contexts"""
    root = Node(node_id=0, file_context=FileContext(repo=repo))
    root.message = "Initial task"

    # Node1: View code action
    node1 = Node(node_id=1)
    node1.action = TestActionArguments()
    node1.file_context = FileContext(repo=repo)
    node1.file_context.add_span_to_context("file1.py", "method1")
    node1.observation = Observation(message="Added method1 to context")
    root.add_child(node1)

    # Node2: Another view action
    node2 = Node(node_id=2)
    node2.action = TestActionArguments()
    node2.file_context = node1.file_context.clone()
    node2.file_context.add_span_to_context("file2.py", "method2")
    node2.observation = Observation(message="Added method2 to context")
    node1.add_child(node2)

    # Node3: Apply change action
    node3 = Node(node_id=3)
    node3.action = StringReplaceArgs(
        path="file1.py",
        old_str='return "original1"',
        new_str='return "modified1"',
        scratch_pad="Modifying method1 return value"
    )
    node3.file_context = node2.file_context.clone()
    node3.file_context.add_file("file1.py").apply_changes("""def method1():
    return "modified1"
""")
    node3.observation = Observation(message="Modified method1")
    node2.add_child(node3)

    # Node4: View another method
    node4 = Node(node_id=4)
    node4.action = TestActionArguments()
    node4.file_context = node3.file_context.clone()
    node4.file_context.add_span_to_context("file2.py", "method3")
    node4.observation = Observation(message="Added method3 to context")
    node3.add_child(node4)

    # Node5: Finish action
    node5 = Node(node_id=5)
    node5.action = FinishArgs(
        scratch_pad="All changes complete",
        finish_reason="Successfully modified the code"
    )
    node5.observation = Observation(message="Task completed successfully", terminal=True)
    node4.add_child(node5)

    return root, node1, node2, node3, node4, node5


def test_messages_history_type(test_tree):
    """Test MESSAGES history type with different configurations"""
    _, _, node2, node3, node4, _ = test_tree
    
    # Basic message history
    generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.MESSAGES,
        include_file_context=True
    )
    messages = list(generator.generate(node2))
    
    # Verify initial message
    assert messages[0].content == "Initial task"
    # Verify action and observation messages
    assert isinstance(messages[1], AssistantMessage)  # Action message
    assert isinstance(messages[2], UserMessage)  # Observation message
    assert "Added method1 to context" in messages[2].content
    
    # With file changes
    messages = list(generator.generate(node3))
    assert len(messages) >= 5
    
    # Debug output
    print("\nMessages for node3:")
    for i, msg in enumerate(messages):
        print(f"Message {i}: {type(msg).__name__} - Content: {msg.content}")
        if hasattr(msg, 'tool_call'):
            print(f"Tool call: {msg.tool_call}")
    
    # Verify file modification is included
    modification_found = any(
        ("modified1" in (m.content or "")) or  # Check content
        (hasattr(m, 'tool_call') and  # Check tool call input
         isinstance(m, AssistantMessage) and
         "modified1" in str(m.tool_call.input))  # Convert input to string to search
        for m in messages
    )
    
    assert modification_found, "Modified content not found in messages"
    
    # With multiple file contexts
    messages = list(generator.generate(node4))
    assert len(messages) >= 7
    assert any("method3" in (m.content or "") for m in messages), "Method3 not found in messages"


def test_react_history_type(test_tree):
    """Test REACT history type generation"""
    _, _, _, node3, _, _ = test_tree
    
    generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.REACT,
        include_file_context=True
    )
    messages = list(generator.generate(node3))  # Convert generator to list
    
    # Verify ReAct format
    assert any("Thought:" in m.content for m in messages), "Missing Thought: in messages"
    assert any("Action:" in m.content for m in messages), "Missing Action: in messages"
    assert any("Observation:" in m.content for m in messages), "Missing Observation: in messages"
    
    # Verify file changes are included
    assert any("modified1" in m.content for m in messages), "Modified file content not found in messages"
    
    # Print messages for debugging if needed
    for msg in messages:
        print(f"Message content: {msg.content}")


def test_react_history_file_updates(test_tree):
    """Test that REACT history shows file contents at their last update point"""
    _, _, _, node3, node4, _ = test_tree

    generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.REACT,
        include_file_context=True
    )
    
    # Test messages up to node3
    messages = generator.generate(node3)
    messages_list = list(messages)

    # Verify correct message sequence
    assert isinstance(messages_list[0], UserMessage)  # Initial task
    
    # Find ViewCode sequence for file1.py
    view_code_index = None
    for i, msg in enumerate(messages_list[1:], 1):
        if (isinstance(msg, AssistantMessage) and 
            "Let's view the content in file1.py" in msg.content):
            view_code_index = i
            break
    
    assert view_code_index is not None, "ViewCode message for file1.py not found"
    # Verify ViewCode pair
    assert isinstance(messages_list[view_code_index], AssistantMessage)
    assert isinstance(messages_list[view_code_index + 1], UserMessage)
    assert "file1.py" in messages_list[view_code_index].content
    assert 'return "original1"' in messages_list[view_code_index + 1].content
    
    # Remove the incorrect StringReplace check and instead verify the correct sequence
    # The last action should be viewing file2.py from node2
    last_action_index = len(messages_list) - 2  # Second to last message should be the last Assistant message
    assert isinstance(messages_list[last_action_index], AssistantMessage)
    assert "Let's view the content in file2.py" in messages_list[last_action_index].content
    assert isinstance(messages_list[last_action_index + 1], UserMessage)
    assert 'return "original2"' in messages_list[last_action_index + 1].content

    # Test messages up to node4
    messages = generator.generate(node4)
    messages_list = list(messages)

    # Find StringReplace action (should be present in node4's history)
    string_replace_index = None
    for i, msg in enumerate(messages_list):
        if (isinstance(msg, AssistantMessage) and 
            "Action: StringReplace" in msg.content):
            string_replace_index = i
            break
    
    assert string_replace_index is not None, "StringReplace action not found in node4's history"
    assert isinstance(messages_list[string_replace_index + 1], UserMessage)
    assert "Modified method1" in messages_list[string_replace_index + 1].content

    # Verify method3 view is not in the history (it's the current node)
    for msg in messages_list:
        assert "method3" not in msg.content, "Current node's action (viewing method3) should not be in history"


def test_summary_history_type(test_tree):
    """Test SUMMARY history type generation"""
    _, _, _, _, node4, _ = test_tree
    
    generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.SUMMARY,
        include_file_context=True
    )
    messages = generator.generate(node4)
    
    assert len(messages) == 1  # Summary should be a single message
    content = messages[0].content
    assert "history" in content
    assert "method1" in content
    assert "method2" in content
    assert "method3" in content



def test_terminal_node_history(test_tree):
    """Test history generation for terminal nodes"""
    _, _, _, _, _, node5 = test_tree
    
    generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.REACT
    )
    messages = list(generator.generate(node5))
    
    # Verify finish action content
    finish_action_found = any(
        isinstance(m, AssistantMessage) and 
        ("Action: Finish" in (m.content or ""))  # Simplified check
        for m in messages
    )
    
    # Verify observation content
    finish_observation_found = any(
        isinstance(m, UserMessage) and 
        "Task completed successfully" in (m.content or "")
        for m in messages
    )
    
    assert finish_action_found, "Finish action message not found"
    assert finish_observation_found, "Finish observation message not found"


def test_empty_history():
    """Test history generation for nodes without history"""
    root = Node(node_id=0)
    root.message = "Initial task"
    
    generator = MessageHistoryGenerator()
    messages = generator.generate(root)
    
    assert len(messages) == 0


def test_message_history_serialization():
    """Test MessageHistoryGenerator serialization"""
    generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.REACT,
        include_file_context=True,
        include_git_patch=False,
        show_full_file=True
    )
    
    # Test serialization
    data = generator.model_dump()
    # The enum is already serialized to string
    assert data["message_history_type"] == "react"
    assert data["include_file_context"] is True
    assert data["include_git_patch"] is False


def test_message_history_dump_and_load():
    """Test MessageHistoryGenerator dump and load functionality"""
    # Create original generator
    original = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.REACT,
        include_file_context=True,
        include_git_patch=False
    )
    
    # Test JSON serialization
    json_str = original.model_dump_json()
    loaded_dict = json.loads(json_str)
    assert loaded_dict["message_history_type"] == "react"
    assert loaded_dict["include_file_context"] is True
    assert loaded_dict["include_git_patch"] is False
    
    # Test model reconstruction from JSON
    loaded = MessageHistoryGenerator.model_validate_json(json_str)
    assert loaded.message_history_type == MessageHistoryType.REACT
    assert loaded.include_file_context is True
    assert loaded.include_git_patch is False
    
    # Test dictionary serialization
    dict_data = original.model_dump()
    loaded_from_dict = MessageHistoryGenerator.model_validate(dict_data)
    assert loaded_from_dict.message_history_type == MessageHistoryType.REACT
    assert loaded_from_dict.include_file_context is True
    assert loaded_from_dict.include_git_patch is False


def test_react_history_max_tokens(test_tree):
    """Test that message history respects max token limit"""
    _, _, _, node3, node4, _ = test_tree
    
    # Set a very low token limit that should only allow a few messages
    generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.REACT,
        include_file_context=True,
        max_tokens=150  # Small limit to force truncation
    )
    
    # Get messages for node4 (which has the most history)
    messages = list(generator.generate(node4))
    
    print(f"\n=== {len(messages)} Messages with token limit ===")
    for i, msg in enumerate(messages):
        print(f"{i}. {'Assistant' if isinstance(msg, AssistantMessage) else 'User'}: {msg.content}")
    
    # Verify basics
    assert len(messages) > 0, "Should have at least some messages"
    assert isinstance(messages[0], UserMessage), "Should start with initial task"
    
    # Verify token count is under limit
    total_content = "".join([m.content for m in messages if m.content is not None])
    tokens = count_tokens(total_content)
    assert tokens <= 150, f"Token count {tokens} exceeds limit of 150"
    
    # Verify messages are properly paired
    assert len(messages) % 2 == 1, "Messages should be in pairs plus initial message"
    for i in range(1, len(messages), 2):
        assert isinstance(messages[i], AssistantMessage), f"Message {i} should be Assistant"
        assert isinstance(messages[i + 1], UserMessage), f"Message {i + 1} should be User"
    
    # Compare with unlimited history
    unlimited_generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.REACT,
        include_file_context=True,
        max_tokens=10000
    )
    unlimited_messages = list(unlimited_generator.generate(node4))
    
    print(f"\n=== {len(unlimited_messages)} Messages without token limit ===")
    for i, msg in enumerate(unlimited_messages):
        print(f"{i}. {'Assistant' if isinstance(msg, AssistantMessage) else 'User'}: {msg.content}")
    
    assert len(unlimited_messages) > len(messages), "Limited messages should be shorter than unlimited"
    
    # Verify that we get the most recent complete message pairs
    assert messages[0].content == unlimited_messages[0].content, "Initial task should be preserved"
    assert messages[-2:][0].content == unlimited_messages[-2:][0].content, "Last message pair should match"
    assert messages[-2:][1].content == unlimited_messages[-2:][1].content, "Last message pair should match" 
    

def test_react_history_file_context_with_view_code_actions(repo):
    """Test that file context is shown correctly with ViewCode and non-ViewCode actions"""
    # Create root node with file context (same as fixture)
    root = Node(node_id=0, file_context=FileContext(repo=repo))
    root.message = "Initial task"
    
    # Create a new branch with ViewCode and StringReplace actions
    node1 = Node(node_id=10)
    node1.action = ViewCodeArgs(
        scratch_pad="Let's look at method1",
        files=[CodeSpan(file_path="file1.py", span_ids=["method1"])]
    )
    node1.file_context = FileContext(repo=repo)  # Use the repo fixture directly
    node1.file_context.add_span_to_context("file1.py", "method1")
    node1.observation = Observation(message="Here's method1's content")
    root.add_child(node1)

    # Add StringReplace action that modifies the viewed file
    node2 = Node(node_id=11)
    node2.action = StringReplaceArgs(
        path="file1.py",
        old_str='return "original1"',
        new_str='return "modified1"',
        scratch_pad="Modifying method1 return value"
    )
    node2.file_context = node1.file_context.clone()
    node2.file_context.add_file("file1.py").apply_changes("""def method1():
    return "modified1"
""")
    node2.observation = Observation(message="Modified method1")
    node1.add_child(node2)

    node3 = Node(node_id=12)
    node3.file_context = node2.file_context.clone()
    node2.add_child(node3)

    generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.REACT,
        include_file_context=True
    )
    
    messages = list(generator.generate(node3))
    
    print("\n=== Messages ===")
    for i, msg in enumerate(messages):
        print(f"{i}. {'Assistant' if isinstance(msg, AssistantMessage) else 'User'}: {msg.content}")
    
    # Find all ViewCode actions
    viewcode_messages = [
        m for m in messages 
        if isinstance(m, AssistantMessage) and 
        isinstance(m.content, str) and
        "Action: ViewCode" in m.content
    ]
    
    # Find all file contents
    file_content_messages = [
        m for m in messages 
        if isinstance(m, UserMessage) and 
        'file1.py' in m.content
    ]
    
    # Find StringReplace action
    stringreplace_messages = [
        m for m in messages 
        if isinstance(m, AssistantMessage) and 
        "Action: StringReplace" in m.content
    ]
    
    # Verify we have exactly one ViewCode action
    assert len(viewcode_messages) == 1, f"Expected one ViewCode action, got {len(viewcode_messages)}"
    
    # Verify we have exactly one file content message
    assert len(file_content_messages) == 1, f"Expected one file content message, got {len(file_content_messages)}"
    
    # Verify we have exactly one StringReplace action
    assert len(stringreplace_messages) == 1, f"Expected one StringReplace action, got {len(stringreplace_messages)}"
    
    # Verify the sequence: StringReplace -> ViewCode -> file content
    viewcode_index = messages.index(viewcode_messages[0])
    content_index = messages.index(file_content_messages[0])
    stringreplace_index = messages.index(stringreplace_messages[0])
    
    assert stringreplace_index < viewcode_index < content_index, (
        "Messages should be in order: StringReplace -> ViewCode -> file content"
    )
    

def test_get_node_messages_with_failed_viewcode(repo):
    """Test get_node_messages with a failed ViewCode action"""
    # Create root node
    root = Node(node_id=0, file_context=FileContext(repo=repo))
    root.message = "Initial task"
    
    # Create node with failed ViewCode action
    node1 = Node(node_id=1)
    node1.action = ViewCodeArgs(
        scratch_pad="Let's look at the voting code",
        files=[CodeSpan(file_path="sklearn/ensemble/_voting.py")]
    )
    node1.file_context = FileContext(repo=repo)
    node1.observation = Observation(
        message="The requested file sklearn/ensemble/_voting.py is not found in the file repository. "
        "Use the search functions to search for the code if you are unsure of the file path."
    )
    root.add_child(node1)
    
    # Create node2 as child of node1
    node2 = Node(node_id=2)
    node2.file_context = node1.file_context.clone()
    node1.add_child(node2)
    
    # Create generator and get messages
    generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.REACT,
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
    

def test_react_history_with_test_results(repo):
    """Test that test results are shown correctly after file modifications"""
    # Create root node with file context
    root = Node(node_id=0, file_context=FileContext(repo=repo))
    root.message = "Initial task"
    
    print("\n=== Setting up test files and results ===")
    
    # Node1: Create the initial file
    node1 = Node(node_id=10)
    node1.action = CreateFileArgs(
        path="src/example.py",
        file_text="""def add(a, b):
    return a + b""",
        scratch_pad="Creating a new example file"
    )
    node1.file_context = FileContext(repo=repo)  # Use the repo fixture directly
    node1.file_context.add_file("src/example.py").apply_changes("""def add(a, b):
    return a + b""")
    node1.observation = Observation(message="File created successfully at: src/example.py")
    root.add_child(node1)
    
    # Node2: View the file
    node2 = Node(node_id=11)
    node2.action = ViewCodeArgs(
        scratch_pad="Let's look at the new file",
        files=[CodeSpan(file_path="src/example.py")]
    )
    node2.file_context = node1.file_context.clone()
    node2.observation = Observation(message="""Here's the content of src/example.py:

def add(a, b):
    return a + b""")
    node1.add_child(node2)

    # Node3: Add test files and modify the file
    node3 = Node(node_id=12)
    node3.action = CreateFileArgs(
        path="tests/test_example.py",
        file_text="""def test_add():
    assert add(2, 2) == 4""",
        scratch_pad="Creating test file"
    )
    node3.file_context = node2.file_context.clone()
    node3.file_context.add_test_file("tests/test_file1.py")
    node3.file_context.add_test_file("tests/test_file2.py")
    node3.observation = Observation(message="Test file created successfully")
    node2.add_child(node3)

    # Node4: View the test file
    node4 = Node(node_id=13)
    node4.action = ViewCodeArgs(
        scratch_pad="Let's look at the test file",
        files=[CodeSpan(file_path="tests/test_example.py")]
    )
    node4.file_context = node3.file_context.clone()
    node4.observation = Observation(message="""Here's the content of tests/test_example.py:

def test_add():
    assert add(2, 2) == 4""")
    node3.add_child(node4)

    # Node5: Modify the original file
    node5 = Node(node_id=14)
    node5.action = CreateFileArgs(
        path="src/example.py",
        file_text="""def add(a, b):
    return 0  # Bug: always returns 0""",
        scratch_pad="Modifying the add function (with a bug)"
    )
    node5.file_context = node4.file_context.clone()
    node5.file_context.add_file("src/example.py").apply_changes("""def add(a, b):
    return 0  # Bug: always returns 0""")
    
    # Add test results to test files
    test_results = [
        TestResult(
            status=TestStatus.FAILED,
            message="AssertionError: Expected add(2, 2) to equal 4, got 0",
            file_path="tests/test_file1.py",
            span_id="test_add",
            line=15
        ),
        TestResult(
            status=TestStatus.PASSED,
            file_path="tests/test_file1.py",
            span_id="test_add_negative"
        ),
        TestResult(
            status=TestStatus.ERROR,
            message="ImportError: Cannot import module 'src.example'",
            file_path="tests/test_file2.py",
            line=5
        )
    ]
    
    print("\nTest files in context:")
    for file_path in node5.file_context._test_files:
        print(f"* {file_path}")
    
    for test_file in node5.file_context._test_files.values():
        test_file.test_results = [r for r in test_results if r.file_path == test_file.file_path]
        print(f"\nResults for {test_file.file_path}:")
        for result in test_file.test_results:
            print(f"- {result.status}: {result.message or 'No message'}")
            
    node5.observation = Observation(message="File modified successfully")
    node4.add_child(node5)

    # Node6: Current node
    node6 = Node(node_id=15)
    node6.file_context = node5.file_context.clone()
    node5.add_child(node6)

    generator = MessageHistoryGenerator(
        message_history_type=MessageHistoryType.REACT,
        include_file_context=True
    )
    
    print("\n=== Generated Messages ===")
    messages = list(generator.get_node_messages(node6))
    for i, (action, observation) in enumerate(messages):
        print(f"\nMessage {i}:")
        print(f"Action: {action.__class__.__name__}")
        print(f"Observation:\n{observation}")
    
    # Find all actions
    viewcode_messages = [m for m in messages if isinstance(m[0], ViewCodeArgs)]
    createfile_messages = [m for m in messages if isinstance(m[0], CreateFileArgs)]
    runtests_messages = [m for m in messages if isinstance(m[0], RunTestsArgs)]
    
    # Verify we have the expected number of each action type
    assert len(viewcode_messages) == 2, f"Expected two ViewCode actions, got {len(viewcode_messages)}"
    assert len(createfile_messages) == 2, f"Expected two CreateFile actions, got {len(createfile_messages)}"
    assert len(runtests_messages) == 1, f"Expected one RunTests action, got {len(runtests_messages)}"
    
    # Get the RunTests message
    run_tests_message = runtests_messages[0][1]
    print("\n=== Test Results Message ===")
    print(run_tests_message)
    
    # Verify test file paths are listed
    assert "Running tests..." in run_tests_message
    assert "* tests/test_file1.py" in run_tests_message
    assert "* tests/test_file2.py" in run_tests_message
    
    # Verify failure details
    assert "FAILED tests/test_file1.py test_add, line: 15" in run_tests_message
    assert "AssertionError: Expected add(2, 2) to equal 4, got 0" in run_tests_message
    assert "ERROR tests/test_file2.py, line: 5" in run_tests_message
    assert "ImportError: Cannot import module 'src.example'" in run_tests_message
    
    # Verify test summary
    assert "1 passed. 1 failed. 1 errors." in run_tests_message
    