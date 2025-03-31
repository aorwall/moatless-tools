import pytest
import pytest_asyncio

from moatless.actions.create_file import CreateFileArgs
from moatless.actions.schema import Observation, ActionArguments
from moatless.actions.string_replace import StringReplaceArgs
from moatless.actions.view_code import CodeSpan, ViewCodeArgs
from moatless.completion.schema import ChatCompletionAssistantMessage, ChatCompletionUserMessage
from moatless.file_context import FileContext
from moatless.message_history.react_compact import ReactCompactMessageHistoryGenerator
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
def test_tree(repo) -> tuple[Node, Node, Node, Node, Node]:
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

    return root, node1, node2, node3, node4


@pytest.mark.asyncio
async def test_react_history_type(test_tree, workspace):
    """Test REACT history type generation"""
    _, _, _, node3, _ = test_tree
    
    generator = ReactCompactMessageHistoryGenerator(
        include_file_context=True
    )
    messages = await generator.generate_messages(node3, workspace)
    messages = list(messages)
    
    # Verify ReAct format
    assert any("Thought:" in str(m) for m in messages), "Missing Thought: in messages"
    assert any("Action:" in str(m) for m in messages), "Missing Action: in messages"
    assert any("Observation:" in str(m) for m in messages), "Missing Observation: in messages"
    
    # Verify file changes are included
    assert any("modified1" in str(m) for m in messages), "Modified file content not found in messages"


@pytest.mark.asyncio
async def test_react_history_file_context_with_view_code_actions(repo, workspace):
    """Test that file context is shown correctly with ViewCode and non-ViewCode actions"""
    # Create root node with file context (same as fixture)
    root = Node(node_id=0, file_context=FileContext(repo=repo))
    root.message = "Initial task"
    
    # Create a new branch with ViewCode and StringReplace actions
    node1 = Node(node_id=10)
    action1 = ViewCodeArgs(
        scratch_pad="Let's look at method1",
        files=[CodeSpan(file_path="file1.py", span_ids=["method1"])]
    )
    node1.action_steps.append(ActionStep(action=action1, observation=Observation(message="Here's method1's content")))
    node1.file_context = FileContext(repo=repo)  # Use the repo fixture directly
    node1.file_context.add_span_to_context("file1.py", "method1")
    root.add_child(node1)

    # Add StringReplace action that modifies the viewed file
    node2 = Node(node_id=11)
    action2 = StringReplaceArgs(
        path="file1.py",
        old_str='return "original1"',
        new_str='return "modified1"',
        scratch_pad="Modifying method1 return value"
    )
    node2.action_steps.append(ActionStep(action=action2, observation=Observation(message="Modified method1")))
    node2.file_context = node1.file_context.clone()
    node2.file_context.add_file("file1.py").apply_changes("""def method1():
    return "modified1"
""")
    node1.add_child(node2)

    node3 = Node(node_id=12)
    node3.file_context = node2.file_context.clone()
    node2.add_child(node3)

    generator = ReactCompactMessageHistoryGenerator(
        include_file_context=True
    )
    
    messages = await generator.generate_messages(node3, workspace)
    messages = list(messages)
    
    print("\n=== Messages ===")
    for i, msg in enumerate(messages):
        print(f"{i}. {str(msg)[:100]}")
    
    # Find all ViewCode actions
    viewcode_messages = [
        i for i, m in enumerate(messages) 
        if "Action: ViewCode" in str(m)
    ]
    
    # Find StringReplace action
    stringreplace_messages = [
        i for i, m in enumerate(messages) 
        if "Action: StringReplace" in str(m)
    ]
    
    # Verify we have exactly one ViewCode action
    assert len(viewcode_messages) == 1, f"Expected one ViewCode action, got {len(viewcode_messages)}"
    
    # Verify we have exactly one StringReplace action
    assert len(stringreplace_messages) == 1, f"Expected one StringReplace action, got {len(stringreplace_messages)}"
    
    # Verify the sequence: ViewCode -> file content -> StringReplace
    viewcode_index = viewcode_messages[0]
    stringreplace_index = stringreplace_messages[0]
    
    assert viewcode_index > stringreplace_index, (
        "Messages should be in order: ViewCode -> StringReplace"
    ) 