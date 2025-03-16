import pytest
import pytest_asyncio

from moatless.actions.create_file import CreateFileArgs
from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.schema import Observation
from moatless.actions.view_code import CodeSpan, ViewCodeArgs
from moatless.file_context import FileContext
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.node import Node, ActionStep
from moatless.repository.repository import InMemRepository
from moatless.runtime.runtime import TestResult, TestStatus
from moatless.workspace import Workspace

# pyright: reportCallIssue=false, reportAttributeAccessIssue=false


@pytest.fixture
def repo():
    repo = InMemRepository()
    return repo


@pytest_asyncio.fixture
async def workspace(repo):
    workspace = Workspace(repository=repo)
    return workspace


@pytest.fixture
def test_tree_with_results(repo):
    """Creates a test tree with test files and results"""
    # Create root node with file context
    root = Node(node_id=0, file_context=FileContext(repo=repo))
    root.message = "Initial task"
    
    # Node1: Create the initial file
    node1 = Node(node_id=10)
    action1 = CreateFileArgs(
        path="src/example.py",
        file_text="""def add(a, b):
    return a + b""",
        scratch_pad="Creating a new example file"
    )
    node1.action_steps.append(ActionStep(action=action1, observation=Observation(message="File created successfully at: src/example.py")))
    node1.file_context = FileContext(repo=repo)
    node1.file_context.add_file("src/example.py").apply_changes("""def add(a, b):
    return a + b""")
    root.add_child(node1)
    
    # Node2: View the file
    node2 = Node(node_id=11)
    action2 = ViewCodeArgs(
        scratch_pad="Let's look at the new file",
        files=[CodeSpan(file_path="src/example.py")]
    )
    node2.action_steps.append(ActionStep(action=action2, observation=Observation(message="""Here's the content of src/example.py:

def add(a, b):
    return a + b""")))
    node2.file_context = node1.file_context.clone()
    node1.add_child(node2)

    # Node3: Add test files and modify the file
    node3 = Node(node_id=12)
    action3 = CreateFileArgs(
        path="tests/test_example.py",
        file_text="""def test_add():
    assert add(2, 2) == 4""",
        scratch_pad="Creating test file"
    )
    node3.action_steps.append(ActionStep(action=action3, observation=Observation(message="Test file created successfully")))
    node3.file_context = node2.file_context.clone()
    node3.file_context.add_test_file("tests/test_file1.py")
    node3.file_context.add_test_file("tests/test_file2.py")
    node2.add_child(node3)

    # Node4: View the test file
    node4 = Node(node_id=13)
    action4 = ViewCodeArgs(
        scratch_pad="Let's look at the test file",
        files=[CodeSpan(file_path="tests/test_example.py")]
    )
    node4.action_steps.append(ActionStep(action=action4, observation=Observation(message="""Here's the content of tests/test_example.py:

def test_add():
    assert add(2, 2) == 4""")))
    node4.file_context = node3.file_context.clone()
    node3.add_child(node4)

    # Node5: Modify the original file
    node5 = Node(node_id=14)
    action5 = CreateFileArgs(
        path="src/example.py",
        file_text="""def add(a, b):
    return 0  # Bug: always returns 0""",
        scratch_pad="Modifying the add function (with a bug)"
    )
    node5.action_steps.append(ActionStep(action=action5, observation=Observation(message="File modified successfully")))
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
    
    for test_file in node5.file_context._test_files.values():
        test_file.test_results = [r for r in test_results if r.file_path == test_file.file_path]
            
    node4.add_child(node5)

    # Node6: Current node
    node6 = Node(node_id=15)
    node6.file_context = node5.file_context.clone()
    node5.add_child(node6)
    
    return root, node1, node2, node3, node4, node5, node6


@pytest.mark.asyncio
async def test_react_history_with_test_results(test_tree_with_results, workspace):
    """Test that test results are shown correctly after file modifications"""
    _, _, _, _, _, _, node6 = test_tree_with_results
    
    generator = MessageHistoryGenerator(
        include_file_context=True
    )
    
    # Since get_node_messages doesn't exist, we'll use generate_messages instead
    messages = await generator.generate_messages(node6, workspace)
    messages = list(messages)
    
    # Find all actions in the messages
    viewcode_actions = []
    createfile_actions = []
    runtests_actions = []
    
    for msg in messages:
        if "ViewCode" in str(msg):
            viewcode_actions.append(msg)
        elif "CreateFile" in str(msg):
            createfile_actions.append(msg)
        elif "RunTests" in str(msg):
            runtests_actions.append(msg)
    
    # Verify we have the expected number of each action type
    assert len(viewcode_actions) >= 2, f"Expected at least two ViewCode actions, got {len(viewcode_actions)}"
    assert len(createfile_actions) >= 2, f"Expected at least two CreateFile actions, got {len(createfile_actions)}"
    
    # Verify test results are included
    test_results_found = False
    for msg in messages:
        msg_str = str(msg)
        if all(term in msg_str for term in ["tests/test_file1.py", "tests/test_file2.py", "FAILED", "PASSED", "ERROR"]):
            test_results_found = True
            break
    
    assert test_results_found, "Test results not found in messages"
    
    # Verify specific test result details
    failure_details_found = False
    for msg in messages:
        msg_str = str(msg)
        if ("AssertionError: Expected add(2, 2) to equal 4, got 0" in msg_str and
            "ImportError: Cannot import module 'src.example'" in msg_str):
            failure_details_found = True
            break
    
    assert failure_details_found, "Test failure details not found in messages" 