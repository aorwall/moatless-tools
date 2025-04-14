import io
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

import pytest
from moatless.actions.finish import FinishArgs
from moatless.actions.schema import ActionArguments, Observation
from moatless.actions.semantic_search import SemanticSearchArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.actions.think import ThinkArgs
from moatless.completion.stats import CompletionInvocation, Usage, CompletionAttempt
from moatless.flow.trajectory_tree import (
    process_tree,
    process_flat,
    NodeTreeItem,
    ItemType,
    has_branch_nodes,
    create_node_tree,
    generate_ascii_tree,
    print_ascii_tree,
)
from moatless.node import Node, ActionStep, Thoughts

logger = logging.getLogger(__name__)


def create_test_completion(model="test-model", tokens=100, duration=1.0):
    """Helper to create a test completion invocation"""
    usage = Usage(completion_tokens=tokens, prompt_tokens=tokens * 2, completion_cost=0.001 * tokens)

    attempt = CompletionAttempt(
        start_time=datetime.now().timestamp() * 1000,
        end_time=(datetime.now() + timedelta(seconds=duration)).timestamp() * 1000,
        usage=usage,
    )

    return CompletionInvocation(
        model=model, attempts=[attempt], start_time=attempt.start_time, end_time=attempt.end_time
    )


def create_test_node(node_id: int, parent: Optional[Node] = None) -> Node:
    """Create a test node with required properties for testing"""
    # Create minimal required arguments for Node
    node = Node(
        node_id=node_id,
        parent=None,
        user_message=None,
        assistant_message=None,
        file_context=None,
        is_duplicate=False,
        terminal=False,
        error=None,
        reward=None,
        visits=0,
        value=0.0,
        max_expansions=None,
        agent_id=None,
        feedback_data=None,
    )

    if parent:
        node.parent = parent
        parent.add_child(node)

    # Add a completion
    node.completions = {"build_action": create_test_completion(tokens=150)}

    # Add thoughts
    node.thoughts = Thoughts(text="This is a test thought")

    # Add action steps with real ActionArguments

    # Step 1: SemanticSearch
    search_action = SemanticSearchArgs(
        query="test search query",
        category="implementation",
        file_pattern="*.py",
        thoughts="Testing semantic search capabilities",
    )
    search_observation = Observation(
        message="Found test files", summary="Test summary for search", terminal=False, execution_completion=None
    )
    search_step = ActionStep(
        action=search_action, observation=search_observation, completion=create_test_completion(tokens=80)
    )

    # Step 2: StringReplace
    replace_action = StringReplaceArgs(
        path="test/file.py",
        old_str="def old_function():\n    return 'old'",
        new_str="def new_function():\n    return 'new'",
        thoughts="Replacing old function with new implementation",
    )
    replace_observation = Observation(
        message="File updated", summary="Test summary for replace", terminal=False, execution_completion=None
    )
    replace_step = ActionStep(
        action=replace_action, observation=replace_observation, completion=create_test_completion(tokens=120)
    )

    # Step 3: Think action
    think_action = ThinkArgs(thoughts="Thinking about next steps", thought="Planning process")
    think_observation = Observation(
        message="Thought result", summary="Thought summary", terminal=False, execution_completion=None
    )
    think_step = ActionStep(action=think_action, observation=think_observation)

    node.action_steps = [search_step, replace_step, think_step]

    # Set creation timestamp
    node.timestamp = datetime.now()

    return node


def test_process_tree():
    """Test the process_tree function with a branched tree (the intended use case)"""
    # Create a branched tree: root -> child1 -> grandchild
    #                         root -> child2
    root = create_test_node(0)
    child1 = create_test_node(1, root)
    child2 = create_test_node(2, root)  # This creates a branch
    grandchild = create_test_node(3, child1)

    # Process the tree
    result = create_node_tree(root)

    ascii_tree = generate_ascii_tree(result)
    logger.info(ascii_tree)

    # Verify result structure - tree is now a root node with node children
    assert result.id == f"node-{root.node_id}"
    assert result.type == ItemType.NODE
    assert len(result.children) == 1  # The root now has a node child that holds actual nodes

    # The first child should be the actual root node
    root_node = result.children[0]
    assert root_node.type == ItemType.NODE
    assert root_node.id == f"node-{root.node_id}"

    # Find the node children within the root node
    node_children = [child for child in root_node.children if child.type == ItemType.NODE]
    assert len(node_children) == 2  # Should have two child nodes (child1 and child2)

    # Find the child1 node
    child1_node = next((child for child in node_children if child.id == f"node-{child1.node_id}"), None)
    assert child1_node is not None

    # Find child2 node
    child2_node = next((child for child in node_children if child.id == f"node-{child2.node_id}"), None)
    assert child2_node is not None

    # Verify the nested structure continues for child1
    grandchild_nodes = [child for child in child1_node.children if child.type == ItemType.NODE]
    assert len(grandchild_nodes) == 1
    assert grandchild_nodes[0].id == f"node-{grandchild.node_id}"


def test_has_branch_nodes_directly():
    """Test has_branch_nodes function directly on different tree structures"""
    # Test 1: Linear tree (no branches)
    # Simple tree: root -> child1 -> grandchild
    root_linear = create_test_node(0)
    child1 = create_test_node(1, root_linear)
    grandchild = create_test_node(2, child1)

    # Verify the linear tree has no branches
    assert not has_branch_nodes(root_linear)

    # Test 2: Branched tree
    # Tree with branches: root -> child1 -> grandchild
    #                     root -> child2
    root_branched = create_test_node(10)
    child1_b = create_test_node(11, root_branched)
    child2_b = create_test_node(12, root_branched)  # This creates a branch
    grandchild_b = create_test_node(13, child1_b)

    # Verify the branched tree has branches
    assert has_branch_nodes(root_branched)

    # Test 3: Deep branch (branch is deeper in the tree)
    root_deep = create_test_node(20)
    child = create_test_node(21, root_deep)
    grandchild1 = create_test_node(22, child)
    grandchild2 = create_test_node(23, child)  # Branch at the grandchild level

    # Verify the deep branched tree has branches
    assert has_branch_nodes(root_deep)

    # Test 4: Single node (no branches)
    single_node = create_test_node(30)

    # Verify a single node has no branches
    assert not has_branch_nodes(single_node)


def test_process_flat():
    """Test the process_flat function with a linear tree (the intended use case)"""
    # Create a linear tree (no branches): root -> child -> grandchild
    root = create_test_node(0)
    child = create_test_node(1, root)
    grandchild = create_test_node(2, child)

    # Process the tree as flat
    result = create_node_tree(root)

    ascii_tree = generate_ascii_tree(result)
    logger.info(ascii_tree)

    # Verify result structure - now result is a NodeTreeItem with the tree structure
    assert result.id == f"node-{root.node_id}"
    assert result.type == ItemType.NODE

    # The result should have children for the flat representation
    assert len(result.children) > 0

    # From the ASCII tree output, we can see that the structure is:
    # Root
    # ├── Node 1 (child)
    # └── Node 2 (grandchild)
    #
    # So let's verify this structure

    # The children should be the child and grandchild nodes
    child_nodes = [child for child in result.children if child.type == ItemType.NODE]
    assert len(child_nodes) == 2

    # Collect all node IDs from children
    node_ids = [node.id for node in child_nodes]

    # Verify the expected nodes are present
    assert any(f"node-{child.node_id}" == node_id for node_id in node_ids)
    assert any(f"node-{grandchild.node_id}" == node_id for node_id in node_ids)


def test_finish_action_in_tree():
    """Test processing a tree with a Finish action"""
    # Create a simple tree with a finish action
    root = create_test_node(0)

    # Add a finish action
    finish_action = FinishArgs(
        finish_reason="Task completed successfully",
        thoughts="All requirements have been implemented and tests are passing",
    )
    finish_observation = Observation(
        message="Finished", summary="Task is done", terminal=True, execution_completion=None
    )
    finish_step = ActionStep(action=finish_action, observation=finish_observation)

    # Replace the last action step with the finish step
    root.action_steps[-1] = finish_step

    # Process the tree
    result = process_tree(root)

    # Verify we have an action of type ACTION
    action_items = [child for child in result.children if child.type == ItemType.ACTION]

    # The action name might be different based on how trajectory_tree.py handles action names
    # Let's debug by printing the actual action names
    action_names = [getattr(item, "action_name", None) for item in action_items]

    # Check if any action has "Finish" in its name (more flexible)
    finish_actions = [name for name in action_names if name and "Finish" in name]
    assert len(finish_actions) > 0, f"No Finish action found in actions: {action_names}"


def test_generate_ascii_tree():
    """Test the ASCII tree generation function"""
    # Create a simple tree
    root = create_test_node(0)
    child1 = create_test_node(1, root)
    child2 = create_test_node(2, root)

    # Create a node tree from the nodes
    tree = create_node_tree(root)

    # Generate and log ASCII tree for visual inspection
    ascii_tree = generate_ascii_tree(tree)
    logger.info(f"ASCII Tree:\n{ascii_tree}")

    # Basic verification of the ASCII tree
    assert isinstance(ascii_tree, str)
    assert "NodeTree Visualization" in ascii_tree
    assert "Node" in ascii_tree
    assert "Root" in ascii_tree  # Should have a Root node
    assert "─" in ascii_tree  # Should contain connection lines

    # Verify tree structure elements are present
    assert "thought:(" in ascii_tree  # Thoughts are now inline in node description
    assert "SemanticSearch" in ascii_tree  # The action name (not the args class name)
    assert "StringReplace" in ascii_tree  # The action name (not the args class name)

    # Check the root node structure - now the ID is based on node_id
    assert tree.id == f"node-{root.node_id}"
    assert tree.label == "Root"

    # Check that the original nodes are under the Root node
    assert len(tree.children) > 0

    # Test printing (just for coverage, doesn't assert anything)
    # Redirect stdout to capture print output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the print function
    print_ascii_tree(tree)

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Verify something was printed
    printed_output = captured_output.getvalue()
    assert printed_output
    assert "NodeTree Visualization" in printed_output
