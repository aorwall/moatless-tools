from datetime import datetime
from enum import Enum
from typing import Optional, Union, cast

from moatless.actions.find_class import FindClassArgs
from moatless.actions.find_function import FindFunctionArgs
from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.schema import ActionArguments
from moatless.actions.semantic_search import SemanticSearchArgs
from moatless.actions.think import ThinkArgs
from moatless.completion.stats import CompletionInvocation
from moatless.node import Node
from pydantic import BaseModel, Field


# Tree View Schema Classes
class ItemType(str, Enum):
    """Type of tree item."""

    COMPLETION = "completion"
    THOUGHT = "thought"
    ACTION = "action"
    NODE = "node"
    ERROR = "error"
    REWARD = "reward"


class BaseTreeItem(BaseModel):
    """Base class for tree items."""

    id: str
    node_id: int
    type: str
    label: str
    detail: Optional[str] = None
    time: Optional[str] = None


class CompletionTreeItem(BaseTreeItem):
    """Represents a completion item in the tree."""

    type: str = ItemType.COMPLETION
    tokens: Optional[int] = None
    action_step_id: Optional[int] = None


class ThoughtTreeItem(BaseTreeItem):
    """Represents a thought item in the tree."""

    type: str = ItemType.THOUGHT


class RewardTreeItem(BaseTreeItem):
    """Represents a reward item in the tree."""

    type: str = ItemType.REWARD


class ErrorTreeItem(BaseTreeItem):
    """Represents an error item in the tree."""

    type: str = ItemType.ERROR


class ActionTreeItem(BaseTreeItem):
    """Represents an action item in the tree."""

    type: str = ItemType.ACTION
    action_name: str
    action_index: int
    children: list[Union["CompletionTreeItem", "ThoughtTreeItem", "ActionTreeItem"]] = Field(default_factory=list)


class NodeTreeItem(BaseTreeItem):
    """Represents a node item in the tree."""

    type: str = ItemType.NODE
    # timestamp: Optional[str] = None
    parent_node_id: Optional[int] = None
    children: list[
        Union[
            "NodeTreeItem",
            "CompletionTreeItem",
            "ThoughtTreeItem",
            "ActionTreeItem",
            "ErrorTreeItem",
            "RewardTreeItem",
        ]
    ] = Field(default_factory=list)


class TreeItem(BaseModel):
    """Tree representation of trajectory data."""

    items: list[NodeTreeItem] = Field(default_factory=list)


def get_timestamp(node: Node) -> str:
    """Generate a timestamp string for a node."""
    # Use created_at if available, otherwise current time
    if node.timestamp:
        timestamp = node.timestamp

    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def get_completion_detail(completion: CompletionInvocation | None) -> str:
    """Generate detail string for a completion."""
    if not completion:
        return ""
    model = completion.model if hasattr(completion, "model") else "unknown"
    return f"({model})"


def get_execution_time(step) -> str:
    """Get execution time as string."""
    # Extract time from step if available
    execution_time = getattr(step, "execution_time", None)
    if execution_time:
        return f"{execution_time:.2f}s"
    return ""


def get_action_detail(action: ActionArguments) -> str:
    """Get detail string for an action."""
    if isinstance(action, ThinkArgs):
        return f'("{action.thought[:20]}...")' if len(action.thought) > 20 else f'("{action.thought}")'

    if isinstance(action, SemanticSearchArgs):
        return f'("{action.query[:40]}...")' if len(action.query) > 40 else f'("{action.query}")'

    if isinstance(action, RunTestsArgs):
        return f"({action.test_files})"

    if isinstance(action, FindClassArgs):
        return f"({action.class_name})"

    if isinstance(action, FindFunctionArgs):
        return f"({action.function_name})"

    # return first property that is set
    for key, value in action.model_dump().items():
        if value and key != "thought":
            return f"({key}={value[:40]}...)" if len(value) > 40 else f"({key}={value})"
    return ""


def create_node_tree_item(node: Node, parent_node_id: int | None = None) -> NodeTreeItem:
    # Create node item
    timestamp = get_timestamp(node)
    node_item = NodeTreeItem(
        id=f"node-{node.node_id}",
        node_id=node.node_id,
        label=f"Node {node.node_id}",
        time=timestamp,
        parent_node_id=parent_node_id,
        children=[],
    )

    # Add completion if available
    if node.completions and node.completions.get("build_action"):
        completion = node.completions["build_action"]

        if completion.usage:
            tokens = completion.usage.completion_tokens + completion.usage.prompt_tokens
        else:
            tokens = None

        completion_time = f"{completion.duration_sec:.2f}s"

        completion_item = CompletionTreeItem(
            id=f"{node_item.id}-completion",
            label="Completion",
            detail=get_completion_detail(completion),
            time=completion_time,
            tokens=tokens,
            node_id=node_item.node_id,
        )
        node_item.children.append(completion_item)

    # Add thought if available
    if node.thoughts and node.thoughts.text:
        thought_item = ThoughtTreeItem(
            id=f"{node_item.id}-thought",
            label="Thought",
            detail=f'("{node.thoughts.text[:20]}...")' if len(node.thoughts.text) > 20 else f'("{node.thoughts.text}")',
            node_id=node_item.node_id,
        )
        node_item.children.append(thought_item)

    # Add actions
    for i, step in enumerate(node.action_steps):
        if isinstance(step.action, ThinkArgs):
            # Already added thought from node.thoughts
            continue

        action_id = f"{node_item.id}-action-{i}"
        action_item = ActionTreeItem(
            id=action_id,
            label=step.action.name,
            detail=get_action_detail(step.action),
            # time=get_execution_time(step),
            action_name=step.action.name,
            action_index=i,
            node_id=node_item.node_id,
            children=[],
        )

        # Add action completion if available
        if step.observation and step.observation.execution_completion:
            completion_time = f"{step.observation.execution_completion.duration_sec:.2f}s"

            if step.observation.execution_completion.usage:
                tokens = (
                    step.observation.execution_completion.usage.completion_tokens
                    + step.observation.execution_completion.usage.prompt_tokens
                )
            else:
                tokens = None

            action_completion = CompletionTreeItem(
                id=f"{action_id}-completion",
                label="Completion",
                detail=get_completion_detail(step.observation.execution_completion),
                # time=completion_time,
                tokens=tokens,
                node_id=node_item.node_id,
                action_step_id=i,
            )
            action_item.children.append(action_completion)

        node_item.children.append(action_item)

    if node.reward and node.reward.value is not None:
        node_item.children.append(
            RewardTreeItem(
                id=f"{node_item.id}-reward",
                label="Reward",
                detail=f"{node.reward.value:.2f}",
                node_id=node_item.node_id,
            )
        )
    if node.error:
        node_item.children.append(
            ErrorTreeItem(
                id=f"{node_item.id}-error",
                label="Error",
                detail=node.error[:100],
                node_id=node_item.node_id,
            )
        )

    return node_item


def process_tree(node: Node, parent_node_id: int | None = None) -> NodeTreeItem:
    """Process nodes preserving tree structure"""

    node_item = create_node_tree_item(node, parent_node_id)

    for child_node in node.children:
        child_node_item = process_tree(child_node, node.node_id)
        node_item.children.append(child_node_item)

    return node_item


def has_branch_nodes(node: Node) -> bool:
    """Check if this node or any descendants have multiple children"""
    if len(node.children) > 1:
        return True
    return any(has_branch_nodes(child) for child in node.children)


def process_flat(node: Node) -> list[NodeTreeItem]:
    """Process nodes into completely flat list"""
    result: list[NodeTreeItem] = []

    parent_id = node.parent.node_id if node.parent else None
    current_node = node
    while current_node.children:
        current_node = current_node.children[0]
        node_item = create_node_tree_item(current_node, parent_id)
        result.append(node_item)

    return result


def create_node_tree(root_node: Node) -> NodeTreeItem:
    """Convert nodes from trajectory data to tree structure for visualization.

    Creates a hierarchical representation of nodes, completions, thoughts, and actions
    that can be used for tree-based visualizations in the UI. The tree is wrapped in a
    special "Root" node that serves as the container for all nodes.

    Returns:
        A NodeTreeItem with "Root" label containing the processed nodes as children
    """
    timestamp = get_timestamp(root_node)
    root_item = NodeTreeItem(
        id=f"node-{root_node.node_id}",
        node_id=root_node.node_id,
        label="Root",
        time=timestamp,
        parent_node_id=None,
        children=[],
    )

    if has_branch_nodes(root_node):
        # For branched trees, use process_tree
        processed_node = process_tree(root_node)
        root_item.children.append(processed_node)
    else:
        # For linear trees, use process_flat which creates a flat structure
        processed_nodes = process_flat(root_node)
        root_item.children.extend(processed_nodes)

    return root_item


def generate_ascii_tree(tree_item: NodeTreeItem) -> str:
    """Generate an ASCII representation of a TreeItem, NodeTreeItem, or list of NodeTreeItems.

    This function creates a human-readable ASCII tree representation
    showing the hierarchical structure of nodes, completions, thoughts, and actions.

    Args:
        tree_item: The TreeItem, NodeTreeItem, or list of NodeTreeItems to visualize

    Returns:
        A string containing the ASCII representation of the tree
    """
    tree_lines = ["NodeTree Visualization"]

    _append_ascii_node(tree_item, "", True, tree_lines)

    return "\n".join(tree_lines)


def _append_ascii_node(
    node_item: NodeTreeItem,
    prefix: str,
    is_last: bool,
    tree_lines: list[str],
) -> None:
    """Recursively append ASCII representation of node items to the given list of lines.

    This function only handles NodeTreeItem objects and displays non-node children
    (completions, thoughts, actions) inline for a more compact representation.

    Args:
        node_item: The node item to visualize
        prefix: Current line prefix (indentation and connecting lines)
        is_last: Whether this is the last item in its list of siblings
        tree_lines: List to append lines to
    """
    # Determine the connection character
    connection = "└── " if is_last else "├── "

    # Create node line
    node_line = f"{node_item.label} ({node_item.time})"

    # Group children by type
    child_nodes = []
    completions = []
    thoughts = []
    actions = []

    for child in node_item.children:
        if child.type == ItemType.NODE:
            child_nodes.append(child)
        elif child.type == ItemType.COMPLETION:
            completions.append(cast(CompletionTreeItem, child))
        elif child.type == ItemType.THOUGHT:
            thoughts.append(cast(ThoughtTreeItem, child))
        elif child.type == ItemType.ACTION:
            actions.append(cast(ActionTreeItem, child))

    # Add inline summary of important non-node children
    inline_info = []

    # Add completion info
    if completions:
        completion = completions[0]  # Just use the first one for inline display
        model = completion.detail.strip("()") if completion.detail else "unknown"
        tokens = completion.tokens or 0
        inline_info.append(f"{model}:{tokens}tk")

    # Add thought info if available
    if thoughts:
        thought_summary = thoughts[0].detail or ""
        if thought_summary:
            inline_info.append(f"thought:{thought_summary}")

    # Add action count
    if actions:
        inline_info.append(f"actions:{len(actions)}")

    # Add inline info to node line if available
    if inline_info:
        node_line = f"{node_line} [{' | '.join(inline_info)}]"

    # Add the node line to the output
    tree_lines.append(f"{prefix}{connection}{node_line}")

    # Calculate the child prefix - should align with the child's content
    child_prefix = prefix + ("    " if is_last else "│   ")

    # Process actions separately (they're important to show)
    for i, action in enumerate(actions):
        is_last_action = (i == len(actions) - 1) and not child_nodes
        action_connection = "└── " if is_last_action else "├── "

        time_info = f" ({action.time})" if action.time else ""
        action_line = f"{action.action_name} {action.detail or ''}{time_info}"

        # Add action line
        tree_lines.append(f"{child_prefix}{action_connection}{action_line}")

        # Handle action children (usually completions) if any
        if action.children:
            action_child_prefix = child_prefix + ("    " if is_last_action else "│   ")
            for j, action_child in enumerate(action.children):
                is_last_action_child = j == len(action.children) - 1
                action_child_connection = "└── " if is_last_action_child else "├── "

                if action_child.type == ItemType.COMPLETION:
                    completion = cast(CompletionTreeItem, action_child)
                    tokens_display = f" ({completion.tokens} tokens)" if completion.tokens else ""
                    action_child_line = f"Completion {completion.detail or ''}{tokens_display}"
                    tree_lines.append(f"{action_child_prefix}{action_child_connection}{action_child_line}")

    # Process node children
    for i, child_node in enumerate(child_nodes):
        _append_ascii_node(
            cast(NodeTreeItem, child_node),
            child_prefix,
            i == len(child_nodes) - 1,
            tree_lines,
        )


def print_ascii_tree(tree_item: NodeTreeItem) -> None:
    """Print an ASCII representation of a NodeTreeItem to the console.

    This is a convenience function that generates and prints the ASCII tree.

    Args:
        tree_item: The NodeTreeItem to visualize
    """
    print(generate_ascii_tree(tree_item))
