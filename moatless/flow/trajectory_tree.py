from enum import Enum
import logging
from typing import Optional, Union, cast

from pydantic import BaseModel, Field

from moatless.actions.find_class import FindClassArgs
from moatless.actions.find_function import FindFunctionArgs
from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.schema import ActionArguments
from moatless.actions.semantic_search import SemanticSearchArgs
from moatless.actions.think import ThinkArgs
from moatless.completion.stats import CompletionInvocation
from moatless.node import ActionStep, Node, Selection

logger = logging.getLogger(__name__)


# Tree View Schema Classes
class ItemType(str, Enum):
    """Type of tree item."""

    COMPLETION = "completion"
    THOUGHT = "thought"
    ACTION = "action"
    NODE = "node"
    ERROR = "error"
    REWARD = "reward"
    EVALUATION = "evaluation"
    SELECTION = "selection"


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
    item_id: Optional[str] = None


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


class EvaluationTreeItem(BaseTreeItem):
    """Represents an evaluation item in the tree."""

    type: str = ItemType.EVALUATION
    resolved: bool


class SelectionTreeItem(BaseTreeItem):
    """Represents a selection item in the tree."""

    type: str = ItemType.SELECTION


class NodeTreeItem(BaseTreeItem):
    """Represents a node item in the tree."""

    type: str = ItemType.NODE
    is_duplicate: bool = False
    parent_node_id: Optional[int] = None
    tokens: Optional[int] = None
    reward: Optional[float] = None

    children: list[
        Union[
            "NodeTreeItem",
            "CompletionTreeItem",
            "ThoughtTreeItem",
            "ActionTreeItem",
            "ErrorTreeItem",
            "RewardTreeItem",
            "EvaluationTreeItem",
            "SelectionTreeItem",
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


def get_execution_time(step: ActionStep) -> str:
    """Get execution time as string."""
    # Extract time from step if available
    if step.start_time and step.end_time:
        execution_time = step.end_time - step.start_time
        return f"{execution_time.total_seconds():.2f}s"
    return ""


def get_action_detail(action: ActionArguments) -> str:
    """Get detail string for an action."""
    if isinstance(action, ThinkArgs):
        return f'("{action.thought[:20]}...")' if len(action.thought) > 20 else f'("{action.thought}")'

    if isinstance(action, SemanticSearchArgs):
        return f'("{action.query[:100]}...")' if len(action.query) > 100 else f'("{action.query}")'

    if isinstance(action, RunTestsArgs):
        return f"({action.test_files})"

    if isinstance(action, FindClassArgs):
        return f"({action.class_name})"

    if isinstance(action, FindFunctionArgs):
        return f"({action.function_name})"

    # return first property that is set
    for key, value in action.model_dump().items():
        if value and isinstance(value, str):
            return f"({key}={value[:100]}...)" if len(value) > 100 else f"({key}={value})"
        elif value:
            return f"({key}={value})"
    return ""


def create_completion_tree_item(
    completion: CompletionInvocation,
    parent_id: str,
    node_id: int,
    action_step_id: Optional[int] = None,
    item_id: Optional[str] = None,
) -> CompletionTreeItem:
    """Create a CompletionTreeItem from a completion invocation.

    Args:
        completion: The completion invocation to create the tree item from
        parent_id: The ID of the parent item (node or action)
        node_id: The ID of the node this completion belongs to
        action_step_id: Optional action step ID if this completion is for an action

    Returns:
        A CompletionTreeItem representing the completion
    """
    completion_time = f"{completion.duration_sec:.2f}s"
    tokens = None
    if completion.usage:
        tokens = completion.usage.completion_tokens + completion.usage.prompt_tokens

    return CompletionTreeItem(
        id=f"{parent_id}-completion",
        label="Completion",
        detail=get_completion_detail(completion),
        time=completion_time,
        tokens=tokens,
        node_id=node_id,
        action_step_id=action_step_id,
        item_id=item_id,
    )


def create_node_tree_item(node: Node, parent_node_id: int | None = None) -> NodeTreeItem:
    # Create node item
    timestamp = get_timestamp(node)

    # Generate detail text for node
    detail_parts = []

    # Add action names if available
    if node.action_steps:
        action_names = [step.action.name for step in node.action_steps if hasattr(step.action, "name")]
        if action_names:
            detail_parts.append(f"Actions: {', '.join(action_names)}")

    # Add reward if available
    if node.reward and node.reward.value is not None:
        detail_parts.append(f"Reward: {node.reward.value:.2f}")
        reward = node.reward.value
    else:
        reward = None

    # Add error if available
    if node.error:
        error_text = node.error[:50] + "..." if len(node.error) > 50 else node.error
        detail_parts.append(f"Error: {error_text}")

    # Add evaluation result if available
    if node.evaluation_result:
        status = "Resolved" if node.evaluation_result.resolved else "Failed"
        detail_parts.append(f"Evaluation: {status}")

    # Join all parts with separator
    detail = " | ".join(detail_parts) if detail_parts else None

    usage = node.usage()
    if usage and usage.prompt_tokens > 0:
        tokens = usage.completion_tokens + usage.prompt_tokens
    else:
        tokens = None

    node_item = NodeTreeItem(
        id=f"node-{node.node_id}",
        node_id=node.node_id,
        label=f"Node {node.node_id}",
        time=timestamp,
        is_duplicate=node.is_duplicate or False,
        parent_node_id=parent_node_id,
        reward=reward,
        detail=detail,
        children=[],
        tokens=tokens,
    )

    # Add completion if available
    if node.completions and node.completions.get("build_action"):
        completion = node.completions["build_action"]
        completion_item = create_completion_tree_item(
            completion=completion, parent_id=node_item.id, node_id=node_item.node_id, item_id="agent"
        )
        node_item.children.append(completion_item)

    # Add thought if available
    if node.thoughts and node.thoughts.text:
        thought_item = ThoughtTreeItem(
            id=f"{node_item.id}-thought",
            label="Thought",
            detail=f'("{node.thoughts.text[:200]}...")'
            if len(node.thoughts.text) > 200
            else f'("{node.thoughts.text}")',
            node_id=node_item.node_id,
        )
        node_item.children.append(thought_item)

    # Add actions
    for i, step in enumerate(node.action_steps):
        if isinstance(step.action, ThinkArgs):
            thought_item = ThoughtTreeItem(
                id=f"{node_item.id}-thought",
                label="Thought",
                detail=f'("{step.action.thought[:200]}...")'
                if len(step.action.thought) > 200
                else f'("{step.action.thought}")',
                node_id=node_item.node_id,
            )
            node_item.children.append(thought_item)
            continue

        action_id = f"{node_item.id}-action-{i}"
        action_item = ActionTreeItem(
            id=action_id,
            label=step.action.name,
            detail=get_action_detail(step.action),
            time=get_execution_time(step),
            action_name=step.action.name,
            action_index=i,
            node_id=node_item.node_id,
            children=[],
        )

        # Add action completion if available
        if step.completion and step.completion.usage:
            action_completion = create_completion_tree_item(
                completion=step.completion, parent_id=action_id, node_id=node_item.node_id, item_id=f"action_{i}"
            )
            action_item.children.append(action_completion)

        node_item.children.append(action_item)

    if node.reward:
        if node.reward.completion:
            reward_completion = create_completion_tree_item(
                completion=node.reward.completion,
                parent_id=node_item.id,
                node_id=node_item.node_id,
                item_id="value_function",
            )
            node_item.children.append(reward_completion)

        if node.reward.value is not None:
            node_item.children.append(
                RewardTreeItem(
                    id=f"{node_item.id}-reward",
                    label="Reward",
                    detail=f"{node.reward.value:.2f}",
                    node_id=node_item.node_id,
                )
            )

    # Add selection info if available
    if node.selection:
        reason = (
            f"Reason: {node.selection.reason[:100]}..."
            if len(node.selection.reason) > 100
            else f"Reason: {node.selection.reason}"
        )

        node_item.children.append(
            SelectionTreeItem(
                id=f"{node_item.id}-selection",
                label="Selection",
                detail=f"Node {node.selection.node_id} ({reason})",
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

    if node.evaluation_result:
        if node.evaluation_result.start_time and node.evaluation_result.end_time:
            time_diff = node.evaluation_result.end_time - node.evaluation_result.start_time
            time = f"{time_diff.total_seconds():.2f}s"
        else:
            time = None

        node_item.children.append(
            EvaluationTreeItem(
                id=f"{node_item.id}-evaluation",
                node_id=node_item.node_id,
                label="Evaluation",
                detail="Resolved" if node.evaluation_result.resolved else "Failed",
                time=time,
                resolved=node.evaluation_result.resolved,
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
