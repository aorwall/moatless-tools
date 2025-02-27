import json
import logging
from typing import Literal, Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field

from moatless.actions.schema import ActionArguments, Observation
from moatless.agent.settings import AgentSettings
from moatless.artifacts.artifact import ArtifactChange
from moatless.completion.model import (
    Usage,
    Completion,
)
from moatless.file_context import FileContext
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment

logger = logging.getLogger(__name__)


class ActionStep(BaseModel):
    action: ActionArguments
    observation: Optional[Observation] = None
    completion: Optional[Completion] = None

    def is_executed(self) -> bool:
        """Check if this action step has been executed by verifying if it has observations."""
        return self.observation is not None

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)

        data["action"] = self.action.model_dump(**kwargs)
        data["action"]["action_args_class"] = f"{self.action.__class__.__module__}.{self.action.__class__.__name__}"

        return data

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> "ActionArguments":
        if isinstance(obj, dict):
            obj = obj.copy()
            obj["action"] = ActionArguments.model_validate(obj["action"])
        return super().model_validate(obj, **kwargs)


class FeedbackData(BaseModel):
    """Structured feedback data model"""

    feedback: str = Field(..., description="Direct feedback to the AI assistant")
    analysis: Optional[str] = Field(None, description="Analysis of the task and alternative branch attempts")
    suggested_node_id: Optional[int] = Field(None, description="ID of the node that should be expanded next (optional)")


class Reward(BaseModel):
    """Reward from value function."""

    explanation: Optional[str] = Field(
        default=None,
        description="An explanation and the reasoning behind the decision.",
    )
    value: int = Field(
        ...,
        description="Reward value.",
        ge=-100,
        le=100,
    )

class ThoughtBlock(BaseModel):
    type: Literal["thinking", "redacted_thinking"] = Field(..., description="The type of the thought block")
    text: str = Field(..., description="The text of the thought block")
    signature: Optional[str] = Field(None, description="The signature of the thought block")
    data: Optional[str] = Field(None, description="The data of the thought block")

class Node(BaseModel):
    node_id: int = Field(..., description="The unique identifier of the node")

    parent: Optional["Node"] = Field(None, description="The parent node")
    children: List["Node"] = Field(default_factory=list, description="The child nodes")

    artifact_changes: List[ArtifactChange] = Field(
        default_factory=list,
        description="The artifact changes associated with the node",
    )

    user_message: Optional[str] = Field(None, description="The user message for this node")
    assistant_message: Optional[str] = Field(None, description="The assistant response for this node")

    thoughts: Optional[List[dict]] = Field(default=None, description="The thoughts associated with the node")

    action_steps: List[ActionStep] = Field(
        default_factory=list,
        description="The sequence of actions and observations for this node",
    )

    file_context: Optional[FileContext] = Field(None, description="The file context state associated with the node")
    # feedback: Optional[str] = Field(None, description="Feedback provided to the node")
    completions: Dict[str, Completion] = Field(default_factory=dict, description="The completions used in this node")
    possible_actions: List[str] = Field(default_factory=list, description="List of possible action types for this node")
    is_duplicate: Optional[bool] = Field(None, description="Flag to indicate if the node is a duplicate")
    terminal: bool = Field(False, description="Flag to indicate if the node is a terminal node")
    error: Optional[str] = Field(None, description="Error when running node")
    reward: Optional[Reward] = Field(None, description="The reward of the node")
    visits: int = Field(0, description="The number of times the node has been visited")
    value: Optional[float] = Field(None, description="The total value (reward) of the node")
    max_expansions: Optional[int] = Field(None, description="The maximum number of expansions")
    agent_settings: Optional[AgentSettings] = Field(None, description="The agent settings associated with the node")
    feedback_data: Optional[FeedbackData] = Field(None, description="Structured feedback data for the node")

    @property
    def action(self) -> Optional[ActionArguments]:
        """Backward compatibility: Get action from the latest action step"""
        if not self.action_steps:
            return None
        return self.action_steps[-1].action if self.action_steps else None

    @action.setter
    def action(self, value: Optional[ActionArguments]):
        """Backward compatibility: Set action on the current/new action step"""

        if not self.action_steps:
            self.action_steps = [ActionStep(action=value)]
        else:
            self.action_steps[-1].action = value

    @property
    def observation(self) -> Optional[Observation]:
        """Backward compatibility: Get observation from the latest action step"""
        if not self.action_steps:
            return None
        return self.action_steps[-1].observation if self.action_steps else None

    @observation.setter
    def observation(self, value: Optional[Observation]):
        """Backward compatibility: Set observation on the current/new action step"""
        if value is None:
            return

        if not self.action_steps:
            # Create new action step if setting observation on empty node
            self.action_steps.append(ActionStep(action=self.action, observation=value))
        else:
            self.action_steps[-1].observation = value

    @property
    def message(self) -> Optional[str]:
        """Backward compatibility: Get message maps to user_message"""
        return self.user_message

    @message.setter
    def message(self, value: Optional[str]):
        """Backward compatibility: Set message maps to user_message"""
        self.user_message = value

    @classmethod
    def stub(cls, **kwargs):
        """Create a stub node with a unique ID."""
        # Get the highest existing node ID from the kwargs or use 0
        existing_nodes = kwargs.get("children", [])
        highest_id = max([n.node_id for n in existing_nodes] + [kwargs.get("node_id", -1), -1]) + 1
        return cls(node_id=highest_id, **kwargs)

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node (no children)."""
        return len(self.children) == 0

    def expanded_count(self) -> int:
        """Get the number of expanded children."""
        return len([child for child in self.children])

    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried and executed from this node."""
        return self.expanded_count() >= (self.max_expansions or 1)

    def is_terminal(self) -> bool:
        """Determine if the current state is a terminal state."""

        return self.terminal

    def is_finished(self) -> bool:
        """Determine if the node is succesfully finished"""
        if self.action and self.action.name == "Finish":
            return True

        return False

    def is_executed(self) -> bool:
        """Determine if the node is executed"""
        if not self.parent:
            # Consider root node as executed
            return True

        return bool(self.action_steps and self.action_steps[-1].is_executed())

    def add_child(self, child_node: "Node"):
        """Add a child node to this node."""
        child_node.parent = self
        self.children.append(child_node)

    def set_parent(self, parent: "Node"):
        if self.node_id == parent.node_id:
            raise ValueError(f"Node can't have same id {self.node_id} parent")
        self.parent = parent
        parent.add_child(self)

    def get_depth(self) -> int:
        depth = 0
        node = self
        while node.parent:
            depth += 1
            node = node.parent
        return depth

    def is_expandable(self) -> bool:
        """Check if the node can be expanded further."""
        return not self.is_terminal() and not self.is_fully_expanded() and not self.is_duplicate

    def find_duplicate(self) -> Optional["Node"]:
        if not self.parent:
            return None

        for child in self.parent.children:
            if child.node_id != self.node_id and child.equals(self):
                return child

        return None

    def get_sibling_nodes(self) -> List["Node"]:
        if not self.parent:
            return []

        return [child for child in self.parent.children if child.node_id != self.node_id]

    def get_trajectory(self) -> List["Node"]:
        nodes = []
        current_node = self
        while current_node is not None:
            nodes.insert(0, current_node)
            current_node = current_node.parent

        return nodes

    def get_expandable_descendants(self) -> List["Node"]:
        """Get all expandable descendants of this node, including self if expandable."""
        expandable_nodes = []
        if self.is_expandable():
            expandable_nodes.append(self)
        for child in self.children:
            expandable_nodes.extend(child.get_expandable_descendants())
        return expandable_nodes

    def get_expanded_descendants(self) -> List["Node"]:
        """Get all expanded descendants of this node, including self if expanded."""
        expanded_nodes = []
        if self.expanded_count() > 0:
            expanded_nodes.append(self)
        for child in self.children:
            expanded_nodes.extend(child.get_expanded_descendants())
        return expanded_nodes

    def get_all_nodes(self) -> List["Node"]:
        if self.parent:
            node = self.get_root()
        else:
            node = self

        return node._get_all_nodes()
    
    def get_node_by_id(self, node_id: int) -> Optional["Node"]:
        for node in self.get_all_nodes():
            if node.node_id == node_id:
                return node
        return None

    def get_leaf_nodes(self) -> List["Node"]:
        """Get all leaf nodes ."""
        return [node for node in self.get_root().get_all_nodes() if node.is_leaf()]

    def _get_all_nodes(self) -> List["Node"]:
        nodes = []
        nodes.append(self)
        for child in self.children:
            nodes.extend(child._get_all_nodes())
        return nodes

    def get_root(self) -> "Node":
        node = self
        while node.parent:
            node = node.parent
        return node

    def calculate_mean_reward(self) -> float:
        """
        Calculate the mean trajectory reward for this node.

        Returns:
            float: The mean reward.
        """
        rewards = []
        node = self
        while node is not None:
            rewards.append(node.value / node.visits if node.visits > 0 else 0)
            node = node.parent

        return sum(rewards) / len(rewards) if rewards else 0

    def total_usage(self) -> Usage:
        """Calculate total token usage all nodes."""
        total_usage = Usage()
        for node in self.get_all_nodes():
            total_usage += node.usage()
        return total_usage

    def usage(self) -> Usage:
        """Calculate total token usage for this node."""
        usage = Usage()

        # Sum usage across all action steps
        for step in self.action_steps:
            if step.completion:
                usage += step.completion.usage

        for completion in self.completions.values():
            if completion:
                usage += completion.usage

        return usage

    def equals(self, other: "Node"):
        if self.action and not other.action:
            return False

        if not self.action and other.action:
            return False

        if self.action.name != other.action.name:
            return False

        return self.action.equals(other.action)

    def reset(self):
        """Reset the node state to be able to execute it again."""

        self.action_steps = []
        self.assistant_message = None
        self.visits = 0
        self.value = 0.0
        self.terminal = False
        self.is_duplicate = False
        self.error = None
        if self.parent and self.parent.file_context:
            self.file_context = self.parent.file_context.clone()
        self.children = []

    def clone_and_reset(self) -> "Node":
        """
        Creates a copy of the node and resets its observation and file context.

        Returns:
            Node: A new node instance with reset state
        """
        # Find highest node ID in the tree to ensure uniqueness
        root = self.get_root()
        all_nodes = root.get_all_nodes()
        highest_id = max(node.node_id for node in all_nodes) + 1

        # Create a new node with same base attributes but new ID
        new_node = Node(
            node_id=highest_id,  # Use new unique ID
            parent=self.parent,
            visits=self.visits,
            value=self.value,
            max_expansions=self.max_expansions,
            user_message=self.user_message,
            is_duplicate=self.is_duplicate,
            action=self.action,
            possible_actions=self.possible_actions.copy() if self.possible_actions else [],
        )

        new_node.reset()
        return new_node

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the node and its descendants.

        Returns:
            Dict[str, Any]: A dictionary representation of the node tree.
        """

        exclude_set = {"parent", "children"}
        if "exclude" in kwargs:
            if isinstance(kwargs["exclude"], set):
                exclude_set.update(kwargs["exclude"])
            elif isinstance(kwargs["exclude"], dict):
                exclude_set.update(kwargs["exclude"].keys())

        new_kwargs = {k: v for k, v in kwargs.items() if k != "exclude"}
        node_dict = super().model_dump(exclude=exclude_set, **new_kwargs)

        if self.completions and "completions" not in exclude_set:
            node_dict["completions"] = {
                key: completion.model_dump(**kwargs) for key, completion in self.completions.items() if completion
            }

        if self.reward and "reward" not in exclude_set:
            node_dict["reward"] = self.reward.model_dump(**kwargs)

        if self.observation and "output" not in exclude_set:
            node_dict["output"] = self.observation.model_dump(**kwargs)

        if self.file_context and "file_context" not in exclude_set:
            node_dict["file_context"] = self.file_context.model_dump(**kwargs)

        node_dict["action_steps"] = [action_step.model_dump(**kwargs) for action_step in self.action_steps]

        if not kwargs.get("exclude") or "children" not in kwargs.get("exclude"):
            node_dict["children"] = [child.model_dump(**kwargs) for child in self.children]

        return node_dict

    @classmethod
    def _reconstruct_node(
        cls,
        node_data: Dict[str, Any],
        repo: Repository | None = None,
        runtime: RuntimeEnvironment | None = None,
    ) -> "Node":
        """Update reconstruction to handle both old and new formats"""

        # Handle legacy format conversion
        if "action" in node_data and not "action_steps" in node_data:
            action = node_data.get("action")
            observation = node_data.get("output")
            completions = node_data.get("completions", {})

            if action or observation or completions:
                node_data["action_steps"] = [
                    {
                        "action": action,
                        "observation": observation,
                        "completions": completions,
                    }
                ]

        if not "user_message" in node_data and node_data.get("message"):
            node_data["user_message"] = node_data.pop("message")

        if node_data.get("action_steps"):
            node_data["action_steps"] = [
                ActionStep.model_validate(step_data) for step_data in node_data["action_steps"]
            ]

            # To keep backward compatiblity
            for step in node_data["action_steps"]:
                if step.observation and step.observation.terminal:
                    node_data["terminal"] = True

        if not "terminal" in node_data:
            node_data["terminal"] = False

        if node_data.get("file_context"):
            node_data["file_context"] = FileContext.from_dict(
                repo=repo, runtime=runtime, data=node_data["file_context"]
            )

        node_data["visits"] = node_data.get("visits", 0)
        node_data["value"] = node_data.get("value", 0.0)

        if node_data.get("feedback_data"):
            node_data["feedback_data"] = FeedbackData.model_validate(node_data["feedback_data"])

        if "children" in node_data:
            children = node_data.get("children", [])

            del node_data["children"]
            node = super().model_validate(node_data)

            for child_data in children:
                child = cls._reconstruct_node(child_data, repo=repo, runtime=runtime)
                child.parent = node
                node.children.append(child)

            return node
        else:
            return cls.model_validate(node_data)

    @classmethod
    def from_file(cls, file_path: str, repo: Repository | None = None, runtime: RuntimeEnvironment | None = None, **kwargs) -> "Node":
        with open(file_path, "r") as f:
            data = json.load(f)

        if "root" in data:
            return Node.reconstruct(data["root"], repo=repo, runtime=runtime)
        elif "nodes" in data:
            return Node.reconstruct(data["nodes"], repo=repo, runtime=runtime)
        else:
            raise ValueError("No root or nodes found in data")

    @classmethod
    def reconstruct(
        cls,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        repo: Repository | None = None,
        runtime: RuntimeEnvironment | None = None,
    ) -> "Node":
        """
        Reconstruct a node tree from either dict (tree) or list format.

        Args:
            data: Either a dict (tree format) or list of dicts (list format)
            parent: Optional parent node (used internally)
            repo: Optional repository reference

        Returns:
            Node: Root node of reconstructed tree
        """
        # Handle list format
        if isinstance(data, list):
            return cls._reconstruct_from_list(data, repo=repo, runtime=runtime)

        # Handle single node reconstruction (dict format)
        return cls._reconstruct_node(data, repo=repo, runtime=runtime)

    @classmethod
    def _reconstruct_from_list(
        cls,
        node_list: List[Dict],
        repo: Repository | None = None,
        runtime: RuntimeEnvironment | None = None,
    ) -> "Node":
        """
        Reconstruct tree from a flat list of nodes.

        Args:
            node_list: List of serialized nodes
            repo: Optional repository reference

        Returns:
            Node: Root node of reconstructed tree
        """
        # Create nodes without relationships first
        nodes_by_id = {}

        for node_data in node_list:
            parent_id = node_data.pop("parent_id", None)
            node = cls._reconstruct_node(node_data, repo=repo, runtime=runtime)
            nodes_by_id[node.node_id] = (node, parent_id)

        # Connect parent-child relationships
        for node, parent_id in nodes_by_id.values():
            if parent_id is not None:
                parent_node = nodes_by_id[parent_id][0]
                parent_node.add_child(node)

        root_nodes = [n for n, p_id in nodes_by_id.values() if p_id is None]
        if not root_nodes:
            raise ValueError("No root node found in data")

        if logger.isEnabledFor(logging.DEBUG):  
            tree = generate_ascii_tree(root_nodes[0])
            logger.debug(f"Reconstructed tree:\n{tree}")
        return root_nodes[0]

    def dump_as_list(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Dump all nodes as a flat list structure.
        """
        nodes = self.get_all_nodes()
        node_list = []

        for node in nodes:
            node_data = node.model_dump(exclude={"parent", "children"}, **kwargs)
            node_data["parent_id"] = node.parent.node_id if node.parent is not None else None
            node_list.append(node_data)

        return node_list

    @classmethod
    def load_from_file(cls, file_path: str, repo: Repository | None = None) -> "Node":
        """
        Load node tree from file, supporting both old tree format and new list format.

        Args:
            file_path (str): Path to the saved node data
            repo (Repository): Optional repository reference

        Returns:
            Node: Root node of the tree
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return cls.reconstruct_from_list(data, repo=repo)
        else:
            # Old tree format
            return cls.reconstruct(data, repo=repo)

    def persist(self, file_path: str, format: str = "list"):
        """
        Persist the node tree to file.

        Args:
            file_path (str): The path to save to
            format (str): Either "list" (new) or "tree" (legacy)
        """
        if format == "list":
            self.persist_as_list(file_path)
        elif format == "tree":
            self.persist_tree(file_path)
        else:
            raise ValueError("Format must be either 'list' or 'tree'")

    def truncate_children_by_id(self, max_id: int):
        """Truncate children to only include nodes with IDs less than or equal to the specified value.

        Args:
            max_id (int): Maximum node ID to keep (inclusive)
        """
        self.children = [child for child in self.children if child.node_id <= max_id]
        # Recursively truncate remaining children
        for child in self.children:
            child.truncate_children_by_id(max_id)

    def has_unexecuted_actions(self) -> bool:
        """Check if any action step in this node has not been executed."""
        return any(not step.is_executed() for step in self.action_steps)


def generate_ascii_tree(
    root: Node,
    current: Optional[Node] = None,
    include_explanation: bool = False,
    include_diffs: bool = False,
    include_feedback: bool = False,
    include_action_details: bool = False,
    use_color: bool = True,
    show_trajectory: bool = False,
) -> str:
    """Create an ASCII representation of the tree."""
    tree_lines = ["MCTS Tree"]
    # Make sure we're starting from the actual root node
    if root.parent:
        root = root.get_root()

    _append_ascii_node(
        root,
        "",
        True,
        tree_lines,
        current,
        include_explanation,
        include_diffs,
        include_feedback,
        include_action_details,
        use_color,
        show_trajectory,
    )
    return "\n".join(tree_lines)


def _append_ascii_node(
    node: Node,
    prefix: str,
    is_last: bool,
    tree_lines: list[str],
    current: Node | None,
    include_explanation: bool = False,
    include_diffs: bool = False,
    include_feedback: bool = False,
    include_action_details: bool = False,
    use_color: bool = True,
    show_trajectory: bool = False,
) -> None:
    # Get current trajectory nodes if we have a current node and trajectory marking is enabled
    current_trajectory_nodes = []
    if current and show_trajectory:
        current_trajectory_nodes = current.get_trajectory()

    # Build node information
    state_params = []
    if node.action_steps:
        # Include all action names from action steps
        action_names = [step.action.name for step in node.action_steps if step.action]
        state_params.extend(action_names)
        # Check if any action step expects correction
        if any(step.observation and step.observation.expect_correction for step in node.action_steps):
            state_params.append("expect_correction")

    # Build reward string
    if not node.reward:
        reward_str = "0"
        node_str = f"Node{node.node_id}"
    else:
        if use_color:
            if node.reward.value >= 75:
                reward_str = color_green(node.reward.value)
                node_str = color_green(f"Node{node.node_id}")
            elif node.reward.value <= 0:
                reward_str = color_red(node.reward.value)
                node_str = color_red(f"Node{node.node_id}")
            else:
                reward_str = color_yellow(node.reward.value)
                node_str = color_yellow(f"Node{node.node_id}")
        else:
            reward_str = str(node.reward.value)
            node_str = f"Node{node.node_id}"

    # Build state info without repeating node ID
    state_info = ""
    if state_params:
        state_info = f"({', '.join(state_params)})"
    else:
        state_info = "()"

    if use_color and current and node.node_id == current.node_id:
        state_info = color_white(state_info)

    # Add expandable status
    expandable_str = "expandable" if node.is_expandable() else "not-expandable"
    if use_color:
        expandable_str = color_green(expandable_str) if node.is_expandable() else color_red(expandable_str)

    # Calculate the current node's connection prefix
    connection = "└── " if is_last else "├── "

    # Add star marker only if trajectory marking is enabled
    trajectory_marker = "* " if (show_trajectory and node in current_trajectory_nodes) else "  "

    # Add the node line with expandable status and optional trajectory marker
    tree_lines.append(
        f"{prefix}{connection}{trajectory_marker}{node_str} {state_info} "
        f"(expansions: {node.expanded_count()}, reward: {reward_str}, "
        f"visits: {node.visits}, {expandable_str})"
    )

    # Calculate the content prefix - should align with the node's content
    content_prefix = prefix + ("    " if is_last else "│   ")

    # Add explanation if available
    if include_explanation and node.reward and node.reward.explanation:
        explanation_text = node.reward.explanation.strip()
        _append_wrapped_text(tree_lines, explanation_text, content_prefix, "│ Explanation: ")

    # Add feedback if available
    if include_feedback and node.feedback_data:
        tree_lines.append(f"{content_prefix}│ Feedback:")
        _append_wrapped_text(
            tree_lines,
            node.feedback_data.feedback,
            content_prefix,
            "│ Direct Feedback: ",
        )
        _append_wrapped_text(tree_lines, node.feedback_data.analysis, content_prefix, "│ Analysis: ")

    # Add diffs if available - only for Finish actions
    if include_diffs and node.file_context and node.action and node.action.name == "Finish":
        patch = node.file_context.generate_git_patch()
        if patch.strip():
            tree_lines.append(f"{content_prefix}│ Changes (git patch):")
            for line in patch.split("\n"):
                if line.strip():
                    prefix_char = "+" if line.startswith("+") else ("-" if line.startswith("-") else " ")
                    formatted_line = line.strip()
                    if use_color:
                        if prefix_char == "+":
                            formatted_line = color_green(formatted_line)
                        elif prefix_char == "-":
                            formatted_line = color_red(formatted_line)
                    tree_lines.append(f"{content_prefix}│  {formatted_line}")

    # Add action details if available
    if include_action_details and node.action_steps:
        tree_lines.append(f"{content_prefix}│ Action Steps:")
        for i, step in enumerate(node.action_steps, 1):
            tree_lines.append(f"{content_prefix}│ Step {i}:")
            tree_lines.append(f"{content_prefix}│  Action: {step.action.name}")
            tree_lines.append(f"{content_prefix}│  Prompt: {step.action.to_prompt()}")
            if step.observation:
                tree_lines.append(f"{content_prefix}│  Output: {step.observation.message}")
                if step.observation.extra:
                    tree_lines.append(f"{content_prefix}│  Extra: {step.observation.extra}")
                if step.observation.expect_correction:
                    tree_lines.append(f"{content_prefix}│  Expects Correction: True")

    # Process children with updated parameters
    child_prefix = prefix + ("    " if is_last else "│   ")
    children = node.children
    for i, child in enumerate(children):
        _append_ascii_node(
            child,
            child_prefix,
            i == len(children) - 1,
            tree_lines,
            current,
            include_explanation,
            include_diffs,
            include_feedback,
            include_action_details,
            use_color,
            show_trajectory,
        )


def _append_wrapped_text(tree_lines: list[str], text: str, prefix: str, header_prefix: str = "│ "):
    """Helper function to wrap and append text with proper prefixes."""
    words = text.split()
    current_line = []
    current_length = 0
    max_line_length = 100 - len(prefix) - len(header_prefix)

    # First line gets the header prefix
    is_first_line = True

    for word in words:
        if current_length + len(word) + 1 <= max_line_length:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            line_prefix = header_prefix if is_first_line else "│   "
            tree_lines.append(f"{prefix}{line_prefix}{' '.join(current_line)}")
            current_line = [word]
            current_length = len(word)
            is_first_line = False

    if current_line:
        line_prefix = header_prefix if is_first_line else "│   "
        tree_lines.append(f"{prefix}{line_prefix}{' '.join(current_line)}")


def color_red(text: Any) -> str:
    return f"\033[91m{text}\033[0m"


def color_green(text: Any) -> str:
    return f"\033[92m{text}\033[0m"


def color_yellow(text: Any) -> str:
    return f"\033[93m{text}\033[0m"


def color_white(text: Any) -> str:
    return f"\033[97m{text}\033[0m"
