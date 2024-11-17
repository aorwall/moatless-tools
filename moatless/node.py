import json
import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Union

from moatless.actions.view_code import ViewCodeArgs, CodeSpan
from pydantic import BaseModel, Field

from moatless.actions.model import ActionArguments, Observation
from moatless.completion.model import (
    Usage,
    Completion,
    Message,
    UserMessage,
    AssistantMessage,
)
from moatless.file_context import FileContext
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)


class MessageHistoryType(Enum):
    MESSAGES = "messages"  # Provides all messages in sequence
    SUMMARY = "summary"  # Generates one message with summarized history
    REACT = "react"


class Node(BaseModel):
    node_id: int = Field(..., description="The unique identifier of the node")
    parent: Optional["Node"] = Field(None, description="The parent node")
    children: List["Node"] = Field(default_factory=list, description="The child nodes")

    action: Optional[ActionArguments] = Field(
        None, description="The action associated with the node"
    )
    observation: Optional[Observation] = Field(
        None, description="The output of the action"
    )
    file_context: Optional[FileContext] = Field(
        None, description="The file context state associated with the node"
    )
    message: Optional[str] = Field(
        None, description="The message associated with the node"
    )
    feedback: Optional[str] = Field(None, description="Feedback provided to the node")
    completions: Dict[str, Completion] = Field(
        default_factory=dict, description="The completions used in this node"
    )
    possible_actions: List[str] = Field(
        default_factory=list, description="List of possible action types for this node"
    )

    @classmethod
    def stub(cls, **kwargs):
        return cls(node_id=0, **kwargs)

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
        if self.observation and self.observation.terminal:
            return True

        return False

    def is_finished(self) -> bool:
        """Determine if the node is succesfully finished"""
        if self.action and self.action.name == "Finish":
            return True

        return False

    def add_child(self, child_node: "Node"):
        """Add a child node to this node."""
        child_node.parent = self
        self.children.append(child_node)

    def get_depth(self) -> int:
        depth = 0
        node = self
        while node.parent:
            depth += 1
            node = node.parent
        return depth

    def is_expandable(self) -> bool:
        """Check if the node can be expanded further."""
        return (
            not self.is_terminal()
            and not self.is_fully_expanded()
            and not self.is_duplicate
        )

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

        return [
            child for child in self.parent.children if child.node_id != self.node_id
        ]

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
        nodes = []
        nodes.append(self)
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

    def get_root(self) -> "Node":
        node = self
        while node.parent:
            node = node.parent
        return node

    def total_usage(self) -> Usage:
        total_usage = Usage()

        for completion in self.completions.values():
            if completion:
                total_usage += completion.usage

        return total_usage

    def generate_message_history(
        self, message_history_type: MessageHistoryType = MessageHistoryType.MESSAGES
    ) -> list[Message]:
        previous_nodes = self.get_trajectory()[:-1]
        if not previous_nodes:
            return []
        logger.info(
            f"Generating message history for Node{self.node_id}: {message_history_type}"
        )
        if message_history_type == MessageHistoryType.SUMMARY:
            messages = self._generate_summary_history(previous_nodes)
        elif message_history_type == MessageHistoryType.REACT:
            messages = self.generate_react_summary(previous_nodes)
        else:  # MessageHistoryType.MESSAGES
            messages = self._generate_message_history(previous_nodes)

        return messages

    def generate_react_summary(
        self,
        previous_nodes: List["Node"],
        include_file_context: bool = True,
        include_git_patch: bool = True,
    ) -> list[Message]:
        """Generate a sequence of messages in ReAct format."""
        messages = [UserMessage(content=self.get_root().message)]

        if len(previous_nodes) <= 1:
            return messages

        for previous_node in previous_nodes[1:]:
            if previous_node.action:
                # Create assistant message with thought and action
                thought = (
                    f"Thought: {previous_node.action.scratch_pad}"
                    if hasattr(previous_node.action, "scratch_pad")
                    else ""
                )
                action = f"Action: {previous_node.action.name}\nAction Input: {previous_node.action.model_dump_json(exclude={'scratch_pad'})}"
                messages.append(AssistantMessage(content=f"{thought}\n{action}"))

                # Create user message with observation
                if previous_node.observation:
                    if (
                        hasattr(previous_node.observation, "summary")
                        and previous_node.observation.summary
                    ):
                        observation = previous_node.observation.summary
                    else:
                        observation = previous_node.observation.message
                else:
                    logger.warning(f"No output found for Node{previous_node.node_id}")
                    observation = "No output found."
                messages.append(UserMessage(content=f"Observation: {observation}"))

        if include_file_context and not self.file_context.is_empty():
            thought = "Thought: I need to see all the code I have viewed so far"
            action = "Action: ShowViewedCode"
            messages.append(AssistantMessage(content=f"{thought}\n{action}"))

            observation = self.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
            messages.append(UserMessage(content=f"Observation: {observation}"))

        if include_git_patch:
            git_patch = self.file_context.generate_git_patch()
            if git_patch:
                thought = "Thought: I need see the changes I done so far"
                action = "Action: GitDiff"
                messages.append(AssistantMessage(content=f"{thought}\n{action}"))

                git_patch = self.file_context.generate_git_patch()
                observation = f"```diff\n{git_patch}\n```"
                messages.append(UserMessage(content=f"Observation: {observation}"))
        return messages

    def _generate_summary_history(
        self,
        previous_nodes: List["Node"],
        include_file_context: bool = True,
        include_git_patch: bool = True,
    ) -> list[Message]:
        """Generate a single message containing summarized history."""
        formatted_history: List[str] = []
        counter = 0

        content = self.get_root().message

        if not previous_nodes:
            return [UserMessage(content=content)]

        for i, previous_node in enumerate(previous_nodes):
            if previous_node.action:
                counter += 1
                formatted_state = (
                    f"\n## {counter}. Action: {previous_node.action.name}\n"
                )
                formatted_state += previous_node.action.to_prompt()

                if previous_node.observation:
                    if (
                        hasattr(previous_node.observation, "summary")
                        and previous_node.observation.summary
                        and i < len(previous_nodes) - 1
                    ):
                        formatted_state += (
                            f"\n\nObservation: {previous_node.observation.summary}"
                        )
                    else:
                        formatted_state += (
                            f"\n\nObservation: {previous_node.observation.message}"
                        )
                else:
                    logger.warning(f"No output found for Node{previous_node.node_id}")
                    formatted_state += "\n\nObservation: No output found."

                formatted_history.append(formatted_state)

        content += "\n\nBelow is the history of previously executed actions and their observations.\n"
        content += "<history>\n"
        content += "\n".join(formatted_history)
        content += "\n</history>\n\n"

        if include_file_context:
            content += "\n\nThe following code has already been viewed:\n"
            content += self.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )

        if include_git_patch:
            git_patch = self.file_context.generate_git_patch()
            if git_patch:
                content += "\n\nThe current git diff is:\n"
                content += "```diff\n"
                content += git_patch
                content += "\n```"

        return [UserMessage(content=content)]

    def _generate_message_history(
        self, previous_nodes: List["Node"], show_full_file: bool = False
    ) -> list[Message]:
        """Generate a sequence of messages representing the full conversation history."""
        messages: list[Message] = []
        last_file_updates = {}

        if show_full_file:
            # Track when each file was last modified to show file contexts optimally.
            # By showing each file's context only in the last message where it was modified,
            # we improve prompt caching since earlier messages won't change when new files are modified.
            for i, node in enumerate(previous_nodes):
                if not node.parent:
                    updated_files = set(
                        [
                            file.file_path
                            for file in node.file_context.get_context_files()
                        ]
                    )
                else:
                    updated_files = node.file_context.get_updated_files(
                        node.parent.file_context
                    )
                    for file in updated_files:
                        last_file_updates[file] = i

        for i, previous_node in enumerate(previous_nodes):
            if previous_node.message:
                messages.append(UserMessage(content=previous_node.message))

            if previous_node.action:
                tool_call = previous_node.action.to_tool_call()
                messages.append(AssistantMessage(tool_call=tool_call))

                content = ""
                if previous_node.observation:
                    if show_full_file and previous_node.observation.summary:
                        content += previous_node.observation.summary
                    else:
                        content += previous_node.observation.message

                messages.append(UserMessage(content=content))

            # Show file context for files that were last updated in this message
            if not previous_node.parent:
                updated_files = set(
                    [
                        file.file_path
                        for file in previous_node.file_context.get_context_files()
                    ]
                )
            else:
                updated_files = previous_node.file_context.get_updated_files(
                    previous_node.parent.file_context
                )

            files_to_show = set(
                [f for f in updated_files if last_file_updates.get(f) == i]
            )

            for file_path in files_to_show:
                context_file = previous_node.file_context.get_context_file(file_path)

                if context_file.show_all_spans:
                    args = ViewCodeArgs(
                        scratch_pad=f"Let's view the content in {file_path}",
                        files=[CodeSpan(file_path=file_path)],
                    )
                elif context_file.span_ids:
                    args = ViewCodeArgs(
                        scratch_pad=f"Let's view the content in {file_path}",
                        files=[
                            CodeSpan(
                                file_path=file_path, span_ids=context_file.span_ids
                            )
                        ],
                    )
                else:
                    continue

                messages.append(AssistantMessage(tool_call=args.to_tool_call()))
                messages.append(
                    UserMessage(
                        content=context_file.to_prompt(
                            show_span_ids=False,
                            show_line_numbers=True,
                            exclude_comments=False,
                            show_outcommented_code=True,
                            outcomment_code_comment="... rest of the code",
                        )
                    )
                )

        feedback = self._format_feedback()
        if feedback:
            messages.append(UserMessage(content=feedback))

        return messages

    def _show_updated_context(
        self,
        previous_node: "Node",
        show_full_file: bool,
        last_file_updates: Dict[str, int],
        i: int,
    ) -> str:
        updated_context = None
        content = ""

        if show_full_file:
            # Show file context for files that were last updated in this message
            if not previous_node.parent:
                updated_files = set(
                    [
                        file.file_path
                        for file in previous_node.file_context.get_context_files()
                    ]
                )
            else:
                updated_files = previous_node.file_context.get_updated_files(
                    previous_node.parent.file_context
                )

            files_to_show = set(
                [f for f in updated_files if last_file_updates.get(f) == i]
            )

            if files_to_show:
                content += f"\n\nThe file context for the following files was updated by this action:\n"

        elif previous_node.parent:
            updated_context = previous_node.file_context.get_context_diff(
                previous_node.parent.file_context
            )
        else:
            updated_context = previous_node.file_context

        if updated_context and not updated_context.is_empty():
            context_prompt = previous_node.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
            content += f"\n\nCode added to context:\n{context_prompt}"

            if not content:
                logger.warning(
                    f"Node{previous_node.node_id}: No content to add to messages"
                )

        return content

    def _format_feedback(self) -> str:
        """Generate formatted string for feedback."""
        if not self.feedback:
            return ""

        return f"\n\n{self.feedback}"

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

        self.action = None
        self.visits = 0
        self.value = 0.0
        self.observation = None
        self.feedback = None
        self.completions = {}
        self.is_duplicate = False
        if self.parent and self.parent.file_context:
            self.file_context = self.parent.file_context.clone()

        self.children = []

    def clone_and_reset(self) -> "Node":
        """
        Creates a copy of the node and resets its observation and file context.

        Returns:
            Node: A new node instance with reset state
        """
        # Create a new node with same base attributes
        new_node = Node(
            node_id=self.node_id,
            parent=self.parent,
            visits=self.visits,
            value=self.value,
            max_expansions=self.max_expansions,
            message=self.message,
            feedback=self.feedback,
            is_duplicate=self.is_duplicate,
            action=self.action,
            possible_actions=self.possible_actions.copy()
            if self.possible_actions
            else [],
        )

        new_node.reset()
        return new_node

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the node and its descendants.

        Returns:
            Dict[str, Any]: A dictionary representation of the node tree.
        """

        def serialize_node(node: "Node") -> Dict[str, Any]:
            exclude_set = {"parent", "children"}
            if "exclude" in kwargs:
                if isinstance(kwargs["exclude"], set):
                    exclude_set.update(kwargs["exclude"])
                elif isinstance(kwargs["exclude"], dict):
                    exclude_set.update(kwargs["exclude"].keys())

            new_kwargs = {k: v for k, v in kwargs.items() if k != "exclude"}
            node_dict = super().model_dump(exclude=exclude_set, **new_kwargs)

            if node.action and "action" not in exclude_set:
                node_dict["action"] = node.action.model_dump(**kwargs)
                node_dict["action"]["action_args_class"] = (
                    f"{node.action.__class__.__module__}.{node.action.__class__.__name__}"
                )

            if node.completions and "completions" not in exclude_set:
                node_dict["completions"] = {
                    key: completion.model_dump(**kwargs)
                    for key, completion in node.completions.items()
                    if completion
                }

            if node.observation and "output" not in exclude_set:
                node_dict["output"] = node.observation.model_dump(**kwargs)

            if node.file_context and "file_context" not in exclude_set:
                node_dict["file_context"] = node.file_context.model_dump(**kwargs)

            if not kwargs.get("exclude") or "children" not in kwargs.get("exclude"):
                node_dict["children"] = [
                    serialize_node(child) for child in node.children
                ]

            return node_dict

        return serialize_node(self)

    @classmethod
    def _reconstruct_node(
        cls,
        node_data: Dict[str, Any],
        repo: Repository | None = None,
    ) -> "Node":
        if node_data.get("action"):
            node_data["action"] = ActionArguments.model_validate(node_data["action"])

        if node_data.get("output"):
            node_data["observation"] = Observation.model_validate(node_data["output"])

        if node_data.get("completions"):
            for key, completion_data in node_data["completions"].items():
                completion = Completion.model_validate(completion_data)
                node_data["completions"][key] = completion

        if node_data.get("file_context"):
            node_data["file_context"] = FileContext.from_dict(
                repo=repo, data=node_data["file_context"]
            )

        if "children" in node_data:
            children = node_data.get("children", [])

            del node_data["children"]
            node = super().model_validate(node_data)

            for child_data in children:
                child = cls._reconstruct_node(child_data, repo=repo)
                child.parent = node
                node.children.append(child)

            return node
        else:
            return super().model_validate(node_data)

    @classmethod
    def reconstruct(
        cls,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        repo: Repository | None = None,
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
            return cls._reconstruct_from_list(data, repo=repo)

        # Handle single node reconstruction (dict format)
        return cls._reconstruct_node(data, repo=repo)

    @classmethod
    def _reconstruct_from_list(
        cls, node_list: List[Dict], repo: Repository | None = None
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
            # Use the core reconstruct method for each node
            node = cls._reconstruct_node(node_data, repo=repo)
            nodes_by_id[node.node_id] = (node, parent_id)

        # Connect parent-child relationships
        for node, parent_id in nodes_by_id.values():
            if parent_id is not None:
                parent_node = nodes_by_id[parent_id][0]
                parent_node.add_child(node)

        root_nodes = [n for n, p_id in nodes_by_id.values() if p_id is None]
        if not root_nodes:
            raise ValueError("No root node found in data")

        return root_nodes[0]

    def dump_as_list(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Dump all nodes as a flat list structure.
        """
        nodes = self.get_all_nodes()
        node_list = []

        for node in nodes:
            node_data = node.model_dump(exclude={"parent", "children"}, **kwargs)
            node_data["parent_id"] = node.parent.node_id if node.parent else None
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


def generate_ascii_tree(root: Node, current: Node | None = None) -> str:
    tree_lines = ["MCTS Tree"]
    _append_ascii_node(root, "", True, tree_lines, current)
    return "\n".join(tree_lines)


def _append_ascii_node(
    node: Node, prefix: str, is_last: bool, tree_lines: list[str], current: Node | None
):
    state_params = []

    if node.action:
        state_params.append(node.action.name)

        if node.observation and node.observation.expect_correction:
            state_params.append("expect_correction")

    state_info = f"Node{node.node_id}"
    if state_params:
        state_info += f"({', '.join(state_params)})"
    else:
        state_info += f"()"

    if current and node.node_id == current.node_id:
        state_info = color_white(state_info)

    node_str = f"Node{node.node_id} [-]"

    tree_lines.append(
        f"{prefix}{'└── ' if is_last else '├── '}{node_str} {state_info}"
    )

    child_prefix = prefix + ("    " if is_last else "│   ")
    children = node.children
    for i, child in enumerate(node.children):
        _append_ascii_node(
            child, child_prefix, i == len(children) - 1, tree_lines, current
        )


def color_red(text: Any) -> str:
    return f"\033[91m{text}\033[0m"


def color_green(text: Any) -> str:
    return f"\033[92m{text}\033[0m"


def color_yellow(text: Any) -> str:
    return f"\033[93m{text}\033[0m"


def color_white(text: Any) -> str:
    return f"\033[97m{text}\033[0m"
