import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

from pydantic import BaseModel, Field, model_validator, ConfigDict

from moatless.actions.action import Action
from moatless.agent.agent import ActionAgent
from moatless.agent.settings import AgentSettings
from moatless.completion.model import Usage
from moatless.discriminator.base import BaseDiscriminator
from moatless.exceptions import RuntimeError, RejectError
from moatless.expander import Expander
from moatless.feedback.base import BaseFeedbackGenerator
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.node import Node, generate_ascii_tree
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.selector.base import BaseSelector
from moatless.value_function.base import BaseValueFunction

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SearchTree(BaseModel):
    root: Node = Field(..., description="The root node of the search tree.")
    selector: Optional[BaseSelector] = Field(..., description="Selector for node selection.")
    agent: ActionAgent = Field(..., description="Agent for generating actions.")
    agent_settings: Optional[AgentSettings] = Field(None, description="Agent settings for the search tree.")
    actions: List[Action] = Field(
        default_factory=list,
        description="Actions that can be used by the agent in the search tree.",
    )
    repository: Optional[Repository] = Field(None, description="Repository for the search tree.")
    expander: Optional[Expander] = Field(None, description="Expander for expanding nodes.")
    value_function: Optional[BaseValueFunction] = Field(None, description="Value function for reward calculation.")
    feedback_generator: Optional[BaseFeedbackGenerator] = Field(None, description="Feedback generator.")
    discriminator: Optional[BaseDiscriminator] = Field(
        None, description="Discriminator for selecting the best trajectory."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the search tree.")
    persist_path: Optional[str] = Field(None, description="Path to persist the search tree.")
    unique_id: int = Field(default=0, description="Unique ID counter for nodes.")

    max_expansions: int = Field(1, description="The maximum number of expansions of one state.")
    max_iterations: int = Field(10, description="The maximum number of iterations to run the tree search.")
    max_cost: Optional[float] = Field(None, description="The maximum cost spent on token before finishing.")
    min_finished_nodes: Optional[int] = Field(
        None,
        description="The minimum number of finished nodes to consider before finishing",
    )
    max_finished_nodes: Optional[int] = Field(
        None,
        description="The maximum number of finished nodes to consider before finishing",
    )
    reward_threshold: Optional[float] = Field(
        None, description="The min reward threshold to consider before finishing."
    )
    max_depth: Optional[int] = Field(20, description="The maximum depth for one trajectory in simulations.")

    event_handlers: List[Callable] = Field(
        default_factory=list, description="Event handlers for tree events", exclude=True
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(
        cls,
        message: Optional[str] = None,
        root: Optional[Node] = None,
        file_context: Optional[FileContext] = None,
        repository: Repository | None = None,
        selector: Optional[BaseSelector] = None,
        agent: Optional[ActionAgent] = None,
        value_function: Optional[BaseValueFunction] = None,
        feedback_generator: Optional[BaseFeedbackGenerator] = None,
        discriminator: Optional[BaseDiscriminator] = None,
        metadata: Optional[Dict[str, Any]] = None,
        persist_path: Optional[str] = None,
        max_expansions: int = 1,
        max_iterations: int = 10,
        max_cost: Optional[float] = None,
        min_finished_nodes: Optional[int] = None,
        max_finished_nodes: Optional[int] = None,
        reward_threshold: Optional[float] = None,
        max_depth: int = 10,
    ) -> "SearchTree":
        if not root and not message:
            raise ValueError("Either a root node or a message must be provided.")

        if not file_context:
            file_context = FileContext(repo=repository)

        if not root:
            root = Node(
                node_id=0,
                max_expansions=max_expansions,
                message=message,
                file_context=file_context,
            )

        return cls(
            root=root,
            selector=selector,
            agent=agent,
            value_function=value_function,
            feedback_generator=feedback_generator,
            discriminator=discriminator,
            metadata=metadata or {},
            persist_path=persist_path,
            max_expansions=max_expansions,
            max_iterations=max_iterations,
            max_cost=max_cost,
            min_finished_nodes=min_finished_nodes,
            max_finished_nodes=max_finished_nodes,
            reward_threshold=reward_threshold,
            max_depth=max_depth,
        )

    @classmethod
    def model_validate(cls, obj: Any, repository: Repository | None = None, runtime: RuntimeEnvironment | None = None):
        if isinstance(obj, dict):
            obj = obj.copy()

            # Remove repository from validation since it's handled separately
            obj.pop("repository", None)

            if "selector" in obj and isinstance(obj["selector"], dict):
                obj["selector"] = BaseSelector.model_validate(obj["selector"])

            if "agent" in obj and isinstance(obj["agent"], dict):
                obj["agent"] = ActionAgent.model_validate(obj["agent"])

            if "value_function" in obj and isinstance(obj["value_function"], dict):
                obj["value_function"] = BaseValueFunction.model_validate(obj["value_function"])

            if "feedback_generator" in obj and isinstance(obj["feedback_generator"], dict):
                obj["feedback_generator"] = BaseFeedbackGenerator.model_validate(obj["feedback_generator"])

            if "discriminator" in obj and isinstance(obj["discriminator"], dict):
                obj["discriminator"] = BaseDiscriminator.model_validate(obj["discriminator"])

            if "root" in obj and isinstance(obj["root"], dict):
                obj["root"] = Node.reconstruct(obj["root"], repo=repository)

        instance = super().model_validate(obj)
        instance.repository = repository
        return instance

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        persist_path: str | None = None,
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        runtime: RuntimeEnvironment | None = None,
    ) -> "SearchTree":
        data = data.copy()
        if persist_path:
            data["persist_path"] = persist_path

        logger.info(f"Repository: {repository}")
        logger.info(f"Code index: {code_index}")
        logger.info(f"Runtime: {runtime}")
        if "agent" in data and isinstance(data["agent"], dict):
            agent_data = data["agent"]
            data["agent"] = ActionAgent.model_validate(
                agent_data,
                repository=repository,
                code_index=code_index,
                runtime=runtime,
            )

        return cls.model_validate(data, repository)

    @classmethod
    def from_file(cls, file_path: str, persist_path: str | None = None, **kwargs) -> "SearchTree":
        with open(file_path, "r") as f:
            tree_data = json.load(f)

        return cls.from_dict(tree_data, persist_path=persist_path or file_path, **kwargs)

    def run_search(self) -> Node | None:
        """Run the MCTS algorithm for a specified number of iterations."""

        self.assert_runnable()

        self.log(logger.info, generate_ascii_tree(self.root))

        if len(self.root.get_all_nodes()) > 1:
            self.log(
                logger.info,
                f"Restarting search tree with {len(self.root.get_all_nodes())} nodes",
            )

        # Emit tree started event
        self.emit_event("tree_started", {})

        while not self.is_finished():
            total_cost = self.total_usage().completion_cost
            self.log(
                logger.info,
                f"Run iteration {len(self.root.get_all_nodes())}",
                cost=total_cost,
            )

            node = self._select(self.root)

            if node:
                new_node = self._expand(node)
                self._simulate(new_node)
                self._backpropagate(new_node)
                self.maybe_persist()
                self.log(logger.info, generate_ascii_tree(self.root, new_node))

                # Emit tree iteration event
                self.emit_event(
                    "tree_iteration",
                    {
                        "iteration": len(self.root.get_all_nodes()),
                        "total_cost": total_cost,
                        "best_reward": max((n.reward.value if n.reward else 0) for n in self.root.get_all_nodes()),
                        "finished_nodes": len(self.get_finished_nodes()),
                        "total_nodes": len(self.root.get_all_nodes()),
                        "best_node_id": self.get_best_trajectory().node_id if self.get_best_trajectory() else None,
                        "action": new_node.action.name if new_node.action else None,
                        "current_node_id": new_node.node_id,
                    },
                )
            else:
                self.log(logger.info, "Search complete: no more nodes to expand.")
                break

        if not len(self.get_finished_nodes()):
            self.log(
                logger.warning,
                f"Search completed with no finished nodes. {len(self.root.get_all_nodes())} nodes created.",
            )
        else:
            self.log(
                logger.info,
                f"Search completed with {len(self.get_finished_nodes())} finished nodes. {len(self.root.get_all_nodes())} nodes created.",
            )

        # Emit tree completed event
        self.emit_event(
            "tree_completed",
            {
                "total_iterations": len(self.root.get_all_nodes()),
                "total_cost": self.total_usage().completion_cost,
                "finished_nodes": len(self.get_finished_nodes()),
                "best_node_id": self.get_best_trajectory().node_id if self.get_best_trajectory() else None,
            },
        )

        return self.get_best_trajectory()

    def _select(self, node: Node) -> Optional[Node]:
        """Select a node for expansion using the UCT algorithm."""
        expandable_nodes = node.get_expandable_descendants()

        if not expandable_nodes:
            self.log(logger.info, "No expandable nodes found.")
            return None

        # If we have a finished node or exceeded depth, use normal selection
        return self.selector.select(expandable_nodes)

    def _expand(self, node: Node, force_expansion: bool = False) -> Node:
        """Expand the node and return a child node."""

        # Check if any action step was not executed, if so return the node
        if node.action_steps and node.has_unexecuted_actions():
            self.log(logger.info, f"Returning Node{node.node_id} with unexecuted actions")
            return node

        child_node = self.expander.expand(node, self, force_expansion)

        # Only add feedback if this is the second expansion from this node
        if self.feedback_generator and len(node.children) >= 2:
            feedback_data = self.feedback_generator.generate_feedback(
                child_node,
                self.agent.actions,
            )

            if feedback_data:
                child_node.feedback_data = feedback_data
                child_node.user_message = feedback_data.feedback

                self.emit_event(
                    "feedback_generated",
                    {
                        "node_id": child_node.node_id,
                        "parent_id": node.node_id,
                        "feedback": child_node.feedback_data.model_dump(),
                        "action": child_node.action.name if child_node.action else None,
                        "depth": child_node.get_depth(),
                    },
                )

        self.log(logger.info, f"Expanded Node{node.node_id} to new Node{child_node.node_id}")
        return child_node

    def _simulate(self, node: Node):
        """Simulate a playout by executing the action and evaluating the result."""

        if node.observation:
            logger.info(f"Node{node.node_id}: Action already executed. Skipping.")
        else:
            self.agent.run(node)
            logger.info(f"Node{node.node_id}: Action executed. Depth: {node.get_depth()} ({self.max_depth})")

            if self.max_depth and node.get_depth() >= self.max_depth and not node.terminal:
                logger.info(f"Node{node.node_id}: Reached max depth {self.max_depth}. Marking as terminal.")
                node.terminal = True

        if self.value_function and not node.is_duplicate and node.observation:
            try:
                logger.info(f"Node{node.node_id}: Evaluating value function")
                node.reward, completion_response = self.value_function.get_reward(node=node)

                if completion_response:
                    node.completions["value_function"] = completion_response

                if node.reward:
                    self.log(
                        logger.info,
                        f"Node{node.node_id}: The value function returned a reward of {node.reward.value}.",
                    )
                    # Emit reward generated event
                    self.emit_event(
                        "reward_generated",
                        {
                            "node_id": node.node_id,
                            "reward": node.reward.value,
                            "action": node.action.name if node.action else None,
                            "depth": node.get_depth(),
                        },
                    )
                else:
                    self.log(
                        logger.info,
                        f"Node{node.node_id}: The value function returned no reward.",
                    )
            except RejectError as e:
                self.log(
                    logger.warning,
                    f"Node{node.node_id}: Value function rejected: {e.message}",
                )
                node.reward = None
            except RuntimeError as e:
                self.log(
                    logger.error,
                    f"Node{node.node_id}: Value function runtime error: {e.message}",
                )
                raise  # Re-raise to abort the entire search

    def _backpropagate(self, node: Node):
        """Backpropagate the reward up the tree."""

        if not node.reward:
            self.log(
                logger.info,
                f"Node{node.node_id} has no evaluation. Skipping backpropagation.",
            )
            return

        reward = node.reward.value
        while node is not None:
            node.visits += 1
            if not node.value:
                node.value = reward
            else:
                node.value += reward
            node = node.parent

    def get_best_trajectory(self) -> Node | None:
        """
        Get the best finished trajectory to return
        """

        nodes = self.get_finished_nodes()
        if not nodes:
            nodes = self.get_leaf_nodes()
            self.log(
                logger.info,
                f"get_best_trajectory() No finished nodes found. Will select from {len(nodes)} leaf nodes.",
            )

        if len(nodes) == 1:
            return nodes[0]

        if self.discriminator is None:
            self.log(
                logger.info,
                "No discriminator provided. Returning the first finished node.",
            )
            return nodes[-1]

        return self.discriminator.select(nodes)

    def is_finished(self):
        # Check max cost
        total_cost = self.total_usage().completion_cost
        if self.max_cost and self.total_usage().completion_cost and total_cost >= self.max_cost:
            logger.info(f"Search finished: Reached max cost {self.max_cost}")
            return True

        # Check max iterations
        if len(self.root.get_all_nodes()) >= self.max_iterations:
            logger.info(f"Search finished: Reached max iterations {self.max_iterations}")
            return True

        finished_nodes = self.get_finished_nodes()
        unique_finished_parents = set()
        for node in finished_nodes:
            unique_finished_parents.add(node.parent.node_id)

        # Check max finished nodes
        if self.max_finished_nodes and len(unique_finished_parents) >= self.max_finished_nodes:
            logger.info(f"Search finished: Reached max finished nodes {self.max_finished_nodes}")
            return True

        # Check reward threshold
        if self.reward_threshold and any(
            node.reward and node.reward.value >= self.reward_threshold for node in finished_nodes
        ):
            if not self.min_finished_nodes or len(unique_finished_parents) >= self.min_finished_nodes:
                logger.info(f"Search finished: Found solution meeting reward threshold {self.reward_threshold}")
                return True

        # Check if there are no more expandable nodes
        expandable_nodes = self.root.get_expandable_descendants()
        if not expandable_nodes:
            logger.info("Search finished: No more expandable nodes")
            return True

        return False

    def get_finished_nodes(self) -> List[Node]:
        """Get all finished nodes in the search tree by uniqe parent node."""
        parent_ids = set()
        finished_nodes = []
        for node in self.root.get_all_nodes():
            # TODO: Pick finished node with highest/avg/lowest reward?
            if node.is_finished() and node.parent.node_id not in parent_ids:
                parent_ids.add(node.parent.node_id)
                finished_nodes.append(node)

        return finished_nodes

    def get_node_by_id(self, node_id: int) -> Node | None:
        return next(
            (node for node in self.root.get_all_nodes() if node.node_id == node_id),
            None,
        )

    def get_leaf_nodes(self) -> List[Node]:
        """Get all leaf nodes in the search tree."""
        return [node for node in self.root.get_all_nodes() if node.is_leaf()]

    def total_usage(self) -> Usage:
        """Calculate total token usage across all nodes."""
        return self.root.total_usage()


    def maybe_persist(self):
        if self.persist_path:
            self.persist(self.persist_path)

    def persist(self, file_path: str, **kwargs):
        """
        Persist the entire SearchTree to a file.

        Args:
            file_path (str): The path to the file where the tree will be saved.
        """
        tree_data = self.model_dump(**kwargs)

        with open(file_path, "w") as f:
            try:
                json.dump(tree_data, f, indent=2)
            except Exception as e:
                logger.exception(f"Error saving search tree to {file_path}: {tree_data}")
                raise e

    def _generate_unique_id(self) -> int:
        self.unique_id += 1
        return self.unique_id

    def assert_runnable(self):
        if self.root is None:
            raise RuntimeError("SearchTree must have a root node.")

        if self.root.file_context is None:
            raise RuntimeError("SearchTree root node must have a file context.")

        if self.agent is None:
            raise RuntimeError("SearchTree must have an agent.")

        if not self.agent.actions:
            raise RuntimeError("SearchTree agent must have actions.")

        return True

    @classmethod
    def create(
        cls,
        message: Optional[str] = None,
        root: Optional[Node] = None,
        file_context: Optional[FileContext] = None,
        repository: Repository | None = None,
        runtime: RuntimeEnvironment | None = None,
        selector: Optional[BaseSelector] = None,
        expander: Optional[Expander] = None,
        agent: Optional[ActionAgent] = None,
        value_function: Optional[BaseValueFunction] = None,
        feedback_generator: Optional[BaseFeedbackGenerator] = None,
        discriminator: Optional[BaseDiscriminator] = None,
        metadata: Optional[Dict[str, Any]] = None,
        persist_path: Optional[str] = None,
        max_expansions: int = 1,
        max_iterations: int = 10,
        max_cost: Optional[float] = None,
        min_finished_nodes: Optional[int] = None,
        max_finished_nodes: Optional[int] = None,
        reward_threshold: Optional[float] = None,
        simulation_depth: int = 1,
        max_depth: Optional[int] = None,
    ) -> "SearchTree":
        if not root and not message:
            raise ValueError("Either a root node or a message must be provided.")

        if not file_context:
            file_context = FileContext(repo=repository, runtime=runtime)

        if not root:
            root = Node(
                node_id=0,
                max_expansions=max_expansions,
                user_message=message,
                file_context=file_context,
            )

        expander = expander or Expander(max_expansions=max_expansions)

        return cls(
            root=root,
            selector=selector,
            expander=expander,
            agent=agent,
            repository=repository,
            value_function=value_function,
            feedback_generator=feedback_generator,
            discriminator=discriminator,
            metadata=metadata or {},
            persist_path=persist_path,
            max_expansions=max_expansions,
            max_iterations=max_iterations,
            max_cost=max_cost,
            min_finished_nodes=min_finished_nodes,
            max_finished_nodes=max_finished_nodes,
            reward_threshold=reward_threshold,
            max_depth=max_depth,
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        persist_path: str | None = None,
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        runtime: RuntimeEnvironment | None = None,
    ) -> "SearchTree":
        data = data.copy()
        if persist_path:
            data["persist_path"] = persist_path

        if "agent" in data and isinstance(data["agent"], dict):
            agent_data = data["agent"]
            data["agent"] = ActionAgent.model_validate(
                agent_data,
                repository=repository,
                code_index=code_index,
                runtime=runtime,
            )

        return cls.model_validate(data, repository, runtime)

    @classmethod
    def from_file(cls, file_path: str, persist_path: str | None = None, **kwargs) -> "SearchTree":
        with open(file_path, "r") as f:
            tree_data = json.load(f)

        return cls.from_dict(tree_data, persist_path=persist_path or file_path, **kwargs)

    @model_validator(mode="after")
    def set_depth(self):
        if self.max_expansions == 1:
            self.max_depth = self.max_iterations
        return self

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the SearchTree.

        Returns:
            Dict[str, Any]: A dictionary representation of the search tree.
        """
        # Get all fields except the ones we'll handle separately
        data = {
            field: getattr(self, field)
            for field in self.model_fields
            if field
            not in [
                "root",
                "selector",
                "repository",
                "agent",
                "value_function",
                "feedback_generator",
                "discriminator",
                "persist_path",
                "event_handlers",
            ]
        }

        data.pop("persist_path", None)

        data["selector"] = self.selector.model_dump(**kwargs)
        data["expander"] = self.expander.model_dump(**kwargs)
        data["agent"] = self.agent.model_dump(**kwargs)
        data["agent_settings"] = self.agent_settings.model_dump(**kwargs) if self.agent_settings else None
        data["repository"] = self.repository.model_dump(**kwargs) if self.repository else None

        if self.value_function:
            data["value_function"] = self.value_function.model_dump(**kwargs)
        if self.feedback_generator:
            data["feedback_generator"] = self.feedback_generator.model_dump(**kwargs)
        if self.discriminator:
            data["discriminator"] = self.discriminator.model_dump(**kwargs)

        data["root"] = self.root.model_dump(**kwargs)

        return data

    def log(self, logger_fn: Callable, message: str, **kwargs):
        """
        Log a message with metadata prefix (if any) and specified log level.

        Args:
            logger_fn: Logger function (logger.debug, logger.info, etc)
            message (str): The message to log
            **kwargs: Additional key-value pairs to include in metadata
        """
        metadata = {**self.metadata, **kwargs}
        metadata_str = " ".join(f"{k}: {str(v)[:20]}" for k, v in metadata.items())
        log_message = f"[{metadata_str}] {message}" if metadata else message

        logger_fn(log_message)

    def add_event_handler(self, handler: Callable):
        """Add an event handler for tree events."""
        self.event_handlers.append(handler)

    def emit_event(self, event_type: str, data: dict):
        """Emit an event to all registered handlers."""
        logger.info(f"Emit event {event_type}")
        for handler in self.event_handlers:
            handler(
                {
                    "event_type": event_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                }
            )
