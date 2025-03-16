import logging
from collections.abc import Callable
from typing import Any, Dict, Optional

from opentelemetry import trace
from pydantic import ConfigDict, Field, model_validator

from moatless.agent.agent import ActionAgent
from moatless.completion.stats import Usage
from moatless.context_data import current_node_id
from moatless.discriminator.base import BaseDiscriminator
from moatless.exceptions import RejectError, RuntimeError
from moatless.expander import Expander
from moatless.feedback.base import BaseFeedbackGenerator
from moatless.flow import AgenticFlow
from moatless.flow.events import (
    FeedbackGeneratedEvent,
    FlowErrorEvent,
    NodeExpandedEvent,
    NodeRewardEvent,
    NodeRewardFailureEvent,
    NodeSelectedEvent,
)
from moatless.node import Node, generate_ascii_tree
from moatless.selector.base import BaseSelector
from moatless.value_function.base import BaseValueFunction

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.search_tree")


class SearchTree(AgenticFlow):
    selector: BaseSelector = Field(..., description="Selector for node selection.")
    expander: Expander = Field(..., description="Expander for expanding nodes.")
    value_function: Optional[BaseValueFunction] = Field(None, description="Value function for reward calculation.")
    feedback_generator: Optional[BaseFeedbackGenerator] = Field(None, description="Feedback generator.")
    discriminator: Optional[BaseDiscriminator] = Field(
        None, description="Discriminator for selecting the best trajectory."
    )

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(
        cls,
        selector: Optional[BaseSelector] = None,
        expander: Optional[Expander] = None,
        agent: Optional[ActionAgent] = None,
        value_function: Optional[BaseValueFunction] = None,
        feedback_generator: Optional[BaseFeedbackGenerator] = None,
        discriminator: Optional[BaseDiscriminator] = None,
        max_expansions: int = 1,
        max_iterations: int = 10,
        max_cost: Optional[float] = None,
        min_finished_nodes: Optional[int] = None,
        max_finished_nodes: Optional[int] = None,
        reward_threshold: Optional[float] = None,
        max_depth: Optional[int] = None,
        **kwargs,
    ):
        expander = expander or Expander(max_expansions=max_expansions)  # type: ignore

        return super().create(
            selector=selector,
            expander=expander,
            agent=agent,
            value_function=value_function,
            feedback_generator=feedback_generator,
            discriminator=discriminator,
            max_expansions=max_expansions,
            max_iterations=max_iterations,
            max_cost=max_cost,
            min_finished_nodes=min_finished_nodes,
            max_finished_nodes=max_finished_nodes,
            reward_threshold=reward_threshold,
            max_depth=max_depth,
            **kwargs,
        )

    @tracer.start_as_current_span("SearchTree._run")
    async def _run(self, message: str | None = None) -> tuple[Node, str]:
        """Run the search tree algorithm with the given node."""
        if not self.root:
            raise ValueError("No node provided to run")

        self.log(logger.info, generate_ascii_tree(self.root))

        if len(self.root.get_all_nodes()) > 1:
            self.log(
                logger.info,
                f"Restarting search tree with {len(self.root.get_all_nodes())} nodes",
            )

        node = self.root
        finish_reason = None
        while not (finish_reason := self.is_finished()):
            total_cost = self.total_usage().completion_cost
            self.log(
                logger.info,
                f"Run iteration {len(self.root.get_all_nodes())}",
                cost=total_cost,
            )

            node = await self._select(node)

            if node:
                new_node = await self._expand(node)
                if new_node:
                    await self._simulate(new_node)
                    self._backpropagate(new_node)
                    node = new_node
                else:
                    self.log(logger.warning, f"No node expanded from Node{node.node_id}")
                    await self._emit_event(
                        FlowErrorEvent(
                            node_id=node.node_id,
                            error="No node expanded",
                        )
                    )
                    break

                self.log(logger.info, generate_ascii_tree(self.root, node))

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

        if finish_reason:
            node.terminal = True

        return self.get_best_trajectory(), finish_reason

    @tracer.start_as_current_span("SearchTree._select")
    async def _select(self, node: Node) -> Optional[Node]:
        """Select a node for expansion using the UCT algorithm."""
        root = node.get_root()
        expandable_nodes = root.get_expandable_descendants()

        if not expandable_nodes:
            self.log(logger.info, "No expandable nodes found.")
            return None

        previous_node_id = node.node_id

        node = await self.selector.select(expandable_nodes)

        await self._emit_event(
            NodeSelectedEvent(
                node_id=node.node_id,
                previous_node_id=previous_node_id,
            )
        )

        return node

    @tracer.start_as_current_span("SearchTree._expand")
    async def _expand(self, node: Node) -> Node | None:
        """Expand the node and return a child node."""
        # Check if any action step was not executed, if so return the node
        if node.action_steps and node.has_unexecuted_actions():
            self.log(logger.info, f"Returning Node{node.node_id} with unexecuted actions")
            return node

        child_node = await self.expander.expand(node)
        if not child_node:
            self.log(logger.warning, f"Returning Node{node.node_id} with no child node")
            return None

        await self.persist()

        await self._emit_event(
            NodeExpandedEvent(
                node_id=node.node_id,
                child_node_id=child_node.node_id,
            )
        )
        # Only add feedback if this is the second expansion from this node
        if self.feedback_generator:
            feedback_data = await self.feedback_generator.generate_feedback(child_node)

            if feedback_data:
                child_node.feedback_data = feedback_data
                child_node.user_message = feedback_data.feedback

                await self._emit_event(
                    FeedbackGeneratedEvent(
                        node_id=child_node.node_id,
                    )
                )

        self.log(logger.info, f"Expanded Node{node.node_id} to new Node{child_node.node_id}")
        return child_node

    @tracer.start_as_current_span("SearchTree._simulate")
    async def _simulate(self, node: Node):
        """Simulate a playout by executing the action and evaluating the result."""
        if node.observation:
            logger.info(f"Node{node.node_id}: Action already executed. Skipping.")
        else:
            current_node_id.set(node.node_id)
            await self.agent.run(node)
            logger.info(f"Node{node.node_id}: Action executed. Depth: {node.get_depth()} ({self.max_depth})")

            if self.max_depth and node.get_depth() >= self.max_depth and not node.terminal:
                logger.info(f"Node{node.node_id}: Reached max depth {self.max_depth}. Marking as terminal.")
                node.terminal = True

        if self.value_function and not node.is_duplicate and node.observation:
            try:
                logger.info(f"Node{node.node_id}: Evaluating value function")
                node.reward, completion_response = await self.value_function.get_reward(node=node)

                if completion_response:
                    node.completions["value_function"] = completion_response

                if node.reward:
                    self.log(
                        logger.info,
                        f"Node{node.node_id}: The value function returned a reward of {node.reward.value}.",
                    )

                    await self._emit_event(
                        NodeRewardEvent(
                            node_id=node.node_id,
                            reward=node.reward.value,
                        )
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
                await self._emit_event(
                    NodeRewardFailureEvent(
                        node_id=node.node_id,
                        error=str(e),
                    )
                )
                node.reward = None
            except RuntimeError as e:
                self.log(
                    logger.error,
                    f"Node{node.node_id}: Value function runtime error: {e.message}",
                )
                raise  # Re-raise to abort the entire search

    @tracer.start_as_current_span("SearchTree._backpropagate")
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

    def is_finished(self) -> str | None:
        # Check max cost
        total_cost = self.total_usage().completion_cost
        if self.max_cost and self.total_usage().completion_cost and total_cost >= self.max_cost:
            logger.info(f"Search finished: Reached max cost {self.max_cost}")
            return "max_cost"

        # Check max iterations
        if len(self.root.get_all_nodes()) >= self.max_iterations:
            logger.info(f"Search finished: Reached max iterations {self.max_iterations}")
            return "max_iterations"

        finished_nodes = self.get_finished_nodes()
        unique_finished_parents = set()
        for node in finished_nodes:
            unique_finished_parents.add(node.parent.node_id)

        # Check max finished nodes
        if self.max_finished_nodes and len(unique_finished_parents) >= self.max_finished_nodes:
            logger.info(f"Search finished: Reached max finished nodes {self.max_finished_nodes}")
            return "max_finished_nodes"

        # Check reward threshold
        if self.reward_threshold and any(
            node.reward and node.reward.value >= self.reward_threshold for node in finished_nodes
        ):
            if not self.min_finished_nodes or len(unique_finished_parents) >= self.min_finished_nodes:
                logger.info(f"Search finished: Found solution meeting reward threshold {self.reward_threshold}")
                return "reward_threshold"

        # Check if there are no more expandable nodes
        expandable_nodes = self.root.get_expandable_descendants()
        if not expandable_nodes:
            logger.info("Search finished: No more expandable nodes")
            return "no_expandable_nodes"

        return None

    def get_finished_nodes(self) -> list[Node]:
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

    def get_leaf_nodes(self) -> list[Node]:
        """Get all leaf nodes in the search tree."""
        return [node for node in self.root.get_all_nodes() if node.is_leaf()]

    def total_usage(self) -> Usage:
        """Calculate total token usage across all nodes."""
        return self.root.total_usage()

    def _generate_unique_id(self) -> int:
        return len(self.root.get_all_nodes()) + 1

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

    @model_validator(mode="after")
    def set_depth(self):
        if self.max_expansions == 1:
            self.max_depth = self.max_iterations
        return self

    @classmethod
    def model_validate(cls, obj: Any):
        if isinstance(obj, dict):
            obj = obj.copy()

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

            return super().model_validate(obj)
        return obj

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Generate a dictionary representation of the SearchTree.

        Returns:
            Dict[str, Any]: A dictionary representation of the search tree.
        """
        data = super().model_dump(**kwargs)
        data["selector"] = self.selector.model_dump(**kwargs)
        data["expander"] = self.expander.model_dump(**kwargs)
        data["agent"] = self.agent.model_dump(**kwargs)

        if self.value_function:
            data["value_function"] = self.value_function.model_dump(**kwargs)
        if self.feedback_generator:
            data["feedback_generator"] = self.feedback_generator.model_dump(**kwargs)
        if self.discriminator:
            data["discriminator"] = self.discriminator.model_dump(**kwargs)

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
