import logging
import random
from typing import List

from pydantic import BaseModel, Field

from moatless.agent.settings import AgentSettings
from moatless.node import Node

logger = logging.getLogger(__name__)


class Expander(BaseModel):
    random_settings: bool = Field(
        False, description="Whether to select agent settings randomly"
    )
    max_expansions: int = Field(
        1, description="The maximum number of children to create for each node"
    )

    agent_settings: List[AgentSettings] = Field(
        [],
        description="The settings for the agent model",
    )

    def expand(
        self, node: Node, search_tree, force_expansion: bool = False
    ) -> None | Node:
        """Handle all node expansion logic in one place"""
        if not force_expansion and node.is_fully_expanded():
            return None

        # Return the first unexecuted child if one exists
        for child in node.children:
            if not child.observation:
                logger.info(
                    f"Found unexecuted child {child.node_id} for node {node.node_id}"
                )
                return child

        num_expansions = node.max_expansions or self.max_expansions
        if not force_expansion and len(node.children) >= num_expansions:
            logger.info(f"Max expansions reached for node {node.node_id}")
            return None

        settings_to_use = self._get_agent_settings(node)

        child_node = Node(
            node_id=search_tree._generate_unique_id(),
            parent=node,
            file_context=node.file_context.clone() if node.file_context else None,
            max_expansions=self.max_expansions,
            agent_settings=settings_to_use[0] if settings_to_use else None,
        )

        node.add_child(child_node)

        logger.info(f"Expanded Node{node.node_id} to new Node{child_node.node_id}")
        return child_node

    def _get_agent_settings(self, node: Node) -> List[AgentSettings]:
        """Get agent settings for a single expansion."""
        if not self.agent_settings:
            return []

        if self.random_settings:
            used_settings = {
                child.agent_settings
                for child in node.children
                if child.agent_settings is not None
            }

            available_settings = [
                setting
                for setting in self.agent_settings
                if setting not in used_settings
            ]

            settings_pool = available_settings or self.agent_settings
            return [random.choice(settings_pool)]
        else:
            num_children = len(node.children)
            return [self.agent_settings[num_children % len(self.agent_settings)]]

    def _generate_unique_id(self, node: Node):
        return len(node.get_root().get_all_nodes())
