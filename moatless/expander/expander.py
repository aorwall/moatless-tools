import logging
import random

from pydantic import BaseModel, Field

from moatless.component import MoatlessComponent
from moatless.node import Node

logger = logging.getLogger(__name__)


class Expander(MoatlessComponent):
    max_expansions: int = Field(1, description="The maximum number of children to create for each node")
    auto_expand_root: bool = Field(
        False, description="Whether to automatically expand the root node with max expansions"
    )

    @classmethod
    def get_component_type(cls) -> str:
        return "expander"

    @classmethod
    def _get_base_class(cls) -> type[MoatlessComponent]:
        return Expander

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.expander"

    async def expand(self, node: Node) -> None | Node:
        if self.max_expansions and not node.max_expansions:
            node.max_expansions = self.max_expansions

        if self.auto_expand_root and not node.parent:
            logger.info(f"Expanding root node {node.node_id} with {self.max_expansions} expansions")
            expansions = self.max_expansions
        else:
            expansions = 1

        self._add_child_nodes(node, expansions)

        logger.info(f"Expanded Node{node.node_id} to {expansions} new nodes")
        return node.children[-1]

    def _add_child_nodes(self, node: Node, expansions: int = 1) -> None:
        for _ in range(expansions):
            child_node = Node(
                node_id=self._generate_unique_id(node),
                parent=node,
                file_context=node.file_context.clone() if node.file_context else None,
                max_expansions=self.max_expansions,
                agent_id=None,
            )  # type: ignore

            node.add_child(child_node)

    def _generate_unique_id(self, node: Node):
        return len(node.get_root().get_all_nodes())
