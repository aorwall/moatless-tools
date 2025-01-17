from typing import List

from pydantic import Field

from moatless.node import Node
from moatless.selector import BaseSelector


class SimpleSelector(BaseSelector):
    """Select first not expanded node that is not terminal and didn't reach max depth. Then select root again if expandable."""

    max_depth: int = Field(
        default=20,
    )

    def select(self, expandable_nodes: List[Node]) -> Node | None:
        if not expandable_nodes:
            return None

        for node in expandable_nodes:
            if node.expanded_count() == 0 and node.terminal and node.get_depth() < self.max_depth:
                return node

        root_node = expandable_nodes[0].get_root()
        if root_node.is_expandable():
            return root_node

        return None
