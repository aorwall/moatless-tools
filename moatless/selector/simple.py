from typing import List

from pydantic import Field

from moatless.node import Node
from moatless.selector import BaseSelector


class SimpleSelector(BaseSelector):
    """Select first not expanded node that is not terminal. Then select root again if expandable."""

    def select(self, expandable_nodes: List[Node]) -> Node | None:
        if not expandable_nodes:
            return None

        for node in expandable_nodes:
            if node.expanded_count() == 0 and not node.terminal:
                return node

        root_node = expandable_nodes[0].get_root()
        if root_node.is_expandable():
            return root_node

        return None
