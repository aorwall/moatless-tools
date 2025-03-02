from typing import List
from moatless.node import Node
from moatless.selector.base import BaseSelector


class SimpleSelector(BaseSelector):
    """
    Selects the first expandable node.
    """

    async def select(self, expandable_nodes: List[Node]) -> Node | None:
        if not expandable_nodes:
            return None

        return expandable_nodes[0]
