from moatless.node import Node, Selection
from moatless.selector.base import BaseSelector


class SimpleSelector(BaseSelector):
    """
    Selects the first expandable node.
    """

    async def select(self, expandable_nodes: list[Node]) -> Selection:
        if not expandable_nodes:
            return Selection(reason="No expandable nodes available")

        return Selection(node_id=expandable_nodes[0].node_id, reason="First expandable node")
