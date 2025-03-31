import logging

from pydantic import ConfigDict

from moatless.context_data import current_node_id
from moatless.exceptions import RejectError, RuntimeError
from moatless.flow import AgenticFlow
from moatless.flow.events import NodeExpandedEvent
from moatless.node import Node, generate_ascii_tree

logger = logging.getLogger(__name__)


class OneOffFlow(AgenticFlow):
    """Just runs one node and returns the result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _run(self, message: str | None = None, node_id: int | None = None) -> tuple[Node, str | None]:
        """Run the agentic loop until completion or max iterations."""

        current_node = self.root.get_all_nodes()[-1]

        if message:  # Assume to continue with a new node if a message is provided
            current_node = self._create_next_node(current_node)
            current_node.user_message = message

        await self.agent.run(current_node)

        return self.get_last_node(), None

    def _create_next_node(self, parent: Node) -> Node:
        """Create a new node as a child of the parent node."""
        child_node = Node(  # type: ignore
            node_id=self._generate_unique_id(),
            parent=parent,
            file_context=parent.file_context.clone() if parent.file_context else None,
        )
        parent.add_child(child_node)
        return child_node

    def get_last_node(self) -> Node:
        """Get the last node in the action sequence."""
        return self.root.get_all_nodes()[-1]
