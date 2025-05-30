import logging

from pydantic import ConfigDict

from moatless.context_data import current_node_id
from moatless.exceptions import RejectError, RuntimeError
from moatless.flow import AgenticFlow
from moatless.flow.events import NodeExpandedEvent
from moatless.node import Node, generate_ascii_tree

logger = logging.getLogger(__name__)


class AgenticLoop(AgenticFlow):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _run(self, message: str | None = None, node_id: int | None = None) -> tuple[Node, str | None]:
        """Run the agentic loop until completion or max iterations."""

        if node_id:
            current_node = self.get_node_by_id(node_id)
            if not current_node:
                raise ValueError(f"Node with ID {node_id} not found")
        else:
            current_node = self.root.get_last_node()

        if message:  # Assume to continue with a new node if a message is provided
            current_node = self._create_next_node(current_node)
            current_node.user_message = message

        finish_reason = None
        while node_id or not (finish_reason := self.is_finished()):
            total_cost = self.total_usage().completion_cost
            iteration = len(self.root.get_all_nodes())

            self.log(
                logger.info,
                f"Run iteration {iteration}",
                cost=total_cost,
            )

            try:
                if current_node.is_expandable() and current_node.is_executed():
                    child_node = self._create_next_node(current_node)
                    await self._emit_event(
                        NodeExpandedEvent(node_id=current_node.node_id, child_node_id=child_node.node_id)
                    )
                    current_node = child_node

                if current_node.is_executed():
                    raise RuntimeError(f"Node {current_node.node_id} has already been executed")

                current_node_id.set(current_node.node_id)
                await self.agent.run(current_node)
                self.log(logger.debug, generate_ascii_tree(self.root, current_node))
            except RejectError as e:
                self.log(logger.error, f"Rejection error: {e}")
            except Exception as e:
                self.log(logger.exception, f"Unexpected error: {e}")
                raise e
            finally:
                pass

            if node_id:
                self.log(logger.info, f"Node{current_node.node_id} finished. Returning.")
                break

        logger.info(
            f"Loop finished with {len(self.root.get_all_nodes())} iterations and {self.total_usage().completion_cost} cost"
        )

        return self.get_last_node(), finish_reason

    def _create_next_node(self, parent: Node) -> Node:
        """Create a new node as a child of the parent node."""
        child_node = Node(  # type: ignore
            node_id=self._generate_unique_id(),
            parent=parent,
            file_context=parent.file_context.clone() if parent.file_context else None,
        )
        parent.add_child(child_node)
        return child_node

    def is_finished(self) -> str | None:
        """Check if the loop should finish."""
        total_cost = self.total_usage().completion_cost
        if self.max_cost and self.total_usage().completion_cost and total_cost >= self.max_cost:
            return "max_cost"

        nodes = self.root.get_all_nodes()
        if len(nodes) >= self.max_iterations:
            return "max_iterations"

        if nodes[-1].is_terminal():
            return "terminal"

        return None

    def get_last_node(self) -> Node:
        """Get the last node in the action sequence."""
        return self.root.get_all_nodes()[-1]
