import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List

from pydantic import BaseModel, Field, ConfigDict

from moatless.agent.agent import ActionAgent
from moatless.agent.code_agent import CodingAgent
from moatless.completion.model import Usage
from moatless.exceptions import RejectError, RuntimeError
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.node import Node, generate_ascii_tree
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.workspace import Workspace
from moatless.completion.base import BaseCompletionModel
from moatless.agentic_system import AgenticSystem

logger = logging.getLogger(__name__)


class AgenticLoop(AgenticSystem):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self) -> Node:
        """Run the agentic loop until completion or max iterations."""

        current_node = self.root.get_all_nodes()[-1]
        self.log(logger.info, generate_ascii_tree(self.root))

        while not self.is_finished():
            total_cost = self.total_usage().completion_cost
            iteration = len(self.root.get_all_nodes())

            self.log(
                logger.info,
                f"Run iteration {iteration}",
                cost=total_cost,
            )

            if self.max_cost and total_cost and total_cost >= self.max_cost:
                self.log(
                    logger.warning,
                    f"Search cost ${total_cost} exceeded max cost of ${self.max_cost}. Finishing search.",
                )
                break

            try:
                current_node = self._create_next_node(current_node)
                self.agent.run(current_node)
                self.maybe_persist()
                self.log(logger.info, generate_ascii_tree(self.root, current_node))
            except RejectError as e:
                self.log(logger.error, f"Rejection error: {e}")
            except Exception as e:
                self.log(logger.exception, f"Unexpected error: {e}")
                raise e
            finally:
                self.maybe_persist()

        completion_data = {
            "total_iterations": len(self.root.get_all_nodes()),
            "total_cost": self.total_usage().completion_cost,
        }

        return self.get_last_node()

    def _create_next_node(self, parent: Node) -> Node:
        """Create a new node as a child of the parent node."""
        child_node = Node(
            node_id=self._generate_unique_id(),
            parent=parent,
            file_context=parent.file_context.clone() if parent.file_context else None,
        )
        parent.add_child(child_node)
        return child_node

    def is_finished(self) -> bool:
        """Check if the loop should finish."""
        total_cost = self.total_usage().completion_cost
        if self.max_cost and self.total_usage().completion_cost and total_cost >= self.max_cost:
            return True

        nodes = self.root.get_all_nodes()
        if len(nodes) >= self.max_iterations:
            return True

        return nodes[-1].is_terminal()

    def get_last_node(self) -> Node:
        """Get the last node in the action sequence."""
        return self.root.get_all_nodes()[-1]

    @classmethod
    def from_file(cls, file_path: str, persist_path: str | None = None, **kwargs) -> "AgenticLoop":
        """Load a loop instance from a file, but don't initialize with the node."""
        with open(file_path, "r") as f:
            data = json.load(f)

        # Remove root/nodes from loaded data since we'll provide it at runtime
        data.pop("root", None)
        data.pop("nodes", None)

        return cls.from_dict(data, persist_path=persist_path or file_path, **kwargs)
