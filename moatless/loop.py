import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List

from pydantic import BaseModel, Field, ConfigDict

from moatless.agent.agent import ActionAgent
from moatless.completion.model import Usage
from moatless.exceptions import RejectError, RuntimeError
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.node import Node, generate_ascii_tree
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class AgenticLoop(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Node = Field(..., description="The root node of the action sequence.")
    agent: ActionAgent = Field(..., description="Agent for generating actions.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the loop.")
    persist_path: Optional[str] = Field(None, description="Path to persist the action sequence.")
    max_iterations: int = Field(10, description="The maximum number of iterations to run.")
    max_cost: Optional[float] = Field(None, description="The maximum cost spent on tokens before finishing.")
    event_handlers: List[Callable] = Field(
        default_factory=list, description="Event handlers for loop events", exclude=True
    )

    @classmethod
    def create(
        cls,
        message: str | None = None,
        root: Node | None = None,
        file_context: Optional[FileContext] = None,
        workspace: Optional[Workspace] = None,
        repository: Repository | None = None,
        runtime: Optional[RuntimeEnvironment] = None,
        agent: Optional[ActionAgent] = None,
        metadata: Optional[Dict[str, Any]] = None,
        persist_path: Optional[str] = None,
        max_iterations: int = 10,
        max_cost: Optional[float] = None,
    ) -> "AgenticLoop":
        """Create a new AgenticLoop instance."""

        if not file_context:
            file_context = FileContext(repo=repository, runtime=runtime)

        if not root:
            root = Node(
                node_id=0,
                user_message=message,
                file_context=file_context,
                workspace=workspace,
            )

        return cls(
            root=root,
            agent=agent,
            metadata=metadata or {},
            persist_path=persist_path,
            max_iterations=max_iterations,
            max_cost=max_cost,
        )

    def run(self):
        """Run the agentic loop until completion or max iterations."""
        self.assert_runnable()

        current_node = self.get_last_node()
        self.log(logger.info, generate_ascii_tree(self.root))

        self.emit_event("loop_started", {})

        while not self.is_finished():
            total_cost = self.total_usage().completion_cost

            self.log(
                logger.info,
                f"Run iteration {len(self.root.get_all_nodes())}",
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

                # Emit iteration event
                self.emit_event(
                    "loop_iteration",
                    {
                        "iteration": len(self.root.get_all_nodes()),
                        "total_cost": total_cost,
                        "action": current_node.action.name if current_node.action else None,
                        "current_node_id": current_node.node_id,
                        "total_nodes": len(self.root.get_all_nodes()),
                    },
                )

            except RejectError as e:
                self.log(logger.error, f"Rejection error: {e}")
                self.emit_event("loop_error", {"error": str(e)})
            except Exception as e:
                self.log(logger.exception, f"Unexpected error: {e}")
                self.emit_event("loop_error", {"error": str(e)})
                raise e
            finally:
                self.maybe_persist()

        self.emit_event(
            "loop_completed",
            {
                "total_iterations": len(self.root.get_all_nodes()),
                "total_cost": self.total_usage().completion_cost,
            },
        )

        return self.get_last_node()

    def _create_next_node(self, parent: Node) -> Node:
        """Create a new node as a child of the parent node."""
        child_node = Node(
            node_id=self._generate_unique_id(),
            parent=parent,
            workspace=parent.workspace.clone() if parent.workspace else None,
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

    def get_node_by_id(self, node_id: int) -> Node | None:
        return next(
            (node for node in self.root.get_all_nodes() if node.node_id == node_id),
            None,
        )

    def total_usage(self) -> Usage:
        """Calculate total token usage across all nodes."""
        return self.root.total_usage()

    def maybe_persist(self):
        """Persist the loop state if a persist path is set."""
        if self.persist_path:
            self.persist(self.persist_path)

    def persist(self, file_path: str):
        """Persist the loop state to a file."""
        tree_data = self.model_dump(exclude_none=True)
        with open(file_path, "w") as f:
            import json

            json.dump(tree_data, f, indent=2)

    def _generate_unique_id(self) -> int:
        """Generate a unique ID for a new node."""
        return len(self.root.get_all_nodes())

    def assert_runnable(self):
        """Verify that the loop is properly configured to run."""
        if self.root is None:
            raise ValueError("AgenticLoop must have a root node.")

        if self.agent is None:
            raise ValueError("AgenticLoop must have an agent.")

        return True

    def log(self, logger_fn: Callable, message: str, **kwargs):
        """Log a message with metadata."""
        metadata = {**self.metadata, **kwargs}
        metadata_str = " ".join(f"{k}: {str(v)[:20]}" for k, v in metadata.items())
        log_message = f"[{metadata_str}] {message}" if metadata else message
        logger_fn(log_message)

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        repository: Repository | None = None,
    ) -> "AgenticLoop":
        """Validate and reconstruct an AgenticLoop from a dictionary."""
        if isinstance(obj, dict):
            obj = obj.copy()

            if "agent" in obj and isinstance(obj["agent"], dict):
                obj["agent"] = ActionAgent.model_validate(obj["agent"])

            if "root" in obj:
                obj["root"] = Node.reconstruct(obj["root"], repo=repository)
            elif "nodes" in obj:
                obj["root"] = Node.reconstruct(obj["nodes"], repo=repository)
                del obj["nodes"]

        return super().model_validate(obj)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        persist_path: str | None = None,
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        runtime: RuntimeEnvironment | None = None,
    ) -> "AgenticLoop":
        """Create an AgenticLoop instance from a dictionary."""
        data = data.copy()
        if persist_path:
            data["persist_path"] = persist_path

        if "agent" in data and isinstance(data["agent"], dict):
            agent_data = data["agent"]
            data["agent"] = ActionAgent.model_validate(
                agent_data,
                repository=repository,
                code_index=code_index,
                runtime=runtime,
            )
        return cls.model_validate(data, repository=repository)

    @classmethod
    def from_file(cls, file_path: str, persist_path: str | None = None, **kwargs) -> "AgenticLoop":
        """Load an AgenticLoop instance from a file."""

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            return cls.from_dict(data, persist_path=persist_path or file_path, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load AgenticLoop from {file_path}: {e}")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Generate a dictionary representation of the AgenticLoop."""
        # Get all fields except the ones we'll handle separately
        data = super().model_dump(exclude={"event_handlers", "agent", "root"})

        data.pop("persist_path", None)
        data["agent"] = self.agent.model_dump(**kwargs)
        data["nodes"] = self.root.dump_as_list(**kwargs)

        return data

    def add_event_handler(self, handler: Callable):
        """Add an event handler for loop events."""
        self.event_handlers.append(handler)

    def emit_event(self, event_type: str, data: dict):
        """Emit an event to all registered handlers."""
        logger.info(f"Emit event {event_type}")
        for handler in self.event_handlers:
            handler(
                {
                    "event_type": event_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                }
            )
