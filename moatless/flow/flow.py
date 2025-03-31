import json
import logging
import uuid
from abc import abstractmethod
from collections.abc import Callable, Awaitable
from pathlib import Path
from typing import Any, Optional

from opentelemetry import trace
from pydantic import ConfigDict, Field, PrivateAttr

from moatless.agent.agent import ActionAgent
from moatless.completion.stats import Usage
from moatless.component import MoatlessComponent
from moatless.context_data import current_project_id, current_trajectory_id
from moatless.events import (
    BaseEvent,
    FlowCompletedEvent,
    FlowErrorEvent,
    FlowStartedEvent,
)
from moatless.flow.schema import FlowStatus
from moatless.node import Node
from moatless.repository.repository import Repository
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.flow")


class AgenticFlow(MoatlessComponent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_id: Optional[str] = Field(None, description="The project ID")
    trajectory_id: Optional[str] = Field(None, description="The trajectory ID.")

    agent: ActionAgent = Field(..., description="Agent for generating actions.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")
    max_iterations: int = Field(10, description="The maximum number of iterations to run.")
    max_cost: Optional[float] = Field(None, description="The maximum cost spent on tokens before finishing.")

    _root: Optional[Node] = PrivateAttr(default=None)
    _on_event: Optional[Callable[[BaseEvent], Awaitable[None]]] = PrivateAttr(default=None)

    @classmethod
    def get_component_type(cls) -> str:
        return "flow"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.flow"

    @classmethod
    def _get_base_class(cls) -> type:
        return AgenticFlow

    @property
    def root(self) -> Node:
        if not self._root:
            raise ValueError("Root node is not set")
        return self._root

    @classmethod
    def create(
        cls,
        message: str | None = None,
        root: Node | None = None,
        trajectory_id: str | None = None,
        project_id: str = "default",
        agent: ActionAgent | None = None,
        agent_id: str | None = None,
        metadata: Optional[dict[str, Any]] = None,
        max_iterations: int = 10,
        max_expansions: Optional[int] = None,
        max_cost: Optional[float] = None,
        shadow_mode: bool = True,
        on_event: Optional[Callable[[BaseEvent], Awaitable[None]]] = None,
        **kwargs,
    ) -> "AgenticFlow":
        if not trajectory_id:
            trajectory_id = str(uuid.uuid4())
        import moatless.settings as settings

        if not agent:
            if not agent_id:
                raise ValueError("Either an agent or an agent ID must be provided.")

            agent = settings.agent_manager.get_agent(agent_id)

        if not root:
            if not message:
                raise ValueError("Either a root node or a message must be provided.")

            root = Node.create_root(
                user_message=message,
                shadow_mode=shadow_mode,
                max_expansions=max_expansions,
            )

        instance = cls(
            trajectory_id=trajectory_id,
            project_id=project_id,
            agent=agent,
            metadata=metadata or {},
            max_iterations=max_iterations,
            max_cost=max_cost,
            **kwargs,
        )

        instance._root = root
        instance._on_event = on_event

        return instance

    @property
    def workspace(self) -> Workspace:
        return self.agent.workspace

    @workspace.setter
    def workspace(self, workspace: Workspace):
        self.agent.workspace = workspace

    async def run(
        self,
        message: str | None = None,
        workspace: Workspace | None = None,
        node_id: int | None = None,
    ) -> Node:
        """Run the system with optional root node."""
        if not self.root:
            raise ValueError("Root node is not set")

        if workspace:
            await self.agent.initialize(workspace)

        self.agent._on_event = self._on_event

        # TODO: Workaround to set repo on existing nodes
        for node in self.root.get_all_nodes():
            if node.file_context and not node.file_context._repo:
                node.file_context.repository = self.agent.workspace.repository
                node.file_context._runtime = self.agent.workspace.runtime

        if not self.agent.workspace:
            raise ValueError("Agent workspace is not set")

        try:
            current_trajectory_id.set(self.trajectory_id)
            current_project_id.set(self.project_id)
            await self._emit_event(FlowStartedEvent())
            node, finish_reason = await self._run(message, node_id)

            await self._emit_event(FlowCompletedEvent())
            return node
        except Exception as e:
            logger.exception(f"Error running flow {self.trajectory_id}")
            await self._emit_event(FlowErrorEvent(data={"error": str(e)}))
            raise

    @abstractmethod
    async def _run(self, message: str | None = None, node_id: int | None = None) -> tuple[Node, str | None]:
        raise NotImplementedError("Subclass must implement _run method")

    def reset_node(self, node_id: int):
        """Reset a specific node.

        Args:
            node_id (int): ID of the node to retry from

        Returns:
            Node: The retried node with new execution results

        Raises:
            ValueError: If node_id is not found
        """
        node = self.get_node_by_id(node_id)
        if not node:
            raise ValueError(f"Node with ID {node_id} not found")

        if not node.parent:
            node.reset()
        else:
            node.parent.children = [child for child in node.parent.children if child.node_id != node.node_id]

    async def _emit_event(self, event: BaseEvent):
        """Emit an event via the event callback if one is configured.

        Args:
            event: The event to emit
        """
        event.project_id = self.project_id
        event.trajectory_id = self.trajectory_id

        if self._on_event:
            await self._on_event(event)
        else:
            # Just log the event if no callback is configured
            logger.debug(f"Event emitted (no callback): {event.event_type} for {self.project_id}/{self.trajectory_id}")

    def get_node_by_id(self, node_id: int) -> Node | None:
        """Get a node by its ID."""
        return next(
            (node for node in self.root.get_all_nodes() if node.node_id == node_id),
            None,
        )

    def total_usage(self) -> Usage:
        """Calculate total token usage across all nodes."""
        return self.root.total_usage()

    @property
    def status(self) -> FlowStatus:
        if self.root.get_all_nodes()[-1].error:
            return FlowStatus.ERROR
        if self.is_finished():
            return FlowStatus.COMPLETED
        elif not self.root.children:
            return FlowStatus.CREATED

        return FlowStatus.RUNNING

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

    def _generate_unique_id(self) -> int:
        """Generate a unique ID for a new node."""
        return len(self.root.get_all_nodes())

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
    ):
        """Validate and reconstruct a system from a dictionary."""
        if isinstance(obj, dict):
            obj = obj.copy()

            if "agent" in obj and isinstance(obj["agent"], dict):
                obj["agent"] = ActionAgent.from_dict(obj["agent"])

            if "root" in obj:
                obj["root"] = Node.reconstruct(obj["root"], repo=repository)
            elif "nodes" in obj:
                obj["root"] = Node.reconstruct(obj["nodes"], repo=repository)
                del obj["nodes"]

        return super().model_validate(obj)

    @classmethod
    def from_dicts(cls, settings: dict[str, Any], trajectory: dict[str, Any]) -> "AgenticFlow":
        """Load a system instance from a dictionary."""
        flow = cls.from_dict(settings)
        flow._root = Node.from_dict(trajectory)
        return flow

    @classmethod
    def from_dir(cls, trajectory_dir: Path) -> "AgenticFlow":
        """Load a system instance from a directory."""
        settings_path = trajectory_dir / "settings.json"
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found in {trajectory_dir}")

        trajectory_path = trajectory_dir / "trajectory.json"
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found in {trajectory_dir}")

        with open(settings_path) as f:
            settings = json.load(f)

        flow = cls.from_dict(settings)
        flow._root = Node.from_file(trajectory_path)
        return flow

    @classmethod
    async def from_trajectory_id(
        cls,
        trajectory_id: str,
        project_id: str,
    ) -> "AgenticFlow":
        from moatless.settings import get_storage

        storage = await get_storage()
        traj_dict = await storage.read_from_trajectory("trajectory.json", project_id, trajectory_id)
        flow_dict = await storage.read_from_trajectory("settings.json", project_id, trajectory_id)
        return cls.from_dicts(flow_dict, traj_dict)

    def get_trajectory_data(self) -> dict:
        """Get trajectory data for persistence."""
        return {
            "nodes": self.root.dump_as_list(exclude_none=True, exclude_unset=True),
        }

    def get_flow_settings(self) -> dict:
        """Get flow settings for persistence."""
        return self.model_dump(exclude_none=True)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Generate a dictionary representation of the system."""
        data = super().model_dump(exclude={"agent", "root"})
        data["agent"] = self.agent.model_dump(**kwargs)
        return data
