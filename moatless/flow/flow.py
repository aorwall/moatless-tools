import json
import logging
import traceback
import uuid
from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from opentelemetry import trace
from pydantic import ConfigDict, Field, PrivateAttr

from moatless.agent.agent import ActionAgent
from moatless.completion.stats import Usage
from moatless.component import MoatlessComponent
from moatless.eventbus.base import BaseEventBus
from moatless.storage.base import BaseStorage
from moatless.context_data import (
    current_project_id,
    current_trajectory_id,
    get_trajectory_dir,
)
from moatless.events import (
    BaseEvent,
    FlowCompletedEvent,
    FlowErrorEvent,
    FlowStartedEvent,
)
from moatless.flow.schema import FlowStatus, FlowStatusInfo
from moatless.node import Node
from moatless.repository.repository import Repository
from moatless.workspace import Workspace


logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.flow")


class AgenticFlow(MoatlessComponent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_id: str = Field(..., description="The project ID")
    trajectory_id: str = Field(..., description="The trajectory ID.")

    agent: ActionAgent = Field(..., description="Agent for generating actions.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")
    max_iterations: int = Field(10, description="The maximum number of iterations to run.")
    max_cost: Optional[float] = Field(None, description="The maximum cost spent on tokens before finishing.")

    _root: Optional[Node] = PrivateAttr(default=None)
    _status: FlowStatusInfo = PrivateAttr(default_factory=FlowStatusInfo)

    _storage: BaseStorage = PrivateAttr()
    _event_bus: BaseEventBus = PrivateAttr()

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import moatless.settings as settings

        self._storage = kwargs.get("storage", settings.storage)
        self._event_bus = kwargs.get("event_bus", settings.event_bus)

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
        storage: BaseStorage | None = None,
        event_bus: BaseEventBus | None = None,
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

        instance._storage = storage or settings.storage
        instance._event_bus = event_bus or settings.event_bus

        instance._root = root
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
        storage: BaseStorage | None = None,
    ) -> Node:
        """Run the system with optional root node."""
        if not self.root:
            raise ValueError("Root node is not set")

        if workspace:
            await self.agent.initialize(workspace)

        if storage:
            self._storage = storage

        # TODO: Workaround to set repo on existing nodes
        for node in self.root.get_all_nodes():
            if node.file_context and not node.file_context._repo:
                node.file_context.repository = self.agent.workspace.repository
                node.file_context._runtime = self.agent.workspace.runtime

        if not self.agent.workspace:
            raise ValueError("Agent workspace is not set")

        with tracer.start_as_current_span(f"flow_{self.trajectory_id}") as span:
            try:
                current_trajectory_id.set(self.trajectory_id)
                current_project_id.set(self.project_id)
                await self._initialize_run_state()
                await self._emit_event(FlowStartedEvent())
                node, finish_reason = await self._run(message)

                # Complete attempt successfully
                self._status.complete_current_attempt("completed")
                self._status.status = FlowStatus.COMPLETED
                self._status.finished_at = datetime.now(timezone.utc)
                await self._emit_event(FlowCompletedEvent())
                return node
            except Exception as e:
                # Complete attempt with error
                logger.exception(f"Error running flow {self.trajectory_id}")
                error_trace = traceback.format_exc()
                self._status.complete_current_attempt("error", str(e), error_trace)
                self._status.status = FlowStatus.ERROR
                self._status.error = str(e)
                self._status.error_trace = error_trace
                self._status.finished_at = datetime.now(timezone.utc)
                await self._emit_event(FlowErrorEvent(data={"error": str(e)}))
                raise
            finally:
                await self.persist()
                await self._save_status()

    @abstractmethod
    async def _run(self, message: str | None = None) -> tuple[Node, str | None]:
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
        event.project_id = self.project_id
        event.trajectory_id = self.trajectory_id

        await self.persist()
        await self._event_bus.publish(event)

    async def _initialize_run_state(self):
        """Initialize or restore system run state and logging"""

        # Initialize or restore status
        if await self._storage.exists_in_trajectory("status", self.project_id, self.trajectory_id):
            try:
                status_data = await self._storage.read_from_trajectory("status", self.project_id, self.trajectory_id)
                existing_status = FlowStatusInfo.model_validate(status_data)

                # Resume previous run
                self._status = existing_status
                self._status.status = FlowStatus.RUNNING
                self._status.restart_count += 1
                self._status.error = None
                self._status.error_trace = None
                self._status.last_restart = datetime.now(timezone.utc)

                # Mark any incomplete attempts as error
                current_attempt = self._status.get_current_attempt()
                if current_attempt and current_attempt.status == "running":
                    current_attempt.status = "error"
                    current_attempt.error = "System interrupted"
                    current_attempt.finished_at = datetime.now(timezone.utc)
                    self._status.current_attempt = None

            except Exception as e:
                logger.error(f"Error loading existing status: {e}")

        if not self._status:
            self._status = FlowStatusInfo()

        self._status.started_at = datetime.now(timezone.utc)
        self._status.status = FlowStatus.RUNNING
        self._status.metadata = self.metadata

        # Start new attempt
        attempt = self._status.start_new_attempt()

        # TODO: Log restart/resume event
        # if self._status.restart_count > 0:
        #    event_bus.publish(self.trajectory_id, BaseEvent(event_type="flow_restarted"))

        await self._save_status()

    async def _save_status(self):
        """Save current status to status.json"""
        self._status.metadata = self.metadata
        await self._storage.write_to_trajectory(
            "status", self._status.model_dump(), self.project_id, self.trajectory_id
        )

    def get_node_by_id(self, node_id: int) -> Node | None:
        """Get a node by its ID."""
        return next(
            (node for node in self.root.get_all_nodes() if node.node_id == node_id),
            None,
        )

    def total_usage(self) -> Usage:
        """Calculate total token usage across all nodes."""
        return self.root.total_usage()

    @abstractmethod
    def is_finished(self) -> str | None:
        raise NotImplementedError("Subclass must implement is_finished method")

    async def persist(self):
        trajectory_data = {
            "nodes": self.root.dump_as_list(exclude_none=True, exclude_unset=True),
        }

        await self._storage.write_to_trajectory("trajectory", trajectory_data, self.project_id, self.trajectory_id)

        # TODO: Only save on creation
        flow_settings = self.model_dump(exclude_none=True)
        await self._storage.write_to_trajectory("settings", flow_settings, self.project_id, self.trajectory_id)

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
    ) -> "AgenticFlow":
        """Validate and reconstruct a system from a dictionary."""
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
    def from_dicts(cls, settings: dict[str, Any], trajectory: dict[str, Any]) -> "AgenticFlow":
        """Load a system instance from a dictionary."""
        flow = cls.model_validate(settings)
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

        status_path = trajectory_dir / "status.json"
        if status_path.exists():
            with open(status_path) as f:
                status = FlowStatusInfo.model_validate_json(f.read())
        else:
            status = FlowStatusInfo(status=FlowStatus.CREATED)

        with open(settings_path) as f:
            settings = json.load(f)

        flow = cls.model_validate(settings)
        flow._root = Node.from_file(trajectory_path)
        flow._status = status
        return flow

    @classmethod
    async def from_trajectory_id(
        cls,
        trajectory_id: str,
        project_id: str | None = None,
        storage: BaseStorage | None = None,
    ) -> "AgenticFlow":
        import moatless.settings as settings

        storage = storage or settings.storage
        traj_dict = await storage.read_from_trajectory("trajectory", project_id, trajectory_id)
        flow_dict = await storage.read_from_trajectory("settings", project_id, trajectory_id)
        return cls.from_dicts(flow_dict, traj_dict)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Generate a dictionary representation of the system."""
        data = super().model_dump(exclude={"agent", "root"})
        data["agent"] = self.agent.model_dump(**kwargs)
        return data

    def get_status(self) -> FlowStatusInfo:
        """Get the current status of the system."""
        return self._status
