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
from moatless.completion.base import BaseCompletionModel
from moatless.completion.model import Usage
from moatless.component import MoatlessComponent
from moatless.config.agent_config import get_agent
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
    event_bus,
)
from moatless.file_context import FileContext
from moatless.flow.events import FlowErrorEvent
from moatless.flow.schema import FlowStatus, FlowStatusInfo
from moatless.index.code_index import CodeIndex
from moatless.node import Node
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.flow")


class AgenticFlow(MoatlessComponent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_id: str | None = Field(None, description="The project ID")
    trajectory_id: str = Field(..., description="The trajectory ID.")

    agent: ActionAgent = Field(..., description="Agent for generating actions.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata."
    )
    persist_path: Optional[str] = Field(
        None, description="Path to persist the system state."
    )
    max_iterations: int = Field(
        10, description="The maximum number of iterations to run."
    )
    max_cost: Optional[float] = Field(
        None, description="The maximum cost spent on tokens before finishing."
    )

    _persist_dir: Optional[Path] = PrivateAttr(default=None)
    _root: Optional[Node] = PrivateAttr(default=None)
    _status: FlowStatusInfo = PrivateAttr(default_factory=FlowStatusInfo)
    _events_file: Optional[Any] = PrivateAttr(default=None)

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
        persist_path: Optional[str] = None,
        persist_dir: Union[str, Path] | None = None,
        max_iterations: int = 10,
        max_expansions: Optional[int] = None,
        max_cost: Optional[float] = None,
        shadow_mode: bool = True,
        **kwargs,
    ) -> "AgenticFlow":
        if not trajectory_id:
            trajectory_id = str(uuid.uuid4())

        if not agent:
            if not agent_id:
                raise ValueError("Either an agent or an agent ID must be provided.")
            agent = get_agent(agent_id)

        if not root:
            if not message:
                raise ValueError("Either a root node or a message must be provided.")

            root = Node.create_root(
                user_message=message,
                shadow_mode=shadow_mode,
                max_expansions=max_expansions,
            )

        if isinstance(persist_dir, str):
            persist_dir = Path(persist_dir)

        if not persist_dir:
            persist_dir = get_trajectory_dir(
                trajectory_id=trajectory_id, project_id=project_id
            )

        instance = cls(
            trajectory_id=trajectory_id,
            project_id=project_id,
            agent=agent,
            metadata=metadata or {},
            persist_path=persist_path,
            max_iterations=max_iterations,
            max_cost=max_cost,
            **kwargs,
        )

        instance._root = root
        instance._persist_dir = persist_dir
        return instance

    @property
    def workspace(self) -> Workspace:
        return self.agent.workspace

    @workspace.setter
    def workspace(self, workspace: Workspace):
        self.agent.workspace = workspace

    async def run(
        self, message: str | None = None, workspace: Workspace | None = None
    ) -> Node:
        """Run the system with optional root node."""
        if not self.root:
            raise ValueError("Root node is not set")

        if workspace:
            await self.agent.initialize(workspace)

        # TODO: Workaround to set repo on existing nodes
        for node in self.root.get_all_nodes():
            if node.file_context and not node.file_context._repo:
                node.file_context.repository = self.agent.workspace.repository

        if not self.agent.workspace:
            raise ValueError("Agent workspace is not set")

        with tracer.start_as_current_span(f"flow_{self.trajectory_id}") as span:
            try:
                current_trajectory_id.set(self.trajectory_id)
                current_project_id.set(self.project_id)
                self._initialize_run_state()
                await event_bus.publish(FlowStartedEvent())
                node, finish_reason = await self._run(message)

                # Complete attempt successfully
                self._status.complete_current_attempt("completed")
                self._status.status = FlowStatus.COMPLETED
                self._status.finished_at = datetime.now(timezone.utc)
                await event_bus.publish(FlowCompletedEvent())
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
                await event_bus.publish(FlowErrorEvent(error=str(e)))
                raise
            finally:
                self.maybe_persist()
                self._save_status()
                if self._events_file:
                    self._events_file.close()
                    self._events_file = None

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
            node.parent.children = [
                child for child in node.parent.children if child.node_id != node.node_id
            ]

    async def emit_event(self, event: BaseEvent):
        self.maybe_persist()
        logger.info(f"Emit event {event.event_type}")
        await event_bus.publish(event)

    def _initialize_run_state(self):
        """Initialize or restore system run state and logging"""
        if not self._persist_dir:
            return

        if not self._persist_dir.exists():
            self._persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or restore status
        status_path = self._persist_dir / "status.json"
        if status_path.exists():
            try:
                existing_status = FlowStatusInfo.model_validate_json(
                    status_path.read_text()
                )

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

        self._save_status()

    def _save_status(self):
        """Save current status to status.json"""
        if self._persist_dir:
            status_path = self._persist_dir / "status.json"
            self._status.metadata = self.metadata
            status_path.write_text(self._status.model_dump_json(indent=2))

    def get_node_by_id(self, node_id: int) -> Node | None:
        """Get a node by its ID."""
        return next(
            (node for node in self.root.get_all_nodes() if node.node_id == node_id),
            None,
        )

    def total_usage(self) -> Usage:
        """Calculate total token usage across all nodes."""
        return self.root.total_usage()

    def maybe_persist(self):
        """Persist the system state if a persist path is set."""
        if self._persist_dir:
            self.persist(self._persist_dir)

    def persist(self, persist_dir: Path | None = None):
        """Persist the system state to a file."""
        if not persist_dir:
            persist_dir = self._persist_dir

        if not persist_dir:
            raise ValueError("Persist directory is not set")

        trajectory_data = {
            "nodes": self.root.dump_as_list(exclude_none=True, exclude_unset=True),
        }

        self._save_file(persist_dir / "trajectory.json", trajectory_data)

        flow_settings = self.model_dump(exclude_none=True)
        self._save_file(persist_dir / "settings.json", flow_settings)

    def _save_file(self, file_path: Path, data: dict[str, Any]):
        with open(file_path, "w") as f:
            try:
                json.dump(data, f, indent=2)
            except Exception as e:
                logger.exception(f"Error saving system to {file_path}: {data}")
                raise e

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
    def from_dir(
        cls, trajectory_dir: Path, workspace: Workspace | None = None
    ) -> "AgenticFlow":
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
        flow._root = Node.from_file(
            trajectory_path,
            repo=workspace.repository if workspace else None,
            runtime=workspace.runtime if workspace else None,
        )

        if workspace:
            flow.workspace = workspace
        flow._persist_dir = trajectory_dir
        flow._status = status
        return flow

    @abstractmethod
    def is_finished(self) -> str | None:
        raise NotImplementedError("Subclass must implement is_finished method")

    @classmethod
    def from_trajectory_id(
        cls,
        trajectory_id: str,
        project_id: str | None = None,
        workspace: Workspace | None = None,
    ) -> "AgenticFlow":
        trajectory_dir = get_trajectory_dir(
            trajectory_id=trajectory_id, project_id=project_id
        )
        if not workspace:
            workspace = Workspace(trajectory_dir=trajectory_dir)

        return cls.from_dir(trajectory_dir, workspace)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Generate a dictionary representation of the system."""
        data = super().model_dump(
            exclude={"agent", "root", "persist_dir", "persist_path"}
        )
        data["agent"] = self.agent.model_dump(**kwargs)
        return data

    def get_status(self) -> FlowStatusInfo:
        """Get the current status of the system."""
        return self._status
