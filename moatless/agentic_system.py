from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable
import uuid
import traceback
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from moatless.agent.agent import ActionAgent
from moatless.completion.base import BaseCompletionModel
from moatless.completion.model import Usage
from moatless.events import BaseEvent, SystemEvent
from moatless.file_context import FileContext
from moatless.node import Node, generate_ascii_tree
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.index.code_index import CodeIndex
from moatless.workspace import Workspace
from moatless.config.agent_config import get_agent
from moatless.events import event_bus

logger = logging.getLogger(__name__)


class RunAttempt(BaseModel):
    """Information about a single run attempt"""
    attempt_id: int
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    status: str = "running"  # running, error, completed
    error: Optional[str] = None
    error_trace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemStatus(BaseModel):
    """System status information"""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    status: str = "running"  # running, error, completed
    error: Optional[str] = None
    error_trace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    restart_count: int = Field(default=0)
    last_restart: Optional[datetime] = None
    run_history: List[RunAttempt] = Field(default_factory=list)
    current_attempt: Optional[int] = None

    def start_new_attempt(self) -> RunAttempt:
        """Start a new run attempt"""
        attempt = RunAttempt(
            attempt_id=len(self.run_history),
            metadata=self.metadata
        )
        self.run_history.append(attempt)
        self.current_attempt = attempt.attempt_id
        return attempt

    def get_current_attempt(self) -> Optional[RunAttempt]:
        """Get the current run attempt"""
        if self.current_attempt is not None:
            return self.run_history[self.current_attempt]
        return None

    def complete_current_attempt(self, status: str = "completed", error: Optional[str] = None, error_trace: Optional[str] = None):
        """Complete the current attempt"""
        if attempt := self.get_current_attempt():
            attempt.finished_at = datetime.now(timezone.utc)
            attempt.status = status
            attempt.error = error
            attempt.error_trace = error_trace


class AgenticSystem(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Node = Field(..., description="The root node of the system.")
    run_id: str = Field(..., description="The run ID of the system.")
    agent: ActionAgent = Field(..., description="Agent for generating actions.")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")
    persist_path: Optional[str] = Field(None, description="Path to persist the system state.")
    max_iterations: int = Field(10, description="The maximum number of iterations to run.")
    max_cost: Optional[float] = Field(None, description="The maximum cost spent on tokens before finishing.")

    persist_dir: Optional[str] = Field(None, description="Directory to persist system state")
    
    _status: SystemStatus = PrivateAttr(default_factory=SystemStatus)
    _events_file: Optional[Any] = PrivateAttr(default=None)

    @classmethod
    def create(
        cls,
        message: str | None = None,
        run_id: str | None = None,
        agent: ActionAgent | None = None,
        agent_id: str | None = None,
        root: Node | None = None,
        file_context: Optional[FileContext] = None,
        completion_model: BaseCompletionModel | None = None,
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        runtime: RuntimeEnvironment | None = None,
        workspace: Workspace | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        persist_path: Optional[str] = None,
        persist_dir: Optional[str] = None,
        max_iterations: int = 10,
        max_cost: Optional[float] = None,
        **kwargs,
    ) -> "AgenticSystem":
        if not root and not message:
            raise ValueError("Either a root node or a message must be provided.")

        if not agent and not agent_id:
            raise ValueError("Either an agent or an agent ID must be provided.")

        if not run_id:
            run_id = str(uuid.uuid4())

        if not agent:
            agent = get_agent(agent_id, completion_model, repository, code_index, runtime)

        if not agent.workspace:
            if not workspace:
                workspace = Workspace(repository=repository, runtime=runtime, code_index=code_index)
            agent.workspace = workspace
        
        if not file_context:
            file_context = FileContext(repo=agent.workspace.repository, runtime=agent.workspace.runtime)

        if not root:
            root = Node(
                node_id=0,
                user_message=message,
                file_context=file_context,
            )

        return cls(
            run_id=run_id,
            root=root,
            agent=agent,
            metadata=metadata or {},
            persist_path=persist_path,
            persist_dir=persist_dir,
            max_iterations=max_iterations,
            max_cost=max_cost,
            **kwargs,
        )

    async def run(self) -> Node:
        """Run the system with optional root node."""
        try:
            self._initialize_run_state()
            self.agent.set_event_handler(self._handle_agent_event)
            result = await self._run()
            
            # Complete attempt successfully
            self._status.complete_current_attempt("completed")
            self._status.status = "completed"
            self._status.finished_at = datetime.now(timezone.utc)
            return result
        except Exception as e:
            # Complete attempt with error
            error_trace = traceback.format_exc()
            self._status.complete_current_attempt("error", str(e), error_trace)
            self._status.status = "error"
            self._status.error = str(e)
            self._status.error_trace = error_trace
            self._status.finished_at = datetime.now(timezone.utc)
            raise
        finally:
            self._save_status()
            if self._events_file:
                self._events_file.close()
                self._events_file = None
            self.agent.remove_event_handler()

    @abstractmethod
    async def _run(self) -> Node:
        raise NotImplementedError("Subclass must implement _run method")

    async def emit_event(self, event: BaseEvent):
        """Emit an event."""
        logger.info(f"Emit event {event.event_type}")
        self._save_event(self.run_id, event)
        await event_bus.publish(self.run_id, event)
        
    async def _handle_agent_event(self, event: BaseEvent):
        """Handle agent events and propagate them to system event handlers"""
        await self.emit_event(event)

    def _initialize_run_state(self):
        """Initialize or restore system run state and logging"""
        if not self.persist_dir:            
            return

        path = Path(self.persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Initialize or restore status
        status_path = path.joinpath('status.json')
        if status_path.exists():
            try:
                existing_status = SystemStatus.model_validate_json(status_path.read_text())
                
            
                # Resume previous run
                self._status = existing_status
                self._status.status = "running"
                self._status.restart_count += 1
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
            self._status = SystemStatus(
                started_at=datetime.now(timezone.utc),
                status="running",
                metadata=self.metadata
            )

        # Start new attempt
        attempt = self._status.start_new_attempt()
        
        # Setup event logging
        events_path = path.joinpath('events.jsonl')
        self._events_file = open(events_path, 'a', encoding='utf-8')
        
        # Log restart/resume event
        if self._status.restart_count > 0:
            restart_event = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'run_id': self.run_id,
                'event_type': 'system_restarted',
                'data': {
                    'restart_count': self._status.restart_count,
                    'previous_start': self._status.started_at.isoformat(),
                    'metadata': self.metadata,
                    'run_history': [attempt.model_dump() for attempt in self._status.run_history],
                    'resume_type': 'fresh' if self._status.status in ["completed", "error"] else 'resume'
                }
            }
            self._events_file.write(json.dumps(restart_event) + '\n')
            self._events_file.flush()

        self._save_status()

    def _save_status(self):
        """Save current status to status.json"""
        if self.persist_dir:
            status_path = Path(self.persist_dir) / 'status.json'
            self._status.metadata = self.metadata
            status_path.write_text(self._status.model_dump_json(indent=2))

    def _save_event(self, run_id: str, event: BaseEvent):
        """Handle and log system events"""
        if not self.persist_dir:
            return

        event_data = None
        try:
            event_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'run_id': run_id,
                **event.model_dump(),
            }
            self._events_file.write(json.dumps(event_data) + '\n')
            self._events_file.flush()
        except Exception as e:
            if event_data:
                logger.exception(f"Error handling event: {event_data}")
            else:
                logger.exception(f"Error handling event: {event}")

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
        if self.persist_path:
            self.persist(self.persist_path)

    def persist(self, file_path: str):
        """Persist the system state to a file."""
        tree_data = self.model_dump(exclude_none=True)
        with open(file_path, "w") as f:
            try:
                json.dump(tree_data, f, indent=2)
            except Exception as e:
                logger.exception(f"Error saving system to {file_path}: {tree_data}")
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
    ) -> "AgenticSystem":
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
    def from_dict(
        cls,
        data: Dict[str, Any],
        persist_path: str | None = None,
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        runtime: RuntimeEnvironment | None = None,
        workspace: Workspace | None = None,
        completion_model: BaseCompletionModel | None = None,
    ):
        data = data.copy()
        if persist_path:
            data["persist_path"] = persist_path

        if "agent" in data and isinstance(data["agent"], dict):
            agent_data = data["agent"]
            # TODO: To keep backward compatibility
            if "completion_model" in agent_data:
                del agent_data["completion_model"]

            agent = ActionAgent.model_validate(agent_data)
            agent.completion_model = completion_model
            if not workspace and (repository or runtime or code_index):
                workspace = Workspace(repository=repository, runtime=runtime, code_index=code_index)
            agent.workspace = workspace

        return cls.model_validate(
            data,
            repository=workspace.repository if workspace else None,
            runtime=workspace.runtime if workspace else None,
        )

    @classmethod
    def from_file(cls, file_path: str, persist_path: str | None = None, **kwargs) -> "AgenticSystem":
        """Load a system instance from a file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data, persist_path=persist_path or file_path, **kwargs)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Generate a dictionary representation of the system."""
        data = super().model_dump(exclude={"event_handlers", "agent", "root"})

        data.pop("persist_path", None)
        data["agent"] = self.agent.model_dump(**kwargs)
        data["nodes"] = self.root.dump_as_list(**kwargs)

        return data

    def get_status(self) -> SystemStatus:
        """Get the current status of the system."""
        return self._status
