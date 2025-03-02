"""Tree search configuration management."""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from moatless.agent.agent import ActionAgent
from moatless.api.trajectory.schema import TrajectoryDTO
from moatless.api.trajectory.trajectory_utils import convert_nodes
from moatless.benchmark.swebench import create_index_async
from moatless.benchmark.swebench.utils import create_repository_async
from moatless.completion.manager import create_completion_model
from moatless.config.agent_config import get_agent
from moatless.context_data import get_projects_dir, get_trajectory_dir
from moatless.discriminator.base import BaseDiscriminator
from moatless.environment.local import LocalBashEnvironment
from moatless.evaluation.utils import get_swebench_instance
from moatless.expander import Expander
from moatless.feedback.base import BaseFeedbackGenerator
from moatless.flow import AgenticFlow, AgenticLoop, SearchTree
from moatless.flow.run_flow import run_flow
from moatless.flow.schema import (
    ExecuteNodeRequest,
    FlowConfig,
    FlowStatus,
    FlowStatusInfo,
    TrajectoryEventDTO,
    TrajectoryListItem,
    TrajectoryResponseDTO,
)
from moatless.repository.git import GitRepository
from moatless.runner.rq import RQRunner
from moatless.runner.runner import JobStatus
from moatless.runtime.testbed import TestbedEnvironment
from moatless.selector.base import BaseSelector
from moatless.utils.moatless import get_moatless_dir, get_moatless_trajectories_dir
from moatless.value_function.base import BaseValueFunction
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class FlowManager:
    """Manages tree search configurations."""

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the FlowManager."""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the tree config manager."""
        self._configs = {}
        self._load_configs()
        self._runner = RQRunner()

    def create_flow(
        self,
        id: str,
        model_id: str,
        message: str | None = None,
        trajectory_id: str | None = None,
        persist_dir: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> AgenticFlow:
        """Create a SearchTree instance from this configuration.

        Args:
            root_node: Optional root node for the search tree. If not provided,
                      the tree will need to be initialized with a root node later.

        Returns:
            SearchTree: A configured search tree instance
        """
        # Create expander if not specified

        config = self.get_flow_config(id)

        agent = get_agent(agent_id=config.agent_id)
        completion_model = create_completion_model(model_id)
        agent.completion_model = completion_model

        if config.flow_type == "loop":
            return AgenticLoop.create(
                message=message,
                trajectory_id=trajectory_id,
                agent=agent,
                max_iterations=config.max_iterations,
                max_cost=config.max_cost,
                persist_dir=persist_dir,
                metadata=metadata,
                **kwargs,
            )
        elif config.flow_type == "tree":
            expander = config.expander or Expander(max_expansions=config.max_expansions)

            if hasattr(config.value_function, "model_id") and not config.value_function.model_id:
                config.value_function.model_id = model_id

            if hasattr(config.feedback_generator, "model_id") and not config.feedback_generator.model_id:
                config.feedback_generator.model_id = model_id

            if hasattr(config.discriminator, "model_id") and not config.discriminator.model_id:
                config.discriminator.model_id = model_id

            if hasattr(config.selector, "model_id") and not config.selector.model_id:
                config.selector.model_id = model_id

            tree = SearchTree.create(
                message=message,
                trajectory_id=trajectory_id,
                agent=agent,
                selector=config.selector,
                expander=expander,
                value_function=config.value_function,
                feedback_generator=config.feedback_generator,
                discriminator=config.discriminator,
                max_iterations=config.max_iterations,
                max_cost=config.max_cost,
                min_finished_nodes=config.min_finished_nodes,
                max_finished_nodes=config.max_finished_nodes,
                reward_threshold=config.reward_threshold,
                max_depth=config.max_depth,
                persist_dir=persist_dir,
                metadata=metadata,
                **kwargs,
            )

        return tree

    def create_loop(
        self, agent_id: str, model_id: str, message: str | None = None, trajectory_id: str | None = None
    ) -> AgenticLoop:
        agent = get_agent(agent_id=agent_id)
        completion_model = create_completion_model(model_id)
        agent.completion_model = completion_model
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_id = f"loop_{agent_id}_{model_id}_{date_str}"

        return AgenticLoop.create(
            message=message, trajectory_id=trajectory_id, agent=agent, max_iterations=20, max_cost=1.0
        )

    def _get_config_path(self) -> Path:
        """Get the path to the config file."""
        return get_moatless_dir() / "flows.json"

    def _get_global_config_path(self) -> Path:
        """Get the path to the global config file."""
        return Path(__file__).parent / "flows.json"

    def _load_configs(self):
        """Load configurations from JSON file."""
        config_path = self._get_config_path()

        # Copy global config to local path if it doesn't exist
        if not config_path.exists():
            try:
                global_path = self._get_global_config_path()
                if global_path.exists():
                    # Copy global config to local path
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(global_path) as f:
                        global_config = json.load(f)
                        with open(config_path, "w") as local_f:
                            json.dump(global_config, local_f, indent=2)
                    logger.info("Copied global config to local path")
                else:
                    logger.info("No global tree configs found")
            except Exception as e:
                logger.error(f"Failed to copy global tree configs: {e}")

        # Load configs from local path
        try:
            if config_path.exists():
                logger.info(f"Loading flow configs from {config_path}")
                with open(config_path) as f:
                    configs = json.load(f)
                logger.info(f"Loaded {len(configs)} flow configs")
                for config in configs:
                    try:
                        self._configs[config["id"]] = config
                        logger.info(f"Loaded flow config {config['id']}")
                    except Exception:
                        logger.exception(f"Failed to load flow config {config['id']} from {config_path}")
            else:
                logger.info(f"No local flow configs found on path {config_path}")
        except Exception as e:
            logger.exception(f"Failed to load flow config from {config_path}")
            raise e

    def _save_configs(self):
        """Save configurations to JSON file."""
        path = self._get_config_path()
        try:
            configs = list(self._configs.values())
            logger.info(f"Saving flow configs to {path}")
            with open(path, "w") as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save flow configs: {e}")

    def get_flow_config(self, id: str) -> FlowConfig:
        """Get a flow configuration by ID."""
        logger.debug(f"Getting flow config {id}")

        if id in self._configs:
            config = self._configs[id]
            # Ensure the id is set in the config
            config["id"] = id
            return FlowConfig.model_validate(config)
        else:
            raise ValueError(f"Flow config {id} not found. Available configs: {list(self._configs.keys())}")

    def get_all_configs(self) -> list[FlowConfig]:
        """Get all flow configurations."""

        configs = []
        for config in self._configs.values():
            try:
                configs.append(FlowConfig.model_validate(config))
            except Exception as e:
                logger.exception(f"Failed to load flow config {config['id']}: {e}")

        configs.sort(key=lambda x: x.id)
        return configs

    def create_config(self, config: FlowConfig) -> FlowConfig:
        """Create a new flow configuration."""
        logger.debug(f"Creating flow config {config.id}")
        if config.id in self._configs:
            raise ValueError(f"Flow config {config.id} already exists")

        self._configs[config.id] = config.model_dump()
        self._save_configs()
        return config

    def update_config(self, config: FlowConfig):
        """Update an existing flow configuration."""
        logger.debug(f"Updating flow config {config.id}")
        if config.id not in self._configs:
            raise ValueError(f"Flow config {config.id} not found")

        self._configs[config.id] = config.model_dump()
        self._save_configs()

    def delete_config(self, id: str):
        """Delete a flow configuration."""
        logger.debug(f"Deleting flow config {id}")
        if id not in self._configs:
            raise ValueError(f"Flow config {id} not found")

        del self._configs[id]
        self._save_configs()

    async def get_trajectory(self, project_id: str, trajectory_id: str) -> "TrajectoryResponseDTO":
        """Get the status, trajectory data, and events for a specific trajectory."""

        try:
            trajectory_dir = get_trajectory_dir(project_id=project_id, trajectory_id=trajectory_id)
            flow = AgenticFlow.from_dir(trajectory_dir)
            if not flow:
                logger.error(f"Trajectory not found: {trajectory_dir}")
                raise ValueError("Trajectory not found")

            nodes = convert_nodes(flow.root)
            flow_status_info = flow.get_status()

            if not flow.is_finished():
                job_status = await self._runner.get_job_status(project_id, trajectory_id)
                if job_status == JobStatus.RUNNING:
                    flow_status = FlowStatus.RUNNING
                elif job_status == JobStatus.QUEUED:
                    flow_status = FlowStatus.PENDING
                elif len(flow.root.children) == 0:
                    flow_status = FlowStatus.CREATED
                else:
                    flow_status = FlowStatus.PAUSED
            elif flow.root.get_last_node() and flow.root.get_last_node().error:
                flow_status = FlowStatus.ERROR
            else:
                flow_status = flow_status_info.status

            logger.info(f"Trajectory {trajectory_id} status: {flow_status}: {flow.total_usage()}")
            return TrajectoryResponseDTO(
                id=trajectory_id,
                trajectory_id=trajectory_id,
                project_id=project_id,
                status=flow_status,
                system_status=flow_status_info,
                agent_id=flow_status_info.metadata.get("agent_id"),
                model_id=flow_status_info.metadata.get("model_id"),
                nodes=nodes,
                usage=flow.total_usage(),
            )
        except Exception as e:
            logger.exception(f"Error getting trajectory data: {str(e)}")
            raise ValueError(f"Error getting trajectory data: {str(e)}")

    async def list_trajectories(self, project_id: str | None = None) -> list:
        """Get all trajectories."""

        moatless_dir = get_moatless_dir()
        trajectories = []
        logger.info(f"Listing trajectories in {moatless_dir}")

        for project_dir in get_projects_dir().iterdir():
            logger.info(f"Project directory: {project_dir}")
            if project_dir.is_dir():
                for trajectory_id in os.listdir(project_dir):
                    trajectory_path = project_dir / trajectory_id
                    project_id = project_dir.name
                    if trajectory_path.is_dir():
                        try:
                            status_path = trajectory_path / "status.json"
                            if status_path.exists():
                                status = FlowStatusInfo.model_validate_json(status_path.read_text())
                            else:
                                status = FlowStatusInfo(status=FlowStatus.CREATED)
                            if status:
                                # Create TrajectoryListItem from status
                                trajectory_item = TrajectoryListItem(
                                    **status.model_dump(), project_id=project_id, trajectory_id=trajectory_id
                                )
                                trajectories.append(trajectory_item)
                        except Exception as e:
                            logger.error(f"Error loading trajectory {trajectory_id}: {e}")

        return trajectories

    async def start_trajectory(self, project_id: str, trajectory_id: str):
        """Start a trajectory."""

        flow = AgenticFlow.from_trajectory_id(trajectory_id, project_id)

        job_status = await self._runner.get_job_status(project_id, trajectory_id)
        if job_status == JobStatus.RUNNING:
            raise ValueError("Flow is already running")

        if flow.root.get_last_node() and flow.root.get_last_node().error:
            logger.info(f"Resetting node {flow.root.get_last_node().node_id} with error")
            flow.root.get_last_node().reset()
            flow.persist()

        await self._runner.start_job(project_id=project_id, trajectory_id=trajectory_id, job_func=run_flow)

    async def resume_trajectory(self, project_id: str, trajectory_id: str, request):
        """Resume a trajectory."""
        from moatless.flow.loop import AgenticLoop
        from moatless.flow.runner import agentic_runner

        system = await agentic_runner.get_run(trajectory_id, project_id)
        if system:
            raise ValueError("Flow is already running")

        agentic_flow = AgenticLoop.from_trajectory_id(trajectory_id, project_id, request.agent_id, request.model_id)

        await agentic_runner.start(agentic_flow, message=request.message)
        return agentic_flow

    async def retry_trajectory(self, project_id: str, trajectory_id: str):
        """Reset and restart a trajectory by removing all children from the root node.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID

        Returns:
            The reset and restarted flow
        """
        logger.info(f"Resetting trajectory {trajectory_id} in project {project_id}")

        job_status = await self._runner.get_job_status(project_id, trajectory_id)
        if job_status == JobStatus.RUNNING:
            raise ValueError("Cannot retry a trajectory that is already running")

        flow = AgenticFlow.from_trajectory_id(trajectory_id, project_id)

        # Reset the trajectory by removing all children from the root node
        if flow.root and flow.root.children:
            logger.info(f"Removing {len(flow.root.children)} children from root node")
            flow.root.children = []
            flow.persist()
            logger.info("Trajectory reset successfully")

        await self._runner.start_job(project_id=project_id, trajectory_id=trajectory_id, job_func=run_flow)
        logger.info(f"Trajectory {trajectory_id} restarted successfully")

        return flow

    async def execute_node(self, project_id: str, trajectory_id: str, request: ExecuteNodeRequest):
        """Execute a specific node in a trajectory."""

        agentic_flow = AgenticFlow.from_trajectory_id(trajectory_id=trajectory_id, project_id=project_id)
        # TODO: This is for testing purposes, the node should be executed by a worker!
        repository = GitRepository(repo_path=str(Path.cwd()))
        workspace = Workspace(repository=repository, environment=LocalBashEnvironment())
        await agentic_flow.agent.initialize(workspace)

        node = agentic_flow.root.get_node_by_id(request.node_id)
        if not node:
            raise ValueError("Node not found")

        # Clone file context from parent node to reset file context
        if node.parent and node.parent.file_context:
            node.file_context = node.parent.file_context.clone()
            logger.info(
                f"Cloned file context from parent node {node.parent.node_id} to node {node.node_id}, setting repo to {repository}"
            )
            node.file_context.repository = repository
        else:
            logger.warning("No parent node found, cannot clone file context")
            node.file_context = None

        # Execute the action step
        return await agentic_flow.agent._execute_action_step(node, node.action_steps[0])

    async def _setup_flow(self, trajectory_id: str, project_id: str):
        """Set up a flow for execution."""
        moatless_instance = get_swebench_instance(trajectory_id)
        if moatless_instance:
            logger.info(f"Setting up swebench for trajectory {trajectory_id}")

            repository = await create_repository_async(moatless_instance)
            code_index = await create_index_async(moatless_instance, repository=repository)

            runtime = TestbedEnvironment(
                repository=repository,
                instance_id=trajectory_id,
                log_dir=str(get_trajectory_dir(trajectory_id=trajectory_id, project_id=project_id) / "testbed_logs"),
                enable_cache=True,
            )
            workspace = Workspace(repository=repository, code_index=code_index, runtime=runtime, legacy_workspace=True)
            return AgenticFlow.from_trajectory_id(trajectory_id, project_id, workspace=workspace)
        else:
            return AgenticFlow.from_trajectory_id(trajectory_id, project_id)

    def get_trajectory_path(self, project_id: str, trajectory_id: str) -> Path:
        """Get the trajectory file path for a run."""
        return get_trajectory_dir(project_id, trajectory_id) / "trajectory.json"

    def load_trajectory_events(self, trajectory_dir: Path) -> list:
        """Load events from events.jsonl file."""

        events_path = trajectory_dir / "events.jsonl"
        events = []

        if events_path.exists():
            try:
                with open(events_path, encoding="utf-8") as f:
                    for line in f:
                        event_data = json.loads(line)
                        # Convert ISO timestamp to milliseconds, ensuring UTC
                        dt = datetime.fromisoformat(event_data["timestamp"])
                        if dt.tzinfo is None:
                            # If timestamp has no timezone, assume UTC
                            dt = dt.replace(tzinfo=timezone.utc)
                        event_data["timestamp"] = int(dt.timestamp() * 1000)
                        events.append(TrajectoryEventDTO(**event_data))
            except Exception as e:
                logger.error(f"Error reading events file: {e}")

        return events

    def load_trajectory_from_file(self, file_path: Path) -> TrajectoryDTO:
        """Load trajectory data from a JSON file."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Convert nodes to DTO format
        root_node = data.get("root", {})
        if root_node:
            data["root"] = convert_nodes(root_node)

        return TrajectoryDTO(**data)

    async def get_trajectory_logs(self, project_id: str, trajectory_id: str, file_name: Optional[str] = None) -> Dict:
        """Get logs for a trajectory.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            file_name: Optional specific log file name to retrieve

        Returns:
            Dict containing the log contents and metadata
        """
        trajectory_dir = get_trajectory_dir(project_id, trajectory_id)
        logs_dir = trajectory_dir / "logs"

        if not logs_dir.exists():
            return {"logs": "", "files": [], "current_file": None}

        log_files = sorted(list(logs_dir.glob("*.log")), key=lambda p: p.stat().st_mtime, reverse=True)

        if not log_files:
            return {"logs": "", "files": [], "current_file": None}

        # Get a list of all log files with their modified times
        file_list = [
            {
                "name": f.name,
                "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
            for f in log_files
        ]

        # If a specific file is requested, try to find it
        if file_name:
            target_file = logs_dir / file_name
            if not target_file.exists() or not target_file.is_file():
                raise ValueError(f"Log file {file_name} not found")
            current_file = target_file
        else:
            # Otherwise return the most recent log file
            current_file = log_files[0]

        # Read the log contents
        with open(current_file, "r") as f:
            log_content = f.read()

        return {
            "logs": log_content,
            "files": file_list,
            "current_file": current_file.name,
        }


_manager = FlowManager.get_instance()

get_flow_config = _manager.get_flow_config
get_all_configs = _manager.get_all_configs
create_flow_config = _manager.create_config
update_flow_config = _manager.update_config
delete_flow_config = _manager.delete_config
create_flow = _manager.create_flow
