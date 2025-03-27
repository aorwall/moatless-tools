"""Tree search configuration management."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from moatless.actions.action import CompletionModelMixin
from moatless.context_data import get_trajectory_dir
from moatless.eventbus import BaseEventBus
from moatless.expander import Expander
from moatless.flow import AgenticFlow, AgenticLoop, SearchTree
from moatless.flow.run_flow import run_flow
from moatless.flow.schema import (
    CompletionDTO,
    FlowConfig,
    TrajectoryEventDTO,
    TrajectoryResponseDTO,
    StartTrajectoryRequest,
)
from moatless.flow.trajectory_tree import create_node_tree
from moatless.node import Node
from moatless.runner.runner import BaseRunner, JobStatus
from moatless.storage.base import BaseStorage
from moatless.utils.moatless import get_moatless_dir

logger = logging.getLogger(__name__)


class FlowManager:
    def __init__(
        self,
        runner: BaseRunner,
        storage: BaseStorage,
        eventbus: BaseEventBus,
        agent_manager,
        model_manager,
    ):
        self._configs = {}
        self._runner = runner
        self._storage = storage
        self._eventbus = eventbus
        self._agent_manager = agent_manager
        self._model_manager = model_manager

    async def initialize(self):
        await self._load_configs()

    async def create_flow(
        self,
        id: str,
        model_id: str,
        message: str | None = None,
        trajectory_id: str | None = None,
        project_id: str | None = None,
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
        if not config:
            raise ValueError(f"Flow config {id} not found")

        agent = self._agent_manager.get_agent(agent_id=config.agent_id)
        completion_model = self._model_manager.create_completion_model(model_id)
        agent.completion_model = completion_model
        for action in agent.actions:
            if isinstance(action, CompletionModelMixin):
                action.completion_model = completion_model

        if not project_id:
            project_id = "default"

        if not trajectory_id:
            trajectory_id = str(uuid.uuid4())

        if config.flow_type == "loop":
            flow = AgenticLoop.create(
                message=message,
                trajectory_id=trajectory_id,
                project_id=project_id,
                agent=agent,
                max_iterations=config.max_iterations,
                max_cost=config.max_cost,
                persist_dir=persist_dir,
                metadata=metadata,
                shadow_mode=False,
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

            flow = SearchTree.create(
                message=message,
                trajectory_id=trajectory_id,
                project_id=project_id,
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

        await self.save_trajectory(project_id, trajectory_id, flow)

        return flow

    def _get_config_path(self) -> Path:
        """Get the path to the config file."""
        return get_moatless_dir() / "flows.json"

    def _get_global_config_path(self) -> Path:
        """Get the path to the global config file."""
        return Path(__file__).parent / "flows.json"

    async def _load_configs(self):
        """Load configurations from JSON file."""
        try:
            configs = await self._storage.read("flows.json")
        except KeyError:
            logger.warning("No flow configs found")
            configs = []

        # Copy global config to local path if it doesn't exist
        if not configs:
            try:
                global_path = self._get_global_config_path()
                if global_path.exists():
                    with open(global_path) as f:
                        global_config = json.load(f)
                        await self._storage.write("flows.json", global_config)
                    logger.info("Copied global config to local path")
                else:
                    logger.info("No global tree configs found")
            except Exception as e:
                logger.error(f"Failed to copy global tree configs: {e}")

        logger.info(f"Loading {len(configs)} flow configs")
        for config in configs:
            try:
                self._configs[config["id"]] = config
                logger.debug(f"Loaded flow config {config['id']}")
            except Exception:
                logger.exception(f"Failed to load flow config {config['id']}")

    async def _save_configs(self):
        """Save configurations to JSON file."""
        path = self._get_config_path()
        try:
            configs = list(self._configs.values())
            await self._storage.write("flows.json", configs)
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

    async def create_config(self, config: FlowConfig) -> FlowConfig:
        """Create a new flow configuration."""
        logger.debug(f"Creating flow config {config.id}")
        if config.id in self._configs:
            raise ValueError(f"Flow config {config.id} already exists")

        self._configs[config.id] = config.model_dump()
        await self._save_configs()
        return config

    async def update_config(self, config: FlowConfig):
        """Update an existing flow configuration."""
        logger.debug(f"Updating flow config {config.id}")
        if config.id not in self._configs:
            raise ValueError(f"Flow config {config.id} not found")

        self._configs[config.id] = config.model_dump()
        await self._save_configs()

    async def delete_config(self, id: str):
        """Delete a flow configuration."""
        logger.debug(f"Deleting flow config {id}")
        if id not in self._configs:
            raise ValueError(f"Flow config {id} not found")

        del self._configs[id]
        await self._save_configs()

    async def get_flow(self, project_id: str, trajectory_id: str) -> AgenticFlow:
        """Get a flow from a trajectory."""
        await self._storage.assert_exists_in_trajectory("trajectory.json", project_id, trajectory_id)

        trajectory_data = await self._storage.read_from_trajectory("trajectory.json", project_id, trajectory_id)
        settings_data = await self._storage.read_from_trajectory("settings.json", project_id, trajectory_id)
        return AgenticFlow.from_dicts(settings_data, trajectory_data)

    async def get_trajectory(self, project_id: str, trajectory_id: str) -> "TrajectoryResponseDTO":
        """Get the status, trajectory data, and events for a specific trajectory."""
        await self._storage.assert_exists_in_trajectory("trajectory.json", project_id, trajectory_id)

        try:
            trajectory_data = await self._storage.read_from_trajectory("trajectory.json", project_id, trajectory_id)
            settings_data = await self._storage.read_from_trajectory("settings.json", project_id, trajectory_id)
            flow = AgenticFlow.from_dicts(settings_data, trajectory_data)

            job_status = await self._runner.get_job_status(project_id=project_id, trajectory_id=trajectory_id)

            return TrajectoryResponseDTO(
                trajectory_id=flow.trajectory_id,
                project_id=flow.project_id,
                status=flow.status,
                job_status=job_status,
                agent_id=flow.agent.agent_id,
                model_id=flow.agent.model_id,
                usage=flow.total_usage(),
            )
        except Exception as e:
            logger.exception(f"Error getting trajectory data: {str(e)}")
            raise ValueError(f"Error getting trajectory data: {str(e)}")

    async def list_trajectories(self, project_id: str | None = None) -> list:
        """Get all trajectories."""

        raise NotImplementedError("Not implemented yet")

    async def start_trajectory(self, project_id: str, trajectory_id: str):
        """Start a trajectory."""

        flow = await AgenticFlow.from_trajectory_id(trajectory_id, project_id)

        job_status = await self._runner.get_job_status(project_id, trajectory_id)
        if job_status == JobStatus.RUNNING:
            raise ValueError("Flow is already running")

        if flow.root.get_last_node() and flow.root.get_last_node().error:
            logger.info(f"Resetting node {flow.root.get_last_node().node_id} with error")
            flow.root.get_last_node().reset()
            trajectory_path = self._storage.get_trajectory_path(project_id, trajectory_id)
            await self._storage.write(f"{trajectory_path}/trajectory.json", flow.get_trajectory_data())

        await self._runner.start_job(project_id=project_id, trajectory_id=trajectory_id, job_func=run_flow)

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

        flow = await AgenticFlow.from_trajectory_id(trajectory_id, project_id)

        # Reset the trajectory by removing all children from the root node
        if flow.root and flow.root.children:
            logger.info(f"Removing {len(flow.root.children)} children from root node")
            flow.root.children = []
            trajectory_path = self._storage.get_trajectory_path(project_id, trajectory_id)
            await self._storage.write(f"{trajectory_path}/trajectory.json", flow.get_trajectory_data())
            logger.info("Trajectory reset successfully")

        await self._runner.start_job(project_id=project_id, trajectory_id=trajectory_id, job_func=run_flow)
        logger.info(f"Trajectory {trajectory_id} restarted successfully")

        return flow

    async def resume_trajectory(self, project_id: str, trajectory_id: str, request: StartTrajectoryRequest):
        """Resume a trajectory with additional parameters.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            request: The request containing parameters to resume the trajectory

        Returns:
            The resumed flow
        """
        logger.info(f"Resuming trajectory {trajectory_id} in project {project_id}")

        job_status = await self._runner.get_job_status(project_id, trajectory_id)
        if job_status == JobStatus.RUNNING:
            raise ValueError("Cannot resume a trajectory that is already running")

        flow = await AgenticFlow.from_trajectory_id(trajectory_id, project_id)

        # Update flow with any new parameters from the request
        if request.message:
            flow.message = request.message

        if request.metadata:
            flow.metadata.update(request.metadata)

        trajectory_path = self._storage.get_trajectory_path(project_id, trajectory_id)
        await self._storage.write(f"{trajectory_path}/trajectory.json", flow.get_trajectory_data())
        logger.info("Trajectory updated with new parameters")

        await self._runner.start_job(project_id=project_id, trajectory_id=trajectory_id, job_func=run_flow)
        logger.info(f"Trajectory {trajectory_id} resumed successfully")

        return flow

    async def execute_node(self, project_id: str, trajectory_id: str, node_id: int):
        """Execute a specific node in a trajectory."""

        agentic_flow = await AgenticFlow.from_trajectory_id(trajectory_id=trajectory_id, project_id=project_id)

        node = agentic_flow.root.get_node_by_id(node_id)
        if not node:
            raise ValueError("Node not found")
        if node.is_executed():
            node = node.clone()
            node.reset(rebuild_action_steps=False)
            logger.info(f"Cloned node {node.node_id} from parent {node.parent.node_id}")

            trajectory_path = self._storage.get_trajectory_path(project_id, trajectory_id)
            await self._storage.write(f"{trajectory_path}/trajectory.json", agentic_flow.get_trajectory_data())

        await self._runner.start_job(
            project_id=project_id,
            trajectory_id=trajectory_id,
            job_func=run_flow,
            node_id=node_id,
        )

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

    async def get_trajectory_logs(self, project_id: str, trajectory_id: str, file_name: Optional[str] = None) -> Dict:
        """Get logs for a trajectory.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            file_name: Optional specific log file name to retrieve

        Returns:
            Dict containing the log contents and metadata
        """
        file_list = []

        # First, try to get logs from the runner if it's a job log file or no specific file is requested
        if not file_name or file_name == "job.log":
            runner_logs = await self._runner.get_job_logs(project_id, trajectory_id)
            logger.info(f"Runner logs: {runner_logs}")
            if runner_logs:
                # Get the list of local log files
                logs_dir = get_trajectory_dir(project_id, trajectory_id) / "logs"
                file_list = [{"name": "job.log"}]

                # Add local log files to the file list if they exist
                if logs_dir.exists():
                    local_log_files = sorted(
                        list(logs_dir.glob("*.log")), key=lambda p: p.stat().st_mtime, reverse=True
                    )
                    file_list.extend([{"name": f.name} for f in local_log_files])

                return {"logs": runner_logs, "files": file_list, "current_file": "job.log", "source": "runner"}

        log_dir_path = f"projects/{project_id}/trajs/{trajectory_id}/logs"

        logger.info(f"Listing log files for {self._storage}")
        log_files = await self._storage.list_paths(log_dir_path)
        logger.info(f"Log files: {log_files}")
        if not log_files:
            return {"logs": "", "files": [], "current_file": None}

        # Get a list of all log files with their modified times
        file_list = [{"name": f.split("/")[-1]} for f in log_files]
        file_list.sort(key=lambda x: x["name"], reverse=True)

        # Add the job log file to the list if we have a runner
        if await self._runner.job_exists(project_id, trajectory_id):
            file_list.insert(0, {"name": "job.log"})

        # If a specific file is requested, try to find it
        if file_name:
            if file_name == "job.log":
                # This case would have been handled above, but if we reached here,
                # it means the runner logs weren't available
                raise ValueError("Job logs not available")

            target_file = f"{log_dir_path}/{file_name}"
            if not await self._storage.exists(target_file):
                raise ValueError(f"Log file {file_name} not found")
            current_file = target_file
        else:
            # Otherwise return the most recent log file
            current_file = log_files[0]

        log_content = await self._storage.read(current_file)
        current_file_name = current_file.split("/")[-1]

        return {"logs": log_content, "files": file_list, "current_file": current_file_name}

    async def get_completions(self, project_id: str, trajectory_id: str, node_id: str, action_step: int | None = None):
        """Get the completions for a specific node.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            node_id: The node ID to get completions for

        Returns:
            A list of parsed CompletionDTO objects
        """
        from moatless.utils.completion_parser import parse_completion

        trajectory_key = self._storage.get_trajectory_path(project_id, trajectory_id)

        if action_step is not None:
            log_keys = await self._storage.list_paths(
                f"{trajectory_key}/completions/node_{node_id}_action_{action_step}/"
            )
        else:
            log_keys = await self._storage.list_paths(f"{trajectory_key}/completions/node_{node_id}/")

        completions = []

        logger.info(f"Found {len(log_keys)} completions for node {node_id} and action {action_step}: {log_keys}")

        for log_key in log_keys:
            logger.info(f"Reading completion from {log_key}")
            raw_completion = await self._storage.read(log_key)

            try:
                completion_dto = parse_completion(raw_completion)
                completions.append(completion_dto)
            except Exception as e:
                logger.exception(f"Error parsing completion: {e}")
                # Still add a basic DTO with original data for debugging
                completions.append(
                    CompletionDTO(
                        original_input=raw_completion.get("original_input"),
                        original_output=raw_completion.get("original_response"),
                    )
                )

        return completions

    async def get_node_tree(self, project_id: str, trajectory_id: str) -> dict:
        node = await self.read_trajectory_node(project_id, trajectory_id)
        tree_node = create_node_tree(node)
        return tree_node.model_dump()

    async def get_node(self, project_id: str, trajectory_id: str, node_id: int) -> dict:
        await self._storage.assert_exists_in_trajectory("trajectory.json", project_id, trajectory_id)

        root_node = await self.read_trajectory_node(project_id, trajectory_id)
        node = root_node.get_node_by_id(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found in trajectory {trajectory_id}")

        return node.model_dump(exclude={"children", "parent"})

    async def read_trajectory_node(self, project_id: str, trajectory_id: str) -> Node:
        await self._storage.assert_exists_in_trajectory("trajectory.json", project_id, trajectory_id)

        trajectory_data = await self._storage.read_from_trajectory("trajectory.json", project_id, trajectory_id)

        return Node.from_dict(trajectory_data)

    async def save_trajectory(self, project_id: str, trajectory_id: str, flow: AgenticFlow):
        trajectory_path = self._storage.get_trajectory_path(project_id, trajectory_id)

        await self._storage.write(f"{trajectory_path}/trajectory.json", flow.get_trajectory_data())
        await self._storage.write(f"{trajectory_path}/settings.json", flow.get_flow_settings())
