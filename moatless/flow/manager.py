"""Tree search configuration management."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from moatless.actions.list_files import ListFilesArgs
from moatless.actions.read_files import ReadFiles, ReadFilesArgs
from moatless.actions.view_diff import ViewDiffArgs
from moatless.completion.json import JsonCompletionModel
from moatless.actions.action import CompletionModelMixin
from moatless.completion.react import ReActCompletionModel
from moatless.context_data import get_trajectory_dir
from moatless.environment.local import LocalBashEnvironment
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
    FlowStatus,
)
from moatless.flow.trajectory_tree import create_node_tree
from moatless.index.code_index import CodeIndex
from moatless.node import ActionStep, Node
from moatless.repository.git import GitRepository
from moatless.runner.runner import BaseRunner, JobStatus
from moatless.storage.base import BaseStorage
from moatless.utils.moatless import get_moatless_dir
from moatless.workspace import Workspace

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

    async def build_flow(
        self,
        id: str,
        model_id: str,
    ) -> AgenticFlow:
        """Create a SearchTree instance from this configuration.

        Args:
            root_node: Optional root node for the search tree. If not provided,
                      the tree will need to be initialized with a root node later.

        Returns:
            SearchTree: A configured search tree instance
        """
        # Create expander if not specified

        flow = self.get_flow_config(id)
        if not flow:
            raise ValueError(f"Flow config {id} not found")

        if not flow.agent:
            if not flow.agent_id:
                raise ValueError(f"Flow config {id} has no agent")
            
            flow.agent = self._agent_manager.get_agent(agent_id=flow.agent_id)
            if not flow.agent:
                raise ValueError(f"Agent {flow.agent_id} not found")
            
        if not flow.agent.completion_model:
            if not model_id:
                raise ValueError(f"Flow config {id} has no model")
            
            flow.agent.completion_model = self._model_manager.create_completion_model(model_id)

        json_completion_model = JsonCompletionModel(
            model=flow.agent.completion_model.model,
            temperature=0.0,
            max_tokens=flow.agent.completion_model.max_tokens,
            timeout=flow.agent.completion_model.timeout,
            few_shot_examples=flow.agent.completion_model.few_shot_examples,
            headers=flow.agent.completion_model.headers,
            params=flow.agent.completion_model.params,
            merge_same_role_messages=flow.agent.completion_model.merge_same_role_messages,
            message_cache=flow.agent.completion_model.message_cache,
            model_base_url=flow.agent.completion_model.model_base_url,
            model_api_key=flow.agent.completion_model.model_api_key,
        )

        for action in flow.agent.actions:
            if isinstance(action, CompletionModelMixin) and not action.completion_model:
                action.completion_model = json_completion_model.clone()

        if hasattr(flow, "value_function") and hasattr(flow.value_function, "completion_model") and not flow.value_function.completion_model:
            flow.value_function.completion_model = json_completion_model.clone()
        
        return flow

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

        json_completion_model = JsonCompletionModel(
            model=completion_model.model,
            temperature=0.0,
            max_tokens=completion_model.max_tokens,
            timeout=completion_model.timeout,
            few_shot_examples=completion_model.few_shot_examples,
            headers=completion_model.headers,
            params=completion_model.params,
            merge_same_role_messages=completion_model.merge_same_role_messages,
            thoughts_in_action=completion_model.thoughts_in_action,
            disable_thoughts=completion_model.disable_thoughts,
            message_cache=completion_model.message_cache,
            model_base_url=completion_model.model_base_url,
            model_api_key=completion_model.model_api_key,
        )

        for action in agent.actions:
            if isinstance(action, CompletionModelMixin):
                action.completion_model = json_completion_model.clone()

        if hasattr(config.value_function, "completion_model") and not config.value_function.completion_model:
            config.value_function.completion_model = json_completion_model.clone()

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
                max_expansions=config.max_expansions or 1,
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

    async def create_loop(
        self,
        trajectory_id: str,
        project_id: str,
        agent_id: str,
        model_id: str,
        message: str,
        repository_path: str | None = None,
        **kwargs,
    ) -> AgenticLoop:
        logger.info(f"Creating loop for project {project_id} with trajectory {trajectory_id}")
        agent = self._agent_manager.get_agent(agent_id=agent_id)
        completion_model = self._model_manager.create_completion_model(model_id)
        agent.completion_model = completion_model

        if isinstance(completion_model, ReActCompletionModel):
            extra_completion_model = JsonCompletionModel(
                model=completion_model.model,
                temperature=0.0,
                max_tokens=completion_model.max_tokens,
                timeout=completion_model.timeout,
                few_shot_examples=completion_model.few_shot_examples,
                headers=completion_model.headers,
                params=completion_model.params,
                merge_same_role_messages=completion_model.merge_same_role_messages,
                thoughts_in_action=completion_model.thoughts_in_action,
                disable_thoughts=completion_model.disable_thoughts,
                message_cache=completion_model.message_cache,
                model_base_url=completion_model.model_base_url,
                model_api_key=completion_model.model_api_key,
            )
        else:
            extra_completion_model = completion_model.clone()

        for action in agent.actions:
            if isinstance(action, CompletionModelMixin):
                action.completion_model = extra_completion_model.clone()

        if repository_path:
            repository = GitRepository(repo_path=repository_path)
            environment = LocalBashEnvironment(cwd=repository_path)

            workspace = Workspace(repository=repository, environment=environment)
            agent.workspace = workspace

        loop = AgenticLoop.create(
            message=message,
            trajectory_id=trajectory_id,
            project_id=project_id,
            agent=agent,
            max_iterations=50,
            max_cost=1.0,
            shadow_mode=False,
            **kwargs,
        )

        project_settings = await self.read_project_settings(project_id)
        if not project_settings:
            project_settings = {
                "project_id": project_id,
                "agent_id": agent_id,
                "model_id": model_id,
                "repository_path": repository_path,
            }

            await self.save_project(project_id, project_settings)
            
        #loop.root.action_steps = [
        #    ActionStep(
        #        action = ViewDiffArgs(
        #        ) # type: ignore
        #    )
        #]
        
        await self.save_trajectory(project_id, trajectory_id, loop)

        return loop

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

    def get_flow_config(self, id: str) -> AgenticFlow:
        """Get a flow configuration by ID."""
        logger.debug(f"Getting flow config {id}")

        if id in self._configs:
            config = self._configs[id]
            # Ensure the id is set in the config
            config["id"] = id
            return AgenticFlow.from_dict(config)
        else:
            logger.error(f"Flow config {id} not found. Available configs: {list(self._configs.keys())}")
            raise ValueError(f"Flow config {id} not found. Available configs: {list(self._configs.keys())}")

    def get_all_configs(self) -> list[AgenticFlow]:
        """Get all flow configurations."""

        configs = []
        for config in self._configs.values():
            try:
                configs.append(AgenticFlow.from_dict(config))
            except Exception as e:
                logger.exception(f"Failed to load flow config {config['id']}: {e}")

        configs.sort(key=lambda x: x.id)
        return configs

    async def create_config(self, config: AgenticFlow) -> AgenticFlow:
        """Create a new flow configuration."""
        logger.debug(f"Creating flow config {config.id}")
        if config.id in self._configs:
            raise ValueError(f"Flow config {config.id} already exists")

        self._configs[config.id] = config.model_dump()
        await self._save_configs()
        return config

    async def update_config(self, config: AgenticFlow):
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

    def get_trajectory_status(self, flow: AgenticFlow, job_status: JobStatus) -> FlowStatus:
        if job_status == JobStatus.RUNNING:
            return FlowStatus.RUNNING
        elif job_status == JobStatus.PENDING:
            return FlowStatus.PENDING
        elif flow.root.get_all_nodes()[-1].error:
            return FlowStatus.ERROR
        elif flow.is_finished():
            return FlowStatus.COMPLETED
        elif not flow.root.children:
            return FlowStatus.CREATED
        else:
            return FlowStatus.PAUSED

    async def get_trajectory(self, project_id: str, trajectory_id: str) -> "TrajectoryResponseDTO":
        """Get the status, trajectory data, and events for a specific trajectory."""
        await self._storage.assert_exists_in_trajectory("trajectory.json", project_id, trajectory_id)

        try:
            trajectory_data = await self._storage.read_from_trajectory("trajectory.json", project_id, trajectory_id)
            
            try:
                settings_data = await self._storage.read_from_trajectory("settings.json", project_id, trajectory_id)
            except Exception:
                settings_data = await self._storage.read_from_project("flow.json", project_id)
            
            flow = AgenticFlow.from_dicts(settings_data, trajectory_data)

            job_status = await self._runner.get_job_status(project_id=project_id, trajectory_id=trajectory_id)

            evaluated_nodes = [
                node
                for node in flow.root.get_leaf_nodes()
                if node.evaluation_result and node.evaluation_result.resolved is not None
            ]
            if evaluated_nodes:
                resolved = all(node.evaluation_result.resolved for node in evaluated_nodes)
            else:
                resolved = None

            return TrajectoryResponseDTO(
                trajectory_id=flow.trajectory_id,
                project_id=flow.project_id,
                status=self.get_trajectory_status(flow, job_status),
                job_status=job_status,
                resolved=resolved,
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

        last_node = flow.root.get_last_node()
        if last_node.is_executed():
            last_node.terminal = False

            child_node = Node(
                node_id=last_node.node_id + 1,
                parent=last_node,
                file_context=last_node.file_context.clone() if last_node.file_context else None,
                max_expansions=last_node.max_expansions,
                agent_id=last_node.agent_id,
                user_message=request.message,
            )  # type: ignore

            last_node.add_child(child_node)

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

        logger.info(f"Executing node {node.node_id} with is_executed: {node.is_executed()}")
        if node.is_executed():
            new_node = node.clone()
            new_node.reset(rebuild_action_steps=False)
            logger.info(f"Cloned node {new_node.node_id} from parent {node.node_id}")

            trajectory_path = self._storage.get_trajectory_path(project_id, trajectory_id)
            await self._storage.write(f"{trajectory_path}/trajectory.json", agentic_flow.get_trajectory_data())

            node_id = new_node.node_id

        await self._runner.start_job(
            project_id=project_id,
            trajectory_id=trajectory_id,
            job_func=run_flow,
            node_id=node_id,
        )

        return node

    async def reset_node(self, project_id: str, trajectory_id: str, node_id: int):
        """Reset a specific node in a trajectory."""
        agentic_flow = await AgenticFlow.from_trajectory_id(trajectory_id=trajectory_id, project_id=project_id)
        node = agentic_flow.root.get_node_by_id(node_id)
        if not node:
            raise ValueError("Node not found")

        node.reset()

        trajectory_path = self._storage.get_trajectory_path(project_id, trajectory_id)
        await self._storage.write(f"{trajectory_path}/trajectory.json", agentic_flow.get_trajectory_data())

        return node

    async def get_trajectory_settings(self, project_id: str, trajectory_id: str) -> dict:
        """Get the settings for a trajectory."""
        trajectory_path = self._storage.get_trajectory_path(project_id, trajectory_id)
        return await self._storage.read(f"{trajectory_path}/settings.json")

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

    async def get_completions(self, project_id: str, trajectory_id: str, node_id: str, item_id: str):
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

        log_keys = await self._storage.list_paths(f"{trajectory_key}/completions/node_{node_id}/{item_id}/")

        completions = []

        logger.info(f"Found {len(log_keys)} completions for node {node_id} and item {item_id}: {log_keys}")

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

    async def read_project_settings(self, project_id: str) -> dict | None:
        if not await self._storage.exists(f"projects/{project_id}/settings.json"):
            return None

        return await self._storage.read(f"projects/{project_id}/settings.json")

    async def save_project(self, project_id: str, settings: dict):
        await self._storage.write(f"projects/{project_id}/settings.json", settings)

    async def save_trajectory(self, project_id: str, trajectory_id: str, flow: AgenticFlow):
        trajectory_path = self._storage.get_trajectory_path(project_id, trajectory_id)

        await self._storage.write(f"{trajectory_path}/trajectory.json", flow.get_trajectory_data())
        await self._storage.write(f"{trajectory_path}/settings.json", flow.get_flow_settings())

    async def get_node_evaluation_files(self, project_id: str, trajectory_id: str, node_id: int) -> dict[str, str]:
        """Get the evaluation files for a specific node.

        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            node_id: The node ID to get evaluation files for

        Returns:
            A dictionary mapping file names to file contents
        """
        # Verify node exists
        await self._storage.assert_exists_in_trajectory("trajectory.json", project_id, trajectory_id)
        root_node = await self.read_trajectory_node(project_id, trajectory_id)
        node = root_node.get_node_by_id(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found in trajectory {trajectory_id}")

        # Get the path to the evaluation files
        trajectory_key = self._storage.get_trajectory_path(project_id, trajectory_id)
        eval_dir = f"{trajectory_key}/evaluation/node_{node_id}/"

        # Check if evaluation directory exists
        if not await self._storage.exists(eval_dir):
            # Return empty if no evaluation files
            return {}

        # List evaluation files
        file_paths = await self._storage.list_paths(eval_dir)

        # Read each file and build result dictionary
        result = {}
        for file_path in file_paths:
            file_name = file_path.split("/")[-1]
            try:
                file_content = await self._storage.read_raw(file_path)
                result[file_name] = file_content
            except Exception as e:
                logger.exception(f"Error reading evaluation file {file_path}: {e}")
                result[file_name] = f"Error reading file: {str(e)}"

        return result

    async def save_trajectory_settings(self, project_id: str, trajectory_id: str, settings: dict):
        """Save settings for a trajectory.
        
        Args:
            project_id: The project ID
            trajectory_id: The trajectory ID
            settings: The settings data to save
            
        Raises:
            ValueError: If the trajectory does not exist
        """
        trajectory_path = self._storage.get_trajectory_path(project_id, trajectory_id)
        
        # Check if the trajectory exists
        if not await self._storage.exists(f"{trajectory_path}/trajectory.json"):
            raise ValueError(f"Trajectory {trajectory_id} not found in project {project_id}")
            
        # Save the settings
        await self._storage.write(f"{trajectory_path}/settings.json", settings)
        
        return settings
