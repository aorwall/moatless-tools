import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from moatless.evaluation.run_golden_patch import evaluate_golden_patch
from moatless.evaluation.run_instance import run_swebench_instance
from moatless.evaluation.schema import (
    Evaluation,
    EvaluationInstance,
    EvaluationStats,
    EvaluationStatus,
    InstanceStatus,
    ExecutionStatus,
    ResolutionStatus,
    EvaluationSummary,
    RepoStats,
)
from moatless.evaluation.utils import get_swebench_instance
from moatless.eventbus.base import BaseEventBus
from moatless.events import BaseEvent
from moatless.flow.flow import AgenticFlow
from moatless.flow.manager import FlowManager
from moatless.flow.search_tree import SearchTree
from moatless.node import EvaluationResult, Node
from moatless.runner.runner import BaseRunner, JobStatus
from moatless.storage.base import BaseStorage
from opentelemetry import trace


logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.evaluation.manager")


class EvaluationManager:
    def __init__(
        self,
        runner: BaseRunner,
        storage: BaseStorage,
        eventbus: BaseEventBus,
        flow_manager: FlowManager,
    ):
        self.runner = runner
        self.storage = storage
        self.eventbus = eventbus
        self._flow_manager = flow_manager
        self._cached_datasets = {}
        self._initialized = False
        self._subscribed_to_events = False

    async def initialize(self):
        if self._initialized:
            logger.info("EvaluationManager already initialized, skipping")
            return

        if not self._subscribed_to_events:
            await self.eventbus.subscribe(self._handle_event)
            self._subscribed_to_events = True
            logger.info("EvaluationManager subscribed to events")

        self._initialized = True

    async def create_evaluation(
        self,
        flow_id: str | None = None,
        model_id: str | None = None,
        litellm_model_name: str | None = None,
        flow_config: AgenticFlow | None = None,
        evaluation_name: str | None = None,
        dataset_name: str | None = None,
        instance_ids: list[str] | None = None,
    ) -> Evaluation:
        """Create a new evaluation and return its ID."""
        if not dataset_name and not instance_ids:
            raise ValueError("Either dataset_name or instance_ids must be provided")
        
        if not dataset_name:
            dataset_name = "instance_ids"

        if not evaluation_name:
            evaluation_name = (
                f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{flow_id}_{dataset_name}"
            )

        logger.info(f"Creating evaluation: {evaluation_name}")

        if not instance_ids:
            instance_ids = self.get_dataset_instance_ids(dataset_name)
            logger.info(f"Found {len(instance_ids)} instances in dataset {dataset_name}")

        if await self.storage.exists_in_project("evaluation.json", project_id=evaluation_name):
            raise ValueError("Evaluation already exists")

        flow = await self._flow_manager.build_flow(flow_id=flow_id, model_id=model_id, litellm_model_name=litellm_model_name)

        evaluation = Evaluation(
            evaluation_name=evaluation_name,
            dataset_name=dataset_name,
            status=EvaluationStatus.PENDING,
            flow_id=flow_id,
            model_id=model_id,
            flow=flow,
            instances=[
                EvaluationInstance(instance_id=instance_id)  # Will default to CREATED state
                for instance_id in instance_ids
            ],
        )

        await self.storage.write_to_project(
            "flow.json",
            flow.model_dump(),
            project_id=evaluation.evaluation_name,
        )

        for instance in evaluation.instances:
            swebench_instance = get_swebench_instance(instance_id=instance.instance_id)
            repo_name = swebench_instance["repo"].split("/")[-1]
            problem_statement = (
                f"You're tasks is to solve an issue reported in the project {repo_name}. The repository is cloned in the directory /testbed which is the current working directory. "
                f"The reported issue is:\n{swebench_instance['problem_statement']}"
            )

            root_node = Node.create_root(user_message=problem_statement)

            trajectory_path = self.storage.get_trajectory_path(evaluation.evaluation_name, instance.instance_id)
            trajectory_data = {
                "trajectory_id": instance.instance_id,
                "project_id": evaluation.evaluation_name,
                "nodes": root_node.dump_as_list(exclude_none=True, exclude_unset=True),
                "metadata": {
                    "instance_id": instance.instance_id,
                },
            }

            await self.storage.write(f"{trajectory_path}/trajectory.json", trajectory_data)

        await self._save_evaluation(evaluation)
        logger.info(f"Evaluation created: {evaluation_name} with {len(evaluation.instances)} instances")
        return evaluation

    async def clone_evaluation(self, evaluation_name: str) -> Evaluation:
        """Clone an existing evaluation."""
        evaluation = await self._load_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")

        new_evaluation_name = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{evaluation.flow_id}_{evaluation.dataset_name}"
        logger.info(
            f"Cloning evaluation {evaluation_name} to {new_evaluation_name} with {len(evaluation.instances)} instances, dataset {evaluation.dataset_name}"
        )

        if evaluation.dataset_name == "instance_ids":
            instance_ids = [i.instance_id for i in evaluation.instances]
            logger.info(f"Cloning evaluation {evaluation_name} with {len(instance_ids)} instances")
        else:
            instance_ids = None

        return await self.create_evaluation(
            evaluation.flow_id,
            evaluation.model_id,
            evaluation_name=new_evaluation_name,
            dataset_name=evaluation.dataset_name,
            instance_ids=instance_ids,
        )

    @tracer.start_as_current_span("EvaluationManager.start_evaluation")
    async def start_evaluation(self, evaluation_name: str) -> Evaluation:
        """Start an evaluation by running all instances."""
        evaluation = await self.get_evaluation(evaluation_name)
        started_instances = 0

        for instance in evaluation.instances:
            if instance.is_evaluated():
                continue

            # Start the job using the wrapper function
            if await self.runner.start_job(
                project_id=evaluation.evaluation_name,
                trajectory_id=instance.instance_id,
                job_func=run_swebench_instance,
            ):
                started_instances += 1
                instance.start()  # This will set execution_status to QUEUED

        if started_instances > 0:
            # Update evaluation status if needed
            if evaluation.status in [
                EvaluationStatus.PENDING,
                EvaluationStatus.PAUSED,
                EvaluationStatus.ERROR,
            ]:
                evaluation.status = EvaluationStatus.RUNNING
                if not evaluation.started_at:
                    evaluation.started_at = datetime.now(timezone.utc)

            # Save the updated evaluation
            await self.save_evaluation(evaluation)

        return evaluation

    async def get_evaluation(self, evaluation_name: str, sync: bool = False) -> Evaluation:
        """Get an evaluation by name."""
        # Load the evaluation from project storage
        evaluation = await self._load_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")

        await self._sync_evaluation_status(evaluation)
        
        evaluation.flow = await self._flow_manager.get_flow(project_id=evaluation.evaluation_name)

        return evaluation
    
    async def _sync_evaluation_status(self, evaluation: Evaluation):
        if evaluation.status == EvaluationStatus.COMPLETED:
            return evaluation
        
        # Process the evaluation status
        evaluation_is_running = False
        evaluation_is_completed = True
        has_updates = False

        for instance in evaluation.instances:
            # Skip if already fully processed
            
            # Get current job status from runner
            job_status = await self.runner.get_job_status(
                project_id=evaluation.evaluation_name, trajectory_id=instance.instance_id
            )
            
            # Check if evaluation is still incomplete
            if not instance.is_finished() or not instance.is_evaluated():
                evaluation_is_completed = False

            # Check if any job is running
            if job_status == JobStatus.RUNNING:
                evaluation_is_running = True
                
                if instance.execution_status != ExecutionStatus.RUNNING:
                    instance.execution_status = ExecutionStatus.RUNNING
                    has_updates = True
                
            if job_status == JobStatus.PENDING:
                if instance.execution_status != ExecutionStatus.QUEUED:
                    instance.execution_status = ExecutionStatus.QUEUED
                    has_updates = True
                    
            # Update instance if job status changed
            if instance.job_status != job_status:
                instance.job_status = job_status
                await self._sync_instance_with_job_status(instance, job_status)
                await self._process_trajectory_results(evaluation, instance)
                has_updates = True
            
            # Clean up completed and evaluated jobs from Kubernetes cluster
            if instance.is_evaluated() and job_status == JobStatus.COMPLETED:
                try:
                    logger.info(f"Instance {instance.instance_id} is completed and evaluated, cleaning up Kubernetes job")
                    await self.runner.cancel_job(evaluation.evaluation_name, instance.instance_id)
                    logger.info(f"Successfully cleaned up completed job for instance {instance.instance_id}")
                    # Update job_status to None since we just deleted it
                    instance.job_status = None
                    has_updates = True
                except Exception as e:
                    logger.warning(f"Failed to clean up completed job for instance {instance.instance_id}: {e}")
                
        # Update evaluation status based on instance states
        if evaluation_is_completed:
            evaluation.status = EvaluationStatus.COMPLETED
            self._set_evaluation_completed(evaluation)
            has_updates = True
        elif not evaluation_is_running and evaluation.status != EvaluationStatus.PAUSED:
            evaluation.status = EvaluationStatus.PAUSED
            has_updates = True
        elif evaluation_is_running and evaluation.status != EvaluationStatus.RUNNING:
            evaluation.status = EvaluationStatus.RUNNING
            has_updates = True
            
        if has_updates:
            await self._save_evaluation(evaluation)


    async def get_config(self, evaluation_name: str) -> dict:
        """Get the config for an evaluation."""
        config = await self.storage.read_from_project("flow.json", project_id=evaluation_name)

        # Handle different return types from storage.read_from_project
        if isinstance(config, dict):
            return config
        elif isinstance(config, str):
            return json.loads(config)
        elif isinstance(config, list) and len(config) > 0 and isinstance(config[0], dict):
            return config[0]
        else:
            return {}

    async def update_config(self, evaluation_name: str, config: dict):
        """Update the config for an evaluation."""
        flow = SearchTree.from_dict(config)
        await self.storage.write_to_project("flow.json", flow.model_dump(), project_id=evaluation_name)

    @tracer.start_as_current_span("EvaluationManager._handle_event")
    async def _handle_event(self, event: BaseEvent):
        """Handle events from evaluation runners."""
        logger.info(f"Received event: {event}")
        if event.scope not in ["flow", "evaluation"]:
            return

        if not event.project_id or not event.trajectory_id:
            logger.error(f"Received event with no project id or trajectory id: {event}")
            return

        evaluation = await self._load_evaluation(event.project_id)
        if not evaluation:
            logger.debug(f"Received event with a project id not found in evaluations: {event.project_id}, skipping")
            return

        instance = evaluation.get_instance(event.trajectory_id)
        if not instance:
            logger.warning(f"Received event for unknown instance: {event.trajectory_id}, skipping")
            return

        if event.scope == "flow" and event.event_type == "started":
            instance.started_at = event.timestamp
            instance.mark_running()
        elif event.scope == "flow" and event.event_type == "completed":
            instance.execution_status = ExecutionStatus.COMPLETED
            instance.completed_at = event.timestamp
            await self._process_trajectory_results(evaluation, instance)
        elif event.scope == "evaluation" and event.event_type == "started":
            instance.start_evaluating_at = event.timestamp
            instance.mark_evaluating()
        elif event.scope == "evaluation" and event.event_type == "completed":
            instance.evaluated_at = event.timestamp
            await self._process_trajectory_results(evaluation, instance)
        elif event.event_type == "error":
            instance.fail(event.data.get("error", "Unknown error") if event.data else "Unknown error")

        await self._save_evaluation(evaluation)

    async def _sync_instance_with_job_status(self, instance: EvaluationInstance, job_status: JobStatus | None):
        """Sync instance execution status with job status from runner"""
        if job_status is None:
            # Job no longer exists - this is expected for completed/deleted jobs
            return
        elif job_status == JobStatus.PENDING:
            if instance.execution_status == ExecutionStatus.CREATED:
                instance.start()  # Move to QUEUED
        elif job_status == JobStatus.RUNNING:
            if instance.execution_status in [ExecutionStatus.CREATED, ExecutionStatus.QUEUED]:
                instance.mark_running()
        elif job_status == JobStatus.COMPLETED:
            if instance.execution_status not in [ExecutionStatus.COMPLETED, ExecutionStatus.ERROR]:
                instance.execution_status = ExecutionStatus.COMPLETED
                instance._sync_legacy_status()
        elif job_status == JobStatus.FAILED:
            instance.fail("Job failed in runner")
        elif job_status == JobStatus.CANCELED:
            instance.fail("Job was canceled")
        elif job_status == JobStatus.STOPPED:
            instance.fail("Job was stopped unexpectedly")

    async def _process_trajectory_results(
        self, evaluation: Evaluation, instance: EvaluationInstance, force_update: bool = True
    ):
        """Process the results of a trajectory and update the instance with the results."""

        if instance.is_finished() and instance.is_evaluated() and not force_update:
            logger.info(f"Instance {instance.instance_id} is already completed and evaluated, skipping processing")
            return instance

        # Check if the instance information exists in storage
        if not await self.storage.exists_in_trajectory(
            "trajectory.json",
            project_id=evaluation.evaluation_name,
            trajectory_id=instance.instance_id,
        ):
            logger.warning(f"Instance {instance.instance_id} not found in storage {self.storage}")
            return instance

        try:
            await self._sync_event_timestamps(evaluation, instance)
            flow = await self._read_flow(
                trajectory_id=instance.instance_id, project_id=evaluation.evaluation_name
            )
            logger.debug(f"Read flow with {len(flow.root.get_all_nodes())} nodes")

            instance.usage = flow.total_usage()
            instance.iterations = len(flow.root.get_all_nodes())

            node = None
            if flow.root.discriminator_result and flow.root.discriminator_result.selected_node_id:
                node = flow.get_node_by_id(flow.root.discriminator_result.selected_node_id)

            if not node:
                node = flow.root.get_last_node()
                
            if node.reward:
                instance.reward = node.reward.value
                
            logger.debug(f"Instance {instance.instance_id} flow is finished: {flow.is_finished()}")
            if flow.is_finished():
                instance.execution_status = ExecutionStatus.COMPLETED
            else:
                
                instance.execution_status = ExecutionStatus.CREATED

            if node.evaluation_result and node.evaluation_result.resolved is not None:
                # Set resolution status based on evaluation
                instance.set_resolution(node.evaluation_result.resolved)
                logger.info(f"Instance {instance.instance_id} resolution: {instance.resolution_status}")
            
            if instance.resolution_status != ResolutionStatus.RESOLVED:
                logger.debug(f"Instance {instance.instance_id} and node {node.node_id} has no evaluation result")
                # Check if any leaf node is resolved for partial resolution
                for leaf_node in flow.root.get_all_nodes():
                    if leaf_node.evaluation_result and leaf_node.evaluation_result.resolved:
                        instance.set_resolution(False, partially_resolved=True)
                        logger.info(f"Instance {instance.instance_id} is partially resolved")
                        break

            if node.error:
                instance.fail(node.error)
                logger.info(f"Instance {instance.instance_id} encountered error: {node.error}")

        except Exception as e:
            logger.exception(f"Error processing trajectory results for instance {instance.instance_id}")
            instance.fail(str(e))
            
        return instance

    async def _read_flow(
        self,
        trajectory_id: str,
        project_id: str,
    ) -> "AgenticFlow":

        traj_dict = await self.storage.read_from_trajectory("trajectory.json", project_id, trajectory_id)
        if not traj_dict:
            raise ValueError("Trajectory not found")
        
        try:
            flow_dict: dict | None = await self.storage.read_from_trajectory("flow.json", project_id, trajectory_id)  # type: ignore
        except Exception:
            flow_dict = await self.storage.read_from_project("flow.json", project_id)  # type: ignore

        if not flow_dict:
            raise ValueError("Flow settings not found")

        return AgenticFlow.from_dicts(flow_dict, traj_dict)

    async def cancel_evaluation(self, evaluation_name: str):
        """Cancel all running jobs for an evaluation."""
        evaluation = await self._load_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")

        # Cancel all jobs for the evaluation
        await self.runner.cancel_job(evaluation_name, None)

        # Set evaluation status to PAUSED
        if evaluation.status == EvaluationStatus.RUNNING:
            evaluation.status = EvaluationStatus.PAUSED
            await self._save_evaluation(evaluation)

        return evaluation

    @tracer.start_as_current_span("EvaluationManager.retry_instance")
    async def retry_instance(self, evaluation_name: str, instance_id: str):
        """
        Retry a specific instance in an evaluation.

        This is useful if an instance failed or you want to re-run it.
        """
        evaluation = await self.get_evaluation(evaluation_name)
        instance = None

        for i in evaluation.instances:
            if i.instance_id == instance_id:
                instance = i
                break

        if not instance:
            raise ValueError(f"Instance {instance_id} not found in evaluation {evaluation_name}")

        # Reset the instance state
        instance.execution_status = ExecutionStatus.CREATED
        instance.resolution_status = ResolutionStatus.PENDING
        instance.job_status = None
        instance.error = None
        instance.error_at = None
        instance.evaluated_at = None
        instance.start_evaluating_at = None
        instance.started_at = None
        instance.completed_at = None
        instance._sync_legacy_status()

        # If the instance was already started, cancel any existing jobs
        await self.runner.cancel_job(evaluation_name, instance_id)

        # Start a new job for this instance using the wrapper function
        await self.runner.start_job(
            project_id=evaluation_name,
            trajectory_id=instance_id,
            job_func=run_swebench_instance,
        )

        await self._save_evaluation(evaluation)

        return evaluation

    def _is_evaluation_completed(self, evaluation: Evaluation) -> bool:
        """Check if all instances in an evaluation have been evaluated."""
        return all(instance.is_finished() and instance.is_evaluated() for instance in evaluation.instances)

    def _set_evaluation_completed(self, evaluation: Evaluation):
        """Set the evaluation status to COMPLETED."""
        if evaluation.status != EvaluationStatus.COMPLETED:
            evaluation.status = EvaluationStatus.COMPLETED
            evaluation.completed_at = datetime.now(timezone.utc)

    async def process_evaluation_results(self, evaluation_name: str) -> Evaluation:
        """Process all instances in an evaluation to ensure results are in sync."""
        evaluation = await self._load_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")

        logger.debug(f"Processing evaluation results for {evaluation_name} with {len(evaluation.instances)} instances")
        for instance in evaluation.instances:
            await self._process_trajectory_results(evaluation, instance)

        all_evaluated = self._is_evaluation_completed(evaluation)
        if all_evaluated and evaluation.status != EvaluationStatus.COMPLETED:
            self._set_evaluation_completed(evaluation)

        await self._save_evaluation(evaluation)
        return evaluation

    async def save_evaluation(self, evaluation: Evaluation):
        """Save evaluation metadata to storage."""
        await self._save_evaluation(evaluation)

    async def _save_evaluation(self, evaluation: Evaluation):
        """Save evaluation data using project storage."""
        try:
            eval_dump = evaluation.model_dump()
            del eval_dump["flow"]
            await self.storage.write_to_project(
                "evaluation.json",
                eval_dump,
                project_id=evaluation.evaluation_name,
            )

            # Mark the evaluation as created in the storage system
            # Use this pattern to avoid linter errors with private attributes
            if hasattr(self.storage, "_created_evaluations"):
                storage_created_evaluations = getattr(self.storage, "_created_evaluations", set())
                storage_created_evaluations.add(evaluation.evaluation_name)

            # Also update the evaluation summary for fast lookups
            await self._update_evaluation_summary(evaluation)
        except Exception as e:
            logger.error(f"Failed to save evaluation {evaluation.evaluation_name}: {e}")

    async def _update_evaluation_summary(self, evaluation: Evaluation):
        """Update or create a lightweight evaluation summary for the evaluation."""
        try:
            summary = EvaluationSummary.from_evaluation(evaluation)

            all_summaries = await self._load_all_summaries()
            all_summaries[evaluation.evaluation_name] = json.loads(summary.model_dump_json())

            await self.storage.write("evaluation_summaries.json", all_summaries)

        except Exception as e:
            logger.error(f"Failed to update evaluation summary for {evaluation.evaluation_name}: {e}")

    async def _load_all_summaries(self) -> dict:
        """Load all evaluation summaries from a single file."""
        try:
            if await self.storage.exists("evaluation_summaries.json"):
                data = await self.storage.read("evaluation_summaries.json")
                # Handle different return types from storage.read
                if isinstance(data, dict):
                    return data
                elif isinstance(data, str):
                    return json.loads(data)
                elif isinstance(data, list):
                    # Convert list to dict if needed
                    return {str(i): item for i, item in enumerate(data)}
                return {}
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to load evaluation summaries: {e}")
            return {}

    async def list_evaluation_summaries(self) -> list[EvaluationSummary]:
        """List all evaluation summaries."""
        summaries = []

        # Check if the evaluation_summaries file exists
        exists = await self.storage.exists("evaluation_summaries.json")
        if not exists:
            return summaries

        # Load all summaries from the single file
        all_summaries = await self._load_all_summaries()

        # Convert dictionary values to EvaluationSummary objects
        for summary_data in all_summaries.values():
            try:
                summary = EvaluationSummary.model_validate(summary_data)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error loading evaluation summary: {e}")

        # Sort summaries by creation time, newest first
        return sorted(
            summaries,
            key=lambda x: x.created_at if x.created_at else datetime.min,
            reverse=True,
        )

    async def _load_evaluation(self, evaluation_name: str) -> Optional[Evaluation]:
        """Load evaluation metadata from storage."""
        if await self.storage.exists_in_project("evaluation.json", project_id=evaluation_name):
            data = await self.storage.read_from_project("evaluation.json", project_id=evaluation_name)
            evaluation = Evaluation.model_validate(data)
            return evaluation
        else:
            return None

    def get_dataset_instance_ids(self, dataset_name: str) -> list[str]:
        """Get instance IDs for a dataset."""
        if dataset_name in self._cached_datasets:
            return self._cached_datasets[dataset_name]

        dataset_path = Path(__file__).parent / "datasets" / f"{dataset_name}_dataset.json"
        if not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_name} not found at {dataset_path}")

        with open(dataset_path) as f:
            dataset = json.load(f)
            self._cached_datasets[dataset_name] = dataset["instance_ids"]
            return self._cached_datasets[dataset_name]

    async def get_evaluation_instance(self, evaluation_name: str, instance_id: str) -> EvaluationInstance:
        """Get an instance from an evaluation."""
        evaluation = await self.get_evaluation(evaluation_name)

        # Try to find the instance in the evaluation object first
        for instance in evaluation.instances:
            if instance.instance_id == instance_id:
                return instance

        # If not found in the evaluation object, try to load it from storage
        try:
            data = await self.storage.read_from_trajectory(
                "instance.json", project_id=evaluation_name, trajectory_id=instance_id
            )
            return EvaluationInstance.model_validate(data)
        except (KeyError, ValueError):
            # If not found in storage either, raise an error
            raise ValueError(f"Instance {instance_id} not found in evaluation {evaluation_name}")

    @tracer.start_as_current_span("EvaluationManager.start_instance")
    async def start_instance(self, evaluation_name: str, instance_id: str) -> Evaluation:
        """Start a specific instance of an evaluation."""
        evaluation = await self.get_evaluation(evaluation_name)
        instance = None

        for i in evaluation.instances:
            if i.instance_id == instance_id:
                instance = i
                break

        if not instance:
            raise ValueError(f"Instance {instance_id} not found in evaluation {evaluation_name}")

        # Reset instance state if it was in error
        if instance.execution_status == ExecutionStatus.ERROR:
            logger.info(f"Evaluation {evaluation_name} has error instance {instance_id}, resetting and retrying")
            instance.execution_status = ExecutionStatus.CREATED
            instance.resolution_status = ResolutionStatus.PENDING
            instance.error = None
            instance.error_at = None
            instance.started_at = None
            instance.completed_at = None
            instance.start_evaluating_at = None
            instance.evaluated_at = None
            instance._sync_legacy_status()

        flow = await self._flow_manager.get_flow(evaluation.evaluation_name, instance.instance_id)
        should_save = False
        for leaf_node in flow.root.get_leaf_nodes():
            if leaf_node.error:
                logger.info(f"Resetting node {leaf_node.node_id} with error")
                leaf_node.reset()
                should_save = True
                
            
        # Skip if already completed and evaluated
        if instance.is_finished() and instance.is_evaluated():
            finish_reason = flow.is_finished()
            if finish_reason:
                logger.info(f"Instance {instance_id} is already completed and evaluated, skipping: {finish_reason}")
                return evaluation


        # TODO: This is a hack to ignore the terminal node if it is not really terminal
        if flow.root.get_last_node().terminal and not flow.root.get_last_node().is_terminal():
            flow.root.get_last_node().terminal = False
            
            should_save = True

        if should_save:
            await self._flow_manager.save_trajectory(evaluation.evaluation_name, instance.instance_id, flow)
        # Start the job using the wrapper function
        await self.runner.start_job(
            project_id=evaluation.evaluation_name,
            trajectory_id=instance.instance_id,
            job_func=run_swebench_instance,
        )
        
        job_status = await self.runner.get_job_status(
            project_id=evaluation.evaluation_name, trajectory_id=instance.instance_id
        )
        
        instance.job_status = job_status
        await self._sync_instance_with_job_status(instance, job_status)
        
        logger.info(f"Started instance {instance_id} with job status: {job_status}")
            
        # Update evaluation status if needed
        if evaluation.status in [
            EvaluationStatus.PENDING,
            EvaluationStatus.PAUSED,
            EvaluationStatus.ERROR,
        ]:
            evaluation.status = EvaluationStatus.RUNNING
            if not evaluation.started_at:
                evaluation.started_at = datetime.now(timezone.utc)

        await self._save_evaluation(evaluation)
        
        return evaluation

    async def list_evaluation_instances(self, evaluation_name: str) -> list[str]:
        """
        List all instance IDs in an evaluation.

        Args:
            evaluation_name: The evaluation name

        Returns:
            A list of instance IDs
        """
        # Check if storage has list_trajectories method and use reflection to avoid linter errors
        if hasattr(self.storage, "list_trajectories"):
            list_trajectories_method = getattr(self.storage, "list_trajectories")
            if callable(list_trajectories_method):
                return await list_trajectories_method(evaluation_name)

        # Fallback: try to get instances from the evaluation object
        evaluation = await self._load_evaluation(evaluation_name)
        if evaluation:
            return [instance.instance_id for instance in evaluation.instances]
        return []

    async def get_all_evaluation_instances(self, evaluation_name: str) -> list[EvaluationInstance]:
        """
        Get all instances in an evaluation directly from storage.

        Args:
            evaluation_name: The evaluation name

        Returns:
            A list of EvaluationInstance objects
        """
        # Get the evaluation to ensure it exists
        evaluation = await self.get_evaluation(evaluation_name)

        # Get all instance IDs for this evaluation
        instance_ids = await self.list_evaluation_instances(evaluation_name)

        # Get the instance data for each ID
        instances = []
        for instance_id in instance_ids:
            try:
                # Try to read the instance data from trajectory storage
                if await self.storage.exists_in_trajectory(
                    "instance.json", project_id=evaluation_name, trajectory_id=instance_id
                ):
                    instance_data = await self.storage.read_from_trajectory(
                        "instance.json",
                        project_id=evaluation_name,
                        trajectory_id=instance_id,
                    )
                    instance = EvaluationInstance.model_validate(instance_data)
                    instances.append(instance)
            except Exception as e:
                logger.error(f"Error loading instance {instance_id}: {e}")

        return instances

    async def _sync_event_timestamps(self, evaluation: Evaluation, instance: EvaluationInstance):
        if instance.started_at and instance.completed_at and instance.start_evaluating_at and instance.evaluated_at:
            return

        events = await self.eventbus.read_events(
            project_id=evaluation.evaluation_name,
            trajectory_id=instance.instance_id,
        )

        if not events:
            instance.last_event_timestamp = None
            return

        instance.last_event_timestamp = events[-1].timestamp

        # Single pass through events to find all required timestamps
        # Track which timestamps we still need to find
        need_started = not instance.started_at
        need_completed = not instance.completed_at
        need_start_evaluating = not instance.start_evaluating_at
        need_evaluated = not instance.evaluated_at

        for event in events:
            # Check for each timestamp we still need
            if need_started and event.scope == "flow" and event.event_type == "started":
                instance.started_at = event.timestamp
                need_started = False
            elif need_completed and event.scope == "flow" and event.event_type == "completed":
                instance.completed_at = event.timestamp
                need_completed = False
            elif need_start_evaluating and event.scope == "evaluation" and event.event_type == "started":
                instance.start_evaluating_at = event.timestamp
                need_start_evaluating = False
            elif need_evaluated and event.scope == "evaluation" and event.event_type == "completed":
                instance.evaluated_at = event.timestamp
                need_evaluated = False

            # Early termination if all timestamps are found
            if not (need_started or need_completed or need_start_evaluating or need_evaluated):
                break

    async def get_evaluation_stats(self, evaluation_name: str) -> EvaluationStats:
        """Get comprehensive statistics for an evaluation, including only finished instances."""

        evaluation = await self.get_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")

        # Calculate basic metrics
        total_instances = len(evaluation.instances)
        resolved_instances = sum(1 for instance in evaluation.instances if instance.resolved is True)
        failed_instances = sum(1 for instance in evaluation.instances if instance.resolved is False)
        success_rate = (resolved_instances / total_instances) * 100 if total_instances > 0 else 0

        # Calculate cost and token metrics
        total_cost = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cache_read_tokens = 0
        total_iterations = 0
        iteration_values = []
        cost_values = []

        for instance in evaluation.instances:
            if instance.usage:
                cost = instance.usage.completion_cost or 0
                total_cost += cost
                cost_values.append(cost)
                total_prompt_tokens += instance.usage.prompt_tokens or 0
                total_completion_tokens += instance.usage.completion_tokens or 0
                total_cache_read_tokens += instance.usage.cache_read_tokens or 0

            if instance.iterations:
                total_iterations += instance.iterations
                iteration_values.append(instance.iterations)

        avg_iterations = total_iterations / total_instances if total_instances > 0 else 0
        avg_cost = total_cost / total_instances if total_instances > 0 else 0
        avg_tokens = (total_prompt_tokens + total_completion_tokens) / total_instances if total_instances > 0 else 0

        # Cost efficiency metrics
        solved_instances_per_dollar = resolved_instances / total_cost if total_cost > 0 else 0
        solved_percentage_per_dollar = success_rate / total_cost if total_cost > 0 else 0

        # Iteration distribution with improved ranges
        iteration_ranges = ["1-20", "20-40", "40-60", "60-80", "80-100", "100+"]
        iterations_distribution = {range_name: 0 for range_name in iteration_ranges}

        for iterations in iteration_values:
            if iterations <= 20:
                iterations_distribution["1-20"] += 1
            elif iterations <= 40:
                iterations_distribution["20-40"] += 1
            elif iterations <= 60:
                iterations_distribution["40-60"] += 1
            elif iterations <= 80:
                iterations_distribution["60-80"] += 1
            elif iterations <= 100:
                iterations_distribution["80-100"] += 1
            else:
                iterations_distribution["100+"] += 1

        # Cost distribution with defined ranges
        cost_ranges = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0+"]
        cost_distribution = {range_name: 0 for range_name in cost_ranges}

        for cost in cost_values:
            if cost <= 0.2:
                cost_distribution["0-0.2"] += 1
            elif cost <= 0.4:
                cost_distribution["0.2-0.4"] += 1
            elif cost <= 0.6:
                cost_distribution["0.4-0.6"] += 1
            elif cost <= 0.8:
                cost_distribution["0.6-0.8"] += 1
            elif cost <= 1.0:
                cost_distribution["0.8-1.0"] += 1
            else:
                cost_distribution["1.0+"] += 1

        # Success distribution
        success_distribution = {"resolved": resolved_instances, "failed": failed_instances}

        # Per-repo statistics
        repo_data = {}
        for instance in evaluation.instances:
            try:
                swebench_instance = get_swebench_instance(instance.instance_id)
                repo = swebench_instance["repo"]

                if repo not in repo_data:
                    repo_data[repo] = {"total": 0, "resolved": 0, "failed": 0}

                repo_data[repo]["total"] += 1
                if instance.resolved is True:
                    repo_data[repo]["resolved"] += 1
                elif instance.resolved is False:
                    repo_data[repo]["failed"] += 1
            except Exception as e:
                logger.warning(f"Could not get repo for instance {instance.instance_id}: {e}")
                continue

        repo_stats = []
        for repo, data in repo_data.items():
            solve_rate = (data["resolved"] / data["total"]) * 100 if data["total"] > 0 else 0
            repo_stats.append(
                RepoStats(
                    repo=repo,
                    total_instances=data["total"],
                    resolved_instances=data["resolved"],
                    failed_instances=data["failed"],
                    solve_rate=solve_rate,
                )
            )

        # Sort repo stats by solve rate descending
        repo_stats.sort(key=lambda x: x.solve_rate, reverse=True)

        return EvaluationStats(
            success_rate=success_rate,
            avg_iterations=avg_iterations,
            avg_cost=avg_cost,
            avg_tokens=int(avg_tokens),
            solved_instances_per_dollar=solved_instances_per_dollar,
            solved_percentage_per_dollar=solved_percentage_per_dollar,
            total_instances=total_instances,
            resolved_instances=resolved_instances,
            failed_instances=failed_instances,
            total_cost=total_cost,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cache_read_tokens=total_cache_read_tokens,
            iteration_range_min=min(iteration_values) if iteration_values else 0,
            iteration_range_max=max(iteration_values) if iteration_values else 0,
            success_distribution=success_distribution,
            iterations_distribution=iterations_distribution,
            cost_distribution=cost_distribution,
            repo_stats=repo_stats,
        )
