import json
import logging
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiofiles
import aiofiles.os
from opentelemetry import trace

from moatless.benchmark.report import to_result
from moatless.evaluation.run_instance import run_instance
from moatless.evaluation.schema import (
    Evaluation,
    EvaluationEvent,
    EvaluationInstance,
    EvaluationStatus,
    InstanceStatus,
)
from moatless.evaluation.utils import get_moatless_instance
from moatless.events import BaseEvent, EventBus, event_bus
from moatless.flow.flow import AgenticFlow
from moatless.flow.manager import create_flow, get_flow_config
from moatless.runner.rq import RQRunner
from moatless.runner.runner import BaseRunner, EvaluationJobStatus, JobInfo, JobStatus
from moatless.storage.base import BaseStorage
from moatless.storage.file_storage import FileStorage
from moatless.utils.moatless import get_moatless_dir, get_moatless_trajectory_dir

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.evaluation.manager")


class EvaluationManager:
    _instance = None

    @classmethod
    async def get_instance(cls) -> "EvaluationManager":
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.initialize()

        return cls._instance

    def __init__(
        self, runner: BaseRunner | None = None, storage: BaseStorage | None = None, eventbus: EventBus | None = None
    ):
        self.runner = runner or BaseRunner.get_instance()
        self.storage = storage or BaseStorage.get_instance()
        self.eventbus = eventbus or event_bus

    async def initialize(self):
        await event_bus.subscribe(self._handle_event)

    async def create_evaluation(
        self,
        flow_id: str,
        model_id: str,
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
                f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{flow_id}_{model_id}_{dataset_name}"
            )

        logger.info(f"Creating evaluation: {evaluation_name}")

        if not instance_ids:
            instance_ids = self.get_dataset_instance_ids(dataset_name)
            logger.info(f"Found {len(instance_ids)} instances in dataset {dataset_name}")

        if await self.storage.exists_in_project("evaluation", evaluation_name):
            raise ValueError("Evaluation already exists")

        evaluation = Evaluation(
            evaluation_name=evaluation_name,
            dataset_name=dataset_name,
            status=EvaluationStatus.PENDING,
            flow_id=flow_id,
            model_id=model_id,
            instances=[
                EvaluationInstance(instance_id=instance_id)  # Will default to CREATED state
                for instance_id in instance_ids
            ],
        )

        for instance in evaluation.instances:
            moatless_instance = get_moatless_instance(instance_id=instance.instance_id)
            problem_statement = f"Solve the following issue:\n{moatless_instance['problem_statement']}"
            flow = create_flow(
                id=evaluation.flow_id,
                message=problem_statement,
                project_id=evaluation.evaluation_name,
                trajectory_id=instance.instance_id,
                model_id=evaluation.model_id,
                metadata={"instance_id": instance.instance_id},
            )
            await flow.persist()

        await self._save_evaluation(evaluation)
        logger.info(f"Evaluation created: {evaluation_name} with {len(evaluation.instances)} instances")
        return evaluation

    async def clone_evaluation(self, evaluation_name: str) -> Evaluation:
        """Clone an existing evaluation."""
        evaluation = await self._load_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")

        new_evaluation_name = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{evaluation.flow_id}_{evaluation.model_id}_{evaluation.dataset_name}"

        return await self.create_evaluation(
            evaluation.flow_id,
            evaluation.model_id,
            evaluation_name=new_evaluation_name,
            dataset_name=evaluation.dataset_name,
        )

    @tracer.start_as_current_span("EvaluationManager.start_evaluation")
    async def start_evaluation(self, evaluation_name: str) -> Evaluation:
        """Start an evaluation by running all instances."""
        evaluation = await self.get_evaluation(evaluation_name)
        started_instances = 0

        for instance in evaluation.instances:
            if instance.status == InstanceStatus.EVALUATED:
                continue

            # Start the job
            if await self.runner.start_job(
                project_id=evaluation.evaluation_name,
                trajectory_id=instance.instance_id,
                job_func=run_instance,
            ):
                started_instances += 1
                instance.status = InstanceStatus.PENDING

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

    async def get_evaluation(self, evaluation_name: str) -> Evaluation:
        """Get an evaluation by name."""
        # Load the evaluation from project storage
        evaluation = await self._load_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")

        if evaluation.status == EvaluationStatus.COMPLETED:
            return evaluation

        # Process the evaluation status
        evaluation_is_running = False
        evaluation_is_completed = True

        for instance in evaluation.instances:
            if instance.status not in [InstanceStatus.EVALUATED, InstanceStatus.ERROR]:
                evaluation_is_completed = False

                job_status = await self.runner.get_job_status(
                    project_id=evaluation_name, trajectory_id=instance.instance_id
                )

                if job_status in [JobStatus.RUNNING, JobStatus.QUEUED]:
                    evaluation_is_running = True

        if evaluation_is_completed:
            evaluation.status = EvaluationStatus.COMPLETED
        elif not evaluation_is_running:
            evaluation.status = EvaluationStatus.PAUSED
        else:
            evaluation.status = EvaluationStatus.RUNNING

        return evaluation

    async def _handle_event(self, event: BaseEvent):
        """Handle events from evaluation runners."""
        if event.scope not in ["flow", "evaluation"]:
            return

        if not event.project_id or not event.trajectory_id:
            logger.error(f"Received event with no project id or trajectory id: {event}")
            return

        evaluation = await self._load_evaluation(event.project_id)
        if not evaluation:
            logger.info(f"Received event with a project id not found in evaluations: {event.project_id}, skipping")
            return

        instance = evaluation.get_instance(event.trajectory_id)
        if not instance:
            logger.warning(f"Received event for unknown instance: {event.trajectory_id}, skipping")
            return

        if event.scope == "flow" and event.event_type == "started":
            instance.status = InstanceStatus.RUNNING
            instance.started_at = event.timestamp
        elif event.scope == "flow" and event.event_type == "completed":
            await self._process_trajectory_results(evaluation, instance)
            instance.status = InstanceStatus.COMPLETED
            instance.completed_at = event.timestamp
        elif event.scope == "evaluation" and event.event_type == "started":
            instance.status = InstanceStatus.EVALUATING
            instance.start_evaluating_at = event.timestamp
        elif event.scope == "evaluation" and event.event_type == "completed":
            await self._process_trajectory_results(evaluation, instance)
            instance.status = InstanceStatus.EVALUATED
            instance.evaluated_at = event.timestamp
        elif event.event_type == "error":
            instance.status = InstanceStatus.ERROR
            if event.data:
                instance.error = event.data.get("error")
            instance.error_at = event.timestamp

        await self._save_evaluation(evaluation)

    async def _process_trajectory_results(self, evaluation: Evaluation, instance: EvaluationInstance):
        """Process the results of a trajectory and update the instance with the results."""
        if instance.status == InstanceStatus.EVALUATED:
            return instance

        # Check if the instance information exists in storage
        instance_key = "instance"
        if not await self.storage.exists_in_trajectory(
            instance_key, project_id=evaluation.evaluation_name, trajectory_id=instance.instance_id
        ):
            logger.warning(f"Instance not found in storage: {instance.instance_id}")
            return instance

        # Get the trajectory directory for backup/legacy access
        trajectory_dir = get_moatless_trajectory_dir(
            trajectory_id=instance.instance_id, project_id=evaluation.evaluation_name
        )

        try:
            # Get usage information from flow
            if trajectory_dir.exists():
                flow = AgenticFlow.from_dir(trajectory_dir=trajectory_dir)
                instance.usage = flow.total_usage()

            # Get events
            events = await event_bus.read_events(
                project_id=evaluation.evaluation_name,
                trajectory_id=instance.instance_id,
            )

            # Set started_at if not already set
            if not instance.started_at:
                event = next(
                    (e for e in events if e.scope == "flow" and e.event_type == "started"),
                    None,
                )
                if event:
                    instance.started_at = event.timestamp

            # After processing, save the updated instance
            await self._save_instance(evaluation.evaluation_name, instance)

            if not instance.completed_at and not instance.evaluated_at:
                event = next(
                    (e for e in events if e.scope == "flow" and e.event_type == "completed"),
                    None,
                )
                if event:
                    instance.completed_at = event.timestamp
                    if instance.status == InstanceStatus.RUNNING:
                        instance.status = InstanceStatus.COMPLETED
                        logger.info(f"Synced instance {instance.instance_id} completed at {instance.completed_at}")

                # TODO: Handle trajectories with multiple leaf nodes
                leaf_nodes = flow.root.get_leaf_nodes()
                node = leaf_nodes[0]
                if node.terminal:
                    if node.reward:
                        instance.reward = node.reward.value

                    if node.error:
                        instance.error = node.error
                        instance.status = InstanceStatus.ERROR
                        instance.error_at = datetime.now(timezone.utc)
                    else:
                        instance.status = InstanceStatus.COMPLETED

                instance.iterations = len(flow.root.get_all_nodes())

            if not instance.evaluated_at or instance.resolved is None or instance.status != InstanceStatus.EVALUATED:
                eval_result_path = trajectory_dir / "eval_result.json"

                event = next(
                    (e for e in events if e.scope == "evaluation" and e.event_type == "started"),
                    None,
                )
                if event:
                    instance.start_evaluating_at = event.timestamp
                    if instance.status == InstanceStatus.COMPLETED:
                        instance.status = InstanceStatus.EVALUATING
                        logger.info(
                            f"Synced instance {instance.instance_id} started evaluating at {instance.start_evaluating_at}"
                        )

                event = next(
                    (e for e in events if e.scope == "evaluation" and e.event_type == "completed"),
                    None,
                )
                if event:
                    instance.evaluated_at = event.timestamp

                if not eval_result_path.exists():
                    logger.debug(f"Evaluation result not found for instance: {instance.instance_id}")
                    return

                with open(eval_result_path) as f:
                    eval_result = json.load(f)

                benchmark_result = to_result(
                    node=flow.root,
                    eval_report=eval_result,
                    instance_id=instance.instance_id,
                )

                instance.benchmark_result = benchmark_result
                instance.resolved = benchmark_result.resolved
                if instance.status != InstanceStatus.EVALUATED:
                    instance.status = InstanceStatus.EVALUATED
                    logger.info(f"Synced instance {instance.instance_id} evaluated at {instance.evaluated_at}")

                # Check if all instances have been evaluated, and update evaluation status
                all_evaluated = self._is_evaluation_completed(evaluation)
                if all_evaluated and evaluation.status != EvaluationStatus.COMPLETED:
                    logger.info(
                        f"All instances evaluated for evaluation {evaluation.evaluation_name}, setting to COMPLETED"
                    )
                    self._set_evaluation_completed(evaluation)
            return instance
        except Exception as e:
            logger.error(f"Error processing trajectory results for instance {instance.instance_id}: {e}")
            return instance

    async def get_evaluation_status(self, evaluation_name: str) -> EvaluationJobStatus:
        """Get the status of all jobs for an evaluation."""
        evaluation = await self._load_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")

        return await self.runner.get_evaluation_status(evaluation)

    async def is_runner_up(self) -> bool:
        """Check if any RQ workers are currently running.

        Returns:
            bool: True if at least one worker is up and running, False otherwise
        """
        return await self.runner.status()

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
        if instance.status == InstanceStatus.EVALUATED:
            # Moving from evaluated back to pending
            instance.status = InstanceStatus.PENDING
            instance.evaluated_at = None
            instance.start_evaluating_at = None

        elif instance.status == InstanceStatus.ERROR:
            # Reset error state
            instance.status = InstanceStatus.PENDING
            instance.error = None
            instance.error_at = None

        else:
            # For any other state, just reset to pending
            instance.status = InstanceStatus.PENDING

        # If the instance was already started, cancel any existing jobs
        await self.runner.cancel_job(evaluation_name, instance_id)

        # Start a new job for this instance
        await self.runner.start_job(project_id=evaluation_name, trajectory_id=instance_id)

        await self._save_evaluation(evaluation)

        return evaluation

    def _is_evaluation_completed(self, evaluation: Evaluation) -> bool:
        """Check if all instances in an evaluation have been evaluated."""
        return all(instance.status == InstanceStatus.EVALUATED for instance in evaluation.instances)

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

        # Also save each instance individually using trajectory storage
        for instance in evaluation.instances:
            await self._save_instance(evaluation.evaluation_name, instance)

    async def _save_evaluation(self, evaluation: Evaluation):
        """Save evaluation data using project storage."""
        try:
            # Save the evaluation data using the project storage
            # Use model_dump_json for serialization
            await self.storage.write_to_project(
                "evaluation", json.loads(evaluation.model_dump_json()), project_id=evaluation.evaluation_name
            )
        except Exception as e:
            logger.error(f"Failed to save evaluation {evaluation.evaluation_name}: {e}")

    async def _save_instance(self, evaluation_name: str, instance: EvaluationInstance):
        """Save instance data using trajectory storage."""
        try:
            # Save the instance data using trajectory storage
            # Use model_dump_json for serialization
            await self.storage.write_to_trajectory(
                "instance",
                json.loads(instance.model_dump_json()),
                project_id=evaluation_name,
                trajectory_id=instance.instance_id,
            )
        except Exception as e:
            logger.error(f"Failed to save instance {instance.instance_id} for evaluation {evaluation_name}: {e}")

    async def _load_evaluation(self, evaluation_name: str) -> Optional[Evaluation]:
        """Load evaluation metadata from storage."""
        try:
            # Check if the evaluation exists in storage
            if await self.storage.exists_in_project("evaluation", evaluation_name):
                # Load the evaluation data from project storage
                data = await self.storage.read_from_project("evaluation", evaluation_name)
                evaluation = Evaluation.model_validate(data)
                return evaluation
            else:
                # Try legacy path
                eval_path = self.get_evaluation_dir(evaluation_name) / "evaluation.json"
                if eval_path.exists():
                    with open(eval_path) as f:
                        data = json.load(f)
                        evaluation = Evaluation.model_validate(data)
                        return evaluation
                return None
        except KeyError:
            return None
        except Exception as e:
            logger.error(f"Failed to load evaluation {evaluation_name}: {e}")
            return None

    async def list_evaluations(self) -> list[Evaluation]:
        """List all evaluations with their metadata."""
        evaluations = []

        # First, get evaluations from storage
        eval_data_list = await self.storage.list_evaluations()
        for eval_data in eval_data_list:
            try:
                evaluation = Evaluation.model_validate(eval_data)
                evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Error parsing evaluation data: {e}")

        # Sort evaluations by creation time, newest first
        return sorted(evaluations, key=lambda x: x.created_at if x.created_at else datetime.min, reverse=True)

    async def _handle_evaluation_event(self, trajectory_id: str | None, project_id: str | None, event: EvaluationEvent):
        """Handle events from evaluation runners."""
        if not isinstance(event, EvaluationEvent):
            return

        # Get evaluation from active runner if exists, otherwise load from file
        evaluation = await self._load_evaluation(event.evaluation_name)

        if not evaluation:
            logger.warning(f"Received event for unknown evaluation: {event.evaluation_name}")
            return

        await self._save_evaluation(evaluation)

    def get_dataset_instance_ids(self, dataset_name: str) -> list[str]:
        """Get instance IDs for a dataset."""
        dataset_path = Path(__file__).parent / "datasets" / f"{dataset_name}_dataset.json"
        if not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_name} not found at {dataset_path}")

        with open(dataset_path) as f:
            dataset = json.load(f)
            return dataset["instance_ids"]

    def get_evaluation_dir(self, evaluation_name: str) -> Path:
        """Get the directory for a specific evaluation."""
        return self.evals_dir / evaluation_name

    def get_instance_dir(self, evaluation_name: str, instance_id: str) -> Path:
        """Get the directory for a specific instance within an evaluation."""
        return self.get_evaluation_dir(evaluation_name) / instance_id

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
                "instance", project_id=evaluation_name, trajectory_id=instance_id
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
        if instance.status == InstanceStatus.ERROR:
            logger.info(f"Evaluation {evaluation_name} has error instance {instance_id}, resetting and retrying")
            if instance.completed_at:
                instance.status = InstanceStatus.COMPLETED
                instance.start_evaluating_at = None
                instance.evaluated_at = None
            else:
                instance.status = InstanceStatus.PENDING
                instance.started_at = None

            instance.error = None
            instance.error_at = None

        # Skip if already evaluated
        if instance.status == InstanceStatus.EVALUATED:
            logger.info(f"Instance {instance_id} is already evaluated, skipping")
            return evaluation

        # Start the job
        if await self.runner.start_job(
            project_id=evaluation.evaluation_name,
            trajectory_id=instance.instance_id,
        ):
            logger.info(f"Started instance {instance_id} for evaluation {evaluation_name}")

            # Update instance status to PENDING
            instance.status = InstanceStatus.PENDING

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
        else:
            logger.error(f"Failed to start instance {instance_id} for evaluation {evaluation_name}")

        return evaluation

    async def list_evaluation_instances(self, evaluation_name: str) -> list[str]:
        """
        List all instance IDs in an evaluation.

        Args:
            evaluation_name: The evaluation name

        Returns:
            A list of instance IDs
        """
        return await self.storage.list_trajectories(evaluation_name)

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
                    "instance", project_id=evaluation_name, trajectory_id=instance_id
                ):
                    instance_data = await self.storage.read_from_trajectory(
                        "instance", project_id=evaluation_name, trajectory_id=instance_id
                    )
                    instance = EvaluationInstance.model_validate(instance_data)
                    instances.append(instance)
            except Exception as e:
                logger.error(f"Error loading instance {instance_id}: {e}")

        return instances
