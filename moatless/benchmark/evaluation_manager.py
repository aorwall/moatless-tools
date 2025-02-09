import json
import os
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from moatless.utils.moatless import get_moatless_dir
from moatless.benchmark.evaluation_runner import EvaluationRunner, TreeSearchSettings, Evaluation, EvaluationInstance
from moatless.benchmark.schema import EvaluationStatus, InstanceStatus
from moatless.events import BaseEvent, event_bus
from moatless.benchmark.schema import EvaluationEvent

logger = logging.getLogger(__name__)

class EvaluationManager:
    def __init__(self):
        self.evals_dir = get_moatless_dir() / "evals"
        self.evals_dir.mkdir(parents=True, exist_ok=True)
        event_bus.subscribe(self._handle_evaluation_event)

    def get_evaluation_dir(self, evaluation_name: str) -> Path:
        """Get the directory for a specific evaluation."""
        return self.evals_dir / evaluation_name

    def get_instance_dir(self, evaluation_name: str, instance_id: str) -> Path:
        """Get the directory for a specific instance within an evaluation."""
        return self.get_evaluation_dir(evaluation_name) / instance_id

    def create_evaluation(
        self,
        dataset_name: str,
        instance_ids: List[str],
        agent_id: str,
        model_id: str,
        num_workers: int = 1,
        max_iterations: int = 10,
        max_expansions: int = 1,
    ) -> str:
        """Create a new evaluation and return its ID."""
        evaluation_name = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{dataset_name}_{agent_id}_{model_id}_iter_{max_iterations}"
        logger.info(f"Creating evaluation: {evaluation_name}")
        
        eval_dir = self.get_evaluation_dir(evaluation_name)
        eval_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Evaluation directory created: {eval_dir}")

        settings = TreeSearchSettings(
            agent_id=agent_id,
            model_id=model_id,
            max_iterations=max_iterations,
            max_expansions=max_expansions
        )

        evaluation = Evaluation(
            evaluation_name=evaluation_name,
            dataset_name=dataset_name,
            status=EvaluationStatus.PENDING,
            num_workers=num_workers,
            settings=settings,
            instances=[
                EvaluationInstance(instance_id=instance_id)
                for instance_id in instance_ids
            ]
        )

        self._save_evaluation(evaluation)

        for instance in evaluation.instances:
            instance_dir = self.get_instance_dir(evaluation_name, instance.instance_id)
            instance_dir.mkdir(parents=True, exist_ok=True)
            self._save_instance(evaluation_name, instance)

        logger.info(f"Evaluation created: {evaluation_name} with {len(evaluation.instances)} instances")
        return evaluation_name

    async def _handle_evaluation_event(self, trajectory_id: str, event: EvaluationEvent):
        """Handle events from the evaluation runner."""
        if not isinstance(event, EvaluationEvent):
            return

        evaluation = self._load_evaluation(event.evaluation_name)
        if not evaluation:
            logger.warning(f"Received event for unknown evaluation: {event.evaluation_name}")
            return

        if event.event_type == "evaluation_started":
            evaluation.status = EvaluationStatus.RUNNING
            evaluation.started_at = datetime.now(timezone.utc)
            self._save_evaluation(evaluation)

        elif event.event_type == "evaluation_completed":
            evaluation.status = EvaluationStatus.COMPLETED
            evaluation.completed_at = datetime.now(timezone.utc)
            self._save_evaluation(evaluation)

        elif event.event_type == "instance_completed":
            instance_id = event.data["instance_id"]
            instance = evaluation.get_instance(instance_id)
            if instance:
                instance.status = InstanceStatus.COMPLETED
                instance.completed_at = datetime.now(timezone.utc)
                self._save_instance(evaluation.evaluation_name, instance)

        elif event.event_type == "instance_error":
            instance_id = event.data["instance_id"]
            error = event.data["error"]
            instance = evaluation.get_instance(instance_id)
            if instance:
                instance.status = InstanceStatus.ERROR
                instance.error = error
                self._save_instance(evaluation.evaluation_name, instance)

        # Log status summary after handling any event
        completed = sum(1 for i in evaluation.instances if i.status == InstanceStatus.COMPLETED)
        running = sum(1 for i in evaluation.instances if i.status == InstanceStatus.STARTED)
        pending = sum(1 for i in evaluation.instances if i.status == InstanceStatus.PENDING)
        errors = sum(1 for i in evaluation.instances if i.status == InstanceStatus.ERROR)
        
        logger.info(
            f"Evaluation {evaluation.evaluation_name} status - "
            f"Completed: {completed}, Running: {running}, Pending: {pending}, Errors: {errors}"
        )

        # Check if all instances are finished
        total_finished = completed + errors
        if total_finished == len(evaluation.instances) and evaluation.status == EvaluationStatus.RUNNING:
            logger.info(f"All instances finished for evaluation {evaluation.evaluation_name}")
            evaluation.status = EvaluationStatus.COMPLETED
            evaluation.completed_at = datetime.now(timezone.utc)
            self._save_evaluation(evaluation)
            
            # Emit completion event
            await event_bus.publish(
                EvaluationEvent(
                    evaluation_name=evaluation.evaluation_name,
                    event_type="evaluation_completed",
                    data={
                        "total_completed": completed,
                        "total_errors": errors
                    }
                )
            )

    async def start_evaluation(
        self,
        evaluation_name: str,
        agent_id: str,
        model_id: str,
        num_workers: int,
        max_iterations: int,
        max_expansions: int,
    ):
        """Start an evaluation run."""
        logger.info(f"Starting evaluation: {evaluation_name}")
        evaluation = self._load_evaluation(evaluation_name)
        if not evaluation:
            logger.error(f"Evaluation {evaluation_name} not found")
            raise ValueError(f"Evaluation {evaluation_name} not found")

        if evaluation.status in [EvaluationStatus.RUNNING, EvaluationStatus.COMPLETED]:
            logger.warning(f"Evaluation {evaluation_name} cannot be started in its current state: {evaluation.status}")
            raise ValueError("Evaluation cannot be started in its current state")

        tree_settings = TreeSearchSettings(
            agent_id=agent_id,
            model_id=model_id,
            max_iterations=max_iterations,
            max_expansions=max_expansions
        )
        
        evaluation.settings = tree_settings
        self._save_evaluation(evaluation)

        runner = EvaluationRunner(
            evaluation=evaluation,
            num_workers=num_workers,
            evaluations_dir=str(self.evals_dir)
        )

        try:
            await runner.run_evaluation()
        except Exception as e:
            evaluation.status = EvaluationStatus.ERROR
            evaluation.error = str(e)
            self._save_evaluation(evaluation)
            raise

    def save_evaluation(self, evaluation: Evaluation):
        """Save evaluation metadata to evaluation.json."""
        self._save_evaluation(evaluation)
        for instance in evaluation.instances:
            self._save_instance(evaluation.evaluation_name, instance)

    def _save_evaluation(self, evaluation: Evaluation):
        """Save evaluation metadata to evaluation.json."""
        eval_path = self.get_evaluation_dir(evaluation.evaluation_name) / "evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(evaluation.model_dump(), f, indent=2, default=str)

    def _save_instance(self, evaluation_name: str, instance: EvaluationInstance):
        """Save instance metadata to instance.json."""
        instance_path = self.get_instance_dir(evaluation_name, instance.instance_id) / "instance.json"
        with open(instance_path, "w") as f:
            json.dump(instance.model_dump(), f, indent=2, default=str)

    def _load_evaluation(self, evaluation_name: str) -> Optional[Evaluation]:
        """Load evaluation metadata from evaluation.json."""
        eval_path = self.get_evaluation_dir(evaluation_name) / "evaluation.json"
        if not eval_path.exists():
            return None
        
        with open(eval_path) as f:
            data = json.load(f)
            return Evaluation.model_validate(data)

    def list_evaluations(self) -> List[Evaluation]:
        """List all evaluations with their metadata."""
        evaluations = []
        for eval_dir in self.evals_dir.glob("eval_*"):
            try:
                eval_file = eval_dir / "evaluation.json"
                if eval_file.exists():
                    with open(eval_file, "r") as f:
                        data = json.load(f)
                        evaluation = Evaluation.model_validate(data)
                        evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Failed to load evaluation {eval_dir.name}: {e}")
                continue
        
        return sorted(evaluations, key=lambda x: x.start_time, reverse=True)

    def get_dataset_instance_ids(self, dataset_name: str) -> List[str]:
        """Get instance IDs for a dataset."""
        dataset_path = f"datasets/{dataset_name}_dataset.json"
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset {dataset_name} not found at {dataset_path}")
            
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            return dataset["instance_ids"]

    async def wait_for_completion(self, evaluation_name: str, check_interval: float = 1.0) -> Evaluation:
        """Wait for an evaluation to complete and return the final evaluation object."""
        last_status = None
        while True:
            evaluation = self._load_evaluation(evaluation_name)
            if not evaluation:
                raise ValueError(f"Evaluation {evaluation_name} not found")

            # Log status changes
            if evaluation.status != last_status:
                logger.info(f"Evaluation {evaluation_name} status changed to: {evaluation.status}")
                last_status = evaluation.status

            if evaluation.status == EvaluationStatus.COMPLETED:
                logger.info(f"Evaluation {evaluation_name} completed successfully")
                return evaluation
            elif evaluation.status == EvaluationStatus.ERROR:
                logger.error(f"Evaluation {evaluation_name} failed with error: {evaluation.error}")
                raise RuntimeError(f"Evaluation failed: {evaluation.error}")

            # Log instance status summary
            completed = sum(1 for i in evaluation.instances if i.status == InstanceStatus.COMPLETED)
            running = sum(1 for i in evaluation.instances if i.status == InstanceStatus.STARTED)
            pending = sum(1 for i in evaluation.instances if i.status == InstanceStatus.PENDING)
            errors = sum(1 for i in evaluation.instances if i.status == InstanceStatus.ERROR)
            
            logger.debug(f"Status summary - Completed: {completed}, Running: {running}, Pending: {pending}, Errors: {errors}")

            await asyncio.sleep(check_interval)