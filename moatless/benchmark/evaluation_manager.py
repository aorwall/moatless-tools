import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import filelock

from moatless.benchmark.evaluation_runner import EvaluationRunner, Evaluation, EvaluationInstance
from moatless.benchmark.schema import EvaluationEvent
from moatless.benchmark.schema import EvaluationStatus
from moatless.events import event_bus
from moatless.utils.moatless import get_moatless_dir
from moatless.flow.manager import get_flow_config

logger = logging.getLogger(__name__)

class EvaluationManager:
    def __init__(self):
        self.evals_dir = get_moatless_dir() / "evals"
        self.evals_dir.mkdir(parents=True, exist_ok=True)
        event_bus.subscribe(self._handle_evaluation_event)
        self.locks_dir = self.evals_dir / ".locks"
        self.locks_dir.mkdir(exist_ok=True)

    def get_evaluation_dir(self, evaluation_name: str) -> Path:
        """Get the directory for a specific evaluation."""
        return self.evals_dir / evaluation_name

    def get_instance_dir(self, evaluation_name: str, instance_id: str) -> Path:
        """Get the directory for a specific instance within an evaluation."""
        return self.get_evaluation_dir(evaluation_name) / instance_id

    def create_evaluation(
        self,
        flow_id: str,
        model_id: str,
        evaluation_name: str | None = None,
        dataset_name: str | None = None,
        instance_ids: Optional[List[str]] = None,
    ) -> Evaluation:
        """Create a new evaluation and return its ID."""
        if not dataset_name and not instance_ids:
            raise ValueError("Either dataset_name or instance_ids must be provided")
        
        if not dataset_name:
            dataset_name = "instance_ids"

        if not evaluation_name:
            evaluation_name = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{flow_id}_{model_id}_{dataset_name}"
        logger.info(f"Creating evaluation: {evaluation_name}")
        if not instance_ids:
            instance_ids = self.get_dataset_instance_ids(dataset_name)
            logger.info(f"Found {len(instance_ids)} instances in dataset {dataset_name}")

        eval_dir = self.get_evaluation_dir(evaluation_name)
        eval_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Evaluation directory created: {eval_dir}")

        flow = get_flow_config(flow_id)
        if not flow:
            raise ValueError(f"Flow {flow_id} not found")
        
        logger.info(f"Flow: {flow}")

        evaluation = Evaluation(
            evaluation_name=evaluation_name,
            dataset_name=dataset_name,
            status=EvaluationStatus.PENDING,
            flow_id=flow_id,
            model_id=model_id,
            instances=[
                EvaluationInstance(instance_id=instance_id)  # Will default to CREATED state
                for instance_id in instance_ids
            ]
        )

        self._save_evaluation(evaluation)
        logger.info(f"Evaluation created: {evaluation_name} with {len(evaluation.instances)} instances")
        return evaluation

    async def _handle_evaluation_event(self, trajectory_id: str | None, event: EvaluationEvent):
        """Handle events from evaluation runners."""
        if not isinstance(event, EvaluationEvent):
            return

        evaluation = self._load_evaluation(event.evaluation_name)
        if not evaluation:
            logger.warning(f"Received event for unknown evaluation: {event.evaluation_name}")
            return

        # Update evaluation based on event
#        self._update_evaluation_from_event(evaluation, event)
        self._save_evaluation(evaluation)

        # Log status summary after handling any event
        summary = evaluation.get_summary()
        logger.info(
            f"Evaluation {evaluation.evaluation_name} status - "
            f"Created: {summary['counts'].get('created', 0)}, "
            f"Setting up: {summary['counts'].get('setting_up', 0)}, "
            f"Pending: {summary['counts'].get('pending', 0)}, "
            f"Running: {summary['counts'].get('running', 0)}, "
            f"Evaluating: {summary['counts'].get('evaluating', 0)}, "
            f"Evaluated: {summary['counts'].get('evaluated', 0)}, "
            f"Errors: {summary['counts'].get('errors', 0)}"
        )

    async def create_evaluation_runner(
        self,
        evaluation_name: str,
        num_concurrent_instances: int
    ) -> EvaluationRunner:
        """Create and configure an evaluation runner."""
        logger.info(f"Creating runner for evaluation: {evaluation_name}")
        evaluation = self._load_evaluation(evaluation_name)
        if not evaluation:
            logger.error(f"Evaluation {evaluation_name} not found")
            raise ValueError(f"Evaluation {evaluation_name} not found")

        if evaluation.status in [EvaluationStatus.RUNNING, EvaluationStatus.COMPLETED]:
            logger.warning(f"Evaluation {evaluation_name} cannot be started in its current state: {evaluation.status}")
            raise ValueError("Evaluation cannot be started in its current state")
        
        self._save_evaluation(evaluation)

        runner = EvaluationRunner(
            evaluation=evaluation,
            num_concurrent_instances=num_concurrent_instances,
            evaluations_dir=str(self.evals_dir)
        )

        return runner

    def save_evaluation(self, evaluation: Evaluation):
        """Save evaluation metadata to evaluation.json."""
        self._save_evaluation(evaluation)

    def _save_evaluation(self, evaluation: Evaluation):
        """Save evaluation metadata to evaluation.json in a thread-safe manner."""
        eval_path = self.get_evaluation_dir(evaluation.evaluation_name) / "evaluation.json"
        lock_path = self.locks_dir / f"{evaluation.evaluation_name}.lock"
        
        with filelock.FileLock(lock_path):
            with open(eval_path, "w") as f:
                json.dump(evaluation.model_dump(), f, indent=2, default=str)

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
        
        return sorted(evaluations, key=lambda x: x.created_at, reverse=True)

    def get_dataset_instance_ids(self, dataset_name: str) -> List[str]:
        """Get instance IDs for a dataset."""
        dataset_path = Path(__file__).parent / "datasets" / f"{dataset_name}_dataset.json"
        if not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_name} not found at {dataset_path}")
            
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            return dataset["instance_ids"]
