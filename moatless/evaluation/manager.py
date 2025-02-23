import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict
import asyncio
import uuid
import aiofiles
import aiofiles.os
import tempfile
import threading

import filelock

from moatless.evaluation.runner import EvaluationRunner, Evaluation, EvaluationInstance
from moatless.evaluation.schema import EvaluationEvent, EvaluationStatus, InstanceStatus
from moatless.events import event_bus
from moatless.telemetry import instrument, set_attribute
from moatless.utils.moatless import get_moatless_dir
from moatless.flow.manager import get_flow_config
from moatless.completion.manager import create_completion_model

logger = logging.getLogger(__name__)

class EvaluationManager:
    _instance = None

    @classmethod
    def get_instance(cls) -> "EvaluationManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.evals_dir = get_moatless_dir() / "evals"
        self.evals_dir.mkdir(parents=True, exist_ok=True)
        event_bus.subscribe(self._handle_evaluation_event)
        self.locks_dir = self.evals_dir / ".locks"
        self.locks_dir.mkdir(exist_ok=True)
        self._active_runners: Dict[str, EvaluationRunner] = {}
        self._initialized = True

    def get_evaluation_dir(self, evaluation_name: str) -> Path:
        """Get the directory for a specific evaluation."""
        return self.evals_dir / evaluation_name

    def get_instance_dir(self, evaluation_name: str, instance_id: str) -> Path:
        """Get the directory for a specific instance within an evaluation."""
        return self.get_evaluation_dir(evaluation_name) / instance_id

    async def create_evaluation(
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
        if eval_dir.exists():
            logger.warning(f"Evaluation directory already exists: {eval_dir}")
            raise ValueError(f"Evaluation already exists: {evaluation_name}")
        
        eval_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Evaluation directory created: {eval_dir}")

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

        await self._save_evaluation(evaluation)
        logger.info(f"Evaluation created: {evaluation_name} with {len(evaluation.instances)} instances")
        return evaluation

    async def _handle_evaluation_event(self, trajectory_id: str | None, project_id: str | None, event: EvaluationEvent):
        """Handle events from evaluation runners."""
        if not isinstance(event, EvaluationEvent):
            return

        # Get evaluation from active runner if exists, otherwise load from file
        evaluation = None
        if event.evaluation_name in self._active_runners:
            evaluation = self._active_runners[event.evaluation_name].evaluation
        else:
            evaluation = await self._load_evaluation(event.evaluation_name)

        if not evaluation:
            logger.warning(f"Received event for unknown evaluation: {event.evaluation_name}")
            return

        # If the event is evaluation_completed or evaluation_error, remove the runner
        if event.event_type in ["evaluation_completed", "evaluation_error"]:
            if event.evaluation_name in self._active_runners:
                del self._active_runners[event.evaluation_name]

        await self._save_evaluation(evaluation)

    @instrument(name="EvaluationManager.start_evaluation")
    async def start_evaluation(
        self,
        evaluation_name: str,
        num_concurrent_instances: int
    ) -> Evaluation:
        set_attribute("evaluation_name", evaluation_name)
        set_attribute("num_concurrent_instances", num_concurrent_instances)
        
        """Start an evaluation and return the updated evaluation object."""
        evaluation = await self._load_evaluation(evaluation_name)
        
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")
            
        if evaluation.status in [EvaluationStatus.RUNNING]:
            raise ValueError(f"Evaluation {evaluation_name} cannot be started in its current state: {evaluation.status}")
        
        # Allow restart if evaluation is completed but has error instances
        if evaluation.status == EvaluationStatus.COMPLETED:

            error_instances = [instance for instance in evaluation.instances if instance.status == InstanceStatus.ERROR]
            if len(error_instances) == 0:
                raise ValueError(f"Evaluation {evaluation_name} is already completed successfully")
            else:
                logger.info(f"Evaluation {evaluation_name} is completed but has error instances, resetting and restarting")
                for instance in error_instances:
                    if instance.completed_at:
                        instance.status = InstanceStatus.COMPLETED
                    else:
                        instance.status = InstanceStatus.PENDING
                        instance.started_at = None
                    
                    instance.error = None
                    instance.resolved = None
                    instance.iterations = None
                    instance.usage = None
                    instance.benchmark_result = None
                    instance.duration = None

        logger.info(f"Starting evaluation: {evaluation_name}")

        flow = get_flow_config(evaluation.flow_id)
        if not flow:
            raise ValueError(f"Flow {evaluation.flow_id} not found")
        
        logger.info(flow)

        completion_model = create_completion_model(evaluation.model_id)
        if not completion_model:
            raise ValueError(f"Model {evaluation.model_id} not found")
        
        logger.info(completion_model)

        evaluation.status = EvaluationStatus.RUNNING
        await self._save_evaluation(evaluation)
        
        # Create runner instance
        runner = EvaluationRunner(
            evaluation=evaluation,
            num_concurrent_instances=num_concurrent_instances,
            evaluations_dir=str(self.evals_dir)
        )
        
        # Get current trace context before spawning thread
        from opentelemetry import trace, context
        current_context = context.get_current()
        current_span = trace.get_current_span()
        
        # Start runner in a separate thread with its own event loop
        def run_evaluation_in_thread():
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Attach the parent trace context
                token = context.attach(current_context)
                try:
                    # Run the evaluation
                    loop.run_until_complete(runner.run_evaluation())
                finally:
                    context.detach(token)
                    loop.close()
            except Exception as e:
                logger.exception("Error in evaluation thread setup")

        # Start the thread
        thread = threading.Thread(target=run_evaluation_in_thread, daemon=True)
        thread.start()
        
        self._active_runners[evaluation.evaluation_name] = runner
        return evaluation

    async def save_evaluation(self, evaluation: Evaluation):
        """Save evaluation metadata to evaluation.json."""
        await self._save_evaluation(evaluation)

    async def _save_evaluation(self, evaluation: Evaluation):
        """Save evaluation metadata atomically using a temporary file approach."""
        eval_path = self.get_evaluation_dir(evaluation.evaluation_name) / "evaluation.json"
        
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp_path = eval_path.with_suffix(f'.json.{uuid.uuid4()}.tmp')
        
        try:
            async with aiofiles.open(tmp_path, "w") as f:
                json_data = json.dumps(evaluation.model_dump(), indent=2, default=str)
                await f.write(json_data)
                await f.flush()  # Ensure all data is written
                
            await aiofiles.os.rename(tmp_path, eval_path)
            
        except Exception as e:
            logger.error(f"Failed to save evaluation {evaluation.evaluation_name}: {e}")
        
        try:
            await aiofiles.os.remove(tmp_path)
        except:
            pass

    def _should_be_paused(self, evaluation: Evaluation, runner: Optional[EvaluationRunner] = None) -> bool:
        """
        Determine if an evaluation should be marked as PAUSED.
        
        An evaluation should be PAUSED if:
        1. It was previously RUNNING
        2. Either has no runner, or has a runner with no active tasks
        3. Hasn't reached a terminal state (COMPLETED/ERROR)
        """
        if evaluation.status != EvaluationStatus.RUNNING:
            return False
        
        if evaluation.status in [EvaluationStatus.COMPLETED, EvaluationStatus.ERROR]:
            return False
        
        if runner:
            # TODO: Check something here?
            return False

        # If no runner provided, then it should be paused
        return True

    async def _load_evaluation(self, evaluation_name: str) -> Optional[Evaluation]:
        """Load evaluation metadata from evaluation.json or active runner if exists."""
        # First check if there's an active runner for this evaluation
        logger.info(f"Loading evaluation: {evaluation_name}, active runners: {list(self._active_runners.keys())}")
        if evaluation_name in self._active_runners:
            runner = self._active_runners[evaluation_name]
            evaluation = runner.evaluation
            
            if self._should_be_paused(evaluation, runner):
                evaluation.status = EvaluationStatus.PAUSED
                
            return evaluation

        # Otherwise load from file
        eval_path = self.get_evaluation_dir(evaluation_name) / "evaluation.json"
        if not eval_path.exists():
            return None
        
        with open(eval_path) as f:
            data = json.load(f)
            evaluation = Evaluation.model_validate(data)
            if self._should_be_paused(evaluation):
                evaluation.status = EvaluationStatus.PAUSED
                await self._save_evaluation(evaluation)
            return evaluation

    async def list_evaluations(self) -> List[Evaluation]:
        """List all evaluations with their metadata."""
        evaluations = []
        for eval_dir in self.evals_dir.glob("eval_*"):
            try:
                evaluation_name = eval_dir.name
                # First try to get from active runners
                if evaluation_name in self._active_runners:
                    runner = self._active_runners[evaluation_name]
                    evaluation = runner.evaluation
                    
                    if self._should_be_paused(evaluation, runner):
                        evaluation.status = EvaluationStatus.PAUSED
                        
                    evaluations.append(evaluation)
                    continue

                # Otherwise load from file
                eval_file = eval_dir / "evaluation.json"
                if eval_file.exists():
                    with open(eval_file, "r") as f:
                        data = json.load(f)
                        evaluation = Evaluation.model_validate(data)
                        if self._should_be_paused(evaluation):
                            evaluation.status = EvaluationStatus.PAUSED
                            await self._save_evaluation(evaluation)
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
