import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid
import aiofiles
import aiofiles.os
import traceback

from moatless.benchmark.report import BenchmarkResult, to_result
from moatless.evaluation.run_instance import run_instance
from moatless.evaluation.schema import (
    EvaluationEvent, 
    EvaluationStatus,
    InstanceStatus, 
    Evaluation, 
    EvaluationInstance
)
from moatless.evaluation.utils import get_moatless_instance
from moatless.events import BaseEvent, event_bus
from moatless.flow.flow import AgenticFlow
from moatless.runner.rq import RQRunner
from moatless.runner.runner import (
    EvaluationJobStatus,
    JobInfo,
    JobStatus
)
from moatless.utils.moatless import get_moatless_dir, get_moatless_trajectory_dir
from moatless.flow.manager import get_flow_config, create_flow

from opentelemetry import trace

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

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.evals_dir = get_moatless_dir() / "evals"
        self.evals_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir = self.evals_dir / ".locks"
        self.locks_dir.mkdir(exist_ok=True)
        self.redis_url = redis_url
        self.runner = RQRunner()

    async def initialize(self):
        await event_bus.subscribe(self._handle_event)
    
    async def _handle_event(self, event: BaseEvent):
        """Handle events from evaluation runners."""
        logger.info(f"Received event: {event.scope}:{event.event_type}")
        if not event.scope in ["flow", "evaluation"]:
            return
        
        if not event.project_id:
            logger.error(f"Received event with no project id: {event}")
            return
        
        evaluation = await self._load_evaluation(event.project_id)
        if not evaluation:
            logger.info(f"Received event with a project id not found in evaluations: {event.project_id}")
            return
        
        instance = evaluation.get_instance(event.trajectory_id)
        if not instance:
            logger.warning(f"Received event for unknown instance: {event.trajectory_id}")
            return
        
        if event.scope == "flow" and event.event_type == "started":
            instance.status = InstanceStatus.RUNNING
            instance.started_at = event.timestamp
        elif event.scope == "flow" and event.event_type == "completed":
            instance = await self._process_trajectory_results(evaluation, instance)
            instance.status = InstanceStatus.COMPLETED
            instance.completed_at = event.timestamp
        elif event.scope == "evaluation" and event.event_type == "started":
            instance.status = InstanceStatus.EVALUATING
            instance.start_evaluating_at = event.timestamp
        elif event.scope == "evaluation" and event.event_type == "completed":
            instance = await self._process_trajectory_results(evaluation, instance)
            instance.status = InstanceStatus.EVALUATED
            instance.evaluated_at = event.timestamp
        elif event.event_type == "error":
            instance.status = InstanceStatus.ERROR
            instance.error = event.data.get("error")
            instance.error_at = event.timestamp

        await self._save_evaluation(evaluation)

    async def _process_trajectory_results(self, evaluation: Evaluation, instance: EvaluationInstance) -> EvaluationInstance:
        if instance.status == InstanceStatus.EVALUATED:
            return instance
        
        trajectory_dir = get_moatless_trajectory_dir(trajectory_id=instance.instance_id, project_id=evaluation.evaluation_name)
        if not trajectory_dir.exists():
            logger.warning(f"Trajectory directory not found for instance: {instance.instance_id}")
            return
        
        flow = AgenticFlow.from_dir(trajectory_dir=trajectory_dir)
        events = await event_bus.read_events(project_id=evaluation.evaluation_name, trajectory_id=instance.instance_id)

        if not instance.completed_at and not instance.evaluated_at:
            # TODO: Handle trajectories with multiple leaf nodes

            if not instance.started_at:
                event = next((e for e in events if e.scope == "flow" and e.event_type == "started"), None)
                if event:
                    instance.started_at = event.timestamp
                else:
                    instance.started_at = datetime.now(timezone.utc)
        
            event = next((e for e in events if e.scope == "flow" and e.event_type == "completed"), None)
            if event:
                instance.completed_at = event.timestamp
            else:
                instance.completed_at = datetime.now(timezone.utc)

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

        if not instance.evaluated_at:
            eval_result_path = trajectory_dir / "eval_result.json"

            event = next((e for e in events if e.scope == "evaluation" and e.event_type == "started"), None)
            if event:
                instance.start_evaluating_at = event.timestamp
            else:
                instance.start_evaluating_at = datetime.now(timezone.utc)

            event = next((e for e in events if e.scope == "evaluation" and e.event_type == "completed"), None)
            if event:
                instance.evaluated_at = event.timestamp
            else:
                instance.evaluated_at = datetime.now(timezone.utc)

            if not eval_result_path.exists():
                logger.info(f"Evaluation result not found for instance: {instance.instance_id}")
                return
            
            with open(eval_result_path, "r") as f:
                eval_result = json.load(f)
                
            benchmark_result = to_result(
                node=flow.root,
                eval_report=eval_result,
                instance_id=instance.instance_id
            )
            
            instance.benchmark_result = benchmark_result
            instance.resolved = benchmark_result.resolved
            instance.status = InstanceStatus.EVALUATED

        return instance

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

        for instance in evaluation.instances:
            moatless_instance = get_moatless_instance(instance_id=instance.instance_id)
            problem_statement = f"Solve the following issue:\n{moatless_instance['problem_statement']}"
            trajectory_dir = get_moatless_trajectory_dir(trajectory_id=instance.instance_id, project_id=evaluation_name)
            flow = create_flow(
                id=evaluation.flow_id,
                message=problem_statement,
                trajectory_id=instance.instance_id,
                persist_dir=trajectory_dir,
                model_id=evaluation.model_id,
                metadata={"instance_id": instance.instance_id},
            )
            flow.persist()

        eval_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Evaluation directory created: {eval_dir}")

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
    async def start_evaluation(
        self,
        evaluation_name: str
    ) -> Evaluation:
        """Start an evaluation and return the updated evaluation object."""
        evaluation = await self._load_evaluation(evaluation_name)
        
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")
        
        started_instances = 0
        for instance in evaluation.instances:
            if instance.status == InstanceStatus.EVALUATED:
                continue

            if instance.status == InstanceStatus.ERROR:
                logger.info(f"Evaluation {evaluation_name} has error instance {instance.instance_id}, resetting and retrying")
                if instance.completed_at:
                    instance.status = InstanceStatus.COMPLETED
                    instance.start_evaluating_at = None
                    instance.evaluated_at = None
                else:
                    instance.status = InstanceStatus.PENDING
                    instance.started_at = None
                    
                instance.error = None
                instance.error_at = None
            
            if await self.runner.start_job(project_id=evaluation.evaluation_name, trajectory_id=instance.instance_id, job_func=run_instance):
                started_instances += 1

        logger.info(f"Started {started_instances} instances for evaluation {evaluation.evaluation_name}")

        if started_instances:
            evaluation.status = EvaluationStatus.RUNNING
            evaluation.started_at = datetime.now(timezone.utc)
            await self._save_evaluation(evaluation)

        return evaluation
    
    async def get_evaluation(self, evaluation_name: str) -> Evaluation:
        """Get an evaluation by name."""
        evaluation = await self._load_evaluation(evaluation_name)
        if evaluation.status == EvaluationStatus.COMPLETED:
            return evaluation
        
        if evaluation.status == EvaluationStatus.RUNNING:
            evaluation_is_running = False
            evaluation_is_completed = True
            for instance in evaluation.instances:
                if instance.status not in [InstanceStatus.EVALUATED, InstanceStatus.ERROR]:
                    
                    evaluation_is_completed = False

                    job_status = await self.get_job_status(evaluation_name, instance.instance_id)

                    if job_status in [JobStatus.RUNNING, JobStatus.QUEUED]:
                        logger.info(f"Evaluation {evaluation_name} is running, instance {instance.instance_id} is {instance.job_status}")
                        evaluation_is_running = True

            if evaluation_is_completed:
                evaluation.status = EvaluationStatus.COMPLETED
            elif not evaluation_is_running:
                evaluation.status = EvaluationStatus.PAUSED

            await self._save_evaluation(evaluation)

        return evaluation

    async def get_evaluation_status(self, evaluation_name: str) -> EvaluationJobStatus:
        """Get the status of all jobs for an evaluation."""
        evaluation = await self._load_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")
        
        return await self.runner.get_evaluation_status(evaluation)
    
    async def get_job_status(self, evaluation_name: str, instance_id: str) -> JobStatus:
        """Get the status of jobs for a specific instance."""
        try:
            job_status = await self.runner.get_job(project_id=evaluation_name, trajectory_id=instance_id)
            if job_status:
                return job_status.status
            else:
                return JobStatus.PENDING
        except Exception as e:
            logger.exception(f"Failed to get job status for evaluation {evaluation_name} and instance {instance_id}: {e}")
            return JobStatus.FAILED
    
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
        
        for instance in evaluation.instances:
            await self.runner.cancel_job(evaluation_name, instance.instance_id)
        
        evaluation.status = EvaluationStatus.PAUSED
        await self._save_evaluation(evaluation)

    @tracer.start_as_current_span("EvaluationManager.retry_instance")
    async def retry_instance(self, evaluation_name: str, instance_id: str):
        """Retry a failed job for a specific instance."""
        evaluation = await self._load_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")
        
        instance = evaluation.get_instance(instance_id)
        if not instance:
            raise ValueError(f"Instance {instance_id} not found in evaluation {evaluation_name}")
        
        # Reset instance state
        instance.status = InstanceStatus.PENDING
        instance.error = None
        instance.error_at = None
        instance.started_at = None
        instance.completed_at = None
        instance.evaluated_at = None
        instance.benchmark_result = None
        
        # If the evaluation was in ERROR state and this was the only failed instance,
        # update the evaluation status to RUNNING
        if evaluation.status == EvaluationStatus.ERROR:
            error_instances = [i for i in evaluation.instances if i.status == InstanceStatus.ERROR]
            if len(error_instances) <= 1:  # This was the only error or there are no more errors
                evaluation.status = EvaluationStatus.RUNNING
                evaluation.error = None
        
        await self.runner.cancel_job(evaluation_name, instance_id)
        await self.runner.start_job(evaluation_name, instance_id, run_instance)
        await self._save_evaluation(evaluation)

    async def process_evaluation_results(self, evaluation_name: str) -> Evaluation:
        """Process all instances in an evaluation to ensure results are in sync."""
        evaluation = await self._load_evaluation(evaluation_name)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_name} not found")
        
        for instance in evaluation.instances:
            if instance.status != InstanceStatus.EVALUATED:
                instance = await self._process_trajectory_results(evaluation, instance)
        
        # Check if all instances have been evaluated, and update evaluation status
        all_evaluated = all(instance.status == InstanceStatus.EVALUATED for instance in evaluation.instances)
        
        if all_evaluated and evaluation.status != EvaluationStatus.COMPLETED:
            evaluation.status = EvaluationStatus.COMPLETED
            evaluation.completed_at = datetime.now(timezone.utc)
            logger.info(f"All instances evaluated, setting evaluation {evaluation_name} status to COMPLETED")
        
        await self._save_evaluation(evaluation)
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

    async def _load_evaluation(self, evaluation_name: str) -> Optional[Evaluation]:
        """Load evaluation metadata from evaluation.json or active runner if exists."""
        # First check if there's an active runner for this evaluation
        # Otherwise load from file
        eval_path = self.get_evaluation_dir(evaluation_name) / "evaluation.json"
        if not eval_path.exists():
            return None
        
        with open(eval_path) as f:
            data = json.load(f)
            evaluation = Evaluation.model_validate(data)
            return evaluation

    async def list_evaluations(self) -> List[Evaluation]:
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

    def get_dataset_instance_ids(self, dataset_name: str) -> List[str]:
        """Get instance IDs for a dataset."""
        dataset_path = Path(__file__).parent / "datasets" / f"{dataset_name}_dataset.json"
        if not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_name} not found at {dataset_path}")
            
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            return dataset["instance_ids"]
