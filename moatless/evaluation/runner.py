import json
import logging
import os
import shutil
import time
import traceback
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict, List
import uuid
from functools import wraps
from redis import Redis
import fcntl

from moatless.benchmark.report import to_result
from moatless.benchmark.swebench.utils import create_repository, create_repository_async, create_index_async
from moatless.evaluation.utils import get_moatless_instance
from moatless.evaluation.schema import (
    Evaluation,
    EvaluationInstance,
    EvaluationStatus,
    InstanceStatus,
    EvaluationEvent,
)
from moatless.events import event_bus
from moatless.flow.flow import AgenticFlow
from moatless.flow.manager import create_flow
from moatless.node import Node
from moatless.runtime.testbed import TestbedEnvironment
from moatless.utils.moatless import get_moatless_dir, get_moatless_trajectory_dir
from moatless.workspace import Workspace
from moatless.telemetry import (
    instrument,
    set_attribute,
    extract_trace_context,
    wrap_with_trace,
    extract_context_data,
    MoatlessQueue,
    run_async
)

logger = logging.getLogger(__name__)

def _load_evaluation(evaluation_name: str) -> Evaluation:
    """Load evaluation data from file."""
    eval_path = get_moatless_dir() / "evals" / evaluation_name / "evaluation.json"
    with open(eval_path) as f:
        return Evaluation.model_validate(json.load(f))

def _save_evaluation(evaluation: Evaluation):
    """Save evaluation data atomically."""
    eval_dir = get_moatless_dir() / "evals" / evaluation.evaluation_name
    eval_path = eval_dir / "evaluation.json"
    
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(eval_path, "w") as f:
            json_data = json.dumps(evaluation.model_dump(), indent=2, default=str)
            f.write(json_data)
            f.flush()
            
    except Exception as e:
        logger.error(f"Failed to save evaluation {evaluation.evaluation_name}: {e}")

def _emit_event(evaluation_name: str, event_type: str, data: Any = None):
    """Emit evaluation event."""
    event = EvaluationEvent(
        evaluation_name=evaluation_name,
        event_type=event_type,
        data=data,
    )
    try:
        if hasattr(event_bus, 'publish_sync'):
            event_bus.publish_sync(event)
    except Exception as e:
        logger.error(f"Failed to publish event {event_type}: {e}")

@instrument(name="EvaluationRunner._update_instance_state")
def _update_instance_state(evaluation_name: str, instance_id: str, new_status: InstanceStatus, event_type: str, error: Optional[str] = None):
    """Update instance state and save with file locking."""
    eval_dir = get_moatless_dir() / "evals" / evaluation_name
    eval_path = eval_dir / "evaluation.json"
    lock_path = eval_dir / "evaluation.lock"
    
    # Ensure directories exist
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            
            evaluation = _load_evaluation(evaluation_name)
            instance = evaluation.get_instance(instance_id)
            if not instance:
                return

            instance.status = new_status
            if error:
                instance.error = error

            if new_status == InstanceStatus.COMPLETED:
                instance.completed_at = datetime.now(timezone.utc)
            elif new_status == InstanceStatus.ERROR:
                instance.error_at = datetime.now(timezone.utc)
            elif new_status == InstanceStatus.RUNNING:
                instance.started_at = datetime.now(timezone.utc)
            elif new_status == InstanceStatus.EVALUATING:
                instance.start_evaluating_at = datetime.now(timezone.utc)
            elif new_status == InstanceStatus.EVALUATED:
                instance.evaluated_at = datetime.now(timezone.utc)
                
            # Check if all instances have reached a final state when this instance is marked as evaluated or error
            if new_status in [InstanceStatus.EVALUATED, InstanceStatus.ERROR]:
                all_final = True
                has_error = False
                for inst in evaluation.instances:
                    if inst.status not in [InstanceStatus.EVALUATED, InstanceStatus.ERROR]:
                        all_final = False
                        break
                    if inst.status == InstanceStatus.ERROR:
                        has_error = True
                
                if all_final:
                    evaluation.status = EvaluationStatus.ERROR if has_error else EvaluationStatus.COMPLETED
                    evaluation.completed_at = datetime.now(timezone.utc)
                
            _save_evaluation(evaluation)
            
            event_data = {"instance_id": instance_id}
            if error:
                event_data["error"] = error
            _emit_event(evaluation_name, event_type, event_data)
            
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

@instrument(name="EvaluationRunner.run_instance")
def run_instance(evaluation_name: str, instance_id: str) -> None:
    """Run an instance's agentic flow."""
    set_attribute("instance_id", instance_id)
    set_attribute("evaluation_name", evaluation_name)
    set_attribute("job.type", "run_instance")
    
    try:
        moatless_instance = get_moatless_instance(instance_id=instance_id)
        repository = create_repository(moatless_instance, repo_base_dir="/tmp/moatless_repos")
        
        problem_statement = f"Solve the following issue:\n{moatless_instance['problem_statement']}"
        
        # Create and run flow
        trajectory_dir = get_moatless_trajectory_dir(instance_id, evaluation_name)
        testbed_log_dir = trajectory_dir / "testbed_logs"
        os.makedirs(testbed_log_dir, exist_ok=True)

        code_index = run_async(
            create_index_async(moatless_instance, repository=repository)
        )

        runtime = TestbedEnvironment(
            repository=repository,
            instance_id=instance_id,
            log_dir=testbed_log_dir,
            enable_cache=True,
        )
        
        workspace = Workspace(
            repository=repository,
            code_index=code_index,
            runtime=runtime,
            legacy_workspace=True
        )

        evaluation = _load_evaluation(evaluation_name)
        flow = create_flow(
            id=evaluation.flow_id,
            message=problem_statement,
            run_id=instance_id,
            persist_dir=trajectory_dir,
            model_id=evaluation.model_id,
            workspace=workspace,
            metadata={"instance_id": instance_id},
        )
        flow.maybe_persist()
        _update_instance_state(
            evaluation_name,
            instance_id,
            InstanceStatus.RUNNING,
            "instance_running"
        )

        node = run_async(flow.run())
        logger.info(f"Flow completed for instance {instance_id}")

        if node.error:
            _update_instance_state(
                evaluation_name,
                instance_id,
                InstanceStatus.ERROR,
                "instance_error",
                error=str(node.error)
            )
        else:
            _update_instance_state(
                evaluation_name,
                instance_id,
                InstanceStatus.COMPLETED,
                "instance_completed"
            )

    except Exception as exc:
        logger.exception(f"Error running instance {instance_id}")
        _update_instance_state(
            evaluation_name,
            instance_id,
            InstanceStatus.ERROR,
            "instance_error",
            error=str(exc)
        )
        raise

@instrument(name="EvaluationRunner.evaluate_instance")
def evaluate_instance(evaluation_name: str, instance_id: str) -> None:
    """Evaluate an instance's results."""
    set_attribute("instance_id", instance_id)
    set_attribute("evaluation_name", evaluation_name)
    set_attribute("job.type", "evaluate_instance")
    
    try:
        evaluation = _load_evaluation(evaluation_name)
        instance = evaluation.get_instance(instance_id)
        if not instance:
            logger.warning(f"Instance {instance_id} not found in evaluation {evaluation_name}")
            return

        trajectory_dir = get_moatless_trajectory_dir(instance_id, evaluation_name)
        path = trajectory_dir / "trajectory.json"
        root_node = Node.from_file(path)
        
        eval_report = _evaluate_nodes(instance, root_node, evaluation)

        instance.benchmark_result = to_result(
            node=root_node,
            eval_report=eval_report,
            instance_id=instance_id
        )
        if instance.benchmark_result is not None:
            instance.resolved = instance.benchmark_result.resolved

        instance.status = InstanceStatus.EVALUATED
        _update_instance_state(
            evaluation_name,
            instance_id,
            InstanceStatus.EVALUATED,
            "instance_evaluated"
        )

    except Exception as exc:
        logger.exception(f"Error evaluating instance {instance_id}")
        _update_instance_state(
            evaluation_name,
            instance_id,
            InstanceStatus.ERROR,
            "instance_error",
            error=str(exc)
        )
        raise

def _evaluate_nodes(instance: EvaluationInstance, root_node: Node, evaluation: Evaluation) -> Optional[Dict[str, Any]]:
    leaf_nodes = root_node.get_leaf_nodes()
    eval_result_path = os.path.join(
        get_moatless_trajectory_dir(instance.instance_id, evaluation.evaluation_name),
        "eval_result.json"
    )
    eval_result: Optional[Dict[str, Any]] = None
    if os.path.exists(eval_result_path):
        try:
            with open(eval_result_path) as f:
                loaded = json.load(f)
            if "node_results" in loaded:
                eval_result = loaded
            elif len(loaded) > 0:
                eval_result = {
                    "node_results": loaded,
                    "status": "started",
                    "start_time": datetime.now(timezone.utc).isoformat(),
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse {eval_result_path}")
    if not eval_result:
        eval_result = {
            "node_results": {},
            "status": "started",
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
    unevaluated_nodes = [
        node for node in leaf_nodes if str(node.node_id) not in eval_result["node_results"]
    ]
    if not unevaluated_nodes:
        logger.info(f"All leaf nodes evaluated for instance {instance.instance_id}")
        return

    testbed_log_dir = os.path.join(
        get_moatless_trajectory_dir(instance.instance_id, evaluation.evaluation_name),
        "testbed_logs"
    )
    os.makedirs(testbed_log_dir, exist_ok=True)

    _update_instance_state(
        evaluation.evaluation_name,
        instance.instance_id,
        InstanceStatus.EVALUATING,
        "instance_evaluating"
    )

    moatless_instance = get_moatless_instance(instance_id=instance.instance_id)
    repository = create_repository(moatless_instance)

    runtime = TestbedEnvironment(
        repository=repository,
        instance_id=instance.instance_id,
        log_dir=testbed_log_dir,
        enable_cache=True,
    )
    for i, leaf_node in enumerate(unevaluated_nodes):
        logger.info(f"Evaluating node {leaf_node.node_id} ({i+1}/{len(unevaluated_nodes)})")
        patch = leaf_node.file_context.generate_git_patch(ignore_tests=True)
        if not patch or not patch.strip():
            logger.info(f"No patch for node {leaf_node.node_id}; skipping.")
            continue
        start_time = time.time()
        try:
            result = run_async(runtime.evaluate(patch=patch))
            if result:
                eval_result["node_results"][str(leaf_node.node_id)] = result.model_dump()
                logger.info(f"Evaluated node {leaf_node.node_id} in {time.time()-start_time:.2f}s")
        except Exception:
            logger.exception(f"Error evaluating node {leaf_node.node_id} for instance {instance.instance_id}")
            eval_result["error"] = traceback.format_exc()
        finally:
            eval_result["duration"] = time.time() - start_time
            with open(eval_result_path, "w") as f:
                json.dump(eval_result, f, indent=2)

    return eval_result


@instrument(name="EvaluationRunner.start_evaluation")
def start_evaluation(evaluation: Evaluation, redis_url: str = "redis://localhost:6379") -> None:
    """Start an evaluation by scheduling all instance jobs."""
    try:
        logger.info(f"Starting evaluation {evaluation.evaluation_name}")
        redis_conn = Redis.from_url(redis_url)
        queue = MoatlessQueue(connection=redis_conn, default_timeout=3600)
        
        evaluation.status = EvaluationStatus.RUNNING
        evaluation.started_at = datetime.now(timezone.utc)
        _save_evaluation(evaluation)
        _emit_event(evaluation.evaluation_name, "evaluation_started")

        logger.info(f"Starting evaluation {evaluation.evaluation_name} with {len(evaluation.instances)} instances")

        # Set evaluation attributes in current span
        set_attribute("evaluation.name", evaluation.evaluation_name)
        set_attribute("evaluation.model_id", evaluation.model_id)
        set_attribute("evaluation.flow_id", evaluation.flow_id)
        set_attribute("evaluation.instance_count", len(evaluation.instances))

        for instance in evaluation.instances:
            try:
                if not instance.completed_at:
                    # Schedule run job with parent span context
                    run_job = queue.enqueue(
                        run_instance,
                        kwargs={
                            "evaluation_name": evaluation.evaluation_name,
                            "instance_id": instance.instance_id,
                        },
                        job_id=f"run_{evaluation.evaluation_name}_{instance.instance_id}",
                        result_ttl=3600,
                        job_timeout=3600
                    )
                    logger.info(f"Scheduled run job for instance {instance.instance_id}, job_id: {run_job.id}")
                else:
                    run_job = None

                if not instance.evaluated_at:                    
                    eval_job = queue.enqueue(
                        evaluate_instance,
                        kwargs={
                            "evaluation_name": evaluation.evaluation_name,
                            "instance_id": instance.instance_id,
                        },
                        job_id=f"eval_{evaluation.evaluation_name}_{instance.instance_id}",
                        depends_on=run_job,
                        result_ttl=3600,
                        job_timeout=1800
                    )
                    logger.info(f"Scheduled eval job for instance {instance.instance_id}, depends on {f'run_job.id: {run_job.id}' if run_job else 'none'}, job_id: {eval_job.id}")
                
            except Exception as exc:
                logger.error(f"Failed to schedule instance {instance.instance_id}: {exc}")
                _update_instance_state(
                    evaluation.evaluation_name,
                    instance.instance_id,
                    InstanceStatus.ERROR,
                    "instance_error",
                    error=str(exc)
                )

    except Exception as exc:
        logger.exception("Critical error starting evaluation")
        evaluation.status = EvaluationStatus.ERROR
        evaluation.error = str(exc)
        _save_evaluation(evaluation)
        _emit_event(evaluation.evaluation_name, "evaluation_error", {
            "error": str(exc),
            "traceback": traceback.format_exc()
        })
        raise
