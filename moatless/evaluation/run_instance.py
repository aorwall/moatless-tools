import asyncio
import fcntl
import json
import logging
import os
import shutil
import time
import traceback
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm
from dotenv import load_dotenv
from opentelemetry import trace
from redis import Redis
from rq.job import Dependency

from moatless.benchmark.swebench.utils import create_index_async, create_repository
from moatless.completion.log_handler import LogHandler
from moatless.context_data import current_project_id, current_trajectory_id
from moatless.evaluation.schema import Evaluation, EvaluationEvent, EvaluationStatus, InstanceStatus
from moatless.evaluation.utils import get_moatless_instance
from moatless.events import event_bus
from moatless.flow.flow import AgenticFlow
from moatless.flow.manager import create_flow
from moatless.node import Node
from moatless.repository.repository import Repository
from moatless.runner.utils import cleanup_job_logging, setup_job_logging
from moatless.runtime.testbed import TestbedEnvironment
from moatless.telemetry import run_async, setup_telemetry
from moatless.utils.moatless import get_moatless_trajectory_dir
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.evaluation.runner")

load_dotenv()
setup_telemetry()


def run_instance(project_id: str, trajectory_id: str) -> None:
    """Run an instance's agentic flow."""
    print(f"Running instance {trajectory_id} for project {project_id}")

    trajectory_dir = get_moatless_trajectory_dir(trajectory_id=trajectory_id, project_id=project_id)
    print(f"Setting up job logging for run in {trajectory_dir}")
    original_handlers = setup_job_logging("run", trajectory_dir)

    logger.info(f"current_project_id: {current_project_id}, current {current_trajectory_id}")

    try:
        swebench_instance = get_moatless_instance(instance_id=trajectory_id)
        repository = create_repository(swebench_instance, repo_base_dir="/tmp/moatless_repos")

        testbed_log_dir = trajectory_dir / "testbed_logs"
        os.makedirs(testbed_log_dir, exist_ok=True)

        runtime = TestbedEnvironment(
            repository=repository,
            instance_id=trajectory_id,
            log_dir=str(testbed_log_dir),
            enable_cache=True,
        )

        flow = run_async(
            _run_instance(
                evaluation_name=project_id,
                instance_id=trajectory_id,
                repository=repository,
                runtime=runtime,
                swebench_instance=swebench_instance,
            )
        )
        evaluate_instance(evaluation_name=project_id, instance_id=trajectory_id, root_node=flow.root, runtime=runtime)
    except Exception as e:
        logger.exception(f"Error running instance {trajectory_id}")
        _emit_event(
            evaluation_name=project_id,
            instance_id=trajectory_id,
            scope="evaluation",
            event_type="error",
            data={"error": str(e)},
        )
        raise e
    finally:
        cleanup_job_logging(original_handlers)


async def _run_instance(
    evaluation_name: str, instance_id: str, repository: Repository, runtime: TestbedEnvironment, swebench_instance: dict
) -> None:
    current_project_id.set(evaluation_name)
    current_trajectory_id.set(instance_id)
    logger.info(f"current_project_id: {current_project_id}, current {current_trajectory_id}")
    with tracer.start_as_current_span(f"run_instance_{instance_id}"):
        trajectory_dir = get_moatless_trajectory_dir(trajectory_id=instance_id, project_id=evaluation_name)
        print(f"Setting up job logging for run in {trajectory_dir}")
        litellm.callbacks = [LogHandler(trajectory_dir=str(trajectory_dir))]

        code_index = await create_index_async(swebench_instance, repository=repository)

        workspace = Workspace(repository=repository, code_index=code_index, runtime=runtime, legacy_workspace=True)

        flow = AgenticFlow.from_dir(trajectory_dir=trajectory_dir)

        leaf_nodes = flow.root.get_leaf_nodes()
        for node in leaf_nodes:
            if node.error:
                logger.info(f"Leaf node {node.node_id} has error: {node.error}, resetting node")
                node.reset()
        flow.persist()

        node = await flow.run(workspace=workspace)
        logger.info(f"Flow completed for instance {instance_id}")


def evaluate_instance(evaluation_name: str, instance_id: str, root_node: Node, runtime: TestbedEnvironment) -> None:
    """Evaluate an instance's results."""
    with tracer.start_as_current_span(f"evaluate_instance_{instance_id}"):
        trajectory_dir = get_moatless_trajectory_dir(trajectory_id=instance_id, project_id=evaluation_name)
        leaf_nodes = root_node.get_leaf_nodes()
        eval_result_path = trajectory_dir / "eval_result.json"

        eval_result: Optional[dict[str, Any]] = None
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
        unevaluated_nodes = [node for node in leaf_nodes if str(node.node_id) not in eval_result["node_results"]]
        if not unevaluated_nodes:
            logger.info(f"All leaf nodes evaluated for instance {instance_id}")
            return eval_result

        _emit_event(evaluation_name, instance_id, "evaluation", "started")

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
                logger.exception(f"Error evaluating node {leaf_node.node_id} for instance {instance_id}")
                eval_result["error"] = traceback.format_exc()
            finally:
                eval_result["duration"] = time.time() - start_time
                with open(eval_result_path, "w") as f:
                    json.dump(eval_result, f, indent=2)

        _emit_event(evaluation_name, instance_id, "evaluation", "completed")


def _emit_event(evaluation_name: str, instance_id: str, scope: str, event_type: str, data: Any = None):
    """Emit evaluation event."""
    event = EvaluationEvent(
        project_id=evaluation_name, trajectory_id=instance_id, scope=scope, event_type=event_type, data=data
    )

    try:
        run_async(event_bus.publish(event))
    except Exception as e:
        logger.error(f"Failed to publish event {event_type}: {e}")
