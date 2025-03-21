import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import litellm
from dotenv import load_dotenv
from opentelemetry import trace
from moatless.storage.base import BaseStorage
from testbeds.schema import SWEbenchInstance

from moatless.completion.log_handler import LogHandler
from moatless.context_data import current_project_id, current_trajectory_id
from moatless.evaluation.schema import EvaluationEvent
from moatless.flow.flow import AgenticFlow
from moatless.index.code_index import CodeIndex
from moatless.node import Node
from moatless.repository.git import GitRepository
from moatless.runtime.local import LocalEnvironment
from moatless.runtime.testbed import TestbedEnvironment
from moatless.telemetry import setup_telemetry
from moatless.workspace import Workspace

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.evaluation.runner")

load_dotenv()
setup_telemetry()


async def run_instance_async(project_id: str, trajectory_id: str):
    from moatless.settings import get_storage, get_event_bus

    storage = await get_storage()
    event_bus = await get_event_bus()

    litellm.callbacks = [LogHandler(storage=storage)]

    logger.info(f"current_project_id: {current_project_id}, current {current_trajectory_id}")

    settings = await storage.read_from_trajectory(key="settings", trajectory_id=trajectory_id, project_id=project_id)
    if not settings:
        logger.error("Trajectory settings not found.")
        raise ValueError("Settings not found")

    try:
        instance_path = os.environ.get("INSTANCE_PATH")
        if not instance_path:
            raise ValueError("INSTANCE_PATH is not set")
        if not Path(instance_path).exists():
            raise ValueError(f"Instance file not found at {instance_path}")
        with open(instance_path) as f:
            swebench_instance = SWEbenchInstance.model_validate(json.load(f))

        logger.info(f"Loaded instance: {swebench_instance.instance_id}")

        repo_path = os.environ.get("REPO_DIR")
        if not repo_path:
            raise ValueError("REPO_DIR is not set")
        logger.info(f"Using repo path: {repo_path}")

        # Set repository path as safe directory to avoid ownership issues
        try:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", repo_path],
                check=True,
                capture_output=True,
            )
            logger.info(f"Added {repo_path} to git safe.directory")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to set safe.directory: {e.stderr.decode('utf-8')}")

        repository = GitRepository(repo_path=repo_path)
        repository.reset()

        runtime = LocalEnvironment(
            repo_path=Path(repo_path),
            instance_id=trajectory_id,
            swebench_instance=swebench_instance,
            storage=storage,
            enable_cache=True,
        )

        index_store_dir = os.environ.get("INDEX_STORE_DIR")
        if not index_store_dir:
            raise ValueError("INDEX_STORE_DIR is not set")

        logger.info(f"Using index store dir: {index_store_dir}")
        code_index = CodeIndex.from_persist_dir(
            persist_dir=index_store_dir,
            file_repo=repository,
        )

        workspace = Workspace(
            repository=repository,
            code_index=code_index,
            runtime=runtime,
            legacy_workspace=True,
        )

        trajectory_dict = await storage.read_from_trajectory(
            key="trajectory", trajectory_id=trajectory_id, project_id=project_id
        )

        flow = AgenticFlow.from_dicts(settings=settings, trajectory=trajectory_dict)
        await flow.run(workspace=workspace)
        logger.info(f"Flow completed for instance {trajectory_id}")

        await evaluate_instance(
            evaluation_name=project_id,
            instance_id=trajectory_id,
            root_node=flow.root,
            runtime=runtime,
            storage=storage,
        )

    except Exception as e:
        logger.exception(f"Error running instance {trajectory_id}")
        await _emit_event(
            evaluation_name=project_id,
            instance_id=trajectory_id,
            scope="evaluation",
            event_type="error",
            data={"error": str(e)},
        )
        raise e


async def evaluate_instance(
    evaluation_name: str, instance_id: str, root_node: Node, runtime: TestbedEnvironment, storage: BaseStorage
) -> None:
    """Evaluate an instance's results."""
    with tracer.start_as_current_span(f"evaluate_instance_{instance_id}"):
        trajectory_key = storage.get_trajectory_key()
        leaf_nodes = root_node.get_leaf_nodes()
        eval_result_path = f"{trajectory_key}/eval_result"

        eval_result: Optional[dict[str, Any]] = None
        if await storage.exists(eval_result_path):
            try:
                eval_result = await storage.read_from_trajectory(
                    key="eval_result", trajectory_id=instance_id, project_id=evaluation_name
                )
            except Exception:
                logger.warning(f"Failed to read {eval_result_path}")

        if eval_result and "node_results" in eval_result:
            eval_result = eval_result
        elif eval_result and len(eval_result) > 0:
            eval_result = {
                "node_results": eval_result,
                "status": "started",
                "start_time": datetime.now(timezone.utc).isoformat(),
            }

        if not eval_result:
            eval_result = {
                "node_results": {},
                "status": "started",
                "start_time": datetime.now(timezone.utc).isoformat(),
            }
        unevaluated_nodes = [node for node in leaf_nodes if str(node.node_id) not in eval_result["node_results"]]
        if not unevaluated_nodes:
            logger.info(f"All leaf nodes evaluated for instance {instance_id}")
            return

        await _emit_event(evaluation_name, instance_id, "evaluation", "started")

        for i, leaf_node in enumerate(unevaluated_nodes):
            logger.info(f"Evaluating node {leaf_node.node_id} ({i+1}/{len(unevaluated_nodes)})")
            if not leaf_node.file_context:
                logger.warning(f"No file context for node {leaf_node.node_id}; skipping.")
                continue

            patch = leaf_node.file_context.generate_git_patch(ignore_tests=True)
            if not patch or not patch.strip():
                logger.info(f"No patch for node {leaf_node.node_id}; skipping.")
                continue
            start_time = time.time()
            try:
                result = await runtime.evaluate(patch=patch)
                if result:
                    eval_result["node_results"][str(leaf_node.node_id)] = result.model_dump()
                    logger.info(f"Evaluated node {leaf_node.node_id} in {time.time()-start_time:.2f}s")
            except Exception:
                logger.exception(f"Error evaluating node {leaf_node.node_id} for instance {instance_id}")
                eval_result["error"] = traceback.format_exc()
            finally:
                eval_result["duration"] = time.time() - start_time
                await storage.write_to_trajectory(
                    key="eval_result", data=eval_result, trajectory_id=instance_id, project_id=evaluation_name
                )

        await _emit_event(evaluation_name, instance_id, "evaluation", "completed")


async def _emit_event(
    evaluation_name: str,
    instance_id: str,
    scope: str,
    event_type: str,
    data: Any = None,
):
    """Emit evaluation event."""
    event = EvaluationEvent(
        project_id=evaluation_name,
        trajectory_id=instance_id,
        scope=scope,
        event_type=event_type,
        data=data,
    )

    try:
        from moatless.settings import event_bus

        await event_bus.publish(event)
    except Exception as e:
        logger.error(f"Failed to publish event {event_type}: {e}")
