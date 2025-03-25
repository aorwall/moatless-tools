import asyncio
import json
import logging
import os
from pathlib import Path

from datetime import datetime
from typing import Any

import litellm
from dotenv import load_dotenv
from opentelemetry import trace
from moatless.runner.utils import cleanup_job_logging, setup_job_logging
from moatless.storage.base import BaseStorage

from moatless.completion.log_handler import LogHandler
from moatless.context_data import current_project_id, current_trajectory_id
from moatless.evaluation.schema import EvaluationEvent
from moatless.flow.flow import AgenticFlow
from moatless.index.code_index import CodeIndex
from moatless.node import Node
from moatless.repository.git import GitRepository
from moatless.runtime.local import SweBenchTestbedEnvironment
from moatless.workspace import Workspace

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


async def run_swebench_instance(project_id: str, trajectory_id: str, node_id: int | None = None):
    from moatless.settings import get_storage, get_event_bus

    load_dotenv()

    storage = await get_storage()
    event_bus = await get_event_bus()
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"/data/logs/logs_{date_str}.log")
    original_handlers = setup_job_logging(log_path=log_path)

    current_project_id.set(project_id)
    current_trajectory_id.set(trajectory_id)

    litellm.callbacks = [LogHandler(storage=storage)]

    logger.info(f"current_project_id: {current_project_id}, current {current_trajectory_id}")

    settings = await storage.read_from_trajectory(
        path="settings.json", trajectory_id=trajectory_id, project_id=project_id
    )
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
            swebench_instance = json.loads(f.read())

        logger.info(f"Loaded instance: {swebench_instance['instance_id']}")

        repo_path = os.environ.get("REPO_DIR")
        if not repo_path:
            raise ValueError("REPO_DIR is not set")
        logger.info(f"Using repo path: {repo_path}")

        repository = GitRepository(repo_path=repo_path)

        runtime = SweBenchTestbedEnvironment(
            repo_path=Path(repo_path),
            swebench_instance=swebench_instance,
            storage=storage,
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
            path="trajectory.json", trajectory_id=trajectory_id, project_id=project_id
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

    finally:
        cleanup_job_logging(original_handlers)

        trajectory_key = storage.get_trajectory_path(project_id=project_id, trajectory_id=trajectory_id)
        await storage.write_raw(f"{trajectory_key}/logs/{log_path.name}", log_path.read_text())


async def evaluate_instance(
    evaluation_name: str, instance_id: str, root_node: Node, runtime: SweBenchTestbedEnvironment, storage: BaseStorage
) -> None:
    """Evaluate an instance's results."""

    evaluation_key = f"projects/{evaluation_name}/trajs/{instance_id}/evaluation"
    leaf_nodes = root_node.get_leaf_nodes()

    unevaluated_nodes = [
        node for node in leaf_nodes if not await storage.exists(f"{evaluation_key}/node_{node.node_id}/report.json")
    ]

    if not unevaluated_nodes:
        logger.info(f"All leaf nodes evaluated for instance {instance_id}")
        return

    await _emit_event(evaluation_name, instance_id, "evaluation", "started")

    any_resolved = False

    for i, leaf_node in enumerate(unevaluated_nodes):
        logger.info(f"Evaluating node {leaf_node.node_id} ({i+1}/{len(unevaluated_nodes)})")
        if not leaf_node.file_context:
            logger.warning(f"No file context for node {leaf_node.node_id}; skipping.")
            continue

        patch = leaf_node.file_context.generate_git_patch(ignore_tests=True)
        if not patch or not patch.strip():
            logger.info(f"No patch for node {leaf_node.node_id}; skipping.")
            continue
        try:
            evaluation_node_key = f"{evaluation_key}/node_{leaf_node.node_id}"
            report = await runtime.swebench_evaluate(evaluation_node_key, patch)
            logger.info(
                f"Evaluation complete for node {leaf_node.node_id}. Resolved: {report[instance_id]['resolved']}"
            )
            if report[instance_id]["resolved"]:
                any_resolved = True

        except Exception:
            logger.exception(f"Error evaluating node {leaf_node.node_id} for instance {instance_id}")

    await _emit_event(evaluation_name, instance_id, "evaluation", "completed", data={"resolved": any_resolved})


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
        from moatless.settings import get_event_bus

        event_bus = await get_event_bus()
        await event_bus.publish(event)
    except Exception as e:
        logger.error(f"Failed to publish event {event_type}: {e}")


if __name__ == "__main__":
    project_id = os.environ.get("PROJECT_ID")
    if not project_id:
        raise ValueError("PROJECT_ID is not set")
    trajectory_id = os.environ.get("TRAJECTORY_ID")
    if not trajectory_id:
        raise ValueError("TRAJECTORY_ID is not set")

    asyncio.run(run_swebench_instance(project_id, trajectory_id))
