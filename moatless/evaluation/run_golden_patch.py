import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv
from moatless.completion.log_handler import LogHandler
from moatless.context_data import current_project_id, current_trajectory_id
from moatless.evaluation.schema import EvaluationEvent
from moatless.flow.flow import AgenticFlow
from moatless.flow.run_flow import (
    handle_flow_event,
    persist_trajectory_data,
    setup_flow,
    setup_swebench_runtime,
    setup_workspace,
)
from moatless.index.code_index import CodeIndex
from moatless.node import EvaluationResult, Node
from moatless.repository.git import GitRepository
from moatless.runner.utils import cleanup_job_logging, setup_job_logging
from moatless.runtime.local import SweBenchLocalEnvironment
from moatless.storage.base import BaseStorage
from moatless.workspace import Workspace

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


async def evaluate_golden_patch(project_id: str, trajectory_id: str, node_id: int | None = None):
    from moatless.settings import get_storage

    load_dotenv()

    storage = await get_storage()
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"/data/logs/logs_{date_str}.log")
    original_handlers = setup_job_logging(log_path=log_path)

    current_project_id.set(project_id)
    current_trajectory_id.set(trajectory_id)

    litellm.callbacks = [LogHandler(storage=storage)]

    logger.info(f"current_project_id: {current_project_id}, current {current_trajectory_id}")

    settings = await storage.read_from_trajectory(path="flow.json", trajectory_id=trajectory_id, project_id=project_id)
    if not settings:
        logger.error("Trajectory settings not found.")
        raise ValueError("Settings not found")

    try:
        flow = await setup_flow(project_id, trajectory_id)

        if not node_id and flow.is_finished():
            logger.warning(f"Flow already finished for instance {trajectory_id}")
            return None

        runtime = await setup_swebench_runtime()

        await evaluate_instance(
            evaluation_name=project_id,
            instance_id=trajectory_id,
            flow=flow,
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
    evaluation_name: str, instance_id: str, flow: AgenticFlow, runtime: SweBenchLocalEnvironment, storage: BaseStorage
) -> None:
    """Evaluate an instance's results."""

    evaluation_key = f"projects/{evaluation_name}/trajs/{instance_id}/evaluation"
    root_node = flow.root

    await _emit_event(evaluation_name, instance_id, "evaluation", "started")

    any_resolved = False

    logger.info(f"Evaluating node {root_node.node_id}")

    instance_path = os.environ.get("INSTANCE_PATH")
    if not instance_path:
        raise ValueError("INSTANCE_PATH is not set")

    if not os.path.exists(instance_path):
        raise FileNotFoundError(f"Instance path {instance_path} does not exist")

    with open(instance_path) as f:
        swebench_instance = json.loads(f.read())

    patch = swebench_instance["patch"]
    try:
        child_node = Node(
            node_id=root_node.get_all_nodes()[-1].node_id + 1,
            parent=root_node,
            file_context=None,
            terminal=True,
        )  # type: ignore

        root_node.add_child(child_node)

        logger.info(f"Evaluating node {child_node.node_id} with patch: {patch}")
        evaluation_node_key = f"{evaluation_key}/node_{child_node.node_id}"
        start_time = datetime.now()
        report = await runtime.swebench_evaluate(evaluation_node_key, patch)
        end_time = datetime.now()

        if instance_id not in report:
            logger.warning(f"Instance {instance_id} not found in report for node {child_node.node_id}: {report}")
            raise ValueError(f"Instance {instance_id} not found in report for node {child_node.node_id}: {report}")

        logger.info(f"Evaluation complete for node {child_node.node_id}. Resolved: {report[instance_id]['resolved']}")

        if report[instance_id]["resolved"]:
            any_resolved = True

        child_node.evaluation_result = EvaluationResult(
            resolved=report[instance_id].get("resolved", False),
            details=report[instance_id],
            start_time=start_time,
            end_time=end_time,
        )

        await persist_trajectory_data(flow)

    except Exception:
        logger.exception(f"Error evaluating node {child_node.node_id} for instance {instance_id}")

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

    asyncio.run(evaluate_golden_patch(project_id, trajectory_id))
