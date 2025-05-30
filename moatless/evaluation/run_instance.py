import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from moatless.completion.log_handler import LogHandler
from moatless.context_data import current_project_id, current_trajectory_id
from moatless.environment.local import LocalBashEnvironment
from moatless.evaluation.schema import EvaluationEvent
from moatless.flow.flow import AgenticFlow
from moatless.flow.run_flow import (
    persist_trajectory_data,
    setup_flow,
    setup_swebench_runtime,
)
from moatless.node import EvaluationResult, Node
from moatless.repository.git import GitRepository
from moatless.runner.utils import cleanup_job_logging, setup_job_logging
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.storage.base import BaseStorage
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


async def run_swebench_instance(project_id: str, trajectory_id: str, node_id: int | None = None):
    logger.info(f"Running swebench instance {project_id} {trajectory_id} {node_id}")

    from moatless.settings import get_storage

    load_dotenv()

    storage = await get_storage()
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"/data/logs/logs_{date_str}.log")
    original_handlers = setup_job_logging(log_path=log_path)

    current_project_id.set(project_id)
    current_trajectory_id.set(trajectory_id)

    import litellm

    litellm.callbacks = [LogHandler(storage=storage)]

    try:
        flow = await setup_flow(project_id, trajectory_id)

        repo_path = os.environ.get("REPO_PATH")
        if not repo_path:
            raise ValueError("REPO_PATH is not set")

        repository = GitRepository(repo_path=repo_path)
        environment = LocalBashEnvironment(cwd=repo_path)

        runtime = await setup_swebench_runtime()

        index_store_dir = os.environ.get("INDEX_STORE_DIR")
        if not index_store_dir:
            raise ValueError("INDEX_STORE_DIR is not set")

        logger.info(f"Using index store dir: {index_store_dir}")
        from moatless.index.code_index import CodeIndex

        code_index = CodeIndex.from_persist_dir(
            persist_dir=index_store_dir,
            file_repo=repository,
        )

        workspace = Workspace(repository=repository, code_index=code_index, runtime=runtime, environment=environment)

        logger.info(f"Flow created for instance {trajectory_id}")

        node = await flow.run(workspace=workspace, node_id=node_id)
        if node.error:
            raise ValueError(f"Node {node.node_id} failed with error: {node.error}")

        logger.info(f"Flow completed for instance {trajectory_id}")

        await evaluate_instance(
            evaluation_name=project_id,
            instance_id=trajectory_id,
            flow=flow,
            selected_node=node,
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
    evaluation_name: str,
    instance_id: str,
    flow: AgenticFlow,
    selected_node: Node,
    runtime: RuntimeEnvironment,
    storage: BaseStorage,
) -> None:
    """Evaluate an instance's results."""

    evaluation_key = f"projects/{evaluation_name}/trajs/{instance_id}/evaluation"
    leaf_nodes = flow.root.get_leaf_nodes()

    await _emit_event(evaluation_name, instance_id, "evaluation", "started")

    resolved = None

    for i, leaf_node in enumerate(leaf_nodes):
        logger.info(f"Evaluating node {leaf_node.node_id} ({i+1}/{len(leaf_nodes)})")

        if not leaf_node.file_context:
            raise ValueError(f"No file context for node {leaf_node.node_id}")

        if not leaf_node.evaluation_result:
            patch = leaf_node.file_context.generate_git_patch(ignore_tests=True)
            try:
                evaluation_node_key = f"{evaluation_key}/node_{leaf_node.node_id}"
                start_time = datetime.now()
                report = await runtime.swebench_evaluate(evaluation_node_key, patch)
                end_time = datetime.now()

                if instance_id not in report:
                    logger.warning(f"Instance {instance_id} not found in report for node {leaf_node.node_id}: {report}")
                    continue

                logger.info(
                    f"Evaluation complete for node {leaf_node.node_id}. Resolved: {report[instance_id]['resolved']}"
                )

                if leaf_node.node_id == selected_node.node_id:
                    resolved = report[instance_id]["resolved"]

                leaf_node.evaluation_result = EvaluationResult(
                    resolved=report[instance_id].get("resolved", False),
                    details=report[instance_id],
                    start_time=start_time,
                    end_time=end_time,
                )

                await persist_trajectory_data(flow)

            except Exception:
                logger.exception(f"Error evaluating node {leaf_node.node_id} for instance {instance_id}")

    await _emit_event(evaluation_name, instance_id, "evaluation", "completed", data={"resolved": resolved})


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
