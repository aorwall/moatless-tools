import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import litellm

from moatless.completion.log_handler import LogHandler
from moatless.context_data import current_project_id, current_trajectory_id
from moatless.environment.local import LocalBashEnvironment
from moatless.events import BaseEvent
from moatless.flow.flow import AgenticFlow
from moatless.index.code_index import CodeIndex
from moatless.repository.git import GitRepository
from moatless.runner.utils import (
    cleanup_job_logging,
    emit_event,
    setup_job_logging,
)
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.settings import get_storage
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


_flow_lock = asyncio.Lock()
_pending_event_tasks = set()


async def setup_flow(project_id: str, trajectory_id: str) -> AgenticFlow:
    from moatless.settings import get_storage

    storage = await get_storage()

    current_project_id.set(project_id)
    current_trajectory_id.set(trajectory_id)

    litellm.callbacks = [LogHandler(storage=storage)]

    logger.info(f"setup_flow(project_id: {project_id}, trajectory_id: {trajectory_id})")

    try:
        settings: dict | None = await storage.read_from_trajectory(
            path="flow.json", trajectory_id=trajectory_id, project_id=project_id
        )  # type: ignore
    except Exception:
        settings = await storage.read_from_project(path="flow.json", project_id=project_id)  # type: ignore

        logger.info(f"Settings not found in trajectory, using project settings")
    else:
        logger.info(f"Settings found in trajectory")

    trajectory_dict: dict | None = await storage.read_from_trajectory(
        path="trajectory.json", trajectory_id=trajectory_id, project_id=project_id
    )  # type: ignore
    flow = AgenticFlow.from_dicts(settings=settings, trajectory=trajectory_dict)

    async def on_event(event: BaseEvent) -> None:
        await handle_flow_event(flow, event)

    flow._on_event = on_event

    return flow


async def setup_swebench_runtime() -> RuntimeEnvironment:
    storage = await get_storage()
    from moatless.runtime.local import SweBenchLocalEnvironment

    instance_path = os.environ.get("INSTANCE_PATH")
    if not instance_path:
        raise ValueError("INSTANCE_PATH is not set")

    if not os.path.exists(instance_path):
        raise FileNotFoundError(f"Instance path {instance_path} does not exist")

    with open(instance_path) as f:
        swebench_instance = json.loads(f.read())

    try:
        return SweBenchLocalEnvironment(
            repo_path=Path("/testbed"),
            swebench_instance=swebench_instance,
            storage=storage,
        )
    except Exception as e:
        logger.error(f"Error setting up SWEbench runtime with instance {swebench_instance}")
        raise e


async def setup_workspace(project_id: str, trajectory_id: str) -> Workspace:
    logger.info(f"setup_workspace(project_id: {project_id}, trajectory_id: {trajectory_id})")

    storage = await get_storage()
    instance_path = os.environ.get("INSTANCE_PATH")
    if instance_path:
        runtime = await setup_swebench_runtime()
        repo_path = "/testbed"
    elif await storage.exists(f"projects/{project_id}/settings.json"):
        project_settings = await storage.read(f"projects/{project_id}/settings.json")
        repo_path = project_settings["repository_path"]
        runtime = None

    elif os.environ.get("REPO_PATH"):
        # TODO: Use Local bash environment
        runtime = None
        repo_path = os.environ.get("REPO_PATH")

    logger.info(f"repo_path: {repo_path}")

    if repo_path:
        repository = GitRepository(repo_path=repo_path, shadow_mode=True)
        environment = LocalBashEnvironment(cwd=repo_path)
    else:
        repository = None
        environment = None

    if os.environ.get("INDEX_STORE_DIR"):
        index_store_dir = os.environ.get("INDEX_STORE_DIR")
    else:
        index_store_dir = None

    if index_store_dir:
        logger.info(f"Using index store dir: {index_store_dir}")
        code_index = CodeIndex.from_persist_dir(
            persist_dir=index_store_dir,
            file_repo=repository,
        )
    else:
        code_index = None
        logger.info("No index store dir provided, skipping code index")

    return Workspace(
        repository=repository,
        code_index=code_index,
        runtime=runtime,
        environment=environment,
        storage=storage,
    )


async def persist_trajectory_data(flow: AgenticFlow) -> None:
    storage = await get_storage()

    trajectory_data = flow.get_trajectory_data()
    await storage.write_to_trajectory("trajectory.json", trajectory_data, flow.project_id, flow.trajectory_id)
    logger.info(
        f"Trajectory data with {len(trajectory_data['nodes'])} nodes written to {flow.project_id}/{flow.trajectory_id}/trajectory.json"
    )


async def handle_flow_event(flow: AgenticFlow, event: BaseEvent) -> None:
    """Handle flow events by persisting data and publishing to event bus.

    Args:
        flow: The flow instance that generated the event
    """
    from moatless.settings import get_event_bus

    try:
        event_bus = await get_event_bus()
        await event_bus.publish(event)
    except Exception as e:
        logger.error(f"Error publishing event: {e}")

    async def process_event_task():
        try:
            async with _flow_lock:
                if (event.scope == "node" and event.event_type == "expanded") or event.scope == "flow":
                    await persist_trajectory_data(flow)

                    logger.info(
                        f"Event {event.scope}:{event.event_type}. Trajectory data written to {flow.project_id}/{flow.trajectory_id}/trajectory.json. "
                    )
                else:
                    logger.info(f"Event {event.scope}:{event.event_type} ignored")

        finally:
            _pending_event_tasks.discard(task)

    task = asyncio.create_task(process_event_task())
    _pending_event_tasks.add(task)


async def run_flow(project_id: str, trajectory_id: str, node_id: int | None = None) -> None:
    """Run an instance's agentic flow."""
    print(f"Running instance {trajectory_id} with node_id {node_id} for project {project_id}")

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path(f"./logs/logs_{date_str}.log")
    original_handlers = setup_job_logging(log_path=log_path)
    storage = await get_storage()

    try:
        flow = await setup_flow(project_id, trajectory_id)

        if not node_id and flow.is_finished():
            logger.warning(f"Flow already finished for instance {trajectory_id}")
            return None

        workspace = await setup_workspace(project_id, trajectory_id)

        logger.info(f"Flow created for instance {trajectory_id}")

        await flow.run(workspace=workspace, node_id=node_id)

        logger.info(f"Flow completed for instance {trajectory_id}")

        if _pending_event_tasks:
            logger.info("Cancelling pending event tasks...")
            for task in _pending_event_tasks:
                task.cancel()

            _pending_event_tasks.clear()

        if flow.is_finished():
            await persist_trajectory_data(flow)

    except Exception as e:
        logger.exception(f"Error running instance {trajectory_id}")
        await emit_event(
            evaluation_name=project_id,
            instance_id=trajectory_id,
            scope="flow",
            event_type="error",
            data={"error": str(e)},
        )
        raise e
    finally:
        cleanup_job_logging(original_handlers)

        trajectory_key = storage.get_trajectory_path(project_id=project_id, trajectory_id=trajectory_id)
        await storage.write_raw(f"{trajectory_key}/logs/{log_path.name}", log_path.read_text())
