import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import litellm

from moatless.completion.log_handler import LogHandler
from moatless.context_data import current_project_id, current_trajectory_id
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


async def setup_flow(project_id: str, trajectory_id: str) -> AgenticFlow:
    from moatless.settings import get_storage

    storage = await get_storage()

    current_project_id.set(project_id)
    current_trajectory_id.set(trajectory_id)

    litellm.callbacks = [LogHandler(storage=storage)]

    logger.info(f"current_project_id: {current_project_id}, current {current_trajectory_id}")

    settings = await storage.read_from_trajectory(
        path="settings.json", trajectory_id=trajectory_id, project_id=project_id
    )
    trajectory_dict = await storage.read_from_trajectory(
        path="trajectory.json", trajectory_id=trajectory_id, project_id=project_id
    )

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

    return SweBenchLocalEnvironment(
        repo_path=Path("/testbed"),
        swebench_instance=swebench_instance,
        storage=storage,
    )


async def setup_workspace() -> Workspace:
    instance_path = os.environ.get("INSTANCE_PATH")
    if instance_path:
        runtime = await setup_swebench_runtime()
    else:
        # TODO: Use Local bash environment
        runtime = None
        repo_path = os.environ.get("REPO_PATH")

    if repo_path:
        repository = GitRepository(repo_path=repo_path)
    else:
        repository = None

    index_store_dir = os.environ.get("INDEX_STORE_DIR")
    if index_store_dir:
        logger.info(f"Using index store dir: {index_store_dir}")
        code_index = CodeIndex.from_persist_dir(
            persist_dir=index_store_dir,
            file_repo=repository,
        )
    else:
        code_index = None
        logger.info("No index store dir provided, skipping code index")

    return Workspace(repository=repository, code_index=code_index, runtime=runtime)


async def handle_flow_event(flow: AgenticFlow, event: BaseEvent) -> None:
    """Handle flow events by persisting data and publishing to event bus.

    Args:
        flow: The flow instance that generated the event
    """
    from moatless.settings import get_storage, get_event_bus

    async def process_event_task():
        async with _flow_lock:
            storage = await get_storage()

            trajectory_data = flow.get_trajectory_data()
            await storage.write_to_trajectory("trajectory.json", trajectory_data, flow.project_id, flow.trajectory_id)

            try:
                event_bus = await get_event_bus()
                await event_bus.publish(event)
            except Exception as e:
                logger.error(f"Error publishing event: {e}")

    asyncio.create_task(process_event_task())


async def run_flow(project_id: str, trajectory_id: str, node_id: int | None = None) -> None:
    """Run an instance's agentic flow."""
    print(f"Running instance {trajectory_id} for project {project_id}")

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path(f"/data/logs/logs_{date_str}.log")
    original_handlers = setup_job_logging(log_path=log_path)
    storage = await get_storage()

    try:
        flow = await setup_flow(project_id, trajectory_id)

        if not node_id and flow.is_finished():
            logger.warning(f"Flow already finished for instance {trajectory_id}")
            return None

        workspace = await setup_workspace()

        logger.info(f"Flow created for instance {trajectory_id}")

        await flow.run(workspace=workspace, node_id=node_id)

        logger.info(f"Flow completed for instance {trajectory_id}")

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
