import json
import logging
import os
from pathlib import Path

import litellm
from moatless.completion.log_handler import LogHandler
from moatless.context_data import current_project_id, current_trajectory_id
from moatless.flow.flow import AgenticFlow
from moatless.index.code_index import CodeIndex
from moatless.repository.git import GitRepository
from moatless.runner.utils import (
    cleanup_job_logging,
    emit_event,
    setup_job_logging,
)
from moatless.runtime.local import SweBenchLocalEnvironment
from moatless.settings import get_storage
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


async def setup_flow(project_id: str, trajectory_id: str) -> AgenticFlow:
    from moatless.settings import get_storage

    storage = await get_storage()

    current_project_id.set(project_id)
    current_trajectory_id.set(trajectory_id)

    logger.info(f"current_project_id: {current_project_id}, current {current_trajectory_id}")

    settings = await storage.read_from_trajectory(
        path="settings.json", trajectory_id=trajectory_id, project_id=project_id
    )
    trajectory_dict = await storage.read_from_trajectory(
        path="trajectory.json", trajectory_id=trajectory_id, project_id=project_id
    )

    return AgenticFlow.from_dicts(settings=settings, trajectory=trajectory_dict)


async def setup_workspace() -> Workspace:
    storage = await get_storage()

    instance_path = os.environ.get("INSTANCE_PATH")
    if instance_path:
        if not os.path.exists(instance_path):
            raise FileNotFoundError(f"Instance path {instance_path} does not exist")

        with open(instance_path) as f:
            swebench_instance = json.loads(f.read())

        logger.info(f"Loaded SWE-Bench instance: {swebench_instance.get('instance_id')}")
        repo_path = "/testbed"
        runtime = SweBenchLocalEnvironment(
            repo_path=Path(repo_path),
            swebench_instance=swebench_instance,
            storage=storage,
        )
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


async def run_flow(project_id: str, trajectory_id: str, node_id: int | None = None) -> None:
    """Run an instance's agentic flow."""
    print(f"Running instance {trajectory_id} for project {project_id}")

    log_path = Path("/data/logs/job.log")
    original_handlers = setup_job_logging(log_path=log_path)
    storage = await get_storage()
    try:
        litellm.callbacks = [LogHandler(storage=storage)]

        flow = await setup_flow(project_id, trajectory_id)

        if not node_id and flow.is_finished():
            logger.warning(f"Flow already finished for instance {trajectory_id}")
            return

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
