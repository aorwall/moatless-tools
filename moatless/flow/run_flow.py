import asyncio
import datetime
import logging
import os
from pathlib import Path

import litellm

from moatless.completion.log_handler import LogHandler
from moatless.context_data import get_trajectory_dir
from moatless.environment.local import LocalBashEnvironment
from moatless.flow.flow import AgenticFlow
from moatless.repository.git import GitRepository
from moatless.runner.utils import (
    cleanup_job_logging,
    emit_event,
    run_async,
    setup_job_logging,
)
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


async def run_flow(project_id: str, trajectory_id: str) -> None:
    """Run an instance's agentic flow."""
    print(f"Running instance {trajectory_id} for project {project_id}")

    from moatless.settings import ensure_managers_initialized, get_storage

    asyncio.run(ensure_managers_initialized())

    storage = await get_storage()

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"/data/logs/logs_{date_str}.log")
    original_handlers = setup_job_logging(log_path=log_path)

    settings = await storage.read_from_trajectory(
        path="settings.json", trajectory_id=trajectory_id, project_id=project_id
    )
    if not settings:
        logger.error("Settings not found")
        raise ValueError("Settings not found")

    repo_path = os.getenv("REPO_PATH", str(Path.cwd()))
    logger.info(f"Using repository path: {repo_path}")
    try:
        repository = GitRepository(repo_path=repo_path)
        workspace = Workspace(repository=repository, environment=LocalBashEnvironment(cwd=repo_path))

        litellm.callbacks = [LogHandler(storage=storage)]

        trajectory_dict = await storage.read_from_trajectory(
            path="trajectory.json", trajectory_id=trajectory_id, project_id=project_id
        )

        flow = AgenticFlow.from_dicts(settings=settings, trajectory=trajectory_dict)

        if flow.is_finished():
            logger.warning(f"Flow already finished for instance {trajectory_id}")
            return

        logger.info(f"Flow created for instance {trajectory_id}")

        run_async(flow.run(workspace=workspace))

        logger.info(f"Flow completed for instance {trajectory_id}")

    except Exception as e:
        logger.exception(f"Error running instance {trajectory_id}")
        emit_event(
            evaluation_name=project_id,
            instance_id=trajectory_id,
            scope="flow",
            event_type="error",
            data={"error": str(e)},
        )
        raise e
    finally:
        cleanup_job_logging(original_handlers)
