import logging
from pathlib import Path

from moatless.flow.flow import AgenticFlow
from moatless.repository.git import GitRepository
from moatless.runner.utils import emit_event, run_async, setup_job_logging, cleanup_job_logging
from moatless.context_data import get_trajectory_dir
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)

def run_flow(project_id: str, trajectory_id: str) -> None:
    """Run an instance's agentic flow."""
    print(f"Running instance {trajectory_id} for project {project_id}")

    trajectory_dir = get_trajectory_dir(trajectory_id=trajectory_id, project_id=project_id)
    print(f"Setting up job logging for run in {trajectory_dir}")
    original_handlers = setup_job_logging("run", trajectory_dir)
    
    try:
        repository = GitRepository(repo_path=str(Path.cwd()))
        workspace = Workspace(repository=repository)

        flow = AgenticFlow.from_dir(trajectory_dir=trajectory_dir, workspace=workspace)
        
        if flow.is_finished():
            logger.warning(f"Flow already finished for instance {trajectory_id}")
            return
        
        logger.info(f"Flow created for instance {trajectory_id}")

        run_async(flow.run())
        
        logger.info(f"Flow completed for instance {trajectory_id}")

    except Exception as e:
        logger.exception(f"Error running instance {trajectory_id}")
        emit_event(
            evaluation_name=project_id,
            instance_id=trajectory_id,
            scope="flow",
            event_type="error"
        )
        raise e
    finally:
        cleanup_job_logging(original_handlers)