import os
import logging
from moatless.runner.job_wrappers import run_evaluation_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    project_id = os.environ.get("PROJECT_ID")
    trajectory_id = os.environ.get("TRAJECTORY_ID")
    if not project_id or not trajectory_id:
        logger.error("PROJECT_ID and TRAJECTORY_ID must be set")
        exit(1)

    logger.info(f"Running worker for project {project_id} and trajectory {trajectory_id}")
    run_evaluation_instance(project_id, trajectory_id)
