"""
Job wrapper functions for RQ.

This module contains wrapper functions for RQ jobs that avoid circular imports.
"""

import logging
import asyncio

logger = logging.getLogger(__name__)


def run_evaluation_instance(project_id: str, trajectory_id: str) -> None:
    """
    Wrapper function to run an evaluation instance.

    This function dynamically imports the run_instance function to avoid circular imports.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
    """
    logger.info(f"Running evaluation instance {trajectory_id} for project {project_id}")

    try:
        # Dynamically import the run_instance function
        # This avoids circular imports that can occur when importing at module level
        from moatless.evaluation.run_instance import run_instance_async

        asyncio.run(run_instance_async(project_id, trajectory_id))

    except ImportError as e:
        logger.error(f"Failed to import run_instance: {e}")
        raise
    except Exception as e:
        logger.error(f"Error running instance {trajectory_id}: {e}")
        raise
