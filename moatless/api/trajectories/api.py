"""API endpoints for run status and trajectory data."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from moatless.flow.manager import FlowManager
from moatless.flow.schema import (
    ExecuteNodeRequest,
    RetryTrajectoryRequest,
    StartTrajectoryRequest,
    TrajectoryListItem,
    TrajectoryResponseDTO,
)

_manager = FlowManager.get_instance()


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=list[TrajectoryListItem])
async def get_trajectories():
    """Get all trajectories."""
    try:
        return await _manager.list_trajectories()
    except Exception as e:
        logger.exception(f"Error getting trajectories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}", response_model=TrajectoryResponseDTO)
async def get_trajectory(project_id: str, trajectory_id: str):
    """Get the status, trajectory data, and events for a specific trajectory."""
    try:
        return await _manager.get_trajectory(project_id, trajectory_id)
    except ValueError as e:
        logger.exception(f"Error getting trajectory data: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting trajectory data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}/logs")
async def get_trajectory_logs(project_id: str, trajectory_id: str, file_name: str = None):
    """Get the log files for a specific trajectory.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        file_name: Optional specific log file name to retrieve

    Returns:
        The log file contents
    """
    try:
        return await _manager.get_trajectory_logs(project_id, trajectory_id, file_name)
    except ValueError as e:
        logger.exception(f"Error getting trajectory logs: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting trajectory logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/start")
async def start_trajectory(project_id: str, trajectory_id: str):
    """Start a trajectory without additional parameters."""
    try:
        await _manager.start_trajectory(project_id, trajectory_id)
        return {"status": "success", "message": f"Started trajectory {trajectory_id}"}
    except ValueError as e:
        logger.exception(f"Error starting trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error starting trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/resume")
async def resume_trajectory(project_id: str, trajectory_id: str, request: StartTrajectoryRequest):
    """Resume a trajectory."""
    try:
        await _manager.resume_trajectory(project_id, trajectory_id, request)
        return {"status": "success", "message": f"Resumed trajectory {trajectory_id}"}
    except ValueError as e:
        logger.exception(f"Error resuming trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error resuming trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/retry")
async def retry_trajectory(project_id: str, trajectory_id: str, request: RetryTrajectoryRequest):
    """Retry a run."""
    try:
        await _manager.retry_trajectory(project_id, trajectory_id, request)
        return {"status": "success", "message": f"Started retry for trajectory {trajectory_id}"}
    except ValueError as e:
        logger.exception(f"Error retrying trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error retrying trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/execute")
async def execute_node(project_id: str, trajectory_id: str, request: ExecuteNodeRequest):
    """Execute a run."""
    try:
        result = await _manager.execute_node(project_id, trajectory_id, request)
        return result
    except ValueError as e:
        logger.exception(f"Error executing node: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error executing node: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
