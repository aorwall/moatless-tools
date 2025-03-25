"""API endpoints for run status and trajectory data."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from moatless.flow.manager import FlowManager
from moatless.flow.schema import (
    ExecuteNodeRequest,
    StartTrajectoryRequest,
    TrajectoryListItem,
    TrajectoryResponseDTO,
)
from moatless.settings import get_flow_manager, get_storage
from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=list[TrajectoryListItem])
async def get_trajectories(flow_manager: FlowManager = Depends(get_flow_manager)):
    """Get all trajectories."""
    try:
        return await flow_manager.list_trajectories()
    except Exception as e:
        logger.exception(f"Error getting trajectories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}", response_model=TrajectoryResponseDTO)
async def get_trajectory(
    project_id: str,
    trajectory_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Get the status, trajectory data, and events for a specific trajectory."""
    try:
        return await flow_manager.get_trajectory(project_id, trajectory_id)
    except ValueError as e:
        logger.exception(f"Error getting trajectory data: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting trajectory data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}/logs")
async def get_trajectory_logs(
    project_id: str,
    trajectory_id: str,
    file_name: str | None = None,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Get the log files for a specific trajectory.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        file_name: Optional specific log file name to retrieve

    Returns:
        The log file contents
    """
    try:
        return await flow_manager.get_trajectory_logs(project_id, trajectory_id, file_name)
    except ValueError as e:
        logger.exception(f"Error getting trajectory logs: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting trajectory logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}/events")
async def get_trajectory_events(
    project_id: str,
    trajectory_id: str,
    storage: BaseStorage = Depends(get_storage),
):
    """Get the events for a specific trajectory.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        event_bus: The event bus instance

    Returns:
        The trajectory events
    """
    key = storage.get_trajectory_path(project_id, trajectory_id)
    key = f"{key}/events.jsonl"

    if not await storage.exists(key):
        logger.info(f"No events found for key {key}")
        return []

    return await storage.read_lines(key)


@router.get("/{project_id}/{trajectory_id}/completions/{node_id}")
async def get_completions(
    project_id: str,
    trajectory_id: str,
    node_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    return await flow_manager.get_completions(project_id, trajectory_id, node_id)


@router.get("/{project_id}/{trajectory_id}/completions/{node_id}/action/{action_step}")
async def get_completions_by_action_step(
    project_id: str,
    trajectory_id: str,
    node_id: str,
    action_step: int,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    return await flow_manager.get_completions(project_id, trajectory_id, node_id, action_step)


@router.post("/{project_id}/{trajectory_id}/start")
async def start_trajectory(
    project_id: str,
    trajectory_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Start a trajectory without additional parameters."""
    try:
        await flow_manager.start_trajectory(project_id, trajectory_id)
        return {"status": "success", "message": f"Started trajectory {trajectory_id}"}
    except ValueError as e:
        logger.exception(f"Error starting trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error starting trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/retry-trajectory")
async def retry_trajectory(
    project_id: str,
    trajectory_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Reset and restart a trajectory by removing all children from the root node."""
    try:
        await flow_manager.retry_trajectory(project_id, trajectory_id)
        return {"status": "success", "message": f"Retried trajectory {trajectory_id}"}
    except ValueError as e:
        logger.exception(f"Error retrying trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error retrying trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/resume")
async def resume_trajectory(
    project_id: str,
    trajectory_id: str,
    request: StartTrajectoryRequest,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Resume a trajectory."""
    try:
        await flow_manager.resume_trajectory(project_id, trajectory_id, request)
        return {"status": "success", "message": f"Resumed trajectory {trajectory_id}"}
    except ValueError as e:
        logger.exception(f"Error resuming trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error resuming trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/execute")
async def execute_node(
    project_id: str,
    trajectory_id: str,
    request: ExecuteNodeRequest,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Execute a run."""
    try:
        result = await flow_manager.execute_node(project_id, trajectory_id, request.node_id)
        return result
    except ValueError as e:
        logger.exception(f"Error executing node: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error executing node: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}/tree")
async def get_node_tree(
    project_id: str,
    trajectory_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    return await flow_manager.get_node_tree(project_id, trajectory_id)


@router.get("/{project_id}/{trajectory_id}/node/{node_id}")
async def get_node(
    project_id: str,
    trajectory_id: str,
    node_id: int,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    return await flow_manager.get_node(project_id, trajectory_id, node_id)
