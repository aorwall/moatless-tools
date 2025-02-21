"""API endpoints for run status and trajectory data."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from moatless.api.trajectory.schema import TrajectoryDTO
from moatless.api.trajectory.trajectory_utils import convert_nodes, load_trajectory_from_file
from moatless.benchmark.swebench.utils import create_index, create_repository_async
from moatless.evaluation.utils import get_moatless_instance
from moatless.flow.runner import agentic_runner
from moatless.flow.flow import AgenticFlow, SystemStatus
from moatless.flow.loop import AgenticLoop
from moatless.runtime.testbed import TestbedEnvironment
from moatless.utils.moatless import get_moatless_trajectories_dir, get_moatless_trajectory_dir
from moatless.workspace import Workspace
from .schema import RetryTrajectoryRequest, StartTrajectoryRequest, TrajectoryEventDTO, TrajectoryListItem, \
    TrajectoryResponseDTO

logger = logging.getLogger(__name__)

router = APIRouter()

def get_trajectory_path(trajectory_id: str) -> str:
    """Get the trajectory file path for a run."""
    return str(get_moatless_trajectory_dir(trajectory_id) / 'trajectory.json')

def load_trajectory_events(trajectory_dir: Path) -> list[TrajectoryEventDTO]:
    """Load events from events.jsonl file."""
    events_path = trajectory_dir / 'events.jsonl'
    events = []
    
    if events_path.exists():
        try:
            with open(events_path, 'r', encoding='utf-8') as f:
                for line in f:
                    event_data = json.loads(line)
                    # Convert ISO timestamp to milliseconds, ensuring UTC
                    dt = datetime.fromisoformat(event_data['timestamp'])
                    if dt.tzinfo is None:
                        # If timestamp has no timezone, assume UTC
                        dt = dt.replace(tzinfo=timezone.utc)
                    event_data['timestamp'] = int(dt.timestamp() * 1000)
                    events.append(TrajectoryEventDTO(**event_data))
        except Exception as e:
            logger.error(f"Error reading events file: {e}")
            
    return events

def load_trajectory_status(trajectory_dir: Path) -> SystemStatus:
    """Load status from status.json file."""
    status_path = trajectory_dir / 'status.json'
    
    if status_path.exists():
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
                # Parse datetime strings
                for dt_field in ['started_at', 'finished_at', 'last_restart']:
                    if status_data.get(dt_field):
                        status_data[dt_field] = datetime.fromisoformat(status_data[dt_field])
                return SystemStatus(**status_data)
        except Exception as e:
            logger.exception(f"Error loading trajectory status: {status_path}")
            return None
    
    return None

@router.get("/", response_model=List[TrajectoryListItem])
async def get_trajectories():
    """Get all trajectories."""
    trajectories_dir = get_moatless_trajectories_dir()
    trajectories = []
    for trajectory_id in os.listdir(trajectories_dir):
        trajectory_path = trajectories_dir / trajectory_id
        if trajectory_path.is_dir():
            status = load_trajectory_status(trajectory_path)
            if status:
                # Create TrajectoryListItem from status
                trajectory_item = TrajectoryListItem(
                    **status.model_dump(),
                    trajectory_id=trajectory_id
                )
                trajectories.append(trajectory_item)
    return trajectories


@router.get("/{trajectory_id}", response_model=TrajectoryResponseDTO)
async def get_trajectory(trajectory_id: str):
    """Get the status, trajectory data, and events for a specific trajectory."""
    try:
        
        # First try to get active run from runner
        system = await agentic_runner.get_run(trajectory_id)
        trajectory_dir = get_moatless_trajectory_dir(trajectory_id)
        if system:
            logger.info(f"Active run found for trajectory {trajectory_id}")

            status = "running"
            nodes = convert_nodes(system.root)

            trajectory = TrajectoryDTO(
                id=trajectory_id,
                nodes=nodes,
                completionCost=system.root.total_usage().completion_cost,
                promptTokens=system.root.total_usage().prompt_tokens,
                completionTokens=system.root.total_usage().completion_tokens,
                cachedTokens=system.root.total_usage().cache_read_tokens,
                flags=getattr(system, "flags", []),
            )

            system_status = system.get_status()

        else:
            trajectory_path = get_trajectory_path(trajectory_id)
            try:
                trajectory = load_trajectory_from_file(trajectory_path)
                status = "finished"
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Run not found")

            system_status = load_trajectory_status(trajectory_dir)
            if system_status.status == "running":
                system_status.status = "stopped"
            
            status = system_status.status

        events = load_trajectory_events(trajectory_dir)

        return TrajectoryResponseDTO(
            id=trajectory_id,
            status=status,
            system_status=system_status,
            agent_id=system_status.metadata.get("agent_id"),
            model_id=system_status.metadata.get("model_id"),
            events=events,
            **trajectory.model_dump()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting trajectory data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 
  

@router.post("/{project_id}/{trajectory_id}/resume")
async def resume(project_id: str, trajectory_id: str, request: StartTrajectoryRequest):
    """Resume a trajectory."""
    system = await agentic_runner.get_run(trajectory_id, project_id)
    if system:
        raise HTTPException(status_code=400, detail="Flow is already running")
    
    agentic_flow = AgenticLoop.from_trajectory_id(trajectory_id, project_id, request.agent_id, request.model_id)
    
    await agentic_runner.start(agentic_flow, message=request.message)


@router.post("/{project_id}/{trajectory_id}/retry")
async def retry(project_id: str, trajectory_id: str, request: RetryTrajectoryRequest):
    """Retry a run."""
    system = await agentic_runner.get_run(trajectory_id, project_id)
    if system:
        raise HTTPException(status_code=400, detail="Flow is already running")
    
    agentic_flow = AgenticFlow.from_trajectory_id(trajectory_id, project_id)

    await swebench_setup(agentic_flow, trajectory_id)

    await agentic_flow.reset_node(request.node_id)
    agentic_flow.persist()
    await agentic_runner.start(agentic_flow)
    
    logger.info(f"Started retry for trajectory {trajectory_id}")

async def swebench_setup(flow: AgenticFlow, trajectory_id: str):
    """Workaround to set up legacy solution for swebench."""

    moatless_instance = get_moatless_instance(trajectory_id)
    if not moatless_instance:
        # No instance found, skip setup
        return
    
    logger.info(f"Setting up swebench for trajectory {trajectory_id}")
    
    repository = await create_repository_async(moatless_instance)
    code_index = create_index(moatless_instance, repository=repository)

    runtime = TestbedEnvironment(
        repository=repository,
        instance_id=trajectory_id,
        log_dir=str(get_moatless_trajectory_dir(trajectory_id) / "testbed_logs"),
        enable_cache=True,
    )
    workspace = Workspace(
        repository=repository,
        code_index=code_index,
        runtime=runtime,
        legacy_workspace=True
    )

    flow.workspace = workspace

