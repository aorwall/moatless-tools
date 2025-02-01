"""API endpoints for run status and trajectory data."""

import os
import json
import logging
from fastapi import APIRouter, HTTPException
from moatless.api.trajectory.schema import TrajectoryDTO
from moatless.runner import agentic_runner
from moatless.api.trajectory.trajectory_utils import convert_nodes, create_trajectory_dto, load_trajectory_from_file
from .schema import RunResponseDTO, RunStatusDTO, RunEventDTO
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

def get_run_dir(run_id: str) -> Path:
    """Get the directory path for a run."""
    moatless_dir = os.getenv('MOATLESS_DIR', '.moatless')
    return Path(moatless_dir) / run_id

def get_trajectory_path(run_id: str) -> str:
    """Get the trajectory file path for a run."""
    return str(get_run_dir(run_id) / 'trajectory.json')

def load_run_events(run_dir: Path) -> list[RunEventDTO]:
    """Load events from events.jsonl file."""
    events_path = run_dir / 'events.jsonl'
    events = []
    
    if events_path.exists():
        try:
            with open(events_path, 'r', encoding='utf-8') as f:
                for line in f:
                    event_data = json.loads(line)
                    # Convert ISO timestamp to milliseconds
                    dt = datetime.fromisoformat(event_data['timestamp'])
                    event_data['timestamp'] = int(dt.timestamp() * 1000)
                    events.append(RunEventDTO(**event_data))
        except Exception as e:
            logger.error(f"Error reading events file: {e}")
            
    return events

def load_run_status(run_dir: Path) -> RunStatusDTO:
    """Load status from status.json file."""
    status_path = run_dir / 'status.json'
    
    if status_path.exists():
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
                # Parse datetime strings
                for dt_field in ['started_at', 'finished_at', 'last_restart']:
                    if status_data.get(dt_field):
                        status_data[dt_field] = datetime.fromisoformat(status_data[dt_field])
                return RunStatusDTO(**status_data)
        except Exception as e:
            logger.error(f"Error reading status file: {e}")
    
    # Return default status if file doesn't exist or has errors
    return RunStatusDTO(
        status="unknown",
        started_at=datetime.utcnow()
    )

@router.get("/{run_id}", response_model=RunResponseDTO)
async def get_run(run_id: str):
    logger.info(f"Getting run {run_id}")
    """Get the status, trajectory data, and events for a specific run."""
    try:
        run_dir = get_run_dir(run_id)
        
        # First try to get active run from runner
        system = agentic_runner.get_run(run_id)
        
        if system:
            # Active run found - get status and trajectory from system
            status = "running"
            nodes = convert_nodes(system.root)

            trajectory = TrajectoryDTO(
                nodes=nodes,
                completionCost=system.root.total_usage().completion_cost,
                promptTokens=system.root.total_usage().prompt_tokens,
                completionTokens=system.root.total_usage().completion_tokens,
                cachedTokens=system.root.total_usage().cache_read_tokens,
                flags=getattr(system, "flags", []),
            )

            system_status = system.get_status()

        else:
            logger.info(f"Run {run_id} not found in runner, trying to load from file")
            # Try to load completed run from trajectory file
            trajectory_path = get_trajectory_path(run_id)
            try:
                trajectory = load_trajectory_from_file(trajectory_path)
                status = "finished"
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Run not found")

            # Load status from file
            system_status = load_run_status(run_dir)
            if system_status.status == "running":
                system_status.status = "stopped"
            
            status = system_status.status

        events = load_run_events(run_dir)

        return RunResponseDTO(
            status=status,
            system_status=system_status,
            trajectory=trajectory,
            events=events
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting run data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 