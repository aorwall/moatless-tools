from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from moatless.agentic_system import SystemStatus
from moatless.api.trajectory.schema import TrajectoryDTO

class RunEventDTO(BaseModel):
    """Single event in a run."""
    timestamp: int  # Changed from datetime to int (milliseconds)
    event_type: str
    node_id: Optional[int] = None
    agent_id: Optional[str] = None
    action_name: Optional[str] = None

class RunResponseDTO(BaseModel):
    """Response containing run status, trajectory data, and events."""
    status: str
    system_status: SystemStatus
    trajectory: Optional[TrajectoryDTO] = None
    events: List[RunEventDTO] = [] 