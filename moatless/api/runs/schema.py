from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from moatless.api.trajectory.schema import TrajectoryDTO

class RunEventDTO(BaseModel):
    """Single event in a run."""
    timestamp: int  # Changed from datetime to int (milliseconds)
    event_type: str
    node_id: Optional[int] = None
    data: Dict[str, Any]
    attempt_id: Optional[int] = None
    restart_count: Optional[int] = None

class RunStatusDTO(BaseModel):
    """Status of a run."""
    status: str = "running"
    error: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None
    restart_count: int = 0
    last_restart: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    current_attempt: Optional[int] = None

class RunResponseDTO(BaseModel):
    """Response containing run status, trajectory data, and events."""
    status: str
    system_status: RunStatusDTO
    trajectory: Optional[TrajectoryDTO] = None
    events: List[RunEventDTO] = [] 