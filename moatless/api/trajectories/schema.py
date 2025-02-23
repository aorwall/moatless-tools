from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from moatless.api.trajectory.schema import TrajectoryDTO
from moatless.flow.flow import SystemStatus


class AttachmentData(BaseModel):
    name: str = Field(description="Original filename of the attachment")
    data: str = Field(description="Base64-encoded data URI of the attachment")

class StartTrajectoryRequest(BaseModel):
    agent_id: Optional[str] = Field(None, description="The agent to use for the loop")
    model_id: Optional[str] = Field(None, description="The model to use for the loop")
    message: str = Field(description="The message to start the loop with")
    attachments: Optional[List[AttachmentData]] = Field(
        default=None,
        description="List of attachments with filename and base64 data"
    )

class RetryTrajectoryRequest(BaseModel):
    agent_id: Optional[str] = Field(None, description="The agent to use for the loop")
    model_id: Optional[str] = Field(None, description="The model to use for the loop")
    node_id: Optional[int] = Field(None, description="The node to start the loop from")
    
class ExecuteNodeRequest(BaseModel):
    agent_id: Optional[str] = Field(None, description="The agent to use for the loop")
    model_id: Optional[str] = Field(None, description="The model to use for the loop")
    node_id: Optional[int] = Field(None, description="The node to start the loop from")

class TrajectoryEventDTO(BaseModel):
    """Single event in a trajectory."""
    timestamp: int  # Changed from datetime to int (milliseconds)
    event_type: str
    trajectory_id: str
    data: Optional[Dict[str, Any]] = None

class TrajectoryResponseDTO(TrajectoryDTO):
    """Response containing run status, trajectory data, and events."""
    id: str
    project_id: Optional[str] = None
    status: str
    agent_id: Optional[str] = None
    model_id: Optional[str] = None
    system_status: SystemStatus
    events: List[TrajectoryEventDTO] = []

class TrajectoryListItem(SystemStatus):
    """Response model for trajectory list items"""
    trajectory_id: str 