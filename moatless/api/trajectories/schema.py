from typing import Any, Optional

from pydantic import BaseModel, Field


class AttachmentData(BaseModel):
    name: str = Field(description="Original filename of the attachment")
    data: str = Field(description="Base64-encoded data URI of the attachment")


class CreateTrajectoryRequest(BaseModel):
    """Request for creating a new trajectory."""
    
    flow_id: Optional[str] = Field(None, description="ID of existing flow configuration to use")
    flow_config: Optional[dict] = Field(None, description="Direct flow configuration to use")
    model_id: Optional[str] = Field(None, description="Model ID to use with the flow")
    message: str = Field(description="Initial message for the trajectory")
    project_id: Optional[str] = Field(None, description="Project ID (defaults to 'default')")
    trajectory_id: Optional[str] = Field(None, description="Trajectory ID (auto-generated if not provided)")
    metadata: Optional[dict[str, Any]] = Field(None, description="Optional metadata for the trajectory")


class StartTrajectoryRequest(BaseModel):
    agent_id: Optional[str] = Field(None, description="The agent to use for the loop")
    model_id: Optional[str] = Field(None, description="The model to use for the loop")
    message: str = Field(description="The message to start the loop with")
    attachments: Optional[list[AttachmentData]] = Field(
        default=None, description="List of attachments with filename and base64 data"
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
    scope: Optional[str] = None
    event_type: str
    trajectory_id: str
    data: Optional[dict[str, Any]] = None
