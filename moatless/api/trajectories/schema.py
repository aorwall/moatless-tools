from typing import Any, Optional

from pydantic import BaseModel, Field


class AttachmentData(BaseModel):
    name: str = Field(description="Original filename of the attachment")
    data: str = Field(description="Base64-encoded data URI of the attachment")


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
