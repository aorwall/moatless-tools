from typing import Optional

from pydantic import BaseModel


class LoopRequestDTO(BaseModel):
    agent_id: str
    model_id: str
    message: str
    attachments: Optional[list[str]] = None
    repository_path: Optional[str] = None
    # Attachments (file uploads) are provided as form-data and are not part of the JSON payload


class LoopResponseDTO(BaseModel):
    project_id: str
    trajectory_id: str
