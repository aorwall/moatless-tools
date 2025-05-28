import base64
import base64
import logging
import mimetypes
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from moatless.flow.manager import FlowManager
from moatless.api.dependencies import get_flow_manager
from moatless.api.loop.schema import LoopResponseDTO
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# You can use an environment variable to override this path if needed.
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")


class AttachmentData(BaseModel):
    name: str = Field(description="Original filename of the attachment")
    data: str = Field(description="Base64-encoded data URI of the attachment")


class LoopRequest(BaseModel):
    agent_id: str = Field(description="The agent to use for the loop")
    model_id: str = Field(description="The model to use for the loop")
    message: str = Field(description="The message to start the loop with")
    attachments: Optional[list[AttachmentData]] = Field(
        default=None, description="List of attachments with filename and base64 data"
    )
    repository_path: str = Field(
        description="The path to the repository to use for the loop"
    )


@router.post("", response_model=LoopResponseDTO)
async def start_loop(
    request: LoopRequest,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    try:
        run_id = f"loop_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        flow = await flow_manager.create_loop(
            trajectory_id=run_id,
            project_id=f"{request.agent_id}_{request.model_id}",
            agent_id=request.agent_id,
            model_id=request.model_id,
            message=request.message,
            repository_path=request.repository_path,
        )
        
        return LoopResponseDTO(
            trajectory_id=run_id,
            project_id=f"{request.agent_id}_{request.model_id}",
        )
    except Exception as e:
        logger.exception(f"Failed to start loop: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
