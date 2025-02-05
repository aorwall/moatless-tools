import mimetypes
from bookkeeper.receipt.artifact import ReceiptArtifactFileHandler
from bookkeeper.verification.artifact import VerificationArtifactFileHandler
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path
import os
import asyncio
import logging
import base64

from moatless.utils.moatless import get_moatless_dir, get_moatless_trajectory_dir
from moatless.api.loop.schema import LoopResponseDTO
from moatless.runner import agentic_runner
from moatless.loop import AgenticLoop
from moatless.artifacts.artifact import ArtifactChange
from moatless.artifacts.file import FileArtifactHandler, FileArtifact
from moatless.config.agent_config import get_agent
from moatless.config.model_config import create_completion_model
from moatless.workspace import Workspace
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# You can use an environment variable to override this path if needed.
ARTIFACTS_DIR = os.getenv('ARTIFACTS_DIR', './artifacts')

class AttachmentData(BaseModel):
    name: str = Field(description="Original filename of the attachment")
    data: str = Field(description="Base64-encoded data URI of the attachment")

class LoopRequest(BaseModel):
    agent_id: str = Field(description="The agent to use for the loop")
    model_id: str = Field(description="The model to use for the loop")
    message: str = Field(description="The message to start the loop with")
    attachments: Optional[List[AttachmentData]] = Field(
        default=None,
        description="List of attachments with filename and base64 data"
    )

@router.post("", response_model=LoopResponseDTO)
async def start_loop(request: LoopRequest):
    try:
        run_id = f"loop_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        files_dir = os.path.join(get_moatless_dir(), "files")
        file_handler = FileArtifactHandler(directory_path=Path(files_dir))
        
        
        # TODO: Move out custom handlers!
        receipt_path = os.path.join(get_moatless_dir(), "receipts.json")
        receipt_handler = ReceiptArtifactFileHandler(file_path=Path(receipt_path))
        verification_path = os.path.join(get_moatless_dir(), "verifications.json")
        verification_handler = VerificationArtifactFileHandler(file_path=Path(verification_path))
        artifact_changes = []

        if request.attachments:
            for attachment in request.attachments:
                file_data = attachment.data.split(',')[1]  # Remove the data URI prefix
                file_bytes = base64.b64decode(file_data)
                mime_type = mimetypes.guess_type(attachment.name)[0]
                logger.info(f"Mime type: {mime_type}")
                logger.info(f"File name: {attachment.name}")

                artifact = FileArtifact(
                    id=attachment.name,
                    name=attachment.name,
                    file_path=attachment.name,
                    content=file_bytes,
                    mime_type=mime_type
                )
                file_handler.create(artifact)
                artifact_changes.append(
                    ArtifactChange(
                        artifact_id=artifact.id,
                        artifact_type=artifact.type,
                        change_type="added",
                        actor="user",
                    )
                )

        workspace = Workspace(artifact_handlers=[file_handler, receipt_handler, verification_handler])
        
        # Get the agent instance and completion model.
        agent = get_agent(agent_id=request.agent_id)
        completion_model = create_completion_model(request.model_id)
        agent.workspace = workspace
        agent.completion_model = completion_model

        # Create an AgenticLoop instance (persist paths can be adjusted as needed).
        persist_dir = f"{get_moatless_trajectory_dir()}/{run_id}"
        loop = AgenticLoop.create(
            message=request.message,
            run_id=run_id,
            agent=agent,
            max_iterations=15,
            persist_dir=persist_dir,
            metadata={
                "agent_id": request.agent_id,
                "model_id": request.model_id,
            }
        )

        # Append any attachment changes to the initial node.
        loop.root.artifact_changes.extend(artifact_changes)

        # Start the loop asynchronously using the runner.
        asyncio.create_task(agentic_runner.start(loop))
        return LoopResponseDTO(run_id=run_id)
    except Exception as e:
        logger.exception(f"Failed to start loop: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 