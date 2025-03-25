import base64
import base64
import logging
import mimetypes
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from moatless.agent.manager import AgentConfigManager
from moatless.api.dependencies import get_agent_manager, get_model_manager
from moatless.api.loop.schema import LoopResponseDTO
from moatless.artifacts.artifact import ArtifactChange
from moatless.artifacts.file import FileArtifact, FileArtifactHandler
from moatless.completion.manager import ModelConfigManager
from moatless.environment.local import LocalBashEnvironment
from moatless.flow.loop import AgenticLoop
from moatless.repository.git import GitRepository
from moatless.utils.moatless import get_moatless_trajectory_dir
from moatless.workspace import Workspace
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
    agent_manager: AgentConfigManager = Depends(get_agent_manager),
    model_manager: ModelConfigManager = Depends(get_model_manager),
):
    try:
        run_id = f"loop_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        trajectory_dir = get_moatless_trajectory_dir(run_id)

        repository = None
        environment = None
        if request.repository_path:
            repository = GitRepository(repo_path=request.repository_path)
            environment = LocalBashEnvironment(cwd=request.repository_path)

        workspace = Workspace(repository=repository, environment=environment)
        file_handler = FileArtifactHandler(trajectory_dir=trajectory_dir)

        artifact_changes = []

        if request.attachments:
            for attachment in request.attachments:
                file_data = attachment.data.split(",")[1]  # Remove the data URI prefix
                file_bytes = base64.b64decode(file_data)
                mime_type = mimetypes.guess_type(attachment.name)[0]

                artifact = FileArtifact(
                    id=attachment.name,
                    name=attachment.name,
                    file_path=attachment.name,
                    content=file_bytes,
                    mime_type=mime_type,
                )
                await file_handler.create(artifact)
                artifact_changes.append(
                    ArtifactChange(
                        artifact_id=artifact.id,
                        artifact_type=artifact.type,
                        change_type="added",
                        actor="user",
                    )
                )

        # Get the agent instance and completion model.
        agent = agent_manager.get_agent(agent_id=request.agent_id)
        completion_model = model_manager.create_completion_model(request.model_id)
        completion_model.metadata = {"trajectory_id": run_id}

        agent.workspace = workspace
        agent.completion_model = completion_model

        trajectory_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        loop = AgenticLoop.create(
            message=request.message,
            trajectory_id=trajectory_id,
            project_id=f"{request.agent_id}_{request.model_id}",
            agent=agent,
            max_iterations=15,
            max_cost=1.0,
        )

        await loop.persist()

        return LoopResponseDTO(
            trajectory_id=trajectory_id,
            project_id=f"{request.agent_id}_{request.model_id}",
        )
    except Exception as e:
        logger.exception(f"Failed to start loop: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
