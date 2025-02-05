import logging
from typing import Dict, List
import uuid
from bookkeeper.fortnox.fortnox_auth import FortnoxAuth
from fastapi import APIRouter, HTTPException
from moatless.artifacts.artifact import ArtifactHandler, ArtifactListItem, ArtifactResponse
from moatless.workspace import Workspace

router = APIRouter()

logger = logging.getLogger(__name__)






@router.get("/{type}/connect")
async def connect_artifact(
    type: str
):
    fortnox_auth = FortnoxAuth()

    state = str(uuid.uuid4())
    authorization_url = fortnox_auth.get_auth_url(state)
    logger.info(f"Create Fortnox connection for handler {type} with authorization URL: {authorization_url}")
    return {"authorization_url": authorization_url}

@router.get("/{type}/callback")
async def fortnox_callback(
    code: str,
    state: str,
    type: str
):
    fortnox_auth = FortnoxAuth()

    try:
        tokens = fortnox_auth.exchange_code_for_token(code)
        await fortnox_auth.save_tokens(tokens)
        logger.info(f"Fortnox account connected successfully for handler {type}")
        return {"success": True}
    except Exception as e:
        logger.exception(f"Failed to connect Fortnox account for handler {type}")
        raise HTTPException(status_code=400, detail="Failed to connect Fortnox account")


@router.get("/{type}", response_model=List[ArtifactListItem])
async def list_artifacts(type: str):
    artifact_handler = ArtifactHandler.get_handler_by_type(type)
    return artifact_handler.get_all_artifacts()
    
@router.get("/{type}/{id}", response_model=ArtifactResponse)
async def get_artifact(type: str, id: str):
    logger.info(f"Getting artifact {id} of type {type}")
    artifact_handler = ArtifactHandler.get_handler_by_type(type)
    artifact = artifact_handler.read(artifact_id=id)
    return artifact.to_ui_representation()

@router.post("/{type}/{id}/persist", response_model=ArtifactResponse)
async def persist_artifact(type: str, id: str):
    try:
        artifact_handler = ArtifactHandler.get_handler_by_type(type)
        artifact = artifact_handler.read(artifact_id=id)
        await artifact_handler.persist(artifact)
        return artifact.to_ui_representation()
    except Exception as e:
        logger.exception(f"Failed to persist artifact {id} of type {type}")
        # Return more specific error details
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "type": type,
                "id": id,
                "error_type": e.__class__.__name__
            }
        )
