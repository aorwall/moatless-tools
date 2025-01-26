from fastapi import FastAPI, HTTPException, UploadFile
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import json
import logging
from moatless.workspace import Workspace
from moatless.artifacts.artifact import ArtifactListItem
from moatless.api.schema import TrajectoryDTO
from moatless.api.trajectory_utils import load_trajectory_from_file, create_trajectory_dto


logger = logging.getLogger(__name__)


def create_api(workspace: Workspace | None = None) -> FastAPI:
    """Create and initialize the API with an optional workspace"""
    api = FastAPI(title="Moatless API")

    # Add CORS middleware with proper configuration
    origins = [
        "http://localhost:5173",  # SvelteKit dev server
        "http://127.0.0.1:5173",  # Alternative local dev URL (IPv4)
        "http://[::1]:5173",  # Alternative local dev URL (IPv6)
    ]

    api.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,  # Cache preflight requests for 1 hour
    )

    if workspace is not None:

        @api.get("/artifacts", response_model=List[ArtifactListItem])
        async def list_all_artifacts():
            """Get all artifacts across all types"""
            return workspace.get_all_artifacts()

        @api.get("/artifacts/{type}", response_model=List[ArtifactListItem])
        async def list_artifacts(type: str):
            try:
                return workspace.get_artifacts_by_type(type)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @api.get("/artifacts/{type}/{id}", response_model=Dict[str, Any])
        async def get_artifact(type: str, id: str):
            try:
                artifact = workspace.get_artifact(type, id)
                if not artifact:
                    raise HTTPException(status_code=404, detail=f"Artifact {id} not found")
                return artifact.to_ui_representation()
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

    @api.get("/trajectory", response_model=TrajectoryDTO)
    async def get_trajectory(file_path: str):
        """Get trajectory data from a file path"""
        try:
            return load_trajectory_from_file(file_path)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @api.post("/trajectory/upload", response_model=TrajectoryDTO)
    async def upload_trajectory(file: UploadFile):
        """Upload and process a trajectory file"""
        try:
            content = await file.read()
            trajectory_data = json.loads(content.decode())
            return create_trajectory_dto(trajectory_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid trajectory file: {str(e)}")

    return api
