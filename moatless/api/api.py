from fastapi import FastAPI, HTTPException, UploadFile, Request
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import json
import logging
from moatless.workspace import Workspace
from moatless.artifacts.artifact import ArtifactListItem
from moatless.api.schema import TrajectoryDTO
from moatless.api.trajectory_utils import load_trajectory_from_file, create_trajectory_dto
from fastapi.staticfiles import StaticFiles
import importlib.resources as pkg_resources
from pathlib import Path


logger = logging.getLogger(__name__)


def create_api(workspace: Workspace | None = None) -> FastAPI:
    """Create and initialize the API with an optional workspace"""
    api = FastAPI(title="Moatless API")

    # Add CORS middleware with proper configuration
    origins = [
        "http://localhost:5173",    # SvelteKit dev server
        "http://127.0.0.1:5173",    # Alternative local dev URL (IPv4)
        "http://[::1]:5173",        # Alternative local dev URL (IPv6)
        "http://localhost:4173",    # SvelteKit preview server
        "http://127.0.0.1:4173",    # Alternative preview URL (IPv4)
        "http://[::1]:4173",        # Alternative preview URL (IPv6)
    ]

    api.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,  # Cache preflight requests for 1 hour
    )

    # Create API router with /api prefix
    router = FastAPI(title="Moatless API")

    if workspace is not None:
        @router.get("/artifacts", response_model=List[ArtifactListItem])
        async def list_all_artifacts():
            """Get all artifacts across all types"""
            return workspace.get_all_artifacts()

        @router.get("/artifacts/{type}", response_model=List[ArtifactListItem])
        async def list_artifacts(type: str):
            try:
                return workspace.get_artifacts_by_type(type)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @router.get("/artifacts/{type}/{id}", response_model=Dict[str, Any])
        async def get_artifact(type: str, id: str):
            try:
                artifact = workspace.get_artifact(type, id)
                if not artifact:
                    raise HTTPException(status_code=404, detail=f"Artifact {id} not found")
                return artifact.to_ui_representation()
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

    @router.get("/trajectory", response_model=TrajectoryDTO)
    async def get_trajectory(file_path: str):
        """Get trajectory data from a file path"""
        try:
            return load_trajectory_from_file(file_path)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.post("/trajectory/upload", response_model=TrajectoryDTO)
    async def upload_trajectory(file: UploadFile):
        """Upload and process a trajectory file"""
        try:
            content = await file.read()
            trajectory_data = json.loads(content.decode())
            return create_trajectory_dto(trajectory_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid trajectory file: {str(e)}")

    # Mount the API router with /api prefix
    api.mount("/api", router)

    # Only serve UI files if API extras are installed
    try:
        import fastapi.staticfiles
        # Try to find UI files in the installed package
        ui_files = pkg_resources.files('moatless_api') / 'ui/dist'
        if ui_files.exists():
            logger.info(f"Found UI files in package at {ui_files}")
            
            # Serve static files from _app directory
            api.mount("/_app", StaticFiles(directory=str(ui_files / "_app")), name="static")
            
            # Create a static files instance for serving index.html
            html_app = StaticFiles(directory=str(ui_files), html=True)
            
            @api.get("/{full_path:path}")
            async def serve_spa(request: Request, full_path: str):
                if full_path.startswith("api/"):
                    raise HTTPException(status_code=404, detail="Not found")
                return await html_app.get_response("index.html", request.scope)
        else:
            # Fallback to development path
            ui_path = Path("ui/dist")
            if ui_path.exists():
                logger.info(f"Found UI files in development path at {ui_path}")
                
                # Serve static files from _app directory
                api.mount("/_app", StaticFiles(directory=str(ui_path / "_app")), name="static")
                
                # Create a static files instance for serving index.html
                html_app = StaticFiles(directory=str(ui_path), html=True)
                
                @api.get("/{full_path:path}")
                async def serve_spa(request: Request, full_path: str):
                    if full_path.startswith("api/"):
                        raise HTTPException(status_code=404, detail="Not found")
                    return await html_app.get_response("index.html", request.scope)
            else:
                logger.info("No UI files found")
    except ImportError:
        logger.info("API extras not installed, UI will not be served")

    return api
