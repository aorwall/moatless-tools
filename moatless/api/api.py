"""Main API module for Moatless."""

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from typing import List, Dict, Any, Set
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
from moatless.workspace import Workspace
from moatless.artifacts.artifact import ArtifactListItem
from moatless.api.trajectory.schema import TrajectoryDTO
from moatless.api.trajectory.trajectory_utils import (
    load_trajectory_from_file,
    create_trajectory_dto,
)
from fastapi.staticfiles import StaticFiles
import importlib.resources as pkg_resources
from pathlib import Path
from moatless.api.models.api import router as model_router
from moatless.api.agents.api import router as agent_router
from moatless.api.swebench.api import router as swebench_router
from moatless.api.runs.api import router as run_router
from moatless.events import event_bus


logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        try:
            logger.info("Accepting WebSocket connection")
            await websocket.accept()
            self.active_connections.add(websocket)
            logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection: {e}")
            raise

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.discard(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Error during WebSocket disconnect: {e}")

    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        logger.info(f"Broadcasting message to {len(self.active_connections)} clients")
        
        connections = self.active_connections.copy()
        disconnected = set()
        
        for connection in connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                disconnected.add(connection)
    
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()


async def handle_system_event(run_id: str, event: dict):
    """Handle system events and broadcast them via WebSocket"""
    message = {
        'run_id': run_id,
        **event.model_dump(exclude_none=True)
    }
    await manager.broadcast_message(message)


def create_api(workspace: Workspace | None = None) -> FastAPI:
    """Create and initialize the API with an optional workspace"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    api = FastAPI(title="Moatless API")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=3600,
    )

    router = FastAPI(title="Moatless API")

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        try:
            await manager.connect(websocket)
            
            while True:
                try:
                    # Wait for messages but don't do anything with them
                    # This keeps the connection alive
                    await websocket.receive_text()
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error in WebSocket connection: {e}")
                    break
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            raise
        finally:
            manager.disconnect(websocket)

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
            logger.info(f"Loading trajectory from {file_path}")
            return load_trajectory_from_file(file_path)
        except ValueError as e:
            logger.exception(f"Failed to load trajectory from {file_path}")
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

    # Include model, agent, and loop configuration routers
    router.include_router(model_router, prefix="/models", tags=["models"])
    router.include_router(agent_router, prefix="/agents", tags=["agents"])
    router.include_router(swebench_router, prefix="/swebench", tags=["swebench"])
    router.include_router(run_router, prefix="/runs", tags=["runs"])

    # Mount the API router with /api prefix
    api.mount("/api", router)

    # Only serve UI files if API extras are installed
    try:
        import fastapi.staticfiles

        # Try to find UI files in the installed package
        ui_files = pkg_resources.files("moatless_api") / "ui/dist"
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

    logger.info("Subscribing to system events")
    # Subscribe to system events
    event_bus.subscribe(handle_system_event)

    return api
