"""Main API module for Moatless."""

import asyncio
import gc
import importlib.resources as pkg_resources
import json
import logging
import os
import tracemalloc
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from moatless.api.agents.api import router as agent_router
from moatless.api.artifacts.api import router as artifact_router
from moatless.api.loop.api import router as loop_router
from moatless.api.models.api import router as model_router
from moatless.api.settings.api import router as settings_router
from moatless.api.swebench.api import router as swebench_router
from moatless.api.swebench.schema import RunnerResponseDTO
from moatless.api.trajectories.api import router as trajectory_router
from moatless.api.trajectory.schema import TrajectoryDTO
from moatless.api.trajectory.trajectory_utils import (
    create_trajectory_dto,
    load_trajectory_from_file,
)
from moatless.artifacts.artifact import ArtifactListItem
from moatless.events import BaseEvent, event_bus
from moatless.logging_config import get_logger, setup_logging
from moatless.runner.rq import RQRunner
from moatless.runner.runner import JobsStatusSummary
from moatless.telemetry import setup_telemetry
from moatless.workspace import Workspace

setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=Path(os.getcwd()) / "logs",
    service_name="moatless-api",
    environment=os.getenv("DEPLOYMENT_ENV", "development"),
    use_json=os.getenv("LOG_FORMAT", "").lower() == "json",
)

logger = get_logger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        try:
            logger.info("Accepting WebSocket connection")
            await websocket.accept()
            self.active_connections.add(websocket)
            logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection: {e}")
            raise

    async def disconnect(self, websocket: WebSocket):
        """Safely disconnect a WebSocket connection."""
        try:
            self.active_connections.discard(websocket)
            if not websocket.client_state.DISCONNECTED:
                await websocket.close()
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Error during WebSocket disconnect: {e}")

    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            logger.debug("No active connections, skipping broadcast")
            return

        logger.debug(f"Broadcasting message to {len(self.active_connections)} clients")

        connections = self.active_connections.copy()
        disconnected = set()

        for connection in connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                disconnected.add(connection)

        for connection in disconnected:
            await self.disconnect(connection)


manager = ConnectionManager()

runner = RQRunner()


async def handle_event(event: BaseEvent):
    """Handle system events and broadcast them via WebSocket"""
    logger.debug(f"Broadcasting event: {event.scope}:{event.event_type}")

    if event.scope == "flow" or event.scope == "evaluation":
        await manager.broadcast_message(event.to_dict())


def create_api(workspace: Workspace | None = None) -> FastAPI:
    """Create and initialize the API with an optional workspace"""

    load_dotenv()

    api = FastAPI(title="Moatless API")

    @api.on_event("startup")
    async def startup_event():
        logger.info("Subscribing to system events")
        setup_telemetry()

        await event_bus.initialize()
        await event_bus.subscribe(handle_event)

    @api.on_event("shutdown")
    async def shutdown_event():
        logger.info("Unsubscribing from system events")
        await event_bus.unsubscribe(handle_event)

    # Initialize FastAPI instrumentation
    FastAPIInstrumentor.instrument_app(api)

    # Update CORS middleware configuration
    origins = [
        "http://localhost:5173",  # Development frontend
        "http://localhost:8000",  # Development API
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ]

    api.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  # Replace "*" with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],  # Add this to expose headers to the client
    )

    router = FastAPI(title="Moatless API")

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        try:
            await manager.connect(websocket)

            while True:
                try:
                    # Receive and process messages
                    message = await websocket.receive_text()
                    data = json.loads(message)

                    # Handle ping message
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    else:
                        logger.info(f"Received message: {data}")

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error in WebSocket connection: {e}")
                    break
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            raise
        finally:
            await manager.disconnect(websocket)

    if workspace is not None:

        @router.get("/artifacts", response_model=list[ArtifactListItem])
        async def list_all_artifacts():
            """Get all artifacts across all types"""
            return workspace.get_all_artifacts()

        @router.get("/artifacts/{type}", response_model=list[ArtifactListItem])
        async def list_artifacts(type: str):
            try:
                return workspace.get_artifacts_by_type(type)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @router.get("/artifacts/{type}/{id}", response_model=dict[str, Any])
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
            return create_trajectory_dto(trajectory_data, trajectory_id=file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid trajectory file: {str(e)}")

    @router.get("/runner", response_model=RunnerResponseDTO)
    async def get_runner() -> RunnerResponseDTO:
        """Get the runner"""
        return RunnerResponseDTO(info=await runner.get_runner_info(), jobs=await runner.get_jobs())

    @router.get("/runner/jobs/summary/{project_id}", response_model=JobsStatusSummary)
    async def get_job_status_summary(project_id: str) -> JobsStatusSummary:
        """Get a summary of job statuses for a project"""
        return await runner.get_job_status_summary(project_id)

    @router.post("/runner/jobs/{project_id}/cancel")
    async def cancel_jobs(project_id: str, request: Request):
        """Cancel jobs for a project"""
        data = (
            await request.json()
            if request.headers.get("content-length") and int(request.headers.get("content-length", "0")) > 0
            else None
        )
        trajectory_id = data.get("trajectory_id") if data else None
        await runner.cancel_job(project_id, trajectory_id)
        return {"status": "success", "message": "Job(s) canceled successfully"}

    @router.post("/runner/jobs/{project_id}/{trajectory_id}/retry")
    async def retry_job(project_id: str, trajectory_id: str):
        """Retry a failed job"""
        success = await runner.retry_job(project_id, trajectory_id)
        if success:
            return {"status": "success", "message": "Job requeued successfully"}
        else:
            raise HTTPException(
                status_code=400, detail="Failed to retry job, it may not exist or not be in a failed state"
            )

    # Include model, agent, and loop configuration routers
    router.include_router(settings_router, prefix="/settings", tags=["settings"])
    router.include_router(model_router, prefix="/models", tags=["models"])
    router.include_router(agent_router, prefix="/agents", tags=["agents"])
    router.include_router(swebench_router, prefix="/swebench", tags=["swebench"])
    router.include_router(trajectory_router, prefix="/trajectories", tags=["trajectories"])
    router.include_router(loop_router, prefix="/loop", tags=["loop"])
    router.include_router(artifact_router, prefix="/artifacts", tags=["artifacts"])

    # Only serve UI files if API extras are installed
    try:
        import fastapi.staticfiles

        # First check moatless_api/ui/dist
        ui_path = Path("moatless_api/ui/dist")
        if ui_path.exists():
            logger.info(f"Found UI files at {ui_path}")

            # Create static files instances
            static_files = StaticFiles(directory=str(ui_path))
            html_app = StaticFiles(directory=str(ui_path), html=True)

            # Mount the API router first
            api.mount("/api", router)

            # Add explicit root path handler
            @api.get("/")
            async def serve_root(request: Request):
                scope = request.scope
                scope.update({"path": "/index.html", "method": "GET", "type": "http"})
                return await html_app.get_response("index.html", scope)

            # Mount static files for assets
            api.mount("/assets", StaticFiles(directory=str(ui_path / "assets")), name="assets")

            # Add the catch-all route for SPA
            @api.get("/{full_path:path}")
            async def serve_spa(request: Request, full_path: str):
                if full_path.startswith("api/"):
                    raise HTTPException(status_code=404, detail="Not found")
                try:
                    return await html_app.get_response("index.html", request.scope)
                except Exception as e:
                    logger.error(f"Error serving SPA: {e}")
                    raise HTTPException(status_code=404, detail="Not found")

        else:
            # Try to find UI files in the installed package
            try:
                ui_files = pkg_resources.files("moatless_api") / "ui/dist"
                if ui_files.exists():
                    logger.info(f"Found UI files in package at {ui_files}")

                    # Create static files instances
                    static_files = StaticFiles(directory=str(ui_files))
                    html_app = StaticFiles(directory=str(ui_files), html=True)

                    # Mount the API router first
                    api.mount("/api", router)

                    # Add explicit root path handler
                    @api.get("/")
                    async def serve_root(request: Request):
                        scope = request.scope
                        scope.update({"path": "/index.html", "method": "GET", "type": "http"})
                        return await html_app.get_response("index.html", scope)

                    # Mount static files for assets
                    api.mount("/assets", StaticFiles(directory=str(ui_files / "assets")), name="assets")

                    # Add the catch-all route for SPA
                    @api.get("/{full_path:path}")
                    async def serve_spa(request: Request, full_path: str):
                        if full_path.startswith("api/"):
                            raise HTTPException(status_code=404, detail="Not found")
                        try:
                            return await html_app.get_response("index.html", request.scope)
                        except Exception as e:
                            logger.error(f"Error serving SPA: {e}")
                            raise HTTPException(status_code=404, detail="Not found")
                else:
                    logger.info("No UI files found in package")
                    # Mount API router even if no UI files
                    api.mount("/api", router)
            except Exception as e:
                logger.error(f"Error accessing package UI files: {e}")
                logger.info("No UI files found")
                # Mount API router even if error occurs
                api.mount("/api", router)
    except ImportError:
        logger.info("API extras not installed, UI will not be served")
        # Mount API router even if no UI support
        api.mount("/api", router)

    return api


def main():
    import uvicorn

    api = create_api()
    uvicorn.run(api, host="0.0.0.0", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()
