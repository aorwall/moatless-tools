"""Main API module for Moatless."""

import importlib.resources as pkg_resources
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    Request,
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
from moatless.api.swebench.api import router as swebench_router
from moatless.api.settings.api import router as settings_router
from moatless.api.trajectories.api import router as trajectory_router
from moatless.api.trajectory.schema import TrajectoryDTO
from moatless.api.trajectory.trajectory_utils import (
    load_trajectory_from_file,
    create_trajectory_dto,
)
from moatless.api.logging_config import setup_logging, get_logger
from moatless.artifacts.artifact import ArtifactListItem
from moatless.events import BaseEvent, event_bus
from moatless.telemetry import setup_telemetry
from moatless.workspace import Workspace

import psutil
import os
import asyncio
from datetime import datetime
import gc
import tracemalloc
from collections import Counter

setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=Path("logs"),
    service_name="moatless-api",
    environment=os.getenv("DEPLOYMENT_ENV", "development"),
    use_json=os.getenv("LOG_FORMAT", "").lower() == "json",
)

logger = get_logger(__name__)


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
            logger.info("No active connections, skipping broadcast")
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


async def handle_system_event(trajectory_id: str, project_id: str, event: BaseEvent):
    """Handle system events and broadcast them via WebSocket"""
    message = {
        'trajectory_id': trajectory_id,
        'project_id': project_id,
        'event_type': event.event_type,
        'data': event.model_dump(exclude_none=True, exclude={'event_type'})
    }
    logger.debug(f"Broadcasting event: {message}")
    await manager.broadcast_message(message)

class MemoryMonitor:
    def __init__(self, interval: int = 300):  # Changed to 5 minutes default
        self.interval = interval
        self._memory_log_dir = Path("profiles/memory")
        self._memory_log_dir.mkdir(parents=True, exist_ok=True)
        self._task = None
        self._running = False
        self._snapshot = None
        self._sample_count = 0
        tracemalloc.start(10)  # Reduced from 25 to 10 frames to lower overhead

    def _should_take_detailed_snapshot(self) -> bool:
        """Only take detailed snapshots periodically to reduce overhead."""
        self._sample_count += 1
        return self._sample_count % 5 == 0  # Detailed snapshot every 5th time

    async def start(self):
        if self._running:
            return
        
        self._running = True
        self._snapshot = tracemalloc.take_snapshot()
        self._task = asyncio.create_task(self._monitor())
        logger.info("Memory monitoring started with tracemalloc")

    async def stop(self):
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        tracemalloc.stop()
        logger.info("Memory monitoring stopped")

    async def _monitor(self):
        while self._running:
            try:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                timestamp = datetime.now().isoformat()
                
                # Basic metrics always collected
                log_entry = {
                    "timestamp": timestamp,
                    "rss": memory_info.rss,
                    "vms": memory_info.vms,
                    "connections": len(manager.active_connections) if hasattr(manager, 'active_connections') else 0
                }

                # Detailed metrics only collected periodically
                if self._should_take_detailed_snapshot():
                    gc.collect()  # Force garbage collection
                    snapshot = tracemalloc.take_snapshot()
                    
                    if self._snapshot:
                        stats = snapshot.compare_to(self._snapshot, 'lineno')
                        significant_changes = [
                            (stat.traceback[0].filename, stat.size_diff)
                            for stat in stats[:3]  # Only track top 3 changes
                            if abs(stat.size_diff) > 100000  # Only track significant changes (>100KB)
                        ]
                        
                        if significant_changes:
                            log_entry["memory_growth"] = {
                                str(filename): size_diff
                                for filename, size_diff in significant_changes
                            }
                    
                    self._snapshot = snapshot
                    
                    # Basic object counting without expensive type checks
                    log_entry["gc_stats"] = {
                        "collections": gc.get_count(),
                        "objects": len(gc.get_objects()),
                        "garbage": len(gc.garbage)
                    }
                
                log_file = self._memory_log_dir / f"api_memory_{datetime.now().strftime('%Y%m%d')}.json"
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                
                await asyncio.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error logging memory usage: {e}")
                await asyncio.sleep(self.interval)

def create_api(workspace: Workspace | None = None) -> FastAPI:
    """Create and initialize the API with an optional workspace"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Initialize OpenTelemetry
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    app_insights_conn_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    
    # Choose telemetry provider based on environment variables
    if app_insights_conn_string:
        setup_telemetry(
            service_name="moatless-api",
            provider="azure",
            logger_name=__name__,
            resource_attributes={
                "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
            }
        )
    else:
        setup_telemetry(
            service_name="moatless-api",
            endpoint=otlp_endpoint,
            resource_attributes={
                "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
            }
        )

    api = FastAPI(title="Moatless API")
    
    # Initialize memory monitor with 5 minute interval
    memory_monitor = MemoryMonitor(interval=300)  # Changed from 60 to 300 seconds

    @api.on_event("startup")
    async def startup_event():
        await memory_monitor.start()

    @api.on_event("shutdown")
    async def shutdown_event():
        await memory_monitor.stop()

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
                    if data.get('type') == 'ping':
                        await websocket.send_json({'type': 'pong'})
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
            return create_trajectory_dto(trajectory_data, trajectory_id=file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid trajectory file: {str(e)}")

    # Include model, agent, and loop configuration routers
    router.include_router(settings_router, prefix="/settings", tags=["settings"])
    router.include_router(model_router, prefix="/models", tags=["models"])
    router.include_router(agent_router, prefix="/agents", tags=["agents"])
    router.include_router(swebench_router, prefix="/swebench", tags=["swebench"])
    router.include_router(trajectory_router, prefix="/trajectories", tags=["trajectories"])
    router.include_router(loop_router, prefix="/loop", tags=["loop"])
    router.include_router(artifact_router, prefix="/artifacts", tags=["artifacts"])

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

    event_bus.subscribe(handle_system_event)

    return api
