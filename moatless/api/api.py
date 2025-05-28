"""Main API module for Moatless."""

import asyncio
import importlib.resources as pkg_resources
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from moatless.eventbus.base import BaseEventBus
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Header,
    Request,
    UploadFile,
    status,
    Depends,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from moatless.api.agents.api import router as agent_router
from moatless.api.loop.api import router as loop_router
from moatless.api.models.api import router as model_router
from moatless.api.settings.api import router as settings_router
from moatless.api.swebench.api import router as swebench_router
from moatless.api.swebench.schema import RunnerResponseDTO, RunnerStatsDTO
from moatless.api.trajectories.api import router as trajectory_router
from moatless.api.trajectory.schema import TrajectoryDTO
from moatless.api.websocket import handle_event, websocket_endpoint
from moatless.artifacts.artifact import ArtifactListItem
from moatless.runner.runner import BaseRunner, JobsStatusSummary, JobStatus, JobDetails
from moatless.settings import get_storage as settings_get_storage, get_event_bus as settings_get_event_bus, get_runner as settings_get_runner
from moatless.telemetry import setup_telemetry
from moatless.utils.warnings import filter_external_warnings
from moatless.workspace import Workspace
from moatless.flow.manager import FlowManager
from moatless.evaluation.manager import EvaluationManager
from moatless.agent.manager import AgentConfigManager
from moatless.completion.manager import ModelConfigManager
from moatless.api.dependencies import get_event_bus, get_runner, get_storage, get_model_manager, get_flow_manager, get_evaluation_manager, get_agent_manager, cleanup_resources

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Filter warnings from external dependencies
filter_external_warnings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
security = HTTPBasic()


# Basic Auth settings
auth_enabled = os.environ.get("MOATLESS_AUTH_ENABLED", "false").lower() == "true"
auth_username = os.environ.get("MOATLESS_AUTH_USERNAME", "admin")
auth_password = os.environ.get("MOATLESS_AUTH_PASSWORD", "password")


async def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify credentials for basic authentication.

    Skip authentication if auth_enabled is False in settings.
    """
    if not auth_enabled:
        return True

    correct_username = secrets.compare_digest(credentials.username, auth_username)
    correct_password = secrets.compare_digest(credentials.password, auth_password)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


def create_api(workspace: Workspace | None = None) -> FastAPI:
    """Create and initialize the API with an optional workspace"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup logic - using lazy initialization with dependencies
        logger.info("API startup event triggered")
        logger.info("Setting up telemetry")
        setup_telemetry()

        # No explicit initialization needed since dependencies will initialize when first used
        yield  # This is where the app runs

        # Shutdown logic - clean up resources
        logger.info("Shutting down API services")
        try:
            await cleanup_resources()
        except Exception as e:
            logger.exception(f"Error during shutdown: {e}")

    # Create FastAPI instance with lifespan handler
    api = FastAPI(
        title="Moatless API",
        json_encoders={datetime: lambda dt: dt.isoformat()},
        model_config={"exclude_none": True},
        lifespan=lifespan,
    )

    # Apply authentication to the main API if enabled
    if auth_enabled:
        # Add authentication middleware to catch all requests
        @api.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Skip authentication for OPTIONS requests (CORS preflight)
            if request.method == "OPTIONS":
                return await call_next(request)

            # Skip authentication for specific paths if needed
            path = request.url.path
            public_paths = [
                "/api/health",  # Health check endpoint
                "/docs",  # Swagger UI
                "/redoc",  # ReDoc UI
                "/openapi.json",  # OpenAPI schema
            ]
            if any(path == public_path for public_path in public_paths):
                return await call_next(request)

            # Get authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Basic "):
                return Response(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    headers={"WWW-Authenticate": "Basic"},
                    content="Authentication required",
                )

            try:
                import base64

                credentials_bytes = base64.b64decode(auth_header[6:])
                credentials_str = credentials_bytes.decode("utf-8")
                username, password = credentials_str.split(":", 1)

                # Verify credentials
                correct_username = secrets.compare_digest(username, auth_username)
                correct_password = secrets.compare_digest(password, auth_password)

                if not (correct_username and correct_password):
                    return Response(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        headers={"WWW-Authenticate": "Basic"},
                        content="Invalid credentials",
                    )
            except Exception:
                return Response(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    headers={"WWW-Authenticate": "Basic"},
                    content="Invalid credentials format",
                )

            # If we get here, credentials are valid
            return await call_next(request)

        logger.info(f"Basic authentication enabled with username: {auth_username}")
    else:
        logger.info("Authentication is disabled")

    router = FastAPI(title="Moatless API")

    # Use the websocket endpoint from the websocket module
    router.websocket("/ws")(websocket_endpoint)

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


    @router.get("/runner", response_model=RunnerResponseDTO)
    async def get_runner_info(
        runner: BaseRunner = Depends(get_runner),
    ) -> RunnerResponseDTO:
        """Get the runner"""
        return RunnerResponseDTO(info=await runner.get_runner_info(), jobs=await runner.get_jobs())

    @router.get("/runner/stats", response_model=RunnerStatsDTO)
    async def get_runner_stats(
        runner: BaseRunner = Depends(get_runner),
    ) -> RunnerStatsDTO:
        """Get lightweight runner stats for the status bar"""
        runner_info = await runner.get_runner_info()
        jobs = await runner.get_jobs()

        # Count jobs by status
        pending_jobs = sum(1 for job in jobs if job.status == JobStatus.PENDING)
        running_jobs = sum(1 for job in jobs if job.status == JobStatus.RUNNING)

        # Get active workers from runner info
        active_workers = runner_info.data.get("ready_nodes", 0)
        total_workers = runner_info.data.get("nodes", 0)
        
        # Get queue size if the runner supports it
        queue_size = 0
        if hasattr(runner, "get_queue_size"):
            queue_size = await runner.get_queue_size()
            
        return RunnerStatsDTO(
            runner_type=runner_info.runner_type,
            status=runner_info.status,
            active_workers=active_workers,
            total_workers=total_workers,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            total_jobs=len(jobs),
            queue_size=queue_size,
        )

    @router.get("/health")
    async def health_check():
        """Health check endpoint for readiness probes.

        Checks that event bus and storage dependencies are available.
        """
        try:
            # Get dependencies through dependency functions
            storage = await get_storage()
            event_bus = await get_event_bus()

            return {"status": "ok", "event_bus": True, "storage": True}
        except Exception as e:
            logger.exception(f"Health check failed: {e}")
            return {"status": "failed", "error": str(e)}

    @router.get("/runner/jobs/summary/{project_id}", response_model=JobsStatusSummary)
    async def get_job_status_summary(project_id: str, runner: BaseRunner = Depends(get_runner)) -> JobsStatusSummary:
        """Get a summary of job statuses for a project"""
        return await runner.get_job_status_summary(project_id)

    @router.post("/runner/jobs/{project_id}/cancel")
    async def cancel_jobs(project_id: str, request: Request, runner: BaseRunner = Depends(get_runner)):
        """Cancel jobs for a project"""
        data = (
            await request.json()
            if request.headers.get("content-length") and int(request.headers.get("content-length", "0")) > 0
            else None
        )
        trajectory_id = data.get("trajectory_id") if data else None
        await runner.cancel_job(project_id, trajectory_id)
        return {"status": "success", "message": "Job(s) canceled successfully"}

    @router.post("/runner/jobs/reset")
    async def reset_jobs(
        request: Request, 
        runner: BaseRunner = Depends(get_runner)
    ):
        """Reset all jobs or jobs for a specific project.
        
        This endpoint will:
        1. Cancel all running jobs
        2. Clear all job history
        
        Accepts an optional project_id in the request body to reset jobs only for that project.
        """
        data = (
            await request.json()
            if request.headers.get("content-length") and int(request.headers.get("content-length", "0")) > 0
            else None
        )
        project_id = data.get("project_id") if data else None
        success = await runner.reset_jobs(project_id)
        
        if success:
            return {"status": "success", "message": f"Jobs reset successfully{f' for project {project_id}' if project_id else ''}"}
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to reset jobs",
            )

    @router.get("/runner/jobs/{project_id}/{trajectory_id}/details")
    async def get_job_details(
        project_id: str, trajectory_id: str, runner: BaseRunner = Depends(get_runner)
    ) -> JobDetails:
        """Get detailed information about a job.

        This endpoint returns detailed information about a job, including:
        - Basic job information (ID, status, timestamps)
        - Runner-specific details organized into sections
        - Error information if the job failed
        """
        job_details = await runner.get_job_details(project_id, trajectory_id)
        if not job_details:
            raise HTTPException(
                status_code=404,
                detail=f"Job details not found for project {project_id}, trajectory {trajectory_id}",
            )

        # Remove raw_data from the response to reduce payload size
        job_details.raw_data = None

        return job_details

    # Include model, agent, and loop configuration routers
    router.include_router(settings_router, prefix="/settings", tags=["settings"])
    router.include_router(model_router, prefix="/models", tags=["models"])
    router.include_router(agent_router, prefix="/agents", tags=["agents"])
    router.include_router(swebench_router, prefix="/swebench", tags=["swebench"])
    router.include_router(trajectory_router, prefix="/trajectories", tags=["trajectories"])
    router.include_router(loop_router, prefix="/loop", tags=["loop"])

    api.mount("/api", router)

    FastAPIInstrumentor.instrument_app(api, excluded_urls="health")

    # Get allowed origins from environment variable or use defaults
    cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    if cors_origins_env:
        origins = cors_origins_env.split(",")
    else:
        origins = [
            "http://localhost:5173",  # Development frontend
            "http://127.0.0.1:5173",
            "http://localhost:8000",  # API server
            "http://127.0.0.1:8000",
        ]

    logger.info(f"CORS allowed origins: {origins}")

    api.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],  # Add this to expose headers to the client
    )

    return api

