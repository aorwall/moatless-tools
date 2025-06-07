"""Main API module for Moatless."""

import asyncio
import importlib.resources as pkg_resources
import json
import logging
import os
import secrets
import traceback
import uuid
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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from moatless.api.agents.api import router as agent_router
from moatless.api.loop.api import router as loop_router
from moatless.api.models.api import router as model_router
from moatless.api.runner.api import router as runner_router
from moatless.api.settings.api import router as settings_router
from moatless.api.swebench.api import router as swebench_router
from moatless.api.trajectories.api import router as trajectory_router

from moatless.telemetry import setup_telemetry
from moatless.utils.warnings import filter_external_warnings
from moatless.workspace import Workspace
from moatless.api.dependencies import (
    get_event_bus,
    get_runner,
    get_storage,
    cleanup_resources,
)

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Filter warnings from external dependencies
filter_external_warnings()

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Filter warnings from external dependencies
filter_external_warnings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler that logs errors with reference codes and returns structured error responses."""
    # Generate a unique reference code for this error
    error_ref = str(uuid.uuid4())[:8]

    # Extract request information for better logging
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    url = str(request.url)

    # Log the full exception with context
    logger.error(
        f"Unhandled exception [REF: {error_ref}] - "
        f"{method} {url} from {client_ip}: {type(exc).__name__}: {str(exc)}\n"
        f"Traceback:\n{traceback.format_exc()}"
    )

    # Return structured error response
    error_response = {
        "error": {
            "message": str(exc),
            "type": type(exc).__name__,
            "reference_code": error_ref,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    }

    # Return appropriate status code based on exception type
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        error_response["error"]["message"] = exc.detail
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        # For non-HTTP exceptions, provide a generic message to avoid exposing internal details
        error_response["error"]["message"] = "An internal server error occurred"
        error_response["error"]["internal_message"] = str(exc)  # Keep original for debugging

    return JSONResponse(status_code=status_code, content=error_response)


def create_api(workspace: Workspace | None = None) -> FastAPI:
    """Create and initialize the API with an optional workspace"""
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)

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
            # For health checks, we want to return a 503 status for failed dependencies
            error_ref = str(uuid.uuid4())[:8]
            logger.error(f"Health check failed [REF: {error_ref}]: {e}")

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"status": "failed", "error": str(e), "reference_code": error_ref},
            )

    # Include model, agent, and loop configuration routers
    router.include_router(runner_router, prefix="/runner", tags=["runner"])
    router.include_router(settings_router, prefix="/settings", tags=["settings"])
    router.include_router(model_router, prefix="/models", tags=["models"])
    router.include_router(agent_router, prefix="/agents", tags=["agents"])
    router.include_router(swebench_router, prefix="/swebench", tags=["swebench"])
    router.include_router(trajectory_router, prefix="/trajectories", tags=["trajectories"])
    router.include_router(loop_router, prefix="/loop", tags=["loop"])

    api.mount("/api", router)

    # Register global exception handler
    api.add_exception_handler(Exception, global_exception_handler)

    FastAPIInstrumentor.instrument_app(api, excluded_urls="health")

    # Get allowed origins from environment variable or use defaults
    cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    if cors_origins_env:
        origins = cors_origins_env.split(",")
    else:
        origins = ["http://localhost:5173", "http://localhost:5174"]

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


# Create the app instance for gunicorn
app = create_api()


def main():
    """Main entry point for running the API server."""
    import uvicorn

    # Load environment variables from .env file
    load_dotenv()

    # Get configuration from environment variables
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    log_level = os.environ.get("LOG_LEVEL", "info")

    logger.info(f"Starting Moatless API server on {host}:{port}")
    logger.info(f"Reload: {reload}, Log level: {log_level}")

    logging.basicConfig(level=logging.INFO)
    # Run the server
    uvicorn.run(
        "moatless.api.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
