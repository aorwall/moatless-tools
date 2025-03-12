"""Moatless API module."""

import os
import sys
import uvicorn
from pathlib import Path

from moatless.utils.warnings import filter_external_warnings


def run_api():
    """Run the Moatless API server."""
    # Filter warnings before importing any modules
    filter_external_warnings()

    # Import here to avoid circular imports
    from moatless.api.api import create_api
    from moatless.logging_config import setup_logging

    # Set up logging
    setup_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=Path(os.getcwd()) / "logs",
        service_name="moatless-api",
        environment=os.getenv("DEPLOYMENT_ENV", "development"),
        use_json=os.getenv("LOG_FORMAT", "").lower() == "json",
    )

    # Create and run the API
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))

    app = create_api()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
