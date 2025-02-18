"""Moatless API initialization."""

import logging

import uvicorn
from dotenv import load_dotenv

from .api import create_api


def run_api(dev_mode: bool = False, host: str = "0.0.0.0", port: int = 8000):
    """Run the Moatless API server.

    Args:
        dev_mode: If True, enables auto-reload and debug features
        host: Host to bind to
        port: Port to listen on
    """
    # Load environment variables from .env file
    load_dotenv()

    if dev_mode:
        logging.info("Starting API server in development mode with auto-reload")
        uvicorn.run(
            "moatless.api.api:create_api",
            host=host,
            port=port,
            reload=True,
            reload_dirs=["moatless"],
            factory=True,
            log_level="info",
        )
    else:
        logging.info("Starting API server")
        api = create_api()
        uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    run_api()
