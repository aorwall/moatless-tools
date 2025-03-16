#!/usr/bin/env python3
"""CLI script to run the Moatless API server."""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv

import uvicorn
from watchfiles import watch
from pathlib import Path

from moatless.api.api import create_api


def main():
    """Run the API server with command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Moatless API server")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with auto-reload"
    )
    parser.add_argument(
        "--debug-watch",
        action="store_true",
        help="Debug file watching to see which files trigger reloads"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    print(f"MOATLESS_DIR: {os.getenv('MOATLESS_DIR')}")
    print(f"MOATLESS_ACTIONS_PATH: {os.getenv('MOATLESS_ACTIONS_PATH')}")
    args = parser.parse_args()

    if args.dev:
        # Add more comprehensive exclusions
        uvicorn.run(
            "moatless.api.api:create_api",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=["moatless"],
            reload_include=[
                "*.py"
            ],
            factory=True,
            log_level="info",
        )
    else:
        logging.info("Starting API server")
        api = create_api()
        uvicorn.run(api, host=args.host, port=args.port)


if __name__ == "__main__":
    main()