#!/usr/bin/env python3
"""CLI script to run the Moatless API server."""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv, find_dotenv

import uvicorn

from moatless.api.api import create_api


def main():
    """Run the API server with command line arguments."""
    # Parse arguments first before any env loading
    parser = argparse.ArgumentParser(description="Run the Moatless API server")
    parser.add_argument("--dev", action="store_true", help="Run in development mode with auto-reload")
    parser.add_argument(
        "--debug-watch", action="store_true", help="Debug file watching to see which files trigger reloads"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--env", choices=["local", "azure"], default="local", help="Environment to use (local or azure)"
    )

    args = parser.parse_args()

    # Load environment variables based on specified environment
    env_file = f".env.{args.env}"
    if not os.path.exists(env_file):
        print(f"Error: Environment file '{env_file}' not found")
        sys.exit(1)

    # Use override=True to ensure our env file takes precedence
    load_dotenv(env_file, override=True)
    print(f"Loaded environment from {env_file}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    print(f"MOATLESS_DIR: {os.getenv('MOATLESS_DIR')}")
    print(f"MOATLESS_RUNNER: {os.getenv('MOATLESS_RUNNER')}")
    print(f"MOATLESS_STORAGE: {os.getenv('MOATLESS_STORAGE')}")

    if args.dev:
        # Add more comprehensive exclusions
        uvicorn.run(
            "moatless.api.api:create_api",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=["moatless"],
            reload_include=["*.py"],
            factory=True,
            log_level="info",
        )
    else:
        logging.info("Starting API server")
        api = create_api()
        uvicorn.run(api, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
