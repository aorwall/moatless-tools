#!/usr/bin/env python3
"""
Command-line script to run a Moatless evaluation with Docker.

This script provides a simplified interface to run a Moatless evaluation
in a Docker container using the DockerRunner.
"""

import asyncio
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from moatless.runner.job_wrappers import run_evaluation_instance

# Add the project root to the path so we can import moatless
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from moatless.runner.docker_runner import DockerRunner
from moatless.evaluation.manager import EvaluationManager
from moatless.flow.manager import FlowManager
import moatless.settings as settings

logger = logging.getLogger(__name__)


async def setup_environment():
    """Create and initialize all required managers and services."""

    # Get the storage and event bus from settings
    storage = await settings.get_storage()
    eventbus = await settings.get_event_bus()

    # Create base directory (ensure we have a local directory even when using other storage)
    base_dir = Path(os.environ.get("MOATLESS_DIR", "./.moatless"))
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)

    # Get the flow manager from settings or create a new one if not initialized
    try:
        flow_manager = await settings.get_flow_manager()
    except RuntimeError:
        # If flow manager is not initialized, create a new one
        flow_manager = FlowManager(storage=storage, eventbus=eventbus)

    return storage, eventbus, flow_manager, base_dir


async def run_docker_evaluation(
    evaluation_name: str,
    instance_id: str,
    model_id: str,
    litellm_model_name: str,
    flow_id: str,
    use_local_source: bool = False,
):
    """Run an evaluation with Docker."""
    # Set MOATLESS_DIR environment variable if not set
    if "MOATLESS_DIR" not in os.environ:
        os.environ["MOATLESS_DIR"] = str(Path("./.moatless").absolute())
        logger.info(f"Setting MOATLESS_DIR to {os.environ['MOATLESS_DIR']}")

    # Validate that both model_id and litellm_model_name are not provided
    if model_id and litellm_model_name:
        raise ValueError("Cannot provide both model_id and litellm_model_name")

    storage, eventbus, flow_manager, base_dir = await setup_environment()

    if not evaluation_name:
        evaluation_name = f"docker-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Create the Docker runner
    moatless_source_dir = None
    if use_local_source:
        # Use the current directory as the source for moatless code
        moatless_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        logger.info(f"Using local moatless source code from: {moatless_source_dir}")

    runner = DockerRunner(moatless_source_dir=moatless_source_dir)

    # Create evaluation manager with the runner
    eval_manager = EvaluationManager(
        runner=runner,
        storage=storage,
        eventbus=eventbus,
        flow_manager=flow_manager,
    )

    try:
        # Set up volume mappings
        volume_mappings = []
        volume_mappings = [f"{os.environ['MOATLESS_DIR']}:/data/moatless"]

        runner.volume_mappings = volume_mappings

        evaluation = await eval_manager.create_evaluation(
            evaluation_name=evaluation_name,
            flow_id=flow_id,
            model_id=model_id,
            litellm_model_name=litellm_model_name,
            instance_ids=[instance_id],
            dataset_name="instance_ids",
        )

        # Start the job
        logger.info(f"Starting Docker job for {instance_id}")

        success = await runner.start_job(
            project_id=evaluation.evaluation_name,
            trajectory_id=instance_id,
            job_func=run_evaluation_instance,
        )

        if success:
            logger.info("Docker job started successfully!")
            container_name = runner._container_name(evaluation.evaluation_name, instance_id)

            logger.info(f"Starting to tail logs for container: {container_name}")

            # Start tailing the logs and wait for container to finish
            process = await asyncio.create_subprocess_exec(
                "docker",
                "logs",
                "-f",
                container_name,
                stdout=None,  # Use parent's stdout/stderr
                stderr=None,
            )

            # Wait for container to finish
            logger.info(f"Waiting for container {container_name} to complete...")

            while True:
                # Check if container is still running
                check_process = await asyncio.create_subprocess_exec(
                    "docker", "ps", "-q", "--filter", f"name={container_name}", stdout=asyncio.subprocess.PIPE
                )
                stdout, _ = await check_process.communicate()

                if not stdout.strip():
                    # Container is no longer running
                    logger.info(f"Container {container_name} has finished")
                    break

                # Wait before checking again
                await asyncio.sleep(5)

            # Make sure the log process is terminated
            try:
                process.terminate()
            except:
                pass

            await eval_manager.process_evaluation_results(evaluation.evaluation_name)

            return True
        else:
            logger.error("Failed to start Docker job")
            return False

    except Exception as e:
        logger.exception(f"Error running Docker evaluation: {e}")
        return False


def main():
    """Parse command-line arguments and run the evaluation."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run a Moatless evaluation with Docker")
    parser.add_argument(
        "--evaluation-name",
        help="Evaluation name",
    )
    parser.add_argument(
        "--instance-id",
        help="Instance ID",
        required=True,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-4o-mini-2024-07-18",
        help="Model ID to use (default: gpt-4o-mini-2024-07-18). For backward compatibility, same as --model-id.",
    )
    parser.add_argument("--model-id", help="Model ID to use (replaces entire completion model configuration)")
    parser.add_argument(
        "--litellm-model-name",
        help="LiteLLM model name to use (overrides only the model field of existing completion model)",
    )
    parser.add_argument("--flow", "-f", default="simple_coding", help="Flow ID to use (default: simple_coding)")
    parser.add_argument(
        "--use-local",
        "-l",
        action="store_true",
        default=True,
        help="Mount the local moatless source code to /opt/moatless in the container",
    )

    args = parser.parse_args()

    # Handle backward compatibility and validation
    model_id = args.model_id or args.model
    litellm_model_name = args.litellm_model_name

    if args.model_id and args.model != "gpt-4o-mini-2024-07-18":  # Default value check
        print("Error: Cannot specify both --model and --model-id")
        sys.exit(1)

    if args.model_id and args.litellm_model_name:
        print("Error: Cannot specify both --model-id and --litellm-model-name")
        sys.exit(1)

    if (
        args.model != "gpt-4o-mini-2024-07-18" and args.litellm_model_name
    ):  # Using non-default --model with --litellm-model-name
        print("Error: Cannot specify both --model and --litellm-model-name")
        sys.exit(1)

    logger.info("Loading environment variables")
    load_dotenv(".env.local")

    asyncio.run(
        run_docker_evaluation(
            evaluation_name=args.evaluation_name,
            instance_id=args.instance_id,
            model_id=model_id,
            litellm_model_name=litellm_model_name,
            flow_id=args.flow,
            use_local_source=args.use_local,
        )
    )


if __name__ == "__main__":
    main()
