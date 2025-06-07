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

from moatless.evaluation.schema import EvaluationStatus, InstanceStatus
from moatless.runner.job_wrappers import run_evaluation_instance
from moatless.runner.scheduler import SchedulerRunner

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
    dataset_split: str,
    model_id: str,
    litellm_model_name: str,
    flow_id: str,
    use_local_source: bool = False,
    num_parallel_jobs: int = 1,
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

    logger.info(
        f"Running evaluation {evaluation_name} with dataset split {dataset_split} and model {model_id or litellm_model_name} and flow {flow_id}"
    )

    # Create the Docker runner
    moatless_source_dir = None
    if use_local_source:
        # Use the current directory as the source for moatless code
        moatless_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        logger.info(f"Using local moatless source code from: {moatless_source_dir}")

    runner = SchedulerRunner(runner_impl=DockerRunner, max_total_jobs=num_parallel_jobs)

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
            dataset_name=dataset_split,
        )

        await eval_manager.start_evaluation(evaluation.evaluation_name)
        logger.info(f"Evaluation {evaluation.evaluation_name} started")

        while True:
            await asyncio.sleep(1)
            await eval_manager.process_evaluation_results(evaluation.evaluation_name)
            evaluation = await eval_manager.get_evaluation(evaluation.evaluation_name)
            # Count instances by status
            status_counts: dict[InstanceStatus, int] = {}
            for instance in evaluation.instances:
                status = instance.status
                status_counts[status] = status_counts.get(status, 0) + 1

            # Print status summary
            print(
                "Instance status summary:",
                ", ".join(f"{status.value}: {count}" for status, count in status_counts.items()),
            )

            if evaluation.status == EvaluationStatus.COMPLETED:
                break

        print(f"Evaluation {evaluation.evaluation_name} completed with status {evaluation.status}")

        # Calculate resolution statistics
        total_instances = len(evaluation.instances)
        resolved_instances = sum(1 for i in evaluation.instances if i.status == InstanceStatus.RESOLVED)
        resolution_percentage = (resolved_instances / total_instances * 100) if total_instances > 0 else 0

        print(
            f"Resolution rate: {resolution_percentage:.1f}% ({resolved_instances}/{total_instances} instances resolved)"
        )

        return True
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
        "--dataset-split",
        help="Dataset split to use",
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
        "--num-parallel-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run (default: 1)",
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
            dataset_split=args.dataset_split,
            model_id=model_id,
            litellm_model_name=litellm_model_name,
            flow_id=args.flow,
            use_local_source=False,  # This parameter doesn't exist in the current script
            num_parallel_jobs=args.num_parallel_jobs,
        )
    )


if __name__ == "__main__":
    main()
