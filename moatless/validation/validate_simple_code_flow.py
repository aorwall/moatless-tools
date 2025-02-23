#!/usr/bin/env python3
"""Script to run simple flow integration tests."""

import logging
import argparse
from datetime import datetime
from pathlib import Path

from moatless.validation.base_code_flow_validation import BaseCodeFlowValidation
from moatless.completion.manager import get_model_config, get_all_configs

logger = logging.getLogger(__name__)


class SimpleFlowValidation(BaseCodeFlowValidation):
    def validate_result(self, node, loop) -> bool:
        """Validate that the loop finished successfully and produced a patch."""
        return node.action and node.action.name == "Finish" and node.file_context.has_patch()

    def create_runtime(self, repository, instance):
        """Create a simple runtime environment without testbed."""
        return None


def main():
    """Main entry point for simple flow validation CLI."""
    parser = argparse.ArgumentParser(description="Run simple flow integration tests for specified models")
    parser.add_argument(
        "--model",
        help="Specific model to test (e.g., 'claude-3-5-sonnet-20241022'). If not provided, all models will be tested.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel test runners (default: 1)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validator = SimpleFlowValidation()
    test_dir = validator.setup_test_directories(timestamp)
    logger.info(f"Running simple flow integration tests... Results will be saved in {test_dir}")

    if args.model:
        model_config = get_model_config(args.model)
        if not model_config:
            logger.error(f"Model '{args.model}' not found in supported models.")
            logger.info("Available models:")
            for model in get_all_configs().keys():
                logger.info(f"  - {model}")
            return
        models_to_test = [model_config]
    else:
        models_to_test = list(get_all_configs().values())

    validator.run_validation(models_to_test, test_dir, args.num_workers)


if __name__ == "__main__":
    main()
