#!/usr/bin/env python3
"""Base script for running integration tests and generating result summaries."""

import argparse
import concurrent.futures
import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import litellm
from dotenv import load_dotenv

from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.model import Usage
from moatless.loop import AgenticLoop
from moatless.model_config import SUPPORTED_MODELS, MODEL_CONFIGS

load_dotenv()
litellm.drop_params = True

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(threadName)s] - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


class BaseCodeFlowValidation(ABC):
    def setup_test_directories(self, timestamp: str) -> Path:
        """Create and return test output directory."""
        base_dir = Path("test_results")
        test_dir = base_dir / f"integration_test_{timestamp}"
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    @abstractmethod
    def create_loop(self, model_config: dict, instance: dict) -> AgenticLoop:
        """Override this method in child classes to create specific loop types."""
        raise NotImplementedError

    @abstractmethod
    def validate_result(self, node, loop) -> bool:
        """Override this method in child classes to implement specific validation logic."""
        raise NotImplementedError

    def run_single_test(self, model_config: dict, test_dir: Path) -> dict:
        """Run a single model test and return results."""
        thread_name = threading.current_thread().name
        logger.info(f"Starting test for model: {model_config['model']}")

        try:
            model_dir = test_dir / model_config["model"].replace("/", "_")
            model_dir.mkdir(exist_ok=True)

            instance = get_moatless_instance("django__django-16527")
            loop = self.create_loop(model_config, instance)

            # Set persist path after loop creation
            persist_path = model_dir / "trajectory.json"
            loop.persist_path = str(persist_path)

            loop.maybe_persist()
            node = loop.run()
            logger.info(f"[{thread_name}] Completed run for {model_config['model']}")
            usage = loop.total_usage()
            logger.info(f"[{thread_name}] Usage for {model_config['model']}: {usage}")
            loop.maybe_persist()

            success = self.validate_result(node, loop)
            result = {
                "success": success,
                "finished": loop.is_finished(),
                "nodes": loop.root.get_all_nodes(),
                "usage": usage.model_dump() if usage else None,
            }

            return result

        except Exception as e:
            logger.exception(f"[{thread_name}] Error running test for {model_config['model']}")
            return {"success": False, "finished": False, "error": str(e)}

    def run_parallel_tests(self, models_to_test: List[dict], test_dir: Path, max_workers: int) -> Dict[str, dict]:
        """Run tests in parallel using ThreadPoolExecutor."""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="TestRunner"
        ) as executor:
            future_to_model = {
                executor.submit(self.run_single_test, model_config, test_dir): model_config
                for model_config in models_to_test
            }

            for future in concurrent.futures.as_completed(future_to_model):
                model_config = future_to_model[future]
                try:
                    results[model_config["model"]] = future.result()
                except Exception as e:
                    logger.exception(f"Test failed for {model_config['model']}")
                    results[model_config["model"]] = {"success": False, "finished": False, "error": str(e)}

        return results

    def generate_summary(self, results: Dict[str, dict], models_to_test: List[dict]) -> List[dict]:
        """Generate a summary of test results."""
        summary = []

        for model_config in models_to_test:
            model_name = model_config["model"]
            result = results.get(model_name, {})

            success = bool(result.get("finished", False))
            iterations = len(result.get("nodes", []))
            usage_data = result.get("usage", {})
            usage = Usage(**usage_data) if usage_data else None

            summary_entry = {
                "model": model_name,
                "success": success,
                "iterations": iterations,
                "cost": f"${usage.completion_cost:.4f}" if usage else "N/A",
                "prompt_tokens": usage.prompt_tokens if usage else "N/A",
                "completion_tokens": usage.completion_tokens if usage else "N/A",
                "cached_tokens": usage.cache_read_tokens if usage else "N/A",
            }
            summary.append(summary_entry)

        return summary

    def print_summary_table(self, summary: List[dict]):
        """Print a formatted table of the summary."""
        widths = {
            "model": max(len(entry["model"]) for entry in summary) + 2,
            "success": 8,
            "iterations": 11,
            "cost": 10,
            "prompt_tokens": 13,
            "completion_tokens": 17,
            "cached_tokens": 14,
        }

        header = (
            f"{'Model':<{widths['model']}} "
            f"{'Success':<{widths['success']}} "
            f"{'Iterations':<{widths['iterations']}} "
            f"{'Cost':<{widths['cost']}} "
            f"{'Prompt Tokens':<{widths['prompt_tokens']}} "
            f"{'Completion Tokens':<{widths['completion_tokens']}} "
            f"{'Cached Tokens':<{widths['cached_tokens']}}"
        )
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))

        for entry in summary:
            print(
                f"{entry['model']:<{widths['model']}} "
                f"{str(entry['success']):<{widths['success']}} "
                f"{entry['iterations']:<{widths['iterations']}} "
                f"{entry['cost']:<{widths['cost']}} "
                f"{str(entry['prompt_tokens']):<{widths['prompt_tokens']}} "
                f"{str(entry['completion_tokens']):<{widths['completion_tokens']}} "
                f"{str(entry['cached_tokens']):<{widths['cached_tokens']}}"
            )
        print("=" * len(header) + "\n")

    def save_summary(self, summary: List[dict], test_dir: Path):
        """Save the summary to a JSON file."""
        output_file = test_dir / "summary.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {output_file}")

    def run(self):
        parser = argparse.ArgumentParser(description="Run integration tests for specified models")
        parser.add_argument(
            "--model",
            help="Specific model to test (e.g., 'claude-3-5-sonnet-20241022'). If not provided, all models will be tested.",
        )
        parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel test runners (default: 1)")
        args = parser.parse_args()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = self.setup_test_directories(timestamp)
        logger.info(f"Running integration tests... Results will be saved in {test_dir}")

        if args.model:
            if args.model not in MODEL_CONFIGS:
                logger.error(f"Model '{args.model}' not found in supported models.")
                logger.info("Available models:")
                for model in MODEL_CONFIGS.keys():
                    logger.info(f"  - {model}")
                return
            models_to_test = [MODEL_CONFIGS[args.model]]
        else:
            models_to_test = SUPPORTED_MODELS

        if args.num_workers > 1:
            logger.info(f"Running tests in parallel with {args.num_workers} workers")
            results = self.run_parallel_tests(models_to_test, test_dir, args.num_workers)
        else:
            logger.info("Running tests sequentially")
            results = {}
            for model_config in models_to_test:
                results[model_config["model"]] = self.run_single_test(model_config, test_dir)

        summary = self.generate_summary(results, models_to_test)
        self.print_summary_table(summary)
        self.save_summary(summary, test_dir)
