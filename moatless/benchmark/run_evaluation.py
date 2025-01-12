import argparse
import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import litellm
from dotenv import load_dotenv
from litellm import InMemoryCache

import moatless.benchmark.evaluation_config as eval_config
from moatless.agent.settings import AgentSettings
from moatless.benchmark.evaluation_factory import (
    create_evaluation,
    create_evaluation_name,
)
from moatless.benchmark.evaluation_runner import EvaluationRunner
from moatless.benchmark.repository import EvaluationFileRepository
from moatless.benchmark.schema import (
    EvaluationDatasetSplit,
    InstanceStatus,
    TreeSearchSettings,
)
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.completion.log_handler import LogHandler
from moatless.schema import MessageHistoryType

# Load environment variables
load_dotenv()

# Automatically get all uppercase configs from evaluation_config
CONFIG_MAP = {
    name.lower().replace("_config", ""): getattr(eval_config, name)
    for name in dir(eval_config)
    if name.isupper()
    and name.endswith("_CONFIG")
    and isinstance(getattr(eval_config, name), dict)
}


def setup_loggers(logs_dir: str):
    """Setup console and file loggers"""

    # Setup console logger (only for this script)
    console_logger = logging.getLogger("scripts.run_evaluation_simple")
    console_logger.setLevel(logging.INFO)
    console_logger.propagate = False  # Don't propagate to root logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_logger.addHandler(console_handler)

    # Setup file logger (for all logs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Main log file (INFO and above)
    file_logger = logging.getLogger()  # Root logger
    file_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(
        os.path.join(logs_dir, f"evaluation_{timestamp}.log")
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    file_logger.addHandler(file_handler)

    # Error log file (ERROR and above)
    error_handler = logging.FileHandler(
        os.path.join(logs_dir, f"evaluation_errors_{timestamp}.log")
    )
    error_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n"
        )
    )
    error_handler.setLevel(logging.ERROR)
    file_logger.addHandler(error_handler)

    # Suppress other loggers from console output
    logging.getLogger("moatless").setLevel(logging.INFO)  # Set level for moatless logs
    for logger_name in logging.root.manager.loggerDict:
        if logger_name != "scripts.run_evaluation_simple":
            logger = logging.getLogger(logger_name)
            logger.propagate = True  # Allow propagation to root logger for file logging
            logger.addHandler(logging.NullHandler())  # Prevent output to console

    return console_logger, file_logger


def load_dataset_split(dataset_name: str) -> Optional[EvaluationDatasetSplit]:
    """Load a dataset split from the datasets directory."""
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "datasets",
        f"{dataset_name}_dataset.json",
    )
    if not os.path.exists(dataset_path):
        return None

    with open(dataset_path) as f:
        data = json.load(f)
        return EvaluationDatasetSplit(**data)


class SimpleEvaluationMonitor:
    def __init__(self, repository, evaluation, console_logger, file_logger):
        self.repository = repository
        self.evaluation = evaluation
        self.start_time = datetime.now()
        self.instances_data = {}
        self.total_cost = 0.0
        self.total_tokens = 0
        self.console = console_logger
        self.logger = file_logger

        # Load initial instances
        for instance in self.repository.list_instances(self.evaluation.evaluation_name):
            self.instances_data[instance.instance_id] = instance

        self._log_settings()

        self.console.info(f"Starting evaluation: {evaluation.evaluation_name}")
        self.console.info(f"Found {len(self.instances_data)} instances in evaluation")
        self.logger.info(
            f"[SimpleEvaluationMonitor] Starting evaluation: {evaluation.evaluation_name}"
        )
        self.logger.info(
            f"[SimpleEvaluationMonitor] Found {len(self.instances_data)} instances in evaluation"
        )

    def _log_settings(self):
        """Log evaluation configuration and settings"""
        eval_dir = os.path.join(
            self.repository.evaluations_dir, self.evaluation.evaluation_name
        )
        settings = self.evaluation.settings

        # Evaluation info
        info_lines = [
            "\nEvaluation Settings:",
            f"Directory: {eval_dir}",
            "\nModel Settings:",
            f"  Model: {settings.model.model}",
            f"  Temperature: {settings.model.temperature}",
            f"  Response Format: {settings.model.response_format.value if settings.model.response_format else 'default'}",
            f"  Thoughts in Action: {settings.model.thoughts_in_action}",
            "\nTree Search Settings:",
            f"  Max Iterations: {settings.max_iterations}",
            f"  Max Expansions: {settings.max_expansions}",
            f"  Max Cost: ${settings.max_cost}",
            "\nAgent Settings:",
            f"  Message History: {settings.agent_settings.message_history_type.value}",
            f"  System Prompt: {'custom' if settings.agent_settings.system_prompt else 'default'}",
            f"  Thoughts in Action: {settings.agent_settings.thoughts_in_action}",
        ]

        for line in info_lines:
            self.console.info(line)
            self.logger.info(line)

    def handle_event(self, event):
        """Handle evaluation events by logging them"""
        event_type = event.event_type
        data = event.data if event.data else {}

        instance_id = data.get("instance_id")

        if event_type == "evaluation_started":
            self.console.info("Evaluation started")
            self.logger.info("Evaluation started")
            return

        if not instance_id and event_type != "evaluation_started":
            self.console.warning(f"Instance ID not found in event data: {data}")
            self.logger.warning(f"Instance ID not found in event data: {data}")
            return

        # Load/reload instance from repository to get latest state
        instance = self.repository.load_instance(
            self.evaluation.evaluation_name, instance_id
        )
        if instance:
            self.instances_data[instance_id] = instance

            if event_type == "instance_started":
                self.console.info(f"Started instance: {instance_id}")
                self.logger.info(f"Started instance: {instance_id}")
            elif event_type == "instance_completed":
                status = "✓" if instance.resolved else "✗"
                self.console.info(f"Completed {instance_id} ({status})")
                self.logger.info(
                    f"Completed {instance_id} (resolved: {instance.resolved})"
                )
                self._log_instance_summary(instance)
            elif event_type == "instance_error":
                error_msg = f"Error in instance {instance_id}: {instance.error}"
                self.console.error(error_msg)
                self.logger.error(error_msg)
            elif event_type == "tree_progress":
                self.console.info(f"Tree progress: {instance_id}")
                self.logger.info(f"Tree progress: {instance_id}")
            else:
                self.console.info(f"Unknown event: {event_type}: {data}")
                self.logger.info(f"Unknown event: {event_type}: {data}")
        else:
            self.console.info(f"Unknown event: {event_type}: {data}")
            self.logger.info(f"Unknown event: {event_type}: {data}")

    def _log_instance_summary(self, instance):
        """Log summary for a completed instance"""
        cost = 0.0
        tokens = 0
        if instance.usage:
            cost = instance.usage.completion_cost
            tokens = (
                instance.usage.prompt_tokens
                + instance.usage.completion_tokens
                + instance.usage.cached_tokens
            )
            self.total_cost += cost
            self.total_tokens += tokens

        summary = (
            f"Instance {instance.instance_id} summary:\n"
            f"  - Resolved: {instance.resolved}\n"
            f"  - Duration: {instance.duration}s\n"
            f"  - Iterations: {instance.iterations}\n"
            f"  - Cost: ${cost:.2f}\n"
            f"  - Tokens: {tokens:,}"
        )

        self.console.info(summary)
        self.logger.info(summary)

        self.log_eval_summary()

    def log_eval_summary(self):
        """Log total instances, completed, errors and resolved instances"""
        total = len(self.instances_data)
        completed = sum(
            1
            for i in self.instances_data.values()
            if i.status == InstanceStatus.COMPLETED
        )
        errors = sum(
            1 for i in self.instances_data.values() if i.status == InstanceStatus.ERROR
        )
        resolved = sum(1 for i in self.instances_data.values() if i.resolved is True)
        summary = (
            f"Evaluation progress summary:\n"
            f"  - Total Instances: {total}\n"
            f"  - Completed: {completed}\n"
            f"  - Errors: {errors}\n"
            f"  - Resolved: {resolved}"
        )
        self.console.info(summary)
        self.logger.info(summary)

    def log_final_summary(self):
        """Log final evaluation summary"""
        duration = datetime.now() - self.start_time
        completed = sum(
            1
            for i in self.instances_data.values()
            if i.status == InstanceStatus.COMPLETED
        )
        errors = sum(
            1 for i in self.instances_data.values() if i.status == InstanceStatus.ERROR
        )
        resolved = sum(1 for i in self.instances_data.values() if i.resolved is True)
        total = len(self.instances_data)

        summary_lines = [
            "\nFinal Evaluation Summary:",
            f"Total Instances: {total}",
            f"Completed: {completed}",
            f"Errors: {errors}",
            f"Success Rate: {(resolved/total*100 if total > 0 else 0):.1f}%",
            f"Total Cost: ${self.total_cost:.2f}",
            f"Total Tokens: {self.total_tokens:,}",
            f"Total Duration: {duration}",
        ]

        for line in summary_lines:
            self.console.info(line)
            self.logger.info(line)


def print_config(config: dict, console_logger: logging.Logger):
    """Print configuration details"""
    config_sections = {
        "Model Settings": [
            ("Model", "model"),
            ("Response Format", "response_format"),
            ("Thoughts in Action", "thoughts_in_action"),
            ("API Key", "api_key"),
            ("Base URL", "base_url"),
        ],
        "Dataset Settings": [
            ("Split", "split"),
            ("Instance IDs", "instance_ids"),
        ],
        "Tree Search Settings": [
            ("Max Iterations", "max_iterations"),
            ("Max Expansions", "max_expansions"),
            ("Max Cost", "max_cost"),
        ],
        "Runner Settings": [
            ("Number of Workers", "num_workers"),
            ("Message History", "message_history"),
        ],
        "Evaluation Settings": [
            ("Evaluation Name", "evaluation_name"),
            ("Rerun Errors", "rerun_errors"),
        ],
        "Environment Settings": [
            ("Repository Dir", "REPO_DIR"),
            ("Index Store Dir", "INDEX_STORE_DIR"),
            ("Index Store URL", "INDEX_STORE_URL"),
            ("Moatless Dir", "MOATLESS_DIR"),
        ],
    }

    console_logger.info("\nConfiguration Settings:")
    console_logger.info("=" * 50)

    for section, settings in config_sections.items():
        console_logger.info(f"\n{section}:")
        console_logger.info("-" * 50)
        for label, key in settings:
            if section == "Environment Settings":
                value = os.getenv(key)
            else:
                value = config.get(key)

            if value is not None:
                if isinstance(value, list) and len(value) > 3:
                    value = f"{value[:3]} ... ({len(value)} items)"
                console_logger.info(f"{label:20}: {value}")

    console_logger.info("\n" + "=" * 50 + "\n")


async def run_evaluation(config: dict):
    """Run evaluation using provided configuration"""

    evaluations_dir = os.getenv("MOATLESS_DIR", "./evals")

    if config.get("evaluation_name"):
        evaluation_name = config["evaluation_name"]
    else:
        # Create evaluation name using the same logic as run_evaluation.py
        evaluation_name = create_evaluation_name(
            model=config["model"],
            temperature=config.get("temperature", 0.0),
            date=datetime.now().strftime("%Y%m%d"),
            max_iterations=config["max_iterations"],
            max_expansions=config["max_expansions"],
            response_format=LLMResponseFormat(config["response_format"])
            if config.get("response_format")
            else None,
            message_history=MessageHistoryType(config["message_history"])
            if config.get("message_history")
            else None,
            thoughts_in_action=config.get("thoughts_in_action", False),
        )

        base_evaluation_name = evaluation_name
        counter = 1
        while os.path.exists(os.path.join(evaluations_dir, evaluation_name)):
            evaluation_name = f"{base_evaluation_name}_{counter}"
            counter += 1

    evaluation_dir = os.path.join(evaluations_dir, evaluation_name)

    # Setup loggers
    logs_dir = os.path.join(evaluation_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    console_logger, file_logger = setup_loggers(logs_dir)

    # Setup prompt logs
    prompt_log_dir = os.path.join(evaluation_dir, "prompt_logs")
    os.makedirs(prompt_log_dir, exist_ok=True)
    prompt_log_callback = LogHandler(log_dir=prompt_log_dir)
    litellm.callbacks = [prompt_log_callback]

    print_config(config, console_logger)

    repository = EvaluationFileRepository(os.getenv("MOATLESS_DIR", "./evals"))

    if config.get("instance_ids"):
        instance_ids = config.get("instance_ids")
    else:
        dataset = load_dataset_split(config["split"])
        if dataset is None:
            console_logger.error(f"Dataset split '{config['split']}' not found")
            file_logger.error(f"Dataset split '{config['split']}' not found")
            sys.exit(1)
        instance_ids = dataset.instance_ids

    model_settings = CompletionModel(
        model=config["model"],
        temperature=0.0,
        max_tokens=3000,
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        response_format=LLMResponseFormat(config["response_format"])
        if config.get("response_format")
        else None,
        thoughts_in_action=config.get("thoughts_in_action", False),
    )

    agent_settings = AgentSettings(
        completion_model=model_settings,
        message_history_type=MessageHistoryType(config["message_history"])
        if config.get("message_history")
        else MessageHistoryType.MESSAGES,
        system_prompt=None,
        thoughts_in_action=config.get("thoughts_in_action", False),
    )

    tree_search_settings = TreeSearchSettings(
        max_iterations=config["max_iterations"],
        max_expansions=config["max_expansions"],
        max_cost=config["max_cost"],
        model=model_settings,
        agent_settings=agent_settings,
    )

    evaluation = create_evaluation(
        repository=repository,
        evaluation_name=evaluation_name,
        settings=tree_search_settings,
        instance_ids=instance_ids,
    )

    # Create monitor with both loggers
    monitor = SimpleEvaluationMonitor(
        repository, evaluation, console_logger, file_logger
    )

    # Create runner with event handler
    runner = EvaluationRunner(
        evaluation=evaluation,
        num_workers=config["num_workers"],
        use_testbed=True,
        rerun_errors=config.get("rerun_errors", False),
    )

    # Add event handler
    runner.add_event_handler(monitor.handle_event)

    # TODO: Just testing to disable cache...
    litellm.in_memory_llm_clients_cache = InMemoryCache(
        max_size_in_memory=-1, default_ttl=-1
    )

    try:
        # Run evaluation
        await loop.run_in_executor(
            ThreadPoolExecutor(),
            lambda: runner.run_evaluation(instance_ids=instance_ids),
        )
        # Log final summary
        monitor.log_final_summary()
    except Exception as e:
        error_msg = f"Fatal error in evaluation: {str(e)}"
        console_logger.error(error_msg)
        file_logger.error(error_msg, exc_info=True)
        sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run evaluation with specified configuration"
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIG_MAP.keys()),
        default="default",
        help="Configuration preset to use",
    )
    parser.add_argument("--split", help="Dataset split to use (overrides config)")
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        help="Specific instance IDs to evaluate (overrides split)",
    )
    parser.add_argument("--model", help="Model to use (overrides config)")
    parser.add_argument(
        "--num-workers", type=int, help="Number of workers (overrides config)"
    )
    parser.add_argument(
        "--max-iterations", type=int, help="Max iterations (overrides config)"
    )
    parser.add_argument(
        "--max-expansions", type=int, help="Max expansions (overrides config)"
    )
    parser.add_argument(
        "--max-cost", type=float, help="Max cost in dollars (overrides config)"
    )
    parser.add_argument(
        "--evaluation-name", help="Name for this evaluation run (overrides config)"
    )
    parser.add_argument(
        "--rerun-errors",
        action="store_true",
        help="Rerun instances that previously errored",
    )
    return parser.parse_args()


def get_config_from_args(args):
    """Get configuration based on command line arguments"""
    config = CONFIG_MAP[args.config].copy()

    # Override with command line arguments if provided
    if args.split:
        config["split"] = args.split
    if args.instance_ids:
        config["instance_ids"] = args.instance_ids
    if args.model:
        config["model"] = args.model
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.max_iterations is not None:
        config["max_iterations"] = args.max_iterations
    if args.max_expansions is not None:
        config["max_expansions"] = args.max_expansions
    if args.max_cost is not None:
        config["max_cost"] = args.max_cost
    if args.evaluation_name:
        config["evaluation_name"] = args.evaluation_name
    if args.rerun_errors:
        config["rerun_errors"] = True

    return config


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Get configuration
    config = get_config_from_args(args)

    # Set up asyncio loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(run_evaluation(config))
    finally:
        loop.close()
