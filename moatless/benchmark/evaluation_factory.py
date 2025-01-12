import json
import logging
import os
import random
from datetime import datetime
from typing import Optional, List

from moatless.benchmark.repository import EvaluationFileRepository
from moatless.benchmark.schema import Evaluation, EvaluationInstance, TreeSearchSettings
from moatless.completion.completion import LLMResponseFormat
from moatless.schema import MessageHistoryType

logger = logging.getLogger(__name__)


def create_evaluation(
    repository: EvaluationFileRepository,
    evaluation_name: str,
    settings: TreeSearchSettings,
    split: str = "lite",
    instance_ids: List[str] | None = None,
    exclude_instance_ids: List[str] | None = None,
    repos: List[str] | None = None,
    ignore_repos: List[str] | None = None,
    min_resolved: Optional[int] = None,
    max_resolved: Optional[int] = None,
) -> Evaluation:
    """Create a new evaluation with filtered instances."""
    # Load and filter instances based on split
    if split == "combo":
        # Load both lite and verified datasets
        lite_path = os.path.join(
            os.path.dirname(__file__), "swebench_lite_all_evaluations.json"
        )
        verified_path = os.path.join(
            os.path.dirname(__file__), "swebench_verified_all_evaluations.json"
        )

        with open(lite_path) as f:
            lite_instances = json.load(f)
        with open(verified_path) as f:
            verified_instances = json.load(f)

        # Get instance IDs that exist in both datasets
        lite_ids = {instance["instance_id"] for instance in lite_instances}
        verified_ids = {instance["instance_id"] for instance in verified_instances}
        common_ids = lite_ids.intersection(verified_ids)

        # Use instances from lite dataset that exist in both
        raw_instances = [
            instance
            for instance in lite_instances
            if instance["instance_id"] in common_ids
        ]
        logger.info(
            f"Found {len(raw_instances)} instances that exist in both lite and verified datasets"
        )
    else:
        file_path = os.path.join(
            os.path.dirname(__file__), f"swebench_{split}_all_evaluations.json"
        )
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path) as f:
            raw_instances = json.load(f)
        logger.info(f"Loaded {len(raw_instances)} instances from {file_path}")

    random.shuffle(raw_instances)

    # Apply all filters
    if instance_ids:
        raw_instances = [
            instance
            for instance in raw_instances
            if instance["instance_id"] in instance_ids
        ]
        logger.info(
            f"Running evaluation for {len(raw_instances)} instances filtered by instance_ids"
        )

    if exclude_instance_ids:
        raw_instances = [
            instance
            for instance in raw_instances
            if instance["instance_id"] not in exclude_instance_ids
        ]
        logger.info(
            f"Running evaluation for {len(raw_instances)} instances filtered by exclude_instance_ids"
        )

    if min_resolved is not None:
        raw_instances = [
            instance
            for instance in raw_instances
            if len(instance["resolved_by"]) >= min_resolved
            or (
                min_resolved == 1
                and instance.get("llm_monkeys", {}).get("resolved_rate", 0) > 0
            )
        ]
        logger.info(
            f"Running evaluation for {len(raw_instances)} instances filtered by min_resolved >= {min_resolved}"
        )

    if max_resolved is not None:
        raw_instances = [
            instance
            for instance in raw_instances
            if len(instance["resolved_by"]) <= max_resolved
        ]
        logger.info(
            f"Running evaluation for {len(raw_instances)} instances filtered by max_resolved <= {max_resolved}"
        )

    if repos:
        raw_instances = [
            instance for instance in raw_instances if instance["repo"] in repos
        ]
        logger.info(
            f"Running evaluation for {len(raw_instances)} instances filtered by repos"
        )

    if ignore_repos:
        raw_instances = [
            instance
            for instance in raw_instances
            if instance["repo"] not in ignore_repos
        ]
        if raw_instances:
            logger.info(
                f"Running evaluation for {len(raw_instances)} instances after filtering by ignore_repos"
            )

    # After all filters, apply random sampling if requested
    if split == "random":
        raw_instances = random.sample(raw_instances, min(50, len(raw_instances)))
        logger.info(
            f"Randomly selected {len(raw_instances)} instances from filtered dataset"
        )

    random.shuffle(raw_instances)

    # Create evaluation object
    evaluation = Evaluation(
        evaluations_dir=repository.evaluations_dir,
        evaluation_name=evaluation_name,
        settings=settings,
    )

    # Save evaluation first
    repository.save_evaluation(evaluation)

    # Create and save instances
    for instance in raw_instances:
        eval_instance = EvaluationInstance(instance_id=instance["instance_id"])
        repository.save_instance(evaluation_name, eval_instance)

    return evaluation


def create_evaluation_name(
    model: str,
    temperature: float = 0.0,
    date: str | None = None,
    max_iterations: int | None = None,
    max_expansions: int | None = None,
    response_format: LLMResponseFormat | None = None,
    message_history: MessageHistoryType | None = None,
    thoughts_in_action: bool | None = None,
) -> str:
    """Create a unique name for an evaluation."""
    if not date:
        date = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean model name
    model_name = model.replace("/", "_").replace("-", "_")

    temp_name = str(temperature).replace(".", "_")

    # Build name components
    components = [date, model_name, temp_name]

    if max_expansions is not None and max_expansions > 1:
        components.append(f"exp_{max_expansions}")

    if max_iterations is not None:
        components.append(f"n_{max_iterations}")

    if response_format:
        components.append(f"fmt_{response_format.value}")

    if message_history:
        components.append(f"hist_{message_history.value}")

    if thoughts_in_action:
        components.append("thoughts-in-action")

    return "_".join(components)
