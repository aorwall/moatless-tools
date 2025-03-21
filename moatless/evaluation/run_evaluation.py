import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import litellm
from dotenv import load_dotenv
from opentelemetry import trace
from moatless.evaluation.run_instance import evaluate_instance
from testbeds.schema import SWEbenchInstance

from moatless.benchmark.swebench.utils import create_file_repository
from moatless.completion.log_handler import LogHandler
from moatless.context_data import current_project_id, current_trajectory_id
from moatless.evaluation.schema import EvaluationEvent
from moatless.evaluation.utils import get_swebench_instance
from moatless.flow.flow import AgenticFlow
from moatless.index.code_index import CodeIndex
from moatless.node import Node
from moatless.repository.file import FileRepository
from moatless.repository.git import GitRepository
from moatless.repository.repository import Repository
from moatless.runner.utils import cleanup_job_logging, setup_job_logging
from moatless.runtime.local import LocalEnvironment
from moatless.runtime.testbed import TestbedEnvironment
from moatless.telemetry import setup_telemetry
from moatless.utils.moatless import get_moatless_trajectory_dir
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.evaluation.runner")

load_dotenv()


async def run_evaluation_async(project_id: str, trajectory_id: str):
    logger.info(f"current_project_id: {current_project_id}, current_trajectory_id: {current_trajectory_id}")
    eval_result_path = "eval_result.json"

    try:
        instance_path = Path("/data/instance.json")
        with open(instance_path) as f:
            swebench_instance = SWEbenchInstance.model_validate(json.load(f))

        logger.info(f"Loaded instance: {swebench_instance.instance_id}")

        repo_path = "/testbed"
        logger.info(f"Using repo path: {repo_path}")

        repository = GitRepository(repo_path=repo_path)
        repository.reset()

        testbed_log_dir = Path("testbed_logs")
        testbed_log_dir.mkdir(parents=True, exist_ok=True)

        runtime = LocalEnvironment(
            repo_path=Path(repo_path),
            instance_id=trajectory_id,
            swebench_instance=swebench_instance,
            log_dir=str(testbed_log_dir),
            enable_cache=True,
        )
        await runtime.setup()

        patch = swebench_instance.patch

        result = await runtime.evaluate(patch=patch)
        logger.info(f"Evaluation result: {result.model_dump_json(indent=2)}")
    except Exception:
        logger.exception(f"Error evaluating instance {swebench_instance.instance_id}")
    finally:
        with open(eval_result_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2)


async def _emit_event(
    evaluation_name: str,
    instance_id: str,
    scope: str,
    event_type: str,
    data: Any = None,
):
    """Emit evaluation event."""
    event = EvaluationEvent(
        project_id=evaluation_name,
        trajectory_id=instance_id,
        scope=scope,
        event_type=event_type,
        data=data,
    )

    try:
        from moatless.settings import event_bus

        await event_bus.publish(event)
    except Exception as e:
        logger.error(f"Failed to publish event {event_type}: {e}")


if __name__ == "__main__":
    asyncio.run(run_evaluation_async("test", "test"))
