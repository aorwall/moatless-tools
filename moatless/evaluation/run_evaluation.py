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
from moatless.settings import get_storage

from moatless.context_data import current_project_id, current_trajectory_id
from moatless.runtime.local import SweBenchTestbedEnvironment

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.evaluation.runner")


async def run_evaluation_async(
    project_id: str, trajectory_id: str | None = None, patch: str | None = None
) -> bool | None:
    logger.info(f"current_project_id: {current_project_id}, current_trajectory_id: {current_trajectory_id}")

    load_dotenv()

    storage = await get_storage()

    instance_path = Path("/data/instance.json")
    if not instance_path.exists():
        raise ValueError(f"Instance file not found at {instance_path}")

    swebench_instance = json.loads(instance_path.read_text())

    logger.info(f"Loaded instance: {swebench_instance['instance_id']}")

    if not trajectory_id:
        trajectory_id = swebench_instance["instance_id"]

    repo_path = os.getenv("REPO_DIR", "/testbed")
    logger.info(f"Using repo path: {repo_path}")

    runtime = SweBenchTestbedEnvironment(
        repo_path=Path(repo_path),
        swebench_instance=swebench_instance,
        storage=storage,
    )

    if not patch:
        patch = swebench_instance["patch"]

    result = await runtime.swebench_evaluate(project_id, trajectory_id, patch)

    if result and swebench_instance["instance_id"] in result:
        return result[swebench_instance["instance_id"]]["resolved"]
    else:
        logger.error(f"Failed to evaluate instance {trajectory_id}: {result}")
        return None


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    current_project_id.set("test")
    current_trajectory_id.set("test")
    resolved = loop.run_until_complete(run_evaluation_async("test"))
    print(f"Resolved: {resolved}")
