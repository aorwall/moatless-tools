import asyncio
import logging
from typing import Dict, Tuple

from moatless.completion.log_handler import LogHandler
from moatless.flow.flow import AgenticFlow
import litellm

logger = logging.getLogger(__name__)


class AgenticRunner:
    _instance = None

    def __init__(self):
        self.active_runs: Dict[str, Tuple[AgenticFlow, asyncio.Task]] = {}
        litellm.callbacks = [LogHandler()]

    @classmethod
    def get_instance(cls) -> "AgenticRunner":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(self, agentic_system: AgenticFlow, message: str | None = None) -> str:
        project_id = agentic_system.project_id
        trajectory_id = agentic_system.trajectory_id

        async def trajectory_wrapper() -> dict:
            result = await agentic_system.run(message)
            # Clean up when done.
            self.active_runs.pop(trajectory_id, None)
            return result

        task_obj = asyncio.create_task(trajectory_wrapper())
        logger.debug(f"Starting run {trajectory_id}")
        self.active_runs[trajectory_id] = (agentic_system, task_obj)
        return trajectory_id

    async def get_run(self, trajectory_id: str, project_id: str | None = None) -> AgenticFlow | None:
        # TODO: Support project_id
        entry = self.active_runs.get(trajectory_id)
        if entry is None:
            return None
        return entry[0]

agentic_runner = AgenticRunner.get_instance()
