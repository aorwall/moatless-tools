import asyncio
import logging
import time
from typing import Callable, Dict, List, Tuple

from moatless.agentic_system import AgenticSystem
from moatless.completion.log_handler import LogHandler


logger = logging.getLogger(__name__)


class AgenticRunner:
    _instance = None

    def __init__(self):
        self.active_runs: Dict[str, Tuple[AgenticSystem, asyncio.Task]] = {}
        import litellm
        litellm.callbacks = [LogHandler()]

    @classmethod
    def get_instance(cls) -> "AgenticRunner":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(self, agentic_system: AgenticSystem, message: str | None = None) -> str:
        run_id = agentic_system.run_id

        async def trajectory_wrapper() -> dict:
            result = await agentic_system.run(message)
            # Clean up when done.
            self.active_runs.pop(run_id, None)
            return result

        task_obj = asyncio.create_task(trajectory_wrapper())
        logger.debug(f"Starting run {run_id}")
        self.active_runs[run_id] = (agentic_system, task_obj)
        return run_id

    async def get_run(self, run_id: str) -> AgenticSystem | None:
        entry = self.active_runs.get(run_id)
        if entry is None:
            return None
        return entry[0]

agentic_runner = AgenticRunner.get_instance()