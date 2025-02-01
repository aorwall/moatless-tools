import asyncio
import logging
import time
from typing import Callable, Dict, List, Tuple

from moatless.agentic_system import AgenticSystem


logger = logging.getLogger(__name__)


class AgenticRunner:
    _instance = None

    def __init__(self):
        self.active_runs: Dict[str, Tuple[AgenticSystem, asyncio.Task]] = {}

    @classmethod
    def get_instance(cls) -> "AgenticRunner":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(self, agentic_system: AgenticSystem) -> str:
        run_id = agentic_system.run_id

        async def trajectory_wrapper() -> dict:
            result = await agentic_system.run()
            # Clean up when done.
            self.active_runs.pop(run_id, None)
            return result

        task_obj = asyncio.create_task(trajectory_wrapper())
        logger.info(f"Starting run {run_id}")
        self.active_runs[run_id] = (agentic_system, task_obj)
        return run_id

    async def get_run(self, run_id: str) -> AgenticSystem | None:
        start_time = time.time()
        entry = self.active_runs.get(run_id)
        if entry is None:
            return None
        logger.info(f"Run {run_id} took {time.time() - start_time} seconds to get run from runner")
        return entry[0]

agentic_runner = AgenticRunner.get_instance()