import asyncio
import logging
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

    def start(self, agentic_system: AgenticSystem) -> str:
        run_id = agentic_system.run_id

        async def trajectory_wrapper() -> dict:
            result = await asyncio.to_thread(agentic_system.run)
            # Clean up when done.
            self.active_runs.pop(run_id, None)
            return result

        task_obj = asyncio.create_task(trajectory_wrapper())
        logger.info(f"Starting run {run_id}")
        self.active_runs[run_id] = (agentic_system, task_obj)
        return run_id

    def get_run(self, run_id: str) -> AgenticSystem | None:
        entry = self.active_runs.get(run_id)
        if entry is None:
            return None
        return entry[0]

    def get_status(self, run_id: str) -> dict:
        entry = self.active_runs.get(run_id)
        if entry is None:
            return {"error": "Run not found or finished."}
        agentic_system, task_obj = entry
        status = agentic_system.get_status()
        if task_obj.done():
            status["result"] = task_obj.result()
            status["status"] = "finished"
        else:
            status["status"] = "running"
        return status

agentic_runner = AgenticRunner.get_instance()