import asyncio
import logging
from typing import Coroutine

logger = logging.getLogger(__name__)

class TaskScheduler:
    """
    A simple central scheduler that manages creation and tracking of asyncio Tasks.
    """
    def __init__(self):
        self.tasks = set()

    def schedule(self, coro: Coroutine, name: str) -> asyncio.Task:
        """
        Create a tracked asyncio Task for the given coroutine.
        """
        task = asyncio.create_task(coro, name=name)
        self.tasks.add(task)
        task.add_done_callback(self._cleanup)
        logger.info(f"Scheduled task: {name}")
        return task

    def _cleanup(self, task: asyncio.Task) -> None:
        """Remove the completed task from the registry."""
        self.tasks.discard(task)

    async def shutdown(self):
        """
        Graceful shutdown of all pending tasks, if needed.
        """
        remaining = [t for t in self.tasks if not t.done()]
        if remaining:
            logger.info(f"Waiting for {len(remaining)} remaining tasks to complete...")
            await asyncio.gather(*remaining, return_exceptions=True)
