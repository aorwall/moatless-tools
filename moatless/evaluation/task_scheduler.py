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
        self._shutting_down = False

    def schedule(self, coro: Coroutine, name: str) -> asyncio.Task:
        """
        Create a tracked asyncio Task for the given coroutine.
        """
        if self._shutting_down:
            logger.warning(f"Scheduler is shutting down, not scheduling task: {name}")
            return None
        task = asyncio.create_task(coro, name=name)
        self.tasks.add(task)
        task.add_done_callback(self._cleanup)
        logger.info(f"Scheduled task: {name}")
        return task

    def _cleanup(self, task: asyncio.Task) -> None:
        """Remove the completed task from the registry."""
        self.tasks.discard(task)

    async def shutdown(self, timeout: float = 5.0):
        """
        Graceful shutdown of all pending tasks with timeout.
        
        Args:
            timeout: Maximum time to wait for tasks to complete gracefully
        """
        self._shutting_down = True
        remaining = [t for t in self.tasks if not t.done()]
        if remaining:
            logger.info(f"Cancelling {len(remaining)} remaining tasks...")
            for task in remaining:
                task.cancel()
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*remaining, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Shutdown timed out after {timeout}s with {len(remaining)} tasks remaining")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
            finally:
                # Force cancel any remaining tasks
                for task in remaining:
                    if not task.done():
                        task.cancel()
                self.tasks.clear()
