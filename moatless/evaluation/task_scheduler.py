import asyncio
import logging
from typing import Any, Coroutine, Dict, Optional, Set
from opentelemetry import trace, context
from asyncio import Task, Future
from collections import defaultdict

logger = logging.getLogger(__name__)

class TaskScheduler:
    """
    A robust central scheduler that manages creation and tracking of asyncio Tasks.
    Features:
    - Per-task completion tracking
    - Graceful shutdown with timeout
    - Task grouping and dependencies
    - Detailed task status monitoring
    """
    def __init__(self):
        self.tasks: Set[Task] = set()
        self._shutting_down = False
        self._active_tasks: Dict[str, Task] = {}
        self._completion_events: Dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._task_results: Dict[str, Future] = {}
        self._task_groups: Dict[str, Set[str]] = defaultdict(set)
        
    def schedule(self, coro: Coroutine, name: str, group: Optional[str] = None) -> Optional[Task]:
        """
        Create a tracked asyncio Task with enhanced monitoring and grouping.
        
        Args:
            coro: Coroutine to schedule
            name: Unique task name
            group: Optional group name for task categorization
        """
        if self._shutting_down:
            logger.warning(f"Scheduler is shutting down, not scheduling task: {name}")
            return None

        # Cancel existing task if present
        self._cancel_existing_task(name)

        # Capture current trace context
        current_context = context.get_current()
        
        # Create completion tracking
        completion_event = self._completion_events[name]
        completion_event.clear()
        
        # Store result future
        result_future: Future = Future()
        self._task_results[name] = result_future
        
        # Add to group if specified
        if group:
            self._task_groups[group].add(name)

        async def wrapped_coro():
            token = context.attach(current_context)
            try:
                result = await coro
                result_future.set_result(result)
                return result
            except asyncio.CancelledError:
                logger.info(f"Task {name} was cancelled")
                result_future.cancel()
                raise
            except Exception as e:
                logger.error(f"Task {name} failed with error: {e}")
                result_future.set_exception(e)
                raise
            finally:
                context.detach(token)
                self._cleanup_task(name)
                completion_event.set()

        task = asyncio.create_task(wrapped_coro(), name=name)
        self.tasks.add(task)
        self._active_tasks[name] = task
        logger.info(f"Scheduled task: {name}")
        return task

    def _cancel_existing_task(self, name: str) -> None:
        """Safely cancel an existing task if present."""
        if name in self._active_tasks:
            old_task = self._active_tasks[name]
            if not old_task.done():
                logger.warning(f"Cancelling existing task {name} before scheduling new one")
                old_task.cancel()

    def _cleanup_task(self, name: str) -> None:
        """Remove completed task and clean up associated resources."""
        if name in self._active_tasks:
            task = self._active_tasks[name]
            self.tasks.discard(task)
            del self._active_tasks[name]
            
            # Clean up group membership
            for group in list(self._task_groups.keys()):
                self._task_groups[group].discard(name)
                if not self._task_groups[group]:
                    del self._task_groups[group]
                    
            logger.debug(f"Cleaned up task: {name}")

    def get_active_count(self, group: Optional[str] = None) -> int:
        """Get count of currently active tasks, optionally filtered by group."""
        if group:
            group_tasks = self._task_groups.get(group, set())
            return len([name for name in group_tasks 
                       if name in self._active_tasks and not self._active_tasks[name].done()])
        return len([t for t in self._active_tasks.values() if not t.done()])

    def is_task_active(self, name: str) -> bool:
        """Check if a task with given name is currently active."""
        return name in self._active_tasks and not self._active_tasks[name].done()

    async def wait_for_task_completion(self, task_name: Optional[str] = None, 
                                     group: Optional[str] = None,
                                     timeout: Optional[float] = None) -> bool:
        """
        Wait for specific task, group of tasks, or any task to complete.
        
        Args:
            task_name: Specific task to wait for
            group: Group of tasks to wait for
            timeout: Maximum time to wait
        """
        try:
            if task_name:
                if task_name not in self._completion_events:
                    return True
                event = self._completion_events[task_name]
                await asyncio.wait_for(event.wait(), timeout=timeout)
                return True
            
            if group:
                group_tasks = self._task_groups.get(group, set())
                events = [self._completion_events[name] for name in group_tasks]
                if not events:
                    return True
                # Create tasks for each event wait
                wait_tasks = [asyncio.create_task(event.wait()) for event in events]
                try:
                    done, pending = await asyncio.wait(
                        wait_tasks,
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    return bool(done)
                except Exception:
                    # Ensure all tasks are cancelled on error
                    for task in wait_tasks:
                        if not task.done():
                            task.cancel()
                    raise
            
            # Wait for any task
            if not self._completion_events:
                return True
                
            # Create tasks for each event wait
            wait_tasks = [asyncio.create_task(event.wait()) 
                         for event in self._completion_events.values()]
            try:
                done, pending = await asyncio.wait(
                    wait_tasks,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                return bool(done)
            except Exception:
                # Ensure all tasks are cancelled on error
                for task in wait_tasks:
                    if not task.done():
                        task.cancel()
                raise
            
        except asyncio.TimeoutError:
            return False

    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Graceful shutdown of all pending tasks with timeout and cleanup.
        
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
                # Force cleanup
                for task in remaining:
                    if not task.done():
                        task.cancel()
                self.tasks.clear()
                self._active_tasks.clear()
                self._completion_events.clear()
                self._task_results.clear()
                self._task_groups.clear()

    async def get_task_result(self, name: str, timeout: Optional[float] = None) -> Any:
        """Get the result of a completed task."""
        if name not in self._task_results:
            raise KeyError(f"No task found with name: {name}")
            
        future = self._task_results[name]
        if timeout is not None:
            return await asyncio.wait_for(future, timeout=timeout)
        return await future
