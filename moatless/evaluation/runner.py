import asyncio
import json
import logging
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict, List
import uuid
import sys
import threading
from contextlib import asynccontextmanager

import aiofiles

from functools import wraps

from moatless.benchmark.report import to_result
from moatless.benchmark.swebench.utils import create_repository_async, create_index_async
from moatless.evaluation.utils import get_moatless_instance
from moatless.evaluation.schema import (
    Evaluation,
    EvaluationInstance,
    EvaluationStatus,
    InstanceStatus,
    EvaluationEvent,
)
from moatless.events import event_bus
from moatless.flow.flow import AgenticFlow
from moatless.flow.manager import create_flow
from moatless.node import Node
from moatless.runtime.testbed import TestbedEnvironment
from moatless.utils.moatless import get_moatless_dir, get_moatless_trajectory_dir
from moatless.workspace import Workspace
from moatless.context_data import current_project_id
from moatless.telemetry import instrument, set_attribute

from moatless.evaluation.state_manager import StateManager
from moatless.evaluation.task_scheduler import TaskScheduler

from moatless.utils.block_detector import BlockingDetector

logger = logging.getLogger(__name__)

class EvaluationRunner:
    """
    Enhanced orchestrator for Evaluation lifecycle with robust error handling and state management.
    """

    def __init__(
        self,
        evaluation: Evaluation,
        repo_base_dir: Optional[str] = None,
        evaluations_dir: Optional[str] = None,
        num_concurrent_instances: int = 1,
        remove_repo_after_evaluation: bool = True,
    ):
        self.evaluation = evaluation
        self.state_manager = StateManager()
        
        # Separate schedulers for different types of tasks
        self.setup_scheduler = TaskScheduler()  # For repository setup
        self.run_scheduler = TaskScheduler()    # For running instances
        self.eval_scheduler = TaskScheduler()   # For evaluation tasks
        
        self.num_concurrent_instances = num_concurrent_instances
        self.remove_repo_after_evaluation = remove_repo_after_evaluation

        # Base directories
        self.repo_base_dir = repo_base_dir or "/tmp/moatless_repos"
        self.evaluations_dir = evaluations_dir or (get_moatless_dir() / "evals")
        self.evaluation_dir = Path(self.evaluations_dir) / self.evaluation.evaluation_name
        self.locks_dir = self.evaluation_dir / ".locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._state_changed = asyncio.Event()
        self._status_log_interval = 30.0
        self._last_status_log = time.time()
        self._instance_locks: Dict[str, asyncio.Lock] = {}
        
        # Monitoring
        self.blocking_detector = BlockingDetector(threshold_ms=1000)
        
        # Statistics
        self._stats = {
            "setup_attempts": 0,
            "run_attempts": 0,
            "eval_attempts": 0,
            "setup_failures": 0,
            "run_failures": 0,
            "eval_failures": 0,
        }

    @asynccontextmanager
    async def instance_lock(self, instance_id: str):
        """Ensure exclusive access to instance state modifications."""
        if instance_id not in self._instance_locks:
            self._instance_locks[instance_id] = asyncio.Lock()
        async with self._instance_locks[instance_id]:
            yield

    @instrument()
    async def run_evaluation(self) -> None:
        """
        Enhanced primary entry point with better error handling and monitoring.
        """
        set_attribute("evaluation_name", self.evaluation.evaluation_name)
        set_attribute("num_instances", len(self.evaluation.instances))
        set_attribute("num_concurrent", self.num_concurrent_instances)
        
        try:
            await self._initialize_evaluation()
            await self._run_main_loop()
            await self._finalize_evaluation()
            
        except Exception as exc:
            logger.exception("Critical error running evaluation")
            await self._handle_evaluation_error(exc)
            raise
            
        finally:
            await self._cleanup()

    async def _initialize_evaluation(self):
        """Initialize evaluation state and monitoring."""
        await self.blocking_detector.start()
        current_project_id.set(self.evaluation.evaluation_name)
        
        self.evaluation.status = EvaluationStatus.RUNNING
        self.evaluation.started_at = datetime.now(timezone.utc)
        await self._save_evaluation()
        await self._emit_event("evaluation_started")

    async def _run_main_loop(self):
        """Main scheduling loop with enhanced error handling."""
        while not self._all_instances_finished():
            try:
                await self._process_instances()
                
                # Create tasks for each scheduler's wait operation
                scheduler_tasks = [
                    asyncio.create_task(
                        self.setup_scheduler.wait_for_task_completion(timeout=5.0),
                        name="setup_scheduler_wait"
                    ),
                    asyncio.create_task(
                        self.run_scheduler.wait_for_task_completion(timeout=5.0),
                        name="run_scheduler_wait"
                    ),
                    asyncio.create_task(
                        self.eval_scheduler.wait_for_task_completion(timeout=5.0),
                        name="eval_scheduler_wait"
                    )
                ]
                
                try:
                    # Wait for any scheduler to complete a task
                    done, pending = await asyncio.wait(
                        scheduler_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Process completed tasks
                    for task in done:
                        try:
                            await task
                        except Exception as e:
                            logger.error(f"Error in scheduler task {task.get_name()}: {e}")
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            logger.error(f"Error cancelling task {task.get_name()}: {e}")
                            
                except Exception as e:
                    logger.error(f"Error waiting for scheduler tasks: {e}")
                    # Ensure all tasks are cancelled
                    for task in scheduler_tasks:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, Exception):
                                pass
                        
            except asyncio.CancelledError:
                logger.info("Main loop received cancellation request")
                raise
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Add a small delay to prevent tight error loops
                await asyncio.sleep(1)

    async def _process_instances(self) -> None:
        """
        Enhanced instance processing with better concurrency management.
        """
        # Process setup tasks
        setup_capacity = self.num_concurrent_instances - self.setup_scheduler.get_active_count()
        if setup_capacity > 0:
            pending_setup = [
                i for i in self.evaluation.instances 
                if i.status == InstanceStatus.PENDING
            ]
            for instance in pending_setup[:setup_capacity]:
                async with self.instance_lock(instance.instance_id):
                    await self._try_schedule_setup(instance)

        # Process run tasks
        run_capacity = self.num_concurrent_instances - self.run_scheduler.get_active_count()
        if run_capacity > 0:
            ready_to_run = [
                i for i in self.evaluation.instances
                if i.status == InstanceStatus.RUNNING
            ]
            for instance in ready_to_run[:run_capacity]:
                async with self.instance_lock(instance.instance_id):
                    await self._try_schedule_run(instance)

        # Process evaluation tasks
        eval_capacity = self.num_concurrent_instances - self.eval_scheduler.get_active_count()
        if eval_capacity > 0:
            ready_for_eval = [
                i for i in self.evaluation.instances
                if i.status == InstanceStatus.COMPLETED
            ]
            for instance in ready_for_eval[:eval_capacity]:
                async with self.instance_lock(instance.instance_id):
                    await self._try_schedule_evaluation(instance)

        self._print_status_with_counts()

    async def _try_schedule_setup(self, instance: EvaluationInstance) -> None:
        """Attempt to schedule setup with proper error handling."""
        self._stats["setup_attempts"] += 1
        try:
            moatless_instance = get_moatless_instance(instance_id=instance.instance_id)
            exists = await self.state_manager.check_repository_exists(moatless_instance, self.repo_base_dir)

            if not exists:
                can_setup = await self.state_manager.can_setup_repo(instance.instance_id)
                if can_setup:
                    await self.state_manager.register_repo_setup(instance.instance_id)
                    self.setup_scheduler.schedule(
                        self._setup_instance(instance.instance_id),
                        f"setup_{instance.instance_id}",
                        group="setup"
                    )
                else:
                    logger.info(f"Repo setup already in progress for {instance.instance_id}")
                return

            # If repo exists, transition to RUNNING
            await self._update_instance_state(
                instance,
                InstanceStatus.RUNNING,
                "instance_started"
            )

        except Exception as exc:
            self._stats["setup_failures"] += 1
            logger.error(f"Failed to schedule setup for {instance.instance_id}: {exc}")
            await self._update_instance_state(
                instance,
                InstanceStatus.ERROR,
                "instance_error",
                error=str(exc)
            )

    async def _try_schedule_run(self, instance: EvaluationInstance) -> None:
        """Attempt to schedule run with proper error handling."""
        self._stats["run_attempts"] += 1
        try:
            self.run_scheduler.schedule(
                self._run_instance(instance.instance_id),
                f"run_{instance.instance_id}",
                group="run"
            )
        except Exception as exc:
            self._stats["run_failures"] += 1
            logger.error(f"Failed to schedule run for {instance.instance_id}: {exc}")
            await self._update_instance_state(
                instance,
                InstanceStatus.ERROR,
                "instance_error",
                error=str(exc)
            )

    async def _try_schedule_evaluation(self, instance: EvaluationInstance) -> None:
        """Attempt to schedule evaluation with proper error handling."""
        self._stats["eval_attempts"] += 1
        try:
            await self._update_instance_state(
                instance,
                InstanceStatus.EVALUATING,
                "instance_evaluating"
            )
            self.eval_scheduler.schedule(
                self._evaluate_instance(instance.instance_id),
                f"evaluate_{instance.instance_id}",
                group="evaluate"
            )
        except Exception as exc:
            self._stats["eval_failures"] += 1
            logger.error(f"Failed to schedule evaluation for {instance.instance_id}: {exc}")
            await self._update_instance_state(
                instance,
                InstanceStatus.ERROR,
                "instance_error",
                error=str(exc)
            )

    async def _finalize_evaluation(self):
        """Finalize evaluation state and emit completion event."""
        self.evaluation.status = EvaluationStatus.COMPLETED
        self.evaluation.completed_at = datetime.now(timezone.utc)
        await self._save_evaluation()
        
        completed = sum(i.status == InstanceStatus.EVALUATED for i in self.evaluation.instances)
        errors = sum(i.status == InstanceStatus.ERROR for i in self.evaluation.instances)
        
        await self._emit_event("evaluation_completed", {
            "completed": completed,
            "errors": errors,
            "statistics": self._stats
        })

    async def _cleanup(self):
        """Comprehensive cleanup of resources."""
        try:
            if self.remove_repo_after_evaluation:
                self._cleanup_repositories()
                
            # Shutdown all schedulers
            await asyncio.gather(
                self.setup_scheduler.shutdown(),
                self.run_scheduler.shutdown(),
                self.eval_scheduler.shutdown()
            )
            
            # Stop monitoring
            await self.blocking_detector.stop()
            stats = self.blocking_detector.get_blocking_statistics()
            if stats:
                logger.info("Final blocking statistics:")
                for location, count in stats.items():
                    logger.info(f"  {location}: blocked {count} times")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def _handle_evaluation_error(self, exc: Exception):
        """Handle critical evaluation errors."""
        self.evaluation.status = EvaluationStatus.ERROR
        self.evaluation.error = str(exc)
        await self._save_evaluation()
        await self._emit_event("evaluation_error", {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "statistics": self._stats
        })

    @instrument()
    async def _setup_instance(self, instance_id: str) -> None:
        """
        Step (1): Setup the repository 
        """
        set_attribute("instance_id", instance_id)
        instance = self.evaluation.get_instance(instance_id)
        if not instance:
            return
        try:
            await self._update_instance_state(
                instance,
                InstanceStatus.SETTING_UP,
                "instance_setup_started"
            )
            
            moatless_instance = get_moatless_instance(instance_id=instance_id)
            await create_repository_async(moatless_instance, repo_base_dir=self.repo_base_dir)
            
            await self.state_manager.unregister_repo_setup(instance_id)
            await self._update_instance_state(
                instance,
                InstanceStatus.PENDING,
                "instance_setup_completed"
            )

        except Exception as exc:
            logger.exception(f"Setup failed for {instance_id}")
            await self.state_manager.unregister_repo_setup(instance_id)
            await self._update_instance_state(
                instance,
                InstanceStatus.ERROR,
                "instance_error",
                error=str(exc)
            )

    @instrument()
    async def _run_instance(self, instance_id: str) -> None:
        """
        Step (2): Run the "agentic flow" or whatever your main "execution" is.
        """
        set_attribute("instance_id", instance_id)
        instance = self.evaluation.get_instance(instance_id)
        if not instance:
            return
        try:
            moatless_instance = get_moatless_instance(instance_id=instance_id)
            # Ensure repository is present (safe to call again)
            await create_repository_async(moatless_instance, repo_base_dir=self.repo_base_dir)

            # Build your problem statement, create the agentic flow, run it
            problem_statement = f"Solve the following issue:\n{moatless_instance['problem_statement']}"
            flow = await self._create_agentic_flow(problem_statement, moatless_instance)

            node = await flow.run()
            logger.info(f"Flow completed for instance {instance_id}")

            if node.error:
                await self._update_instance_state(
                    instance,
                    InstanceStatus.ERROR,
                    "instance_error",
                    error=str(node.error)
                )
            else:
                await self._update_instance_state(
                    instance,
                    InstanceStatus.COMPLETED,
                    "instance_completed"
                )

        except Exception as exc:
            logger.exception(f"Error running instance {instance_id}")
            await self._update_instance_state(
                instance,
                InstanceStatus.ERROR,
                "instance_error",
                error=str(exc)
            )

    @instrument()
    async def _evaluate_instance(self, instance_id: str) -> None:
        """
        Step (3): Evaluate the completed instance (e.g., test the patch).
        """
        set_attribute("instance_id", instance_id)
        instance = self.evaluation.get_instance(instance_id)
        if not instance:
            return
        try:
            root_node = self._read_node(instance_id)
            await self._evaluate_nodes(instance, root_node)

            # Create a final benchmark result
            eval_result_path = os.path.join(
                get_moatless_trajectory_dir(instance_id, self.evaluation.evaluation_name),
                "eval_result.json"
            )
            if os.path.exists(eval_result_path):
                with open(eval_result_path, "r") as f:
                    eval_report = json.load(f)
            else:
                eval_report = None

            instance.benchmark_result = to_result(
                node=root_node,
                eval_report=eval_report,
                instance_id=instance_id
            )
            if instance.benchmark_result is not None:
                instance.resolved = instance.benchmark_result.resolved

            await self._update_instance_state(
                instance,
                InstanceStatus.EVALUATED,
                "instance_evaluated",
                event_data={
                    "resolved": instance.resolved
                }
            )

        except Exception as exc:
            logger.exception(f"Error evaluating {instance_id}")
            await self._update_instance_state(
                instance,
                InstanceStatus.ERROR,
                "instance_error",
                error=str(exc)
            )

    @instrument()
    async def _evaluate_nodes(self, instance: EvaluationInstance, root_node: Node) -> None:
        set_attribute("instance_id", instance.instance_id)

        try:
            leaf_nodes = root_node.get_leaf_nodes()
            eval_result_path = os.path.join(
                get_moatless_trajectory_dir(instance.instance_id, self.evaluation.evaluation_name),
                "eval_result.json"
            )
            eval_result: Optional[Dict[str, Any]] = None
            if os.path.exists(eval_result_path):
                try:
                    with open(eval_result_path) as f:
                        loaded = json.load(f)
                    if "node_results" in loaded:
                        eval_result = loaded
                    elif len(loaded) > 0:
                        eval_result = {
                            "node_results": loaded,
                            "status": "started",
                            "start_time": datetime.now(timezone.utc).isoformat(),
                        }
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse {eval_result_path}")
            if not eval_result:
                eval_result = {
                    "node_results": {},
                    "status": "started",
                    "start_time": datetime.now(timezone.utc).isoformat(),
                }
            unevaluated_nodes = [
                node for node in leaf_nodes if str(node.node_id) not in eval_result["node_results"]
            ]
            if not unevaluated_nodes:
                logger.info(f"All leaf nodes evaluated for instance {instance.instance_id}")
                return

            testbed_log_dir = os.path.join(
                get_moatless_trajectory_dir(instance.instance_id, self.evaluation.evaluation_name),
                "testbed_logs"
            )
            os.makedirs(testbed_log_dir, exist_ok=True)

            moatless_instance = get_moatless_instance(instance_id=instance.instance_id)
            repository = await create_repository_async(moatless_instance, repo_base_dir=self.repo_base_dir)

            runtime = TestbedEnvironment(
                repository=repository,
                instance_id=instance.instance_id,
                log_dir=testbed_log_dir,
                enable_cache=True,
            )
            for i, leaf_node in enumerate(unevaluated_nodes):
                logger.info(f"Evaluating node {leaf_node.node_id} ({i+1}/{len(unevaluated_nodes)})")
                patch = leaf_node.file_context.generate_git_patch(ignore_tests=True)
                if not patch or not patch.strip():
                    logger.info(f"No patch for node {leaf_node.node_id}; skipping.")
                    continue
                start_time = time.time()
                try:
                    result = await runtime.evaluate(patch=patch)
                    if result:
                        eval_result["node_results"][str(leaf_node.node_id)] = result.model_dump()
                        logger.info(f"Evaluated node {leaf_node.node_id} in {time.time()-start_time:.2f}s")
                except Exception:
                    logger.exception(f"Error evaluating node {leaf_node.node_id} for instance {instance.instance_id}")
                    eval_result["error"] = traceback.format_exc()
                finally:
                    eval_result["duration"] = time.time() - start_time
                    with open(eval_result_path, "w") as f:
                        json.dump(eval_result, f, indent=2)

        except Exception as e:
            logger.exception(f"Error evaluating nodes for instance {instance.instance_id}")
            await self._update_instance_state(
                instance,
                InstanceStatus.ERROR,
                "instance_error",
                error=str(e)
            )
            raise

    #
    # Helper Methods
    #

    async def _update_instance_state(
        self,
        instance: EvaluationInstance,
        new_status: InstanceStatus,
        event_type: str,
        event_data: Optional[Dict] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Helper method to update instance state, emit event, save evaluation and trigger state change.
        """
        try:
            await self.state_manager.set_status(instance, new_status, error)
            await self._save_evaluation()

            # Merge instance_id into event data
            event_data = event_data or {}
            event_data["instance_id"] = instance.instance_id
            if error:
                event_data["error"] = error
            
            await self._emit_event(event_type, event_data)
            self._state_changed.set()

            self._print_status_with_counts()
            
        except Exception as exc:
            logger.exception(f"Error updating state for instance {instance.instance_id}")
            # If we fail during a state update, try to set error state
            if new_status != InstanceStatus.ERROR:
                await self.state_manager.set_status(
                    instance, 
                    InstanceStatus.ERROR, 
                    f"Failed to update state: {str(exc)}"
                )
            raise

    async def _create_agentic_flow(self, problem_statement: str, moatless_instance: dict):
        """
        Create the agentic flow (simplified stub).
        """
        instance_id = moatless_instance["instance_id"]
        trajectory_dir = get_moatless_trajectory_dir(instance_id, self.evaluation.evaluation_name)
        testbed_log_dir = os.path.join(
            get_moatless_trajectory_dir(instance_id, self.evaluation.evaluation_name), "testbed_logs"
        )
        os.makedirs(testbed_log_dir, exist_ok=True)

        repository = await create_repository_async(moatless_instance, repo_base_dir=self.repo_base_dir)
        code_index = await create_index_async(moatless_instance, repository=repository)

        runtime = TestbedEnvironment(
            repository=repository,
            instance_id=instance_id,
            log_dir=testbed_log_dir,
            enable_cache=True,
        )
        workspace = Workspace(
            repository=repository,
            code_index=code_index,
            runtime=runtime,
            legacy_workspace=True
        )

        settings_path = trajectory_dir / "settings.json"
        if settings_path.exists():
            flow = AgenticFlow.from_dir(trajectory_dir, workspace=workspace)

            last_node = flow.root.get_all_nodes()[-1]
            if last_node.error:
                logger.info(f"Last node in flow {instance_id} has error, will remove it and try again")
                flow.reset_node(last_node.node_id)

        else:
            flow = create_flow(
            id=self.evaluation.flow_id,
            message=problem_statement,
            run_id=instance_id,
            persist_dir=get_moatless_trajectory_dir(instance_id, self.evaluation.evaluation_name),
            model_id=self.evaluation.model_id,
            workspace=workspace,
            metadata={"instance_id": instance_id},
        )
        return flow

    def _read_node(self, instance_id: str) -> Node:
        """
        Retrieve the root node from the trajectory file.
        """
        trajectory_dir = get_moatless_trajectory_dir(instance_id, self.evaluation.evaluation_name)
        path = trajectory_dir / "trajectory.json"
        return Node.from_file(path)

    def _get_instance_counts(self) -> dict:
        counts = {st.value: 0 for st in InstanceStatus}
        for inst in self.evaluation.instances:
            counts[inst.status] += 1
        return counts

    def _all_instances_finished(self) -> bool:
        total = len(self.evaluation.instances)
        finished = sum(i.status in (InstanceStatus.EVALUATED, InstanceStatus.ERROR) 
                       for i in self.evaluation.instances)
        return finished == total

    def _cleanup_repositories(self) -> None:
        """
        Removes each instance's repository folder if specified.
        """
        logger.info("Cleaning up repositories...")
        for instance in self.evaluation.instances:
            repo_path = Path(self.repo_base_dir) / instance.instance_id
            if repo_path.exists():
                logger.info(f"Removing repo for {instance.instance_id}")
                shutil.rmtree(repo_path, ignore_errors=True)

    async def _save_evaluation(self):
        """Save evaluation metadata atomically using a temporary file approach."""
        eval_path = self.evaluation_dir / "evaluation.json"
        
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp_path = eval_path.with_suffix(f'.json.{uuid.uuid4()}.tmp')
        
        try:
            async with aiofiles.open(tmp_path, "w") as f:
                json_data = json.dumps(self.evaluation.model_dump(), indent=2, default=str)
                await f.write(json_data)
                await f.flush()  # Ensure all data is written
                
            await aiofiles.os.rename(tmp_path, eval_path)
            
        except Exception as e:
            logger.error(f"Failed to save evaluation {self.evaluation.evaluation_name}: {e}")
        
        try:
            await aiofiles.os.remove(tmp_path)
        except:
            pass

    async def _emit_event(self, event_type: str, data: Any = None) -> None:
        """
        Publishes an event with the updated evaluation data.
        """
        logger.info(f"Emitting event: {event_type}")
        event = EvaluationEvent(
            evaluation_name=self.evaluation.evaluation_name,
            event_type=event_type,
            data=data,
        )
        await event_bus.publish(event)

    async def _periodic_status_logger(self) -> None:
        """
        Logs status every N seconds.
        """
        while True:
            await asyncio.sleep(self._status_log_interval)
            self._last_status_log = time.time()

    def _print_status_with_counts(self) -> None:
        """
        Pretty-print the current distribution of instance statuses.
        """
        counts = self._get_instance_counts()
        total = len(self.evaluation.instances)
        if total == 0:
            logger.info("No instances to run in this evaluation.")
            return

        # Compute percentages for each status
        percentages = {
            status: (count / total) * 100 if total else 0
            for status, count in counts.items()
        }

        max_status_length = max(len(status) for status in counts.keys())
        logger.info(f"\nEvaluation: {self.evaluation.evaluation_name}")
        logger.info("=" * 50)
        for status_enum in InstanceStatus:
            status = status_enum.value
            count = counts[status]
            percentage = percentages[status]
            bar_length = 20
            filled_length = int(bar_length * percentage / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            logger.info(
                f"{status.rjust(max_status_length)}: "
                f"{str(count).rjust(3)}/{total} "
                f"[{bar}] {percentage:5.1f}%"
            )
        logger.info("=" * 50)
