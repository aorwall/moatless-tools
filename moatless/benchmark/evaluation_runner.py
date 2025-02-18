import json
import asyncio
import json
import logging
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import filelock

from moatless.benchmark.report import to_result
from moatless.benchmark.schema import (
    Evaluation,
    EvaluationInstance,
    EvaluationStatus,
    InstanceStatus,
    EvaluationEvent,
)
from moatless.benchmark.swebench import create_index
from moatless.benchmark.swebench.utils import create_repository_async
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion import BaseCompletionModel
from moatless.config.agent_config import get_agent
from moatless.config.model_config import create_completion_model
from moatless.context_data import current_evaluation_name
from moatless.flow.manager import create_flow
from moatless.events import event_bus
from moatless.flow import AgenticFlow
from moatless.node import Node
from moatless.runtime.testbed import TestbedEnvironment
from moatless.utils.moatless import get_moatless_dir, get_moatless_trajectory_dir
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that handles datetime objects by converting them to ISO format strings.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class EvaluationRunner:
    """
    Orchestrates the lifecycle of an Evaluation: setting up each EvaluationInstance, running it,
    and performing the testbed evaluation of results.

    Attributes:
        evaluation: The Evaluation configuration (list of instances, tree search settings, etc.).
        repo_base_dir: Base directory to store local clones of repos under test.
        evaluations_dir: Where to store evaluation metadata and output JSON.
        num_concurrent_instances: Maximum number of instances that can run (setup or execution) in parallel.
        remove_repo_after_evaluation: If True, the local repository clones are removed after evaluation completes.
    """

    def __init__(
        self,
        evaluation: Evaluation,
        repo_base_dir: Optional[str] = None,
        evaluations_dir: Optional[str] = None,
        num_concurrent_instances: int = 1,
        remove_repo_after_evaluation: bool = True,
    ) -> None:
        self.evaluation = evaluation
        self.num_concurrent_instances = num_concurrent_instances
        self.remove_repo_after_evaluation = remove_repo_after_evaluation

        # Base directory for storing this evaluation's metadata
        if evaluations_dir:
            self.evaluations_dir = evaluations_dir
        else:
            self.evaluations_dir = get_moatless_dir() / "evals"

        self.evaluation_dir = Path(self.evaluations_dir) / self.evaluation.evaluation_name
        self.locks_dir = self.evaluation_dir / ".locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)

        # Base directory for storing repos under test
        self.repo_base_dir = repo_base_dir or os.getenv("MOATLESS_REPO_DIR", "/tmp/repos")

        # Async concurrency controls
        self._state_lock = asyncio.Lock()
        self._state_changed = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._active_runs: Dict[str, Tuple[AgenticFlow, asyncio.Task]] = {}
        self._last_status_log = time.time()
        self._status_log_interval = 30.0  # seconds

        # Holds a prepared agentic system for each instance
        self._agentic_systems: Dict[str, AgenticFlow] = {}

    async def run_evaluation(self) -> None:
        """
        Start and manage the evaluation process, including looping until all instances are complete.
        """
        status_monitor = None
        try:
            current_evaluation_name.set(self.evaluation.evaluation_name)
            logger.info(f"Running evaluation: {self.evaluation.evaluation_name}")

            self.evaluation.status = EvaluationStatus.RUNNING
            self.evaluation.started_at = datetime.now(timezone.utc)
            self._save_evaluation(self.evaluation)
            await self.emit_event("evaluation_started")

            # Start a background task to monitor and log statuses at intervals
            status_monitor = self._create_tracked_task(self._monitor_status(), "status_monitor")

            # Main loop: process states until the evaluation completes or errors
            while True:
                await self._process_instances()
                if await self._check_evaluation_completion():
                    break
                # If no changes, wait briefly for the next iteration
                try:
                    await asyncio.wait_for(self._state_changed.wait(), timeout=5.0)
                    self._state_changed.clear()
                except asyncio.TimeoutError:
                    pass

            # Cancel the status monitor task since we're done
            if status_monitor and not status_monitor.done():
                status_monitor.cancel()
                try:
                    await status_monitor
                except asyncio.CancelledError:
                    pass

            # Wait for any final tasks
            remaining_tasks = [t for t in self._tasks if t is not status_monitor]
            if remaining_tasks:
                logger.info(f"Waiting for {len(remaining_tasks)} remaining tasks to complete...")
                await asyncio.gather(*remaining_tasks, return_exceptions=True)

        except Exception as e:
            logger.exception("Error running evaluation.")
            self.evaluation.status = EvaluationStatus.ERROR
            self.evaluation.error = str(e)
            self._save_evaluation(self.evaluation)
            await self.emit_event("evaluation_error", {"error": str(e)})
            raise
        finally:
            # Make sure to cancel the status monitor if it's still running
            if status_monitor and not status_monitor.done():
                status_monitor.cancel()
                try:
                    await status_monitor
                except asyncio.CancelledError:
                    pass

            # Clean up repositories if requested
            if self.remove_repo_after_evaluation:
                self._cleanup_repositories()

    async def _process_instances(self) -> None:
        """
        Looks at the current state of all instances and transitions them if capacity is available.
        """
        current_states = await self._get_instance_states()
        state_counts = {status.value: 0 for status in InstanceStatus}
        for st in current_states.values():
            state_counts[st] += 1

        # If there's capacity, start new setups
        if state_counts[InstanceStatus.SETTING_UP] < self.num_concurrent_instances:
            await self._start_new_setups(state_counts[InstanceStatus.SETTING_UP])

        # If there's capacity, start new runs
        if state_counts[InstanceStatus.RUNNING] < self.num_concurrent_instances:
            await self._start_new_runs(state_counts[InstanceStatus.RUNNING])

        # If there's capacity, start new evaluations
        if state_counts[InstanceStatus.EVALUATING] < self.num_concurrent_instances:
            await self._start_new_evaluations(state_counts[InstanceStatus.EVALUATING])

    async def _start_new_setups(self, current_setups: int) -> None:
        """
        Launches setup tasks for CREATED instances if there's capacity.
        Ensures only one instance per repository is being set up at a time.
        """
        available_slots = self.num_concurrent_instances - current_setups
        if available_slots <= 0:
            return

        # Get all instances that need setup
        created_instances = [
            i for i in self.evaluation.instances 
            if i.status == InstanceStatus.CREATED
        ]

        # Track which repos are currently being processed
        active_instances = {
            i.instance_id: get_moatless_instance(i.instance_id)["repo"]
            for i in self.evaluation.instances
            if i.status in [InstanceStatus.SETTING_UP]
        }
        active_repos = set(active_instances.values())

        # Group instances by repo to process them efficiently
        instances_by_repo = {}
        for instance in created_instances:
            moatless_instance = get_moatless_instance(instance_id=instance.instance_id)
            repo = moatless_instance["repo"]
            if repo not in instances_by_repo:
                instances_by_repo[repo] = []
            instances_by_repo[repo].append(instance)

        # Select instances to setup while respecting limits
        instances_to_setup = []
        for repo, repo_instances in instances_by_repo.items():
            if len(instances_to_setup) >= available_slots:
                break
            
            if repo not in active_repos:
                # Take one instance from this repo
                instance = repo_instances[0]
                instances_to_setup.append(instance)
                active_repos.add(repo)

        # Start setup for selected instances
        for instance in instances_to_setup:
            try:
                logger.info(f"Starting setup for instance {instance.instance_id}")
                await self._update_instance_state(instance, InstanceStatus.SETTING_UP)
                
                # Create a unique setup task for this instance
                setup_task = self._create_tracked_task(
                    self.setup_instance(instance.instance_id),
                    f"setup_{instance.instance_id}"
                )
                
                # Add error handling for the setup task
                def handle_setup_completion(task):
                    try:
                        task.result()  # Will raise exception if setup failed
                    except Exception as e:
                        logger.error(f"Setup failed for instance {instance.instance_id}: {str(e)}")
                        # Cleanup will be handled by the task's error handler
                
                setup_task.add_done_callback(handle_setup_completion)
                
            except Exception as e:
                logger.error(f"Failed to start setup for instance {instance.instance_id}: {str(e)}")
                await self._update_instance_state(
                    instance, 
                    InstanceStatus.ERROR,
                    f"Failed to start setup: {str(e)}"
                )
                raise

    async def _start_new_runs(self, current_runs: int) -> None:
        """
        Launches run tasks for PENDING instances if there's capacity.
        """
        available_slots = self.num_concurrent_instances - current_runs
        if available_slots <= 0:
            return

        pending_instances = [
            i for i in self.evaluation.instances if i.status == InstanceStatus.PENDING
        ][:available_slots]

        for instance in pending_instances:
            logger.info(f"Starting run for instance {instance.instance_id}")
            instance.started_at = datetime.now(timezone.utc)
            await self._update_instance_state(instance, InstanceStatus.RUNNING)
            self._create_tracked_task(
                self.create_and_run_instance(instance.instance_id),
                instance.instance_id
            )

    async def _start_new_evaluations(self, current_evaluations: int) -> None:
        """
        Launches evaluation tasks for COMPLETED instances if there's capacity.
        """
        available_slots = self.num_concurrent_instances - current_evaluations
        if available_slots <= 0:
            return

        completed_instances = [
            i for i in self.evaluation.instances if i.status == InstanceStatus.COMPLETED
        ][:available_slots]

        for instance in completed_instances:
            logger.info(f"Starting evaluation for instance {instance.instance_id}")
            await self._update_instance_state(instance, InstanceStatus.EVALUATING)
            self._create_tracked_task(
                self._evaluate_instance(instance),
                instance.instance_id
            )

    async def _check_evaluation_completion(self) -> bool:
        """
        Checks whether all instances are finished (EVALUATED or ERROR).
        If so, marks the evaluation as COMPLETED and emits completion events.
        """
        if self.evaluation.status != EvaluationStatus.RUNNING:
            return True

        total_instances = len(self.evaluation.instances)
        finished_instances = sum(
            1 for i in self.evaluation.instances
            if i.status in [InstanceStatus.EVALUATED, InstanceStatus.ERROR]
        )

        # If all finished, finalize the evaluation
        if finished_instances == total_instances:
            logger.info(f"All instances finished for evaluation {self.evaluation.evaluation_name}")
            
            # Print final status before completing
            counts = await self._get_instance_counts()
            logger.info("Final evaluation status:")
            self._print_status_with_counts(counts)
            
            self.evaluation.status = EvaluationStatus.COMPLETED
            self.evaluation.completed_at = datetime.now(timezone.utc)
            self._save_evaluation(self.evaluation)

            # Summaries for the completion event
            await self.emit_event("evaluation_completed", {
                "total_completed": sum(1 for s in counts.values() if s == InstanceStatus.EVALUATED),
                "total_errors": sum(1 for s in counts.values() if s == InstanceStatus.ERROR),
            })
            return True

        return False

    def _cleanup_repositories(self) -> None:
        """
        Deletes repository directories for each instance if remove_repo_after_evaluation is True.
        """
        logger.info("Cleaning up repositories...")
        for instance in self.evaluation.instances:
            repo_path = Path(self.repo_base_dir) / instance.instance_id
            if repo_path.exists():
                logger.info(f"Removing repository for instance {instance.instance_id} at {repo_path}")
                shutil.rmtree(repo_path, ignore_errors=True)

    async def _get_instance_states(self) -> Dict[str, InstanceStatus]:
        """
        Thread-safe snapshot of instance_id -> instance status.
        """
        async with self._state_lock:
            return {
                instance.instance_id: instance.status
                for instance in self.evaluation.instances
            }

    async def _update_instance_state(
        self,
        instance: EvaluationInstance,
        new_status: InstanceStatus,
        error: Optional[str] = None
    ) -> None:
        """
        Updates the instance state in a thread-safe manner and persists the evaluation object.
        """
        async with self._state_lock:
            old_status = instance.status
            if old_status != new_status:
                instance.status = new_status
                if error:
                    instance.error = error
                instance.completed_at = datetime.now(timezone.utc) if new_status in [InstanceStatus.EVALUATED, InstanceStatus.ERROR] else None
                logger.info(
                    f"Instance {instance.instance_id} transitioned: {old_status} -> {new_status}"
                )
                self._save_evaluation(self.evaluation)
                self._state_changed.set()
                self._last_status_log = time.time()  # Reset status log timer

                # Immediately log after a transition
                counts = await self._get_instance_counts()
                logger.info(counts)
                self._print_status_with_counts(counts)

    async def setup_instance(self, instance_id: str) -> None:
        """
        Sets up the environment for a single EvaluationInstance:
        - Prepares a local repository clone
        - Creates an AgenticSystem (but does not start it)
        - Moves the instance to PENDING on success
        """
        logger.info(f"Setting up instance {instance_id}")
        instance = self.evaluation.get_instance(instance_id)
        if not instance:
            raise ValueError(f"Instance {instance_id} not found in evaluation")

        try:
            await self.emit_event("instance_setup_started", {"instance_id": instance_id})

            moatless_instance = get_moatless_instance(instance_id=instance_id)
            problem_statement = (
                f"<task>\nSolve the following reported issue in {moatless_instance['repo']} "
                f"repository:\n\n{moatless_instance['problem_statement']}\n</task>"
            )

            # Create the agent system but don't run it yet
            agentic_system = await self.create_agentic_flow(
                problem_statement=problem_statement,
                moatless_instance=moatless_instance,
            )
            self._store_agentic_system(instance_id, agentic_system)

            # Move to PENDING after setup completes
            await self._update_instance_state(instance, InstanceStatus.PENDING)
            await self.emit_event("instance_setup_completed", {"instance_id": instance_id})

        except Exception as e:
            logger.exception(f"Error setting up instance {instance_id}")
            await self._update_instance_state(instance, InstanceStatus.ERROR, str(e))
            await self.emit_event("instance_error", {"instance_id": instance_id, "error": str(e)})
            raise

    async def create_and_run_instance(self, instance_id: str) -> None:
        """
        Retrieves the pre-built AgenticSystem from setup_instance and starts it.
        Moves the instance through RUNNING -> COMPLETED states.
        """
        instance = self.evaluation.get_instance(instance_id)
        if not instance:
            raise ValueError(f"Instance {instance_id} not found in evaluation")

        logger.info(f"Starting evaluation flow for instance {instance_id}")
        await self.emit_event("instance_started", {"instance_id": instance_id})

        try:
            agentic_system = self._agentic_systems.get(instance_id)
            if not agentic_system:
                raise ValueError(f"No prepared agentic system found for instance {instance_id}")

            # Run the flow and wait for completion
            result = await agentic_system.run()
            
            # When agent completes successfully, transition to COMPLETED
            logger.info(f"Agent completed for instance {instance_id}")
            instance.completed_at = datetime.now(timezone.utc)
            await self._update_instance_state(instance, InstanceStatus.COMPLETED)
            await self.emit_event("instance_completed", {"instance_id": instance_id})

        except Exception as e:
            logger.exception(f"Error running instance {instance_id}")
            await self._update_instance_state(instance, InstanceStatus.ERROR, str(e))
            await self.emit_event("instance_error", {"instance_id": instance_id, "error": str(e)})
            raise

    async def create_agentic_flow(
        self,
        problem_statement: str,
        moatless_instance: Dict[str, Any],
    ) -> AgenticFlow:
        """
        Creates an AgenticSystem (SearchTree or simple AgenticLoop) for a particular instance.
        """
        logger.info(f"Creating agentic flow for instance {moatless_instance['instance_id']}")
        repository = await create_repository_async(
            moatless_instance,
            repo_base_dir=self.repo_base_dir
        )
        logger.info(f"Created repository for instance {moatless_instance['instance_id']}")
        code_index = create_index(moatless_instance, repository=repository)

        instance_id = moatless_instance["instance_id"]
        completion_model: BaseCompletionModel = create_completion_model(
            self.evaluation.model_id
        )
        completion_model.metadata = {"instance_id": instance_id}

        testbed_log_dir = os.path.join(
            get_moatless_trajectory_dir(instance_id, self.evaluation.evaluation_name),
            "testbed_logs"
        )
        os.makedirs(testbed_log_dir, exist_ok=True)

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

        persist_dir = get_moatless_trajectory_dir(instance_id, self.evaluation.evaluation_name)
        logger.info(f"Persist dir for instance {instance_id}: {persist_dir}")

        return create_flow(
            id=self.evaluation.flow_id,
            message=problem_statement,
            run_id=instance_id,
            persist_dir=persist_dir,
            model_id=self.evaluation.model_id,
            workspace=workspace,
            metadata={"instance_id": instance_id},
        )

    def _store_agentic_system(self, instance_id: str, system: AgenticFlow) -> None:
        """
        Stores the prepared agentic system in memory. This is used after setup_instance,
        so that a subsequent call to create_and_run_instance can retrieve it.
        """
        self._agentic_systems[instance_id] = system

    async def evaluate_nodes(self, instance_id: str, root_node: Node) -> None:
        """
        Evaluates each leaf node in the search tree with the testbed, preserving results in eval_result.json.
        """
        try:
            leaf_nodes = root_node.get_leaf_nodes()
            eval_result_path = os.path.join(
                get_moatless_trajectory_dir(instance_id, self.evaluation.evaluation_name),
                "eval_result.json"
            )

            # Attempt to load any existing evaluation result
            eval_result: Optional[Dict[str, Any]] = None
            if os.path.exists(eval_result_path):
                try:
                    with open(eval_result_path) as f:
                        loaded = json.load(f)
                        # Basic format check
                        if "node_results" in loaded:
                            eval_result = loaded
                            logger.info(f"Found existing node_results in {eval_result_path}")
                        elif len(loaded) > 0:
                            logger.info("Eval result missing 'node_results' key, normalizing format.")
                            eval_result = {
                                "node_results": loaded,
                                "status": "started",
                                "start_time": datetime.now(timezone.utc).isoformat(),
                            }
                        else:
                            eval_result = None
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse {eval_result_path}, ignoring old data.")
                    eval_result = None

            if not eval_result:
                eval_result = {
                    "node_results": {},
                    "status": "started",
                    "start_time": datetime.now(timezone.utc).isoformat(),
                }

            unevaluated_nodes = [
                node for node in leaf_nodes
                if str(node.node_id) not in eval_result["node_results"]
            ]
            if not unevaluated_nodes:
                logger.info(
                    f"All {len(leaf_nodes)} leaf nodes for instance {instance_id} have been evaluated."
                )
                return

            # Prepare the testbed environment
            testbed_log_dir = os.path.join(
                get_moatless_trajectory_dir(instance_id, self.evaluation.evaluation_name),
                "testbed_logs"
            )
            os.makedirs(testbed_log_dir, exist_ok=True)

            instance = get_moatless_instance(instance_id=instance_id)
            repository = await create_repository_async(instance=instance, repo_base_dir=self.repo_base_dir)

            runtime = TestbedEnvironment(
                repository=repository,
                instance_id=instance_id,
                log_dir=testbed_log_dir,
                enable_cache=True,
            )

            for i, leaf_node in enumerate(unevaluated_nodes):
                logger.info(
                    f"Evaluating Node {leaf_node.node_id} ({i+1}/{len(unevaluated_nodes)}) "
                    f"for instance {instance_id}"
                )
                if str(leaf_node.node_id) in eval_result["node_results"]:
                    logger.info(f"Node {leaf_node.node_id} already evaluated; skipping.")
                    continue

                patch = leaf_node.file_context.generate_git_patch(ignore_tests=True)
                if not patch or not patch.strip():
                    logger.info(f"Node {leaf_node.node_id} has no patch; skipping evaluation.")
                    continue

                start_time = time.time()
                try:
                    result = await runtime.evaluate(patch=patch)
                    if not result:
                        logger.error(f"Error returned for testbed evaluation (no result).")
                        continue
                    eval_result["node_results"][str(leaf_node.node_id)] = result.model_dump()
                    logger.info(
                        f"Evaluated patch for node {leaf_node.node_id} in "
                        f"{time.time() - start_time:.2f}s (resolved: {result.resolved})."
                    )
                except Exception as e:
                    logger.exception(
                        f"Error in testbed evaluation of node {leaf_node.node_id} "
                        f"for instance {instance_id}."
                    )
                    eval_result["error"] = traceback.format_exc()
                finally:
                    eval_result["duration"] = time.time() - start_time
                    with open(eval_result_path, "w") as f:
                        json.dump(eval_result, f, indent=2)

            # Emit an event for the completion of node evaluations
            await self.emit_event("instance_evaluated", {
                "instance_id": instance_id,
                "eval_result": eval_result_path
            })

        except Exception as e:
            logger.exception(f"Error evaluating nodes for instance {instance_id}")
            await self.emit_event("instance_error", {"instance_id": instance_id, "error": str(e)})
            raise

    async def _evaluate_instance(self, instance: EvaluationInstance) -> None:
        """
        After the flow completes, evaluate the final patches against the testbed.
        """
        try:
            root_node = self.read_node(instance.instance_id)
            await self.evaluate_nodes(instance.instance_id, root_node)

            # Load the final eval result JSON if available
            eval_result_path = os.path.join(
                get_moatless_trajectory_dir(instance.instance_id, self.evaluation.evaluation_name),
                "eval_result.json"
            )

            eval_result: Optional[Dict[str, Any]] = None
            if os.path.exists(eval_result_path):
                with open(eval_result_path) as f:
                    eval_result = json.load(f)

            # Convert final results to a benchmark_result
            instance.benchmark_result = to_result(
                node=root_node,
                eval_report=eval_result,
                instance_id=instance.instance_id,
            )
            instance.resolved = instance.benchmark_result.resolved
            instance.evaluated_at = datetime.now(timezone.utc)

            # Transition to EVALUATED
            await self._update_instance_state(instance, InstanceStatus.EVALUATED)
            await self.emit_event("instance_evaluated", {
                "instance_id": instance.instance_id,
                "resolved": instance.resolved
            })

        except Exception as e:
            logger.exception(f"Error in instance evaluation: {str(e)}")
            await self._update_instance_state(instance, InstanceStatus.ERROR, str(e))
            await self.emit_event(
                "instance_error",
                {"instance_id": instance.instance_id, "error": str(e)}
            )

    def read_node(self, instance_id: str) -> Node:
        """
        Loads the trajectory.json file from the instance's trajectory directory and parses it into a Node.
        """
        trajectory_dir = get_moatless_trajectory_dir(instance_id, self.evaluation.evaluation_name)
        trajectory_path = trajectory_dir / "trajectory.json"
        return Node.from_file(trajectory_path)

    def _save_evaluation(self, evaluation: Evaluation) -> None:
        """
        Persists the entire Evaluation model to (evaluation_dir)/evaluation.json, using file locking.
        """
        eval_path = self.evaluation_dir / "evaluation.json"
        lock_path = self.locks_dir / f"{evaluation.evaluation_name}.lock"
        eval_path.parent.mkdir(parents=True, exist_ok=True)

        with filelock.FileLock(lock_path):
            with open(eval_path, "w") as f:
                json.dump(evaluation.model_dump(), f, indent=2, default=str)

    async def emit_event(self, event_type: str, data: Any = None) -> None:
        """
        Wrapper to emit an EvaluationEvent with the given type and data via the event bus,
        and then persist the evaluation.
        """
        logger.info(f"Emitting event: {event_type}")
        self._save_evaluation(self.evaluation)
        event = EvaluationEvent(
            evaluation_name=self.evaluation.evaluation_name,
            event_type=event_type,
            data=data,
        )
        await event_bus.publish(event)

    async def wait_for_completion(self, check_interval: float = 1.0) -> Evaluation:
        """
        Blocking call (awaitable) that waits until the evaluation is completed or errored.
        Returns the final Evaluation object or raises if the evaluation fails.
        """
        last_status = None
        while True:
            summary = self.evaluation.get_summary()
            if summary["status"] != last_status:
                logger.info(
                    f"Evaluation {self.evaluation.evaluation_name} status changed to: {summary['status']}"
                )
                last_status = summary["status"]

            if summary["status"] == EvaluationStatus.COMPLETED:
                logger.info(f"Evaluation {self.evaluation.evaluation_name} completed successfully.")
                return self.evaluation
            elif summary["status"] == EvaluationStatus.ERROR:
                logger.error(
                    f"Evaluation {self.evaluation.evaluation_name} failed with error: "
                    f"{summary['error']}"
                )
                raise RuntimeError(f"Evaluation failed: {summary['error']}")

            counts = summary["counts"]
            logger.info(
                f"[Status] Completed: {counts['completed']} | Running: {counts['running']} | "
                f"Evaluating: {counts['evaluating']} | Pending: {counts['pending']} | "
                f"Errors: {counts['errors']}"
            )
            await asyncio.sleep(check_interval)

    def _create_tracked_task(self, coro: asyncio.coroutines, instance_id: str) -> asyncio.Task:
        """
        Creates and tracks an asyncio Task, registering a cleanup callback to remove it from _tasks.
        """
        task = asyncio.create_task(coro)
        self._tasks.append(task)

        def _cleanup(fut: asyncio.Future) -> None:
            self._tasks.remove(task)
            # Only log non-cancellation errors
            if not fut.cancelled():
                exc = fut.exception()
                if exc:
                    logger.error(f"Task for instance {instance_id} failed: {exc}")

        task.add_done_callback(_cleanup)
        return task

    async def _monitor_status(self) -> None:
        """
        Periodically logs the current instance state counts if no state change has occurred lately.
        """
        while True:
            await asyncio.sleep(self._status_log_interval)
            current_time = time.time()
            if (current_time - self._last_status_log) >= self._status_log_interval:
                counts = await self._get_instance_counts()
                self._print_status_with_counts(counts)
                self._last_status_log = current_time

    async def _get_instance_counts(self) -> Dict[str, int]:
        """
        Returns a dict mapping each InstanceStatus value to the count of instances in that status.
        """
        counts = {status.value: 0 for status in InstanceStatus}
        for instance in self.evaluation.instances:
            counts[instance.status] += 1
        return counts

    def _print_status_with_counts(self, counts: Dict[str, int]) -> None:
        """
        Pretty-print the current distribution of instance statuses.
        """
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
