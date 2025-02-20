import asyncio
import json
import logging
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict

import filelock

from moatless.benchmark.report import to_result
from moatless.benchmark.swebench.utils import create_repository_async, create_index
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

from moatless.evaluation.state_manager import StateManager
from moatless.evaluation.task_scheduler import TaskScheduler

logger = logging.getLogger(__name__)

class EvaluationRunner:
    """
    Orchestrates an Evaluation's lifecycle.  
    - Uses TaskScheduler for concurrency.  
    - Relies on StateManager for instance states.  
    - Schedules separate tasks for setup, run, and evaluate.
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
        self.scheduler = TaskScheduler()

        self.num_concurrent_instances = num_concurrent_instances
        self.remove_repo_after_evaluation = remove_repo_after_evaluation

        # Base directories
        self.repo_base_dir = repo_base_dir or "/tmp/moatless_repos"
        self.evaluations_dir = evaluations_dir or (get_moatless_dir() / "evals")
        self.evaluation_dir = Path(self.evaluations_dir) / self.evaluation.evaluation_name
        self.locks_dir = self.evaluation_dir / ".locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)

        # Internal
        self._state_changed = asyncio.Event()
        self._status_log_interval = 30.0
        self._last_status_log = time.time()

    async def run_evaluation(self) -> None:
        """
        Primary entry point to run an entire evaluation, in the background.
        """

        try:
            current_project_id.set(self.evaluation.evaluation_name)
            self.evaluation.status = EvaluationStatus.RUNNING
            self.evaluation.started_at = datetime.now(timezone.utc)
            self._save_evaluation()
            await self._emit_event("evaluation_started")

            # Main scheduling loop: until we see all done or an error
            while True:
                await self._process_instances()
                if self._all_instances_finished():
                    break

                # Wait a short time or until a state change is triggered
                try:
                    await asyncio.wait_for(self._state_changed.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                finally:
                    self._state_changed.clear()

            # Mark evaluation completed
            self.evaluation.status = EvaluationStatus.COMPLETED
            self.evaluation.completed_at = datetime.now(timezone.utc)
            self._save_evaluation()
            await self._emit_event("evaluation_completed", {
                "completed": sum(i.status == InstanceStatus.EVALUATED for i in self.evaluation.instances),
                "errors": sum(i.status == InstanceStatus.ERROR for i in self.evaluation.instances),
            })

        except Exception as exc:
            logger.exception("Error running evaluation.")
            self.evaluation.status = EvaluationStatus.ERROR
            self.evaluation.error = str(exc)
            self._save_evaluation()
            await self._emit_event("evaluation_error", {"error": str(exc)})
            raise
        finally:
            # Cleanup
            if self.remove_repo_after_evaluation:
                self._cleanup_repositories()
                
            # Wait for any scheduled tasks to finish
            await self.scheduler.shutdown()

    async def _process_instances(self) -> None:
        """
        Checks concurrency constraints, schedules tasks for each phase: setup, run, evaluate.
        """
        # Count how many are in "RUNNING" vs. capacity
        counts = self._get_instance_counts()

        # (1) Setup or Start running pending instances, if there's capacity
        running = counts.get(InstanceStatus.RUNNING.value, 0)
        capacity_left = self.num_concurrent_instances - running

        if capacity_left > 0:
            # Grab pending instances
            pending = [i for i in self.evaluation.instances if i.status == InstanceStatus.PENDING]
            for instance in pending[:capacity_left]:
                await self._schedule_setup_or_run(instance)

        # (2) Evaluate completed instances, if there's capacity
        eval_running = counts.get(InstanceStatus.EVALUATING.value, 0)
        eval_capacity_left = self.num_concurrent_instances - eval_running

        if eval_capacity_left > 0:
            # Grab completed (but not yet EVALUATING/EVALUATED) instances
            ready_for_eval = [
                i for i in self.evaluation.instances 
                if i.status == InstanceStatus.COMPLETED
            ]
            for instance in ready_for_eval[:eval_capacity_left]:
                await self._update_instance_state(
                    instance,
                    InstanceStatus.EVALUATING,
                    "instance_evaluating"
                )
                self.scheduler.schedule(self._evaluate_instance(instance.instance_id), f"evaluate_{instance.instance_id}")

    async def _schedule_setup_or_run(self, instance: EvaluationInstance) -> None:
        """
        Decide if the instance needs a repo setup or can immediately go RUNNING.
        """
        # Check if repo is already available
        try:
            moatless_instance = get_moatless_instance(instance_id=instance.instance_id)
            exists = await self.state_manager.check_repository_exists(moatless_instance, self.repo_base_dir)

            if not exists:
                # Need to set up the repository
                can_setup = await self.state_manager.can_setup_repo(instance.instance_id)
                if can_setup:
                    await self.state_manager.register_repo_setup(instance.instance_id)
                    self.scheduler.schedule(
                        self._setup_instance(instance.instance_id), 
                        f"setup_{instance.instance_id}"
                    )
                else:
                    logger.info(f"Repo setup already in progress for {instance.instance_id}")
                return

            # If repo exists, go straight to RUNNING
            await self._update_instance_state(
                instance,
                InstanceStatus.RUNNING,
                "instance_started"
            )
            self.scheduler.schedule(
                self._run_instance(instance.instance_id),
                f"run_{instance.instance_id}"
            )

        except Exception as exc:
            logger.error(f"Failed to schedule setup/run for {instance.instance_id}: {exc}")
            await self._update_instance_state(
                instance,
                InstanceStatus.ERROR,
                "instance_error",
                error=str(exc)
            )

    #
    # Separate Steps
    #

    async def _setup_instance(self, instance_id: str) -> None:
        """
        Step (1): Setup the repository (clone, install deps, etc.).
        """
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

    async def _run_instance(self, instance_id: str) -> None:
        """
        Step (2): Run the "agentic flow" or whatever your main "execution" is.
        """
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

    async def _evaluate_instance(self, instance_id: str) -> None:
        """
        Step (3): Evaluate the completed instance (e.g., test the patch).
        """
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

    async def _evaluate_nodes(self, instance: EvaluationInstance, root_node: Node) -> None:
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
            self._save_evaluation()

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
        code_index = create_index(moatless_instance, repository=repository)

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
            flow.workspace = workspace

            last_node = flow.root.get_all_nodes()[-1]
            if last_node.error:
                logger.info(f"Last node in flow {instance_id} has error, will remove it and try again")
                last_node.parent.remove_child(last_node)
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

    def _save_evaluation(self) -> None:
        """
        Persists the current state of the evaluation to disk.
        """
        eval_path = self.evaluation_dir / "evaluation.json"
        lock_path = self.locks_dir / f"{self.evaluation.evaluation_name}.lock"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with filelock.FileLock(lock_path):
            with open(eval_path, "w") as f:
                json.dump(self.evaluation.model_dump(), f, indent=2, default=str)

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
