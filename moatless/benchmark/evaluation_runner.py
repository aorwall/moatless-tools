import concurrent.futures
import gc
import json
import logging
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Union, Callable, List

from moatless.events import BaseEvent, event_bus
from moatless.config.model_config import create_completion_model
from moatless.config.agent_config import get_agent
from moatless.agent.code_agent import CodingAgent
from moatless.agentic_system import AgenticSystem
from moatless.benchmark.schema import (
    TreeSearchSettings,
    Evaluation,
    EvaluationInstance,
    EvaluationStatus,
    InstanceStatus,
    EvaluationEvent,
)
from moatless.benchmark.swebench import (
    create_repository,
    create_index,
)
from moatless.benchmark.swebench.utils import create_repository_async, instance_repo_path
from moatless.benchmark.utils import get_moatless_instance, load_moatless_datasets
from moatless.completion import BaseCompletionModel
from moatless.events import SystemEvent
from moatless.file_context import FileContext
from moatless.loop import AgenticLoop
from moatless.node import Node
from moatless.runtime.testbed import TestbedEnvironment
from moatless.search_tree import SearchTree

import asyncio
from asyncio import Task
from moatless.context_data import current_evaluation_name

from moatless.utils.moatless import get_moatless_dir, get_moatless_trajectory_dir
from moatless.runner import agentic_runner
from moatless.workspace import Workspace
from testbeds.sdk.sdk import TestbedSDK

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


logger = logging.getLogger(__name__)

__all__ = [
    "TreeSearchSettings",
    "Evaluation",
    "InstanceStatus",
    "EvaluationStatus",
    "EvaluationEvent",
]


class EvaluationRunner:
    def __init__(
        self,
        evaluation: Evaluation,
        repo_base_dir: Union[str, None] = None,
        evaluations_dir: Union[str, None] = None,
        num_workers: int = 1,
        use_testbed: bool = True,
        rerun_errors: bool = True,
        remove_repo_after_evaluation: bool = True,
    ):
        self.tree_search_settings = evaluation.settings
        self.evaluation = evaluation

        if evaluations_dir:
            self.evaluations_dir = evaluations_dir
        else:
            self.evaluations_dir = os.getenv("MOATLESS_DIR", "./evals")

        self.evaluation_dir = os.path.join(self.evaluations_dir, self.evaluation.evaluation_name)

        self.repo_base_dir = repo_base_dir or os.getenv("MOATLESS_REPO_DIR", "./repos")
        self.num_workers = num_workers
        self.use_testbed = use_testbed
        self.rerun_errors = rerun_errors
        self.remove_repo_after_evaluation = remove_repo_after_evaluation
        self.agentic_runner = agentic_runner
        self._pending_evaluations = {}  # track {run_id: (instance_id, moatless_instance)}
        event_bus.subscribe(self._handle_flow_event)

    async def run_evaluation(self):
        """Start the evaluation process."""
        logger.info(f"Running evaluation: {self.evaluation.evaluation_name}")
        await self.emit_event("evaluation_started")
        
        instance_ids = [instance.instance_id for instance in self.evaluation.instances]
        logger.info(f"Processing {len(instance_ids)} instances with {self.num_workers} concurrent tasks")
        
        # Start initial batch of instances up to num_workers
        initial_instances = instance_ids[:self.num_workers]
        async with asyncio.TaskGroup() as tg:
            for instance_id in initial_instances:
                logger.info(f"Starting initial instance {instance_id}")
                tg.create_task(self.create_and_run_instance(instance_id))

        # Start background task to manage remaining instances
        asyncio.create_task(self._manage_instances(instance_ids[self.num_workers:]))

    async def _manage_instances(self, remaining_instances: List[str]):
        """Background task to manage starting new instances as slots become available."""
        for instance_id in remaining_instances:
            while True:
                # Wait for a slot to become available
                if len(self._pending_evaluations) < self.num_workers:
                    logger.info(f"Starting next instance {instance_id}")
                    await self.create_and_run_instance(instance_id)
                    break
                await asyncio.sleep(1)

    async def _handle_flow_event(self, trajectory_id: str, event: BaseEvent):
        """Handle flow events for evaluation."""
        if event.event_type == "flow_completed":
            if trajectory_id in self._pending_evaluations:
                instance_id, moatless_instance = self._pending_evaluations[trajectory_id]
                logger.info(f"Flow completed for instance {instance_id}, starting evaluation")

                instance = self.evaluation.get_instance(instance_id)
                instance.status = InstanceStatus.COMPLETED
                instance.completed_at = datetime.now(timezone.utc)
                
                try:
                    # Get the completed system
                    system = await self.agentic_runner.get_run(trajectory_id)
                    if not system:
                        raise ValueError(f"Could not find completed system for {trajectory_id}")

                    if self.use_testbed:
                        await self.evaluate_nodes(
                            instance_id=instance_id,
                            instance=moatless_instance,
                            root_node=system.root
                        )

                    instance = self.evaluation.get_instance(instance_id)
                    instance.status = InstanceStatus.EVALUATED
                    instance.evaluated_at = datetime.now(timezone.utc)
                    await self.emit_event(
                        "instance_completed",
                        {"instance_id": instance_id},
                    )
                except Exception as e:
                    logger.exception(f"Error evaluating completed flow: {str(e)}")
                    instance = self.evaluation.get_instance(instance_id)
                    instance.fail(error=str(e))
                    await self.emit_event("instance_error", {"instance_id": instance_id, "error": str(e)})
                finally:
                    # Clean up tracking
                    del self._pending_evaluations[trajectory_id]

        elif event.event_type == "flow_error":
            if trajectory_id in self._pending_evaluations:
                instance_id, _ = self._pending_evaluations[trajectory_id]
                logger.error(f"Flow failed for instance {instance_id}: {event.error}")
                instance = self.evaluation.get_instance(instance_id)
                instance.fail(error=event.error)
                await self.emit_event("instance_error", {"instance_id": instance_id, "error": event.error})
                del self._pending_evaluations[trajectory_id]

    async def create_and_run_instance(self, instance_id: str):
        """Create and run a single instance."""
        logger.info(f"Starting evaluation of instance {instance_id}")
        instance = self.evaluation.get_instance(instance_id)
        if not instance:
            logger.error(f"Instance {instance_id} not found")
            raise ValueError(f"Instance {instance_id} not found")

        try:
            instance.status = InstanceStatus.STARTED
            instance.started_at = datetime.now(timezone.utc)

            moatless_instance = get_moatless_instance(instance_id=instance_id)
            problem_statement = f"<task>\nSolve the following reported issue in the {moatless_instance['repo']} repository:\n\n{moatless_instance['problem_statement']}\n</task>"

            agentic_system = await self.create_agentic_flow(
                problem_statement=problem_statement,
                instance=instance,
                moatless_instance=moatless_instance,
            )

            run_id = await self.agentic_runner.start(agentic_system)
            logger.info(f"Started agentic system with run ID {run_id}")

            # Track this run for evaluation when it completes
            self._pending_evaluations[run_id] = (instance_id, moatless_instance)
            
        except Exception as e:
            logger.exception(f"Error running instance {instance_id}")
            instance.status = InstanceStatus.ERROR
            instance.error = str(e)
            raise

    async def create_agentic_flow(
        self,
        problem_statement: str,
        instance: EvaluationInstance,
        moatless_instance: dict,
    ) -> AgenticSystem:
        """Create an agentic system for the instance."""
        repository = await create_repository_async(moatless_instance, repo_base_dir=self.repo_base_dir)
        code_index = create_index(moatless_instance, repository=repository)
        
        completion_model = create_completion_model(self.tree_search_settings.model_id)
        completion_model.metadata = {"instance_id": instance.instance_id}

        runtime = None
        if self.use_testbed:
            from moatless.runtime.testbed import TestbedEnvironment
            testbed_log_dir = os.path.join(self.evaluation_dir, instance.instance_id, "testbed_logs")
            if not os.path.exists(testbed_log_dir):
                os.makedirs(testbed_log_dir)

            runtime = TestbedEnvironment(
                repository=repository,
                instance_id=instance.instance_id,
                log_dir=testbed_log_dir,
                enable_cache=True,
            )

        agent = get_agent(agent_id=self.tree_search_settings.agent_id)
        agent.completion_model = completion_model
        agent.workspace = Workspace(repository=repository, code_index=code_index, runtime=runtime, legacy_workspace=True)

        persist_dir = get_moatless_trajectory_dir(instance.instance_id)

        if self.tree_search_settings.max_expansions > 1:
            return SearchTree.create(
                message=problem_statement,
                agent=agent,
                repository=repository,
                metadata={"instance_id": instance.instance_id},
                max_iterations=self.tree_search_settings.max_iterations,
                max_expansions=self.tree_search_settings.max_expansions,
                persist_dir=persist_dir,
            )
        else:
            return AgenticLoop.create(
                message=problem_statement,
                run_id=instance.instance_id,
                agent=agent,
                max_iterations=self.tree_search_settings.max_iterations,
                metadata={"instance_id": instance.instance_id},
                persist_dir=persist_dir,
            )

    async def evaluate_nodes(
        self,
        instance_id: str,
        instance: dict,
        root_node: Node,
        ) -> dict:
        """Evaluate all leaf nodes using the testbed."""
        leaf_nodes = root_node.get_leaf_nodes()

        # Load existing eval results if any
        eval_result = None

        eval_result_path = os.path.join(self.get_evaluation_dir(), instance_id, "eval_result.json")
        if os.path.exists(eval_result_path):
            try:
                with open(eval_result_path) as f:
                    eval_result = json.load(f)
                    logger.info(f"Loading eval_result from {eval_result_path}")
                    if "node_results" not in eval_result:
                        if len(eval_result) > 0:
                            logger.info(f"Found node results with {eval_result.keys()} on root, fix format")
                            eval_result = {
                                "node_results": eval_result,
                                "status": "started",
                                "start_time": datetime.now(timezone.utc).isoformat(),
                            }
                        else:
                            logger.info("No node_results found")
                            eval_result = None
                    else:
                        logger.info(f"Found evaluated nodes {eval_result['node_results'].keys()}")
            except json.JSONDecodeError:
                pass

        if not eval_result:
            eval_result = {
                "node_results": {},
                "status": "started",
                "start_time": datetime.now(timezone.utc).isoformat(),
            }

        # Filter out already evaluated nodes
        unevaluated_nodes = [
            node for node in leaf_nodes if str(node.node_id) not in eval_result.get("node_results", {})
        ]

        if not unevaluated_nodes:
            logger.info(f"All {len(leaf_nodes)} nodes for instance {instance_id} have already been evaluated")
            return eval_result

        logger.info(
            f"Found {len(leaf_nodes) - len(unevaluated_nodes)} already evaluated nodes, "
            f"will evaluate remaining {len(unevaluated_nodes)} nodes for instance {instance_id}"
        )

        repository = create_repository(instance, repo_base_dir=self.repo_base_dir)
        async with TestbedEnvironment(repository=repository, instance_id=instance_id) as runtime:
            for i, leaf_node in enumerate(unevaluated_nodes):
                logger.info(f"Evaluate Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id}")

                if str(leaf_node.node_id) in eval_result["node_results"]:
                    logger.info(
                        f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} that has already been evaluated"
                    )
                    continue

                patch = leaf_node.file_context.generate_git_patch(ignore_tests=True)
                if patch and patch.strip():
                    start_time = time.time()
                    try:
                        result = await runtime.evaluate(patch=patch)
                        if not result:
                            logger.error(f"Error in evaluating patch for {instance_id}")
                            continue

                        eval_result["node_results"][str(leaf_node.node_id)] = result.model_dump()
                        logger.info(
                            f"Evaluated patch for node {leaf_node.node_id} in {time.time() - start_time} seconds (resolved: {result.resolved})"
                        )
                    except Exception as e:
                        logger.error(f"Error in testbed evaluation for instance {instance_id}: {str(e)}")
                        eval_result["error"] = traceback.format_exc()
                    finally:
                        eval_result["duration"] = time.time() - start_time
                        with open(eval_result_path, "w") as f:
                            json.dump(eval_result, f, indent=2)
                else:
                    logger.info(
                        f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} with no patch."
                    )

        return eval_result

    def _clean_error_nodes(self, node: Node, instance_id: str):
        """Clean error nodes if rerun_errors is enabled."""
        last_node = node.get_all_nodes()[-1]

        # Remove error nodes
        if last_node.error or (last_node.action and last_node.action.name == "Error" and last_node.parent):
            last_node.parent.children = [c for c in last_node.parent.children if c.node_id != last_node.node_id]
            logger.info(
                f"Removed error node {last_node.node_id} from parent {last_node.parent.node_id} on instance {instance_id}"
            )

        # Remove nodes with unexecuted actions
        last_node = node.get_all_nodes()[-1]
        if last_node.action and all(not step.observation for step in last_node.action_steps):
            last_node.parent.children = [c for c in last_node.parent.children if c.node_id != last_node.node_id]
            logger.info(
                f"Removed last node {last_node.node_id} from instance {instance_id} because it has no observation"
            )

    def get_evaluation_dir(self) -> str:
        """Get the directory path for an evaluation."""
        return self.evaluation_dir

    async def emit_event(self, event_type: str, data: Any = None):
        """Emit an event to all registered handlers"""
        logger.info(f"Emitting event {event_type}")
        event = EvaluationEvent(
            evaluation_name=self.evaluation.evaluation_name,
            event_type=event_type,
            data=data,
        )
        await event_bus.publish(event)

    async def _run_instance(self, instance: EvaluationInstance):
        """Run a single instance evaluation."""
        # Your instance evaluation logic here
        pass
