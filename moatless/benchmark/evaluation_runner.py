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

from moatless.config.model_config import create_completion_model
from moatless.config.agent_config import create_agent
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
from moatless.benchmark.swebench.utils import instance_repo_path
from moatless.benchmark.utils import get_moatless_instance, load_moatless_datasets
from moatless.completion import BaseCompletionModel
from moatless.events import SystemEvent
from moatless.file_context import FileContext
from moatless.loop import AgenticLoop
from moatless.node import Node
from moatless.runtime.testbed import TestbedEnvironment
from moatless.search_tree import SearchTree


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
        tree_search_settings: TreeSearchSettings,
        repo_base_dir: Union[str, None] = None,
        evaluations_dir: Union[str, None] = None,
        num_workers: int = 1,
        use_testbed: bool = False,
        rerun_errors: bool = True,
        remove_repo_after_evaluation: bool = True,
    ):
        self._event_handlers: List[Callable[[EvaluationEvent], None]] = []
        self.tree_search_settings = tree_search_settings

        self.evaluation = evaluation

        if evaluations_dir:
            self.evaluations_dir = evaluations_dir
        else:
            self.evaluations_dir = os.getenv("MOATLESS_DIR", "./evals")

        self.repo_base_dir = repo_base_dir or os.getenv("MOATLESS_REPO_DIR", "./repos")
        self.num_workers = num_workers
        self.use_testbed = use_testbed
        self.rerun_errors = rerun_errors
        self.remove_repo_after_evaluation = remove_repo_after_evaluation

    def add_event_handler(self, handler: Callable[[EvaluationEvent], None]):
        """Add an event handler to receive evaluation events"""
        self._event_handlers.append(handler)

    def emit_event(self, event_type: str, data: Any = None):
        """Emit an event to all registered handlers"""
        logger.info(f"Emitting event {event_type}")
        event = EvaluationEvent(
            evaluation_name=self.evaluation.evaluation_name,
            event_type=event_type,
            data=data,
        )
        for handler in self._event_handlers:
            handler(event)

    def run_evaluation(self, instance_ids: List[str] | None = None):
        """Run the evaluation process."""

        os.makedirs(self.get_evaluation_dir(), exist_ok=True)

        if not self.evaluation.start_time:
            self.evaluation.start_time = datetime.now(timezone.utc)

        self.evaluation.status = EvaluationStatus.RUNNING

        self.emit_event("evaluation_started")
        error = 0

        # TODO: Filter out instances from evaluation + instance_ids

        logger.info(
            f"Processing {len(instance_ids)} instances with {self.num_workers} workers. Rerun error {self.rerun_errors}"
        )

        load_moatless_datasets()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.evaluate_instance, instance_id) for instance_id in instance_ids]

            for future in futures:
                try:
                    future.result()

                except Exception:
                    error += 1
                    logger.exception("Error in processing instance")
                    self.emit_event("instance_error", {"error": traceback.format_exc()})

        logger.info(f"Completed processing with {error} errors")
        self.evaluation.status = EvaluationStatus.COMPLETED if error == 0 else EvaluationStatus.ERROR
        self.evaluation.finish_time = datetime.now(timezone.utc)

        self.emit_event(
            "evaluation_completed",
            {"total_instances": len(instance_ids), "errors": error},
        )

    def evaluate_instance(self, instance_id: str):
        """Evaluate a single instance."""
        logger.info(f"Starting evaluation of instance {instance_id}")
        runtime = None
        repository = None
        agentic_loop = None
        eval_result = None

        instance_dir = os.path.join(self.get_evaluation_dir(), instance_id)
        trajectory_path = os.path.join(instance_dir, "trajectory.json")
        os.makedirs(instance_dir, exist_ok=True)
        instance = self.evaluation.get_instance(instance_id)
        if not instance:
            instance = EvaluationInstance(instance_id=instance_id)
            self.evaluation.instances.append(instance)

        try:
            logger.info(f"Loading moatless instance {instance_id}")
            moatless_instance = get_moatless_instance(instance_id=instance_id)
            problem_statement = f"<task>\nSolve the following reported issue in the {moatless_instance['repo']} repository:\n\n{moatless_instance['problem_statement']}\n</task>"

            root_node = self.create_and_run(
                problem_statement=problem_statement,
                instance=instance,
                moatless_instance=moatless_instance,
                trajectory_path=trajectory_path,
            )
            logger.info(f"Completed agentic loop for instance {instance_id}")

            eval_result_path = os.path.join(instance_dir, "eval_result.json")
            if self.use_testbed:
                logger.info(f"Starting testbed evaluation for instance {instance_id}")
                eval_result = self.evaluate_nodes(
                    instance_id=instance_id,
                    instance=moatless_instance,
                    root_node=root_node,
                    eval_result_path=eval_result_path,
                )
                logger.info(f"Completed testbed evaluation for instance {instance_id}")

            instance.complete()
            self.emit_event(
                "instance_completed",
                {"instance_id": instance_id, "eval_result": eval_result},
            )
            return

        except Exception as e:
            stacktrace = traceback.format_exc()
            instance.fail(error=stacktrace)
            self.emit_event("instance_error", {"instance_id": instance_id, "error": str(e)})
            raise
        finally:
            if self.remove_repo_after_evaluation:
                repo_path = instance_repo_path(instance_id, self.repo_base_dir)
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path)

            # Clean up
            del runtime
            del repository
            del agentic_loop
            del eval_result
            gc.collect()

    def create_and_run(
        self,
        problem_statement: str,
        instance: EvaluationInstance,
        moatless_instance: dict,
        trajectory_path: str,
    ) -> Node:
        """Create and run an agentic loop for the given problem instance."""
        metadata: dict[str, Any] = {
            "evaluation_name": self.evaluation.evaluation_name,
            "instance_id": instance.instance_id,
        }

        start_time = time.time()

        root_node = None
        rerun_tree = False
        if os.path.exists(trajectory_path):
            try:
                root_node = Node.from_file(
                    trajectory_path,
                )

                if self.rerun_errors:
                    for node in root_node.get_all_nodes():
                        if node.error or (node.action and node.action.name == "Error"):
                            rerun_tree = True
                            break

                # TODO: Calculate if is ifinished
                # if persisted_node.is_finished() and not rerun_tree:
                #    logger.info(f"Found completed search tree for {instance.instance_id}")
                #    return persisted_node
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse search tree from {trajectory_path}. Will remove file to start over. Error: {e}"
                )
                os.remove(trajectory_path)

        repository = create_repository(moatless_instance, repo_base_dir=self.repo_base_dir)
        code_index = create_index(moatless_instance, repository=repository)

        runtime = None
        if self.use_testbed:
            from moatless.runtime.testbed import TestbedEnvironment

            runtime = TestbedEnvironment(
                repository=repository,
                instance=moatless_instance,
            )

        if not root_node:
            file_context = FileContext(repo=repository, runtime=runtime)
            root_node = Node(node_id=0, user_message=problem_statement, file_context=file_context)

        completion_model = create_completion_model(self.tree_search_settings.model_id)
        completion_model.metadata = {"instance_id": instance.instance_id}

        agent = create_agent(
            self.tree_search_settings.agent_id,
            completion_model=completion_model,
            repository=repository,
            code_index=code_index,
            runtime=runtime,
        )

        if self.tree_search_settings.max_expansions > 1:
            agentic_system = SearchTree.create(
                message=problem_statement,
                repository=repository,
                runtime=runtime,
                selector=self.tree_search_settings.selector,
                agent=agent,
                value_function=self.tree_search_settings.value_function,
                feedback_generator=self.tree_search_settings.feedback_generator,
                max_iterations=self.tree_search_settings.max_iterations,
                max_expansions=self.tree_search_settings.max_expansions,
                max_cost=self.tree_search_settings.max_cost,
                max_depth=self.tree_search_settings.max_depth,
                min_finished_nodes=self.tree_search_settings.min_finished_nodes,
                max_finished_nodes=self.tree_search_settings.max_finished_nodes,
                reward_threshold=self.tree_search_settings.reward_threshold,
                persist_path=trajectory_path,
                metadata=metadata,
            )
        else:
            agentic_system = AgenticLoop.create(
                message=problem_statement,
                repository=repository,
                runtime=runtime,
                agent=agent,
                max_iterations=self.tree_search_settings.max_iterations,
                max_cost=self.tree_search_settings.max_cost,
                persist_path=trajectory_path,
                metadata=metadata,
            )

        self._clean_error_nodes(root_node, instance.instance_id)

        def event_handler(event: SystemEvent):
            logger.info(f"Got event {event.event_type}")

            self.emit_event(
                event.event_type,
                {"instance_id": instance.instance_id, **event.model_dump()},
            )

        instance.start()
        self.emit_event("loop_started", {"instance_id": instance.instance_id})

        agentic_system.add_event_handler(event_handler)
        agentic_system.run(root_node)

        duration = time.time() - start_time
        self.emit_event(
            "loop_completed",
            {"instance_id": instance.instance_id, "duration": duration},
        )

        return root_node

    def evaluate_nodes(
        self,
        instance_id: str,
        instance: dict,
        root_node: Node,
        eval_result_path: str,
    ) -> dict:
        """Evaluate all leaf nodes using the testbed."""
        leaf_nodes = root_node.get_leaf_nodes()

        # Load existing eval results if any
        eval_result = None
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
        runtime = TestbedEnvironment(
            repository=repository,
            instance=instance,
        )

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
                    result = runtime.evaluate(patch=patch)
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
        return os.path.join(self.evaluations_dir, self.evaluation.evaluation_name)
