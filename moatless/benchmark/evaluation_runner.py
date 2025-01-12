import concurrent.futures
import gc
import hashlib
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Union, Callable, List

from moatless.agent.code_agent import CodingAgent
from moatless.benchmark.report import (
    create_sha256_hash,
    to_result,
)
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
from moatless.benchmark.utils import get_moatless_instance
from moatless.exceptions import RuntimeError
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
        repo_base_dir: Union[str, None] = None,
        evaluations_dir: Union[str, None] = None,
        num_workers: int = 1,
        use_testbed: bool = False,
        rerun_errors: bool = True,
    ):
        self._event_handlers: List[Callable[[EvaluationEvent], None]] = []

        self.evaluation = evaluation

        if evaluations_dir:
            self.evaluations_dir = evaluations_dir
        else:
            self.evaluations_dir = os.getenv("MOATLESS_DIR", "./evals")

        self.repo_base_dir = repo_base_dir
        self.num_workers = num_workers
        self.use_testbed = use_testbed
        self.rerun_errors = rerun_errors

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

        self.emit_event(self.evaluation.evaluation_name, "evaluation_started")
        error = 0

        results = []

        # Filter out instances from evaluation + instance_ids

        logger.info(
            f"Processing {len(instance_ids)} instances with {self.num_workers} workers. Rerun error {self.rerun_errors}"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = [
                executor.submit(self.evaluate_instance, instance_id)
                for instance_id in instance_ids
            ]

            for future in futures:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception:
                    error += 1
                    logger.exception("Error in processing instance")
                    self.emit_event("instance_error", {"error": traceback.format_exc()})

        logger.info(f"Completed processing with {error} errors")
        self.evaluation.status = (
            EvaluationStatus.COMPLETED if error == 0 else EvaluationStatus.ERROR
        )
        self.evaluation.finish_time = datetime.now(timezone.utc)

        self.emit_event(
            "evaluation_completed",
            {"total_instances": len(instance_ids), "errors": error},
        )

    def evaluate_instance(self, instance_id: str):
        """Evaluate a single instance."""
        runtime = None
        repository = None
        search_tree = None
        eval_result = None

        instance_dir = os.path.join(self.get_evaluation_dir(), instance_id)
        trajectory_path = os.path.join(instance_dir, "trajectory.json")
        eval_result_path = os.path.join(instance_dir, "eval_result.json")
        os.makedirs(instance_dir, exist_ok=True)
        instance = self.evaluation.get_instance(instance_id)
        if not instance:
            instance = EvaluationInstance(instance_id=instance_id)
            self.evaluation.instances.append(instance)

        try:
            moatless_instance = get_moatless_instance(instance_id=instance_id)
            problem_statement = f"<task>\nSolve the following reported issue in the {moatless_instance['repo']} repository:\n\n{moatless_instance['problem_statement']}\n</task>"

            eval_result = {
                "node_results": {},
                "status": "started",
                "start_time": datetime.now(timezone.utc).isoformat(),
            }

            if os.path.exists(eval_result_path):
                try:
                    with open(eval_result_path) as f:
                        logger.info(f"Loading eval_result from {eval_result_path}")
                        eval_result = json.load(f)
                except json.JSONDecodeError:
                    pass

            search_tree = self.create_and_run_search_tree(
                problem_statement=problem_statement,
                instance=instance,
                moatless_instance=moatless_instance,
                trajectory_path=trajectory_path,
            )

            start_time = time.time()
            try:
                if self.use_testbed:
                    logger.info(f"Evaluating nodes for instance {instance_id}")
                    eval_result = self.evaluate_nodes(
                        instance_id=instance_id,
                        instance=moatless_instance,
                        search_tree=search_tree,
                        eval_result=eval_result,
                    )
            except RuntimeError as e:
                raise e

            except Exception as e:
                eval_result["status"] = "error"
                eval_result["error"] = traceback.format_exc()
                eval_result["duration"] = time.time() - start_time
                logger.exception(
                    f"Error when evaluating nodes for instance {instance_id}"
                )

            benchmark_result = to_result(search_tree, eval_report=eval_result)

            # Complete instance with result
            instance.complete(
                resolved=benchmark_result.resolved, benchmark_result=benchmark_result
            )
            self.emit_event(
                "instance_completed",
                {
                    "instance_id": instance_id,
                    "resolved": instance.resolved,
                    "benchmark_result": benchmark_result.dict()
                    if benchmark_result
                    else None,
                },
            )
            return benchmark_result

        except Exception as e:
            stacktrace = traceback.format_exc()
            instance.fail(error=stacktrace)
            self.emit_event(
                "instance_error", {"instance_id": instance_id, "error": str(e)}
            )
            raise
        finally:
            if eval_result:
                # Save evaluation result
                with open(eval_result_path, "w") as f:
                    json.dump(eval_result, f, indent=2)

            # Clean up
            del runtime
            del repository
            del search_tree
            del eval_result
            gc.collect()

    def evaluate_nodes(
        self,
        instance_id: str,
        instance: dict,
        search_tree: SearchTree,
        eval_result: dict,
    ):
        """Evaluate all leaf nodes using the testbed."""
        leaf_nodes = search_tree.get_leaf_nodes()
        patch_results = {}

        # Filter out already evaluated nodes
        unevaluated_nodes = [
            node
            for node in leaf_nodes
            if str(node.node_id) not in eval_result.get("node_results", {})
        ]

        if not unevaluated_nodes:
            logger.info(
                f"All {len(leaf_nodes)} nodes for instance {instance_id} have already been evaluated"
            )
            return eval_result

        logger.info(
            f"Found {len(leaf_nodes) - len(unevaluated_nodes)} already evaluated nodes, "
            f"will evaluate remaining {len(unevaluated_nodes)} nodes for instance {instance_id}"
        )
        repository = create_repository(instance, repo_base_dir=self.repo_base_dir)
        # TODO: Set run_id on testbed environment
        run_id = hashlib.sha256(self.evaluation.evaluation_name.encode()).hexdigest()[
            :8
        ]
        runtime = TestbedEnvironment(
            repository=repository,
            instance=instance,
            # run_id=run_id,
        )

        for i, leaf_node in enumerate(unevaluated_nodes):
            logger.info(
                f"Evaluate Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id}"
            )

            if str(leaf_node.node_id) in eval_result["node_results"]:
                logger.info(
                    f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} that has already been evaluated"
                )
                continue

            patch = leaf_node.file_context.generate_git_patch(ignore_tests=True)
            if patch and patch.strip():
                patch_hash = create_sha256_hash(patch)

                if patch_hash in patch_results:
                    logger.info(
                        f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} as patch has already been evaluated."
                    )
                    eval_result["node_results"][str(leaf_node.node_id)] = patch_results[
                        patch_hash
                    ]
                else:
                    start_time = time.time()
                    result = runtime.evaluate(patch=patch)
                    if not result:
                        logger.error(f"Error in evaluating patch for {instance_id}")
                        continue

                    eval_result["node_results"][str(leaf_node.node_id)] = (
                        result.model_dump()
                    )
                    patch_results[patch_hash] = eval_result["node_results"][
                        str(leaf_node.node_id)
                    ]
                    logger.info(
                        f"Evaluated patch for node {leaf_node.node_id} in {time.time() - start_time} seconds (resolved: {result.resolved})"
                    )
            else:
                logger.info(
                    f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} with no patch."
                )

        return eval_result

    def create_and_run_search_tree(
        self,
        problem_statement: str,
        instance: EvaluationInstance,
        moatless_instance: dict,
        trajectory_path: str,
    ) -> SearchTree:
        """Create and run a search tree for the given problem instance."""
        metadata: dict[str, Any] = {
            "evaluation_name": self.evaluation.evaluation_name,
            "instance_id": instance.instance_id,
        }

        search_tree = None
        rerun_tree = False
        if os.path.exists(trajectory_path):
            try:
                persisted_tree = SearchTree.from_file(
                    trajectory_path,
                )

                if self.rerun_errors:
                    leaf_nodes = persisted_tree.get_leaf_nodes()
                    for leaf_node in leaf_nodes:
                        if leaf_node.error or (
                            leaf_node.action and leaf_node.action.name == "Error"
                        ):
                            rerun_tree = True
                            break

                if persisted_tree.is_finished() and not rerun_tree:
                    logger.info(
                        f"Found completed search tree for {instance.instance_id}"
                    )
                    return persisted_tree
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse search tree from {trajectory_path}. Will remove file to start over. Error: {e}"
                )
                os.remove(trajectory_path)

        repository = create_repository(
            moatless_instance, repo_base_dir=self.repo_base_dir
        )
        code_index = create_index(moatless_instance, repository=repository)

        runtime = None
        if self.use_testbed:
            from moatless.runtime.testbed import TestbedEnvironment

            run_id = hashlib.sha256(
                self.evaluation.evaluation_name.encode()
            ).hexdigest()[:8]
            runtime = TestbedEnvironment(
                repository=repository,
                instance=moatless_instance,
                run_id=run_id,
            )

        # Load search tree from file again and set repository, runtime and code index
        if os.path.exists(trajectory_path):
            search_tree = SearchTree.from_file(
                trajectory_path,
                repository=repository,
                runtime=runtime,
                code_index=code_index,
            )
            completion_model = (
                self.evaluation.settings.agent_settings.completion_model.clone()
            )
            completion_model.metadata = {"instance_id": instance.instance_id}

            search_tree.agent = CodingAgent.create(
                completion_model=completion_model,
                repository=repository,
                code_index=code_index,
                runtime=runtime,
                message_history_type=self.evaluation.settings.agent_settings.message_history_type,
                thoughts_in_action=self.evaluation.settings.agent_settings.thoughts_in_action,
            )
        else:
            completion_model = (
                self.evaluation.settings.agent_settings.completion_model.clone()
            )
            completion_model.metadata = {"instance_id": instance.instance_id}

            agent = CodingAgent.create(
                completion_model=completion_model,
                repository=repository,
                code_index=code_index,
                runtime=runtime,
                message_history_type=self.evaluation.settings.agent_settings.message_history_type,
                thoughts_in_action=self.evaluation.settings.agent_settings.thoughts_in_action,
            )
            search_tree = SearchTree.create(
                message=problem_statement,
                repository=repository,
                runtime=runtime,
                selector=self.evaluation.settings.selector,
                agent=agent,
                value_function=self.evaluation.settings.value_function,
                max_iterations=self.evaluation.settings.max_iterations,
                max_expansions=self.evaluation.settings.max_expansions,
                max_cost=self.evaluation.settings.max_cost,
                persist_path=trajectory_path,
                metadata=metadata,
            )

        if self.rerun_errors:
            leaf_nodes = search_tree.get_leaf_nodes()
            for leaf_node in leaf_nodes:
                if (
                    leaf_node.error
                    or (leaf_node.action and leaf_node.action.name == "Error")
                ) and leaf_node.parent:
                    # Remove error node from parent's children
                    leaf_node.parent.children = [
                        c
                        for c in leaf_node.parent.children
                        if c.node_id != leaf_node.node_id
                    ]
                    logger.info(
                        f"Removed error node {leaf_node.node_id} from parent {leaf_node.parent.node_id} on instance {instance.instance_id}"
                    )

        def tree_event_handler(event):
            logger.info(f"Got event {event['event_type']}")
            if event["event_type"] == "tree_iteration":
                instance.usage = search_tree.total_usage()
                instance.iterations = len(search_tree.root.get_all_nodes())

                logger.info("Emit event tree_progress")
                self.emit_event(
                    "tree_progress",
                    {
                        "instance_id": instance.instance_id,
                    },
                )

        instance.start()
        self.emit_event("instance_started", {"instance_id": instance.instance_id})

        search_tree.add_event_handler(tree_event_handler)
        search_tree.run_search()

        self.emit_event("instance_completed", {"instance_id": instance.instance_id})

        return search_tree

    def get_evaluation_dir(self) -> str:
        """Get the directory path for an evaluation."""
        return os.path.join(self.evaluations_dir, self.evaluation.evaluation_name)
