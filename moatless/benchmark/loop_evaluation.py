import concurrent.futures
import gc
import json
import logging
import os
import random
import shutil
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Any

import litellm
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from moatless.agent.agent import ActionAgent, MessageHistoryType
from moatless.agent.code_agent import CodingAgent

from moatless.benchmark.swebench import (
    create_repository,
    create_index,
)
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.completion import CompletionModel
from moatless.completion.log_handler import LogHandler
from moatless.loop import AgenticLoop

logger = logging.getLogger(__name__)


class Evaluation:
    def __init__(
        self,
        evaluations_dir: str,
        evaluation_name: str,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        repo_base_dir: str | None = None,
        report_mode: str | None = None,
        num_workers: int = 1,
        use_testbed: bool = False,
        completion_model: CompletionModel | None = None,
        agent: ActionAgent | None = None,
        max_iterations: int = 30,
        max_cost: float = 1.0,
        evaluate_results: bool = False
    ):
        if not completion_model and not agent:
            raise RuntimeError("Either completion_model or agent must be provided")

        self.evaluations_dir = evaluations_dir
        self.num_workers = num_workers
        self.report_mode = report_mode
        self.dataset_name = dataset_name
        self.evaluation_name = evaluation_name
        self.evaluate_results = evaluate_results

        self.use_testbed = use_testbed

        self.max_iterations = max_iterations
        self.max_cost = max_cost

        self.agent = agent
        self.completion_model = completion_model

        self.evaluation_dir = f"{evaluations_dir}/{evaluation_name}"
        logger.info(f"Evaluation directory: {self.evaluation_dir}")
        if not os.path.exists(self.evaluation_dir):
            os.makedirs(self.evaluation_dir)

        self.predictions_path = f"{self.evaluation_dir}/all_preds.jsonl"

        self.repo_base_dir = repo_base_dir or os.getenv("REPO_DIR", "/tmp/repos")

        completion_log_dir = f"{self.evaluation_dir}/completion_logs"
        if not os.path.exists(completion_log_dir):
            os.makedirs(completion_log_dir)
        litellm.callbacks = [LogHandler(completion_log_dir)]

        self.status_file = f"{self.evaluation_dir}/status_summary.json"
        self.event_file = f"{self.evaluation_dir}/event_log.json"
        self.file_lock = threading.Lock()
        self.statuses = defaultdict(dict)
        self.events = defaultdict(list)

    def update_status(self, instance_id: str, status: str):
        with self.file_lock:
            if instance_id not in self.statuses:
                self.statuses[instance_id] = {
                    "created": datetime.now().isoformat(),
                }

            self.statuses[instance_id].update(
                {"last_updated": datetime.now().isoformat(), "status": status}
            )
            self._save_statuses()

    def log_event(self, instance_id: str, event: str):
        with self.file_lock:
            self.events[instance_id].append(
                {"timestamp": datetime.now().isoformat(), "event": event}
            )
            self._save_events()

    def _save_statuses(self):
        with open(self.status_file, "w") as f:
            json.dump(self.statuses, f, indent=2)

    def _save_events(self):
        with open(self.event_file, "w") as f:
            json.dump(self.events, f, indent=2)

    def run_evaluation(
        self,
        split: str = "lite",
        instance_ids: list[str] | None = None,
        exclude_instance_ids: list[str] | None = None,
        repos: list[str] | None = None,
        ignore_repos: list[str] | None = None,
        min_resolved: Optional[int] = None,
        max_resolved: Optional[int] = None,
    ):
        file_path = os.path.join(
            os.path.dirname(__file__), f"swebench_{split}_all_evaluations.json"
        )
        with open(file_path) as f:
            instances = json.load(f)

        random.shuffle(instances)

        logger.info(f"Loaded {len(instances)} instances from {file_path}")

        if instance_ids:
            instances = [
                instance
                for instance in instances
                if instance["instance_id"] in instance_ids
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by instance_ids"
            )

        if exclude_instance_ids:
            instances = [
                instance
                for instance in instances
                if instance["instance_id"] not in exclude_instance_ids
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by exclude_instance_ids"
            )

        if min_resolved is not None:
            instances = [
                instance
                for instance in instances
                if len(instance["resolved_by"]) >= min_resolved
                or (
                    min_resolved == 1
                    and instance.get("llm_monkeys", {}).get("resolved_rate", 0) > 0
                )
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by min_resolved >= {min_resolved}"
            )

        if max_resolved is not None:
            instances = [
                instance
                for instance in instances
                if len(instance["resolved_by"]) <= max_resolved
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by max_resolved <= {max_resolved}"
            )

        if repos:
            instances = [
                instance for instance in instances if instance["repo"] in repos
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by repos"
            )

        if ignore_repos:
            instances = [
                instance
                for instance in instances
                if instance["repo"] not in ignore_repos
            ]

            if instances:
                logger.info(
                    f"Running evaluation for {len(instances)} instances after filtering by ignore_repos"
                )

        return self._run_evaluation(instances)

    def evaluate_instance(self, instance: dict):
        instance_id = instance["instance_id"]
        instance_dir = os.path.join(self.evaluation_dir, f"{instance_id}")
        trajectory_path = os.path.join(instance_dir, "trajectory.json")

        if not os.path.exists(self.evaluation_dir):
            os.makedirs(trajectory_path)

        log_dir = os.path.join(instance_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        eval_result_path = os.path.join(instance_dir, "eval_result.json")
        if os.path.exists(eval_result_path):
            try:
                with open(eval_result_path) as f:
                    eval_result = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse eval result from {eval_result_path}. Will remove file to start over. Error: {e}"
                )
                os.remove(eval_result_path)
                eval_result = {
                    "node_results": {},
                }
        else:
            eval_result = {
                "node_results": {},
            }

        logger.info(f"Evaluating {instance_id}")
        problem_statement = f"<task>\n{instance['problem_statement']}\n</task>"

        runtime = None
        repository = None

        self.update_status(instance_id, "started")
        self.log_event(instance_id, "evaluate_instance_initiated")

        try:
            loop = None

            if os.path.exists(trajectory_path):
                try:
                    persisted_loop = AgenticLoop.from_file(trajectory_path)
                    if persisted_loop.is_finished():
                        logger.info(f"Found completed trajectory for {instance_id}")
                        loop = persisted_loop
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse trajectory from {trajectory_path}. Will remove file to start over. Error: {e}"
                    )
                    os.remove(trajectory_path)

            if not loop:
                self.log_event(instance_id, "workspace_created")

                metadata: dict[str, Any] = {
                    "evaluation_name": self.evaluation_name,
                    "instance_id": instance["instance_id"],
                }

                repository = create_repository(
                    instance, repo_base_dir=self.repo_base_dir
                )
                code_index = create_index(instance, repository=repository)

                if self.use_testbed:
                    from moatless.runtime.testbed import TestbedEnvironment

                    runtime = TestbedEnvironment(
                        repository=repository,
                        instance=instance,
                        log_dir=log_dir,
                        dataset_name=self.dataset_name,
                    )
                else:
                    runtime = None

                if os.path.exists(trajectory_path):
                    loop = AgenticLoop.from_file(
                        trajectory_path,
                        repository=repository,
                        runtime=runtime,
                        code_index=code_index,
                    )
                else:
                    agent = self.agent or CodingAgent.create(
                        completion_model=self.completion_model,
                        repository=repository,
                        code_index=code_index,
                        runtime=runtime,
                    )

                    agent_role = f"""You are an autonomous AI assistant and a core member of the development team for the {instance["repo"]} project. As a senior developer on the team, you have deep knowledge of the codebase and best practices."""
                    agent.system_prompt = f"{agent_role}\n\n{agent.system_prompt}"

                    loop = AgenticLoop.create(
                        message=problem_statement,
                        repository=repository,
                        agent=agent,
                        max_iterations=self.max_iterations,
                        max_cost=self.max_cost,
                        metadata=metadata,
                        persist_path=trajectory_path,
                    )
                self.log_event(instance_id, "agent_loop_execution_started")

                if loop and "error" in eval_result:
                    del eval_result["error"]
                    with open(eval_result_path, "w") as f:
                        json.dump(eval_result, f, indent=2)

                loop.run()
                self.log_event(instance_id, "agent_loop_execution_completed")

            start_time = time.time()
            try:
                last_node = loop.get_last_node()
                if not last_node:
                    logger.error(f"No last node found for {instance_id}")
                    eval_result["status"] = "no_last_node"
                    return eval_result
                else:
                    patch = last_node.file_context.generate_git_patch()
                    if not patch:
                        logger.error(f"No patch generated for {instance_id} and last node {last_node.node_id}. File context: {last_node.file_context.model_dump()}")

                eval_result["status"] = "completed"
                if not patch:
                    logger.warning(f"No patch generated for {instance_id}")
                    eval_result["status"] = "no_patch"
                    return eval_result
                else:
                    self.save_prediction(instance_id, patch)

                    if not self.evaluate_results:
                        return eval_result

                    if "node_results" not in eval_result:
                        eval_result["node_results"] = {}

                    if str(last_node.node_id) in eval_result["node_results"]:
                        return eval_result

                    if self.use_testbed and patch:

                        if not runtime:
                            repository = create_repository(
                                instance, repo_base_dir=self.repo_base_dir
                            )
                            from testbeds.sdk import TestbedSDK
                            from moatless.runtime.testbed import TestbedEnvironment

                            runtime = TestbedEnvironment(
                                testbed_sdk=TestbedSDK(),
                                repository=repository,
                                instance=instance,
                                log_dir=log_dir,
                                enable_cache=True,
                            )

                            start_time = time.time()
                            result = runtime.evaluate(patch=patch)
                            if not result:
                                logger.error(f"Error in evaluating patch for {instance_id}")
                            else:
                                eval_result["node_results"][str(last_node.node_id)] = (
                                    result.model_dump()
                                )
                                eval_result["status"] = "resolved" if result.resolved else "failed"

            except Exception:
                eval_result["error"] = traceback.format_exc()
                eval_result["status"] = "error"
                logging.exception(f"Error in evaluation of {instance['instance_id']} ")
            finally:
                eval_result["duration"] = time.time() - start_time
                loop.persist(trajectory_path)

                with open(eval_result_path, "w") as f:
                    json.dump(eval_result, f, indent=2)
                self.log_event(instance_id, "evaluation_completed")
                self.update_status(instance_id, eval_result["status"])

            return eval_result

        except Exception:
            logger.exception(f"Error in processing instance {instance_id}")
            self.log_event(instance_id, "evaluation_error")
            self.update_status(instance_id, "error")
            return None

        finally:
            with open(eval_result_path, "w") as f:
                json.dump(eval_result, f, indent=2)

            # Clean up
            if repository:
                shutil.rmtree(repository.repo_dir, ignore_errors=True)

            del runtime
            del repository
            del loop
            gc.collect()

    def save_prediction(self, instance_id, submission):
        with self.file_lock:
            prediction = {
                "model_name_or_path": self.evaluation_name,
                "instance_id": instance_id,
                "model_patch": submission,
            }
            with open(self.predictions_path, "a") as file:
                json_string = json.dumps(prediction)
                file.write(json_string + "\n")

    def _run_evaluation(self, instances: list[dict]):
        error = 0

        with open(self.predictions_path, "w") as file:
            file.write("")

        results = []

        logger.info(
            f"Processing {len(instances)} instances with {self.num_workers} workers"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = [
                executor.submit(self.evaluate_instance, instance)
                for instance in instances
            ]

            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))

            for future in pbar:
                try:
                    result = future.result()
                except Exception:
                    error += 1
                    logger.exception("Error in processing instance")

        logger.info(f"Completed processing with {error} errors")
        self.update_status("all", "evaluation_completed")

    def read_trajectory(self, path) -> dict | None:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        else:
            return None

    def get_actions(self, trajectory: dict):
        actions = []
        for transition in trajectory["transitions"]:
            for action in transition["actions"]:
                actions.append(action["action"])
        return actions
