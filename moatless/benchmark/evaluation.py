import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Tuple

import instructor
import litellm
import pandas as pd
from tqdm.auto import tqdm

from moatless.benchmark.report_v2 import to_result, generate_md_report, BenchmarkResult, to_dataframe
from moatless.trajectory import Trajectory
from moatless.transition_rules import TransitionRules
from moatless.benchmark.swebench import (
    found_in_alternative_spans,
    found_in_expected_spans,
    get_repo_dir_name,
    load_instance,
    setup_swebench_repo,
    sorted_instances,
    create_workspace,
)
from moatless.benchmark.utils import (
    get_missing_files,
    trace_metadata,
)
from moatless.file_context import FileContext
from moatless.loop import AgenticLoop
from moatless.repository import FileRepository, GitRepository
from moatless.workspace import Workspace


logger = logging.getLogger(__name__)


class Evaluation:
    def __init__(
        self,
        evaluations_dir: str,
        evaluation_name: str,
        transitions: TransitionRules,
        workspace: Workspace | None = None,
        report_mode: str | None = None,
        max_cost: float = 0.5,
        max_transitions: int = 25,
        prefill_file_context_tokens: int = 0,
        reward_threshold: Optional[float] = None,
        max_file_context_tokens: int = 16000,
        markdown_report: bool = False,
        litellm_callback: Optional[str] = None,
        previous_trajectory_dir: Optional[str] = None,
        retry_state: Optional[str] = None,
        num_workers: int = 1,
        **kwargs,
    ):
        self.evaluations_dir = evaluations_dir
        self.num_workers = num_workers
        self.markdown_report = markdown_report
        self.report_mode = report_mode

        self.prefill_file_context_tokens = prefill_file_context_tokens

        self.evaluation_name = evaluation_name
        self.max_file_context_tokens = max_file_context_tokens
        self.max_cost = max_cost
        self.max_transitions = max_transitions
        self.reward_threshold = reward_threshold

        self.transitions = transitions
        self.workspace = workspace

        litellm.drop_params = True

        self.evaluation_dir = f"{evaluations_dir}/{evaluation_name}"
        self.repo_base_dir = f"{self.evaluation_dir}/repos"
        if not os.path.exists(self.repo_base_dir):
            os.makedirs(self.repo_base_dir)
        self.predictions_path = f"{self.evaluation_dir}/all_preds.jsonl"
        logger.info(f"Evaluation directory: {self.evaluation_dir}")

        self.previous_trajectory_dir = previous_trajectory_dir
        logger.info(f"Previous trajectory directory: {self.previous_trajectory_dir}")
        self.retry_state = retry_state

        if litellm_callback:
            litellm.success_callback = [litellm_callback]
            litellm.failure_callback = [litellm_callback]

        # This is only to set instances as resolved after all evaluations have been run to generate the report
        # TODO: Run swe-bench-docker after the prediction is generated
        result_file = f"{self.evaluation_dir}/result.json"
        if os.path.exists(result_file):
            with open(os.path.join(result_file)) as f:
                self.report = json.load(f)
        else:
            self.report = {"resolved_ids": []}

    def run_evaluation(
        self,
        split: str = "lite",
        resolved_by: Optional[int] = None,
        instance_ids: list[str] | None = None,
    ):
        file_path = os.path.join(
            os.path.dirname(__file__), f"swebench_{split}_all_evaluations.json"
        )
        with open(file_path) as f:
            instances = json.load(f)

        instances = sorted(instances, key=lambda x: len(x["resolved_by"]), reverse=True)
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

        if resolved_by:
            instances = [
                instance
                for instance in instances
                if len(instance["resolved_by"]) >= resolved_by
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by resolved_by >= {resolved_by}"
            )

        return self._run_evaluation(instances)

    def run_single_instance(
        self,
        instance_id: str,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split="test",
    ) -> BenchmarkResult:
        instance = load_instance(instance_id, dataset, split)
        trajectory = self._evaluate_instance(instance)
        return to_result(instance, trajectory, self.report)

    def _evaluate_instance(self, instance: dict, retry: bool = False) -> Trajectory:
        instance_id = instance["instance_id"]

        trajectory_path = os.path.join(
            self.evaluation_dir, f"{instance_id}/trajectory.json"
        )
        if not os.path.exists(self.evaluation_dir):
            os.makedirs(trajectory_path)

        if os.path.exists(trajectory_path) and not retry:
            # TODO: Retry when failed or not finished?
            trajectory = Trajectory.load(trajectory_path, skip_workspace=True)
            if trajectory.info.get("status"):
                logger.info(
                    f"Skipping {instance_id} because it has already been evaluated with status {trajectory.info.get('status')}"
                )
                return trajectory

        problem_statement = instance["problem_statement"]

        workspace = create_workspace(instance, repo_base_dir=self.repo_base_dir, max_file_context_tokens=self.max_file_context_tokens)

        if self.prefill_file_context_tokens:
            results = workspace.code_index.semantic_search(query=instance["problem_statement"], max_results=1000)

            # Flatten and sort the search results
            flattened_results = []
            for hit in results.hits:
                for span in hit.spans:
                    flattened_results.append((hit.file_path, span.span_id, span.rank, span.tokens))

            # Sort by rank (ascending) and then by tokens (descending)
            flattened_results.sort(key=lambda x: (x[2], -x[3]))

            # Add spans to context in the new order
            for file_path, span_id, _, tokens in flattened_results:
                if tokens + workspace.file_context.context_size() > self.prefill_file_context_tokens:
                    break

                workspace.file_context.add_spans_to_context(file_path, [span_id])

        previous_actions = None
        if self.previous_trajectory_dir:
            previous_trajectory_path = os.path.join(
                self.previous_trajectory_dir, f"{instance_id}/trajectory.json"
            )
            previous_trajectory = Trajectory.load(previous_trajectory_path)
            previous_actions = previous_trajectory.get_mocked_actions()

        metadata = trace_metadata(
            instance_id=instance_id,
            session_id=self.evaluation_name,
            trace_name="moatless",
        )

        loop = AgenticLoop(
            transition_rules=self.transitions,
            initial_message=problem_statement,
            workspace=workspace,
            metadata=metadata,
            reset_mocks_at_state=self.retry_state,
            mocked_actions=previous_actions,
            continue_after_mocks=True,
            trajectory_path=trajectory_path,
            max_cost=self.max_cost,
            max_transitions=self.max_transitions,
        )

        info = {
            "evaluation_name": self.evaluation_name,
            "instance_id": instance["instance_id"],
        }
        loop.trajectory.save_info(info)

        start_time = time.time()
        try:
            response = loop.run()
            info["status"] = response.status
        except Exception:
            info["error"] = traceback.format_exc()
            info["status"] = "error"
            logging.exception(f"Error in evaluation of {instance['instance_id']} ")

        info["duration"] = time.time() - start_time
        info["total_cost"] = loop.total_cost()

        if isinstance(workspace.file_repo, GitRepository):
            diff = workspace.file_repo.diff()
        else:
            workspace.save()

            output = subprocess.run(
                ["git", "diff"],
                capture_output=True,
                text=True,
                cwd=workspace.file_repo.repo_dir,
            )

            if output:
                diff = output.stdout
            else:
                diff = None

        if diff and not diff.endswith("\n"):
            diff += "\n"

        info["submission"] = diff
        loop.trajectory.save_info(info)

        return loop.trajectory

    def _process_repo_group(self, repo: str, instances: list[dict]):
        logger.info(f"Processing {len(instances)} instances in {repo}")

        results = []
        for i, instance in enumerate(instances):
            logger.info(
                f"Processing {instance['instance_id']} ({i+1}/{len(instances)} in {repo})"
            )

            trajectory = self._evaluate_instance(instance)
            if not trajectory:
                return None, None

            result = to_result(instance, trajectory, self.report)
            results.append(result)

            if self.markdown_report:
                try:
                    md_report = generate_md_report(trajectory, instance)
                    if not os.path.exists(f"{self.evaluation_dir}/reports"):
                        os.makedirs(f"{self.evaluation_dir}/reports")
                    with open(
                        f"{self.evaluation_dir}/reports/{instance['instance_id']}.md",
                        "w",
                    ) as file:
                        file.write(md_report)
                except Exception:
                    logging.exception(
                        f"Error in generating report for {instance['instance_id']} "
                    )

            prediction = {
                "model_name_or_path": self.evaluation_name,
                "instance_id": instance["instance_id"],
                "model_patch": trajectory.info.get("submission", ""),
            }

            with open(self.predictions_path, "a") as file:
                json_string = json.dumps(prediction)
                file.write(json_string + "\n")

        return results

    def _to_csv_report(self, results: list[BenchmarkResult]):
        df = to_dataframe(self.report_mode, results)
        df.to_csv(
            f"{self.evaluation_dir}/result.csv",
            index=False,
            sep=",",
            decimal=",",
            quoting=1,
        )

    def _run_evaluation(self, instances: list[dict]):
        if not os.path.exists(self.evaluation_dir):
            os.makedirs(self.evaluation_dir)

        if self.num_workers > 1:
            self._run_evaluation_threads(instances)
        else:
            self._run_evaluation_simple(instances)

        #if self.repo_base_dir is in evaluations_dir
        if self.repo_base_dir in self.evaluations_dir:
            shutil.rmtree(self.repo_base_dir)

    def _run_evaluation_threads(self, instances: list[dict]):
        error = 0

        with open(self.predictions_path, "w") as file:
            file.write("")

        repo_groups = defaultdict(list)
        for instance in instances:
            repo_groups[instance.get("repo")].append(instance)

        results = []

        logger.info(
            f"Processing {len(instances)} instances with {len(repo_groups)} repos with {self.num_workers} workers"
        )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = []
            for repo, group in repo_groups.items():
                logger.info(json.dumps(group, indent=2))
                futures.append(executor.submit(self._process_repo_group, repo, group))

            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))

            for future in pbar:
                try:
                    group_results = future.result()
                    if not group_results:
                        logger.warning("Error in processing repo group")
                        error += 1
                        continue
                except Exception:
                    error += 1
                    logger.exception("Error in processing repo group")
                    continue

                results.extend(group_results)
                self._to_csv_report(results)

    def _run_evaluation_simple(self, instances: list[dict]):
        with open(self.predictions_path, "w") as file:
            file.write("")

        results = []
        stats = {}
        pbar = tqdm(instances)
        for instance in pbar:
            trajectory = self._evaluate_instance(instance)
            if not trajectory:
                continue

            result = to_result(instance, trajectory, report=self.report)
            results.append(result)
            self._to_csv_report(results)
            self._save_json_report(results)

            stats["avg_duration"] = sum(r.duration for r in results) / len(results)
            stats["avg_cost"] = sum(r.total_cost for r in results) / len(results)
            stats["total_cost"] = sum(r.total_cost for r in results)

            identified = sum(
                1
                for r in results
                if r.status in ["identified", "planned", "edited", "resolved"]
            )
            generated = sum(1 for r in results if r.status in ["edited", "resolved"])
            error = sum(1 for r in results if r.status == "error")

            if identified > 0:
                stats["identified"] = f"{(identified / len(results)) * 100:.2f}%"
            if generated > 0:
                stats["generated"] = f"{(generated / len(results)) * 100:.2f}%"
            stats["error"] = error

            pbar.set_postfix(stats)

            prediction = {
                "model_name_or_path": self.evaluation_name,
                "instance_id": instance["instance_id"],
                "model_patch": trajectory.info.get("submission", ""),
            }

            with open(self.predictions_path, "a") as file:
                json_string = json.dumps(prediction)
                file.write(json_string + "\n")

            if self.markdown_report:
                try:
                    md_report = generate_md_report(trajectory, instance)
                    if not os.path.exists(f"{self.evaluation_dir}/reports"):
                        os.makedirs(f"{self.evaluation_dir}/reports")
                    with open(
                        f"{self.evaluation_dir}/reports/{instance['instance_id']}.md",
                        "w",
                    ) as file:
                        file.write(md_report)
                except Exception:
                    logging.exception(
                        f"Error in generating report for {instance['instance_id']} "
                    )

    def _save_json_report(self, results: list[BenchmarkResult]):
        json_results = [result.model_dump() for result in results]
        with open(f"{self.evaluation_dir}/report.json", "w") as f:
            json.dump(json_results, f, indent=2)


def create_evaluation_name(
    name: str,
    model: str,
):
    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    model_name = model.split("/")[-1]
    return f"{date_str}_{name}_{model_name}"