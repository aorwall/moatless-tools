import concurrent.futures
import json
import logging
import os
import subprocess
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import instructor
import litellm
import pandas as pd
from tqdm.auto import tqdm

from moatless.benchmark.report_v2 import to_result, generate_md_report
from moatless.transition_rules import TransitionRules
from moatless.benchmark.swebench import (
    found_in_alternative_spans,
    found_in_expected_spans,
    get_repo_dir_name,
    load_instance,
    setup_swebench_repo,
    sorted_instances,
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

TEST_SUBSET = [
    "astropy__astropy-14995",
    "django__django-10914",
    "django__django-11039",
    "django__django-11179",
    "django__django-12286",
    "django__django-12453",
    "django__django-12983",
    "django__django-13230",
    "django__django-13710",
    "django__django-13757",
    "django__django-14915",
    "django__django-14999",
    "django__django-15789",
    "matplotlib__matplotlib-23913",
    "matplotlib__matplotlib-23964",
    "pydata__xarray-5131",
    "pytest-dev__pytest-11143",
    "pytest-dev__pytest-5692",
    "pytest-dev__pytest-7373",
    "scikit-learn__scikit-learn-13142",
    "scikit-learn__scikit-learn-13241",
    "scikit-learn__scikit-learn-13439",
    "scikit-learn__scikit-learn-13496",
    "scikit-learn__scikit-learn-13779",
    "scikit-learn__scikit-learn-14894",
    "scikit-learn__scikit-learn-25570",
    "sympy__sympy-13480",
    "sympy__sympy-13647",
    "sympy__sympy-20212",
    "sympy__sympy-24213",
]


class Evaluation:
    def __init__(
        self,
        index_store_dir: str,
        repo_base_dir: str,
        evaluations_dir: str,
        evaluation_name: str,
        transitions: TransitionRules,
        instructor_mode: instructor.Mode | None = None,
        max_cost: float = 0.5,
        max_transitions: int = 25,
        max_expansions: int = 2,
        max_file_context_tokens: int = 16000,
        litellm_callback: Optional[str] = None,
        previous_trajectory_dir: Optional[str] = None,
        retry_state: Optional[str] = None,
        num_workers: int = 1,
        detailed_report: bool = False,
    ):
        self.index_store_dir = index_store_dir
        self.repo_base_dir = repo_base_dir
        self.evaluations_dir = evaluations_dir
        self.num_workers = num_workers
        self.detailed_report = detailed_report

        self.evaluation_name = evaluation_name
        self.max_file_context_tokens = max_file_context_tokens
        self.max_cost = max_cost
        self.max_expansions = max_expansions
        self.max_transitions = max_transitions
        self.instructor_mode = instructor_mode

        self.transitions = transitions

        litellm.drop_params = True

        self.evaluation_dir = f"{evaluations_dir}/{evaluation_name}"
        self.trajectory_dir = f"{self.evaluations_dir}/{evaluation_name}/trajs"
        self.logs_dir = f"{self.evaluations_dir}/{evaluation_name}/prompt_logs"
        self.predictions_path = f"{self.evaluation_dir}/all_preds.jsonl"

        self.previous_trajectory_dir = previous_trajectory_dir
        self.retry_state = retry_state

        logger.info(f"Save trajectories to directory: {self.trajectory_dir}")
        if not os.path.exists(self.trajectory_dir):
            os.makedirs(self.trajectory_dir)

        logger.info(f"Save logs to directory: {self.logs_dir}")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

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

    def run_evaluation_with_moatless_dataset(
        self,
        resolved_by: Optional[int] = None,
        use_test_subset: bool = False,
        instance_ids: list[str] | None = None,
    ):
        file_path = os.path.join(
            os.path.dirname(__file__), "swebench_lite_all_evaluations.json"
        )
        with open(file_path) as f:
            instances = json.load(f)

        instances = sorted(instances, key=lambda x: len(x["resolved_by"]), reverse=True)

        if use_test_subset:
            instances = [
                instance
                for instance in instances
                if instance["instance_id"] in TEST_SUBSET
            ]

        if instance_ids:
            instances = [
                instance
                for instance in instances
                if instance["instance_id"] in instance_ids
            ]

        if resolved_by:
            instances = [
                instance
                for instance in instances
                if len(instance["resolved_by"]) >= resolved_by
            ]

        return self._run_evaluation(instances)

    def run_swebench_evaluation(
        self,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split="test",
        instance_ids: list[str] | None = None,
    ):
        instances = sorted_instances(dataset, split)

        if instance_ids:
            instances = [
                instance
                for instance in instances
                if instance["instance_id"] in instance_ids
            ]

        return self._run_evaluation_simple(instances)

    def run_single_instance(
        self,
        instance_id: str,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split="test",
    ):
        instance = load_instance(instance_id, dataset, split)
        return self._evaluate_instance(instance)

    def _evaluate_instance(self, instance: dict, retry: bool = False) -> dict:
        instance_id = instance["instance_id"]
        trajectory_path = os.path.join(self.trajectory_dir, f"{instance_id}.json")
        prompt_log_dir = os.path.join(self.logs_dir, f"{instance_id}")
        if not os.path.exists(prompt_log_dir):
            os.makedirs(prompt_log_dir)

        if os.path.exists(trajectory_path) and not retry:
            with open(trajectory_path) as file:
                trajectory = json.load(file)
            if trajectory["info"].get("status") or trajectory["info"].get("error"):
                return trajectory

        repo_dir = setup_swebench_repo(instance)
        persist_dir = os.path.join(self.index_store_dir, get_repo_dir_name(instance_id))
        workspace = Workspace.from_dirs(
            repo_path=repo_dir, index_dir=persist_dir, max_file_context_tokens=16000
        )

        problem_statement = instance["problem_statement"]

        previous_actions = []
        if self.previous_trajectory_dir:
            previous_trajectory_path = os.path.join(
                self.previous_trajectory_dir, f"{instance_id}.json"
            )
            previous_trajectory = self.read_trajectory(previous_trajectory_path)
            if previous_trajectory:
                previous_actions = self.get_actions(previous_trajectory)

        metadata = trace_metadata(
            instance_id=instance_id,
            session_id=self.evaluation_name,
            trace_name="moatless",
        )

        loop = AgenticLoop(
            transition_rules=self.transitions,
            workspace=workspace,
            metadata=metadata,
            mocked_actions=previous_actions,
            reset_mocks_at_state=self.retry_state,
            trajectory_path=trajectory_path,
            prompt_log_dir=prompt_log_dir,
            max_cost=self.max_cost,
            max_transitions=self.max_transitions,
            max_actions=self.max_expansions,
            instructor_mode=self.instructor_mode,
        )

        info = {
            "evaluation_name": self.evaluation_name,
            "instance_id": instance["instance_id"],
        }

        start_time = time.time()
        try:
            response = loop.run(problem_statement)
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
                cwd=repo_dir,
            )

            if output:
                diff = output.stdout
            else:
                diff = None

        info["submission"] = diff

        loop.trajectory.save_info(info)
        return loop.trajectory.to_dict()

    def _process_instance(self, instance):
        trajectory = self._evaluate_instance(instance)
        if not trajectory:
            return None, None, None

        result, transition_result = to_result(instance, trajectory, self.report)
        submission = trajectory["info"].get("submission", "")

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

        return result, transition_result, submission

    def _process_repo_group(self, repo, instances):
        results = []
        transition_results = []
        for i, instance in enumerate(instances):
            logger.info(
                f"Processing {instance['instance_id']} ({i+1}/{len(instances)} in {repo})"
            )

            trajectory = self._evaluate_instance(instance)
            if not trajectory:
                return None, None

            result, transition_result = to_result(instance, trajectory, report=self.report)
            results.append(result)
            transition_results.extend(transition_result)

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
                "instance_id": result["instance_id"],
                "model_patch": trajectory["info"].get("submission", ""),
            }

            with open(self.predictions_path, "a") as file:
                json_string = json.dumps(prediction)
                file.write(json_string + "\n")

        return results, transition_results

    def _run_evaluation(self, instances: list[dict]):
        if self.detailed_report or self.num_workers > 1:
            self._run_evaluation_detailed(instances)
        else:
            self._run_evaluation_simple(instances)

    def _run_evaluation_detailed(self, instances: list[dict]):
        error = 0

        with open(self.predictions_path, "w") as file:
            file.write("")

        repo_groups = defaultdict(list)
        for instance in instances:
            repo_groups[instance.get("repo")].append(instance)

        results = []
        transition_results = []

        logger.info(f"Processing {len(instances)} instances with {len(repo_groups)} repos with {self.num_workers} workers")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = []
            for repo, group in repo_groups.items():
                futures.append(executor.submit(self._process_repo_group, repo, group))

            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))

            for future in pbar:
                try:
                    group_results, group_transition_results = future.result()
                    if not group_results:
                        logger.warning("Error in processing repo group")
                        error += 1
                        continue
                except Exception:
                    error += 1
                    logger.exception("Error in processing repo group")
                    continue

                results.extend(group_results)
                transition_results.extend(group_transition_results)

                df = pd.DataFrame(results)
                df.to_csv(
                    f"{self.evaluation_dir}/result.csv",
                    index=False,
                    sep=",",
                    decimal=",",
                    quoting=1,
                )

                avg_duration = df["duration"].mean()
                avg_cost = df["total_cost"].mean()
                total_identified = df["identified"].sum()
                total_processed = len(df)

                logger.info(f"Average duration: {avg_duration:.2f} seconds")
                logger.info(f"Average cost: ${avg_cost:.4f}")
                logger.info(f"Total identified: {total_identified}")
                logger.info(f"Total processed: {total_processed}")
                logger.info(f"Error count: {error}")

                if transition_results:
                    df_search = pd.DataFrame(transition_results)
                    df_search.to_csv(
                        f"{self.evaluation_dir}/transition_results.csv",
                        index=False,
                        sep=",",
                        decimal=",",
                        quoting=1,
                    )

    def _run_evaluation_simple(self, instances: list[dict]):
        with open(self.predictions_path, "w") as file:
            file.write("")

        count = 0
        identified = 0
        generated = 0
        error = 0

        sum_duration = 0
        sum_total_cost = 0

        stats = {}
        pbar = tqdm(instances)
        for instance in pbar:
            trajectory = self._evaluate_instance(instance)
            if not trajectory:
                continue

            result, transition_result = to_result(instance, trajectory, report=self.report)

            sum_duration += result["duration"]
            sum_total_cost += result["total_cost"]

            if result["status"] == "error":
                error += 1

            if result["status"] in ["generated", "failed", "resolved"]:
                generated += 1

            if result["identified"] is not None:
                identified += 1

            count += 1

            if sum_duration > 0:
                stats["avg_duration"] = sum_duration / count

            if sum_total_cost > 0:
                stats["avg_cost"] = sum_total_cost / count
                stats["total_cost"] = sum_total_cost

            if identified > 0:
                success_rate = (identified / count) * 100
                stats["identified"] = f"{success_rate:.2f}%"

            if generated > 0:
                success_rate = (generated / count) * 100
                stats["generated"] = f"{success_rate:.2f}%"

            stats["error"] = error

            pbar.set_postfix(stats)

            prediction = {
                "model_name_or_path": self.evaluation_name,
                "instance_id": instance["instance_id"],
                "model_patch": trajectory["info"].get("submission", ""),
            }

            with open(self.predictions_path, "a") as file:
                json_string = json.dumps(prediction)
                file.write(json_string + "\n")


    def read_trajectory(self, path) -> Optional[dict]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        else:
            return None

    def get_actions(self, trajectory: dict):
        actions = []
        for transition in trajectory["transitions"]:
            for action in transition["actions"]:
                actions.append(action)
        return actions


def create_evaluation_name(
    name: str,
    model: str,
):
    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    model_name = model.split("/")[-1]
    return f"{date_str}_{name}_{model_name}"

