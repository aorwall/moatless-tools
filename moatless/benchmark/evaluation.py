import concurrent.futures
import datetime
import json
import logging
import os
import subprocess
import time
import traceback
from collections import defaultdict
from typing import Optional, Tuple

import instructor
import litellm
import pandas as pd
from tqdm.auto import tqdm

from moatless import Workspace, FileRepository
from moatless.benchmark.swebench import (
    setup_swebench_repo,
    get_repo_dir_name,
    found_in_expected_spans,
    found_in_alternative_spans,
    sorted_instances,
    load_instance,
)
from moatless.benchmark.utils import (
    trace_metadata,
    get_missing_files,
)
from moatless.file_context import FileContext
from moatless.loop import Transitions, AgenticLoop

logger = logging.getLogger("Evaluator")

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
        transitions: Transitions,
        instructor_mode: Optional[instructor.Mode] = None,
        max_cost: float = 0.5,
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
        self.instructor_mode = instructor_mode

        self.transitions = transitions

        litellm.drop_params = True

        self.evaluation_dir = f"{evaluations_dir}/{evaluation_name}"
        self.trajectory_dir = f"{self.evaluations_dir}/{evaluation_name}/trajs"
        self.logs_dir = f"{self.evaluations_dir}/{evaluation_name}/prompt_logs"
        self.predictions_path = f"{self.evaluation_dir}/all_preds.jsonl"

        self.previous_trajectory_dir = previous_trajectory_dir
        self.retry_state = retry_state

        if not os.path.exists(self.trajectory_dir):
            os.makedirs(self.trajectory_dir)

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        if litellm_callback:
            litellm.success_callback = [litellm_callback]
            litellm.failure_callback = [litellm_callback]

        # This is only to set instances as resolved after all evaluations have been run to generate the report
        # TODO: Run swe-bench-docker after the prediction is generated
        result_file = f"{self.evaluation_dir}/result.json"
        if os.path.exists(result_file):
            with open(os.path.join(result_file), "r") as f:
                self.report = json.load(f)
        else:
            self.report = {"resolved": []}

    def run_evaluation_with_moatless_dataset(
        self,
        resolved_by: Optional[int] = None,
        use_test_subset: bool = False,
        instance_ids: Optional[list[str]] = None,
    ):
        file_path = os.path.join(
            os.path.dirname(__file__), "swebench_lite_all_evaluations.json"
        )
        with open(file_path, "r") as f:
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
        instance_ids: Optional[list[str]] = None,
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
            repo_dir=repo_dir, index_dir=persist_dir, max_file_context_tokens=16000
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
            transitions=self.transitions,
            workspace=workspace,
            metadata=metadata,
            mocked_actions=previous_actions,
            reset_mocks_at_state=self.retry_state,
            trajectory_path=trajectory_path,
            prompt_log_dir=prompt_log_dir,
            max_cost=self.max_cost,
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
        except Exception as e:
            info["error"] = traceback.format_exc()
            info["status"] = "error"
            logging.exception(f"Error in evaluation of {instance['instance_id']} ")

        info["duration"] = time.time() - start_time
        info["total_cost"] = loop.trajectory.total_cost()

        workspace.save()

        output = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        )

        info["submission"] = output.stdout
        loop.trajectory.save_info(info)
        return loop.trajectory.to_dict()

    def _process_instance(self, instance):
        trajectory = self._evaluate_instance(instance)
        if not trajectory:
            return None, None, None

        result, transition_result = self.to_result(instance, trajectory)
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
        except Exception as e:
            logging.exception(
                f"Error in generating report for {instance['instance_id']} "
            )

        return result, transition_result, submission

    def _process_repo_group(self, repo, instances):
        results = []
        transition_results = []
        for i, instance in enumerate(instances):
            print(
                f"Processing {instance['instance_id']} ({i+1}/{len(instances)} in {repo})"
            )

            trajectory = self._evaluate_instance(instance)
            if not trajectory:
                return None, None

            result, transition_result = self.to_result(instance, trajectory)
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
            except Exception as e:
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
                        print("Error in processing repo group")
                        error += 1
                        continue
                except Exception as e:
                    error += 1
                    logger.exception(f"Error in processing repo group")
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

                print(f"Average duration: {avg_duration:.2f} seconds")
                print(f"Average cost: ${avg_cost:.4f}")
                print(f"Total identified: {total_identified}")
                print(f"Total processed: {total_processed}")
                print(f"Error count: {error}")

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

            result, transition_result = self.to_result(instance, trajectory)

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

    def to_result(self, instance: dict, trajectory: dict) -> Tuple[list, list]:
        info = trajectory["info"]

        resolved = info.get("instance_id", "") in self.report["resolved"]

        try:
            transitions = []
            result = {
                "instance_id": instance["instance_id"],
                "duration": info.get("duration", 0),
                "total_cost": info.get("total_cost", 0),
                "resolved_by": (len(instance.get("resolved_by", []))),
                "status": None,
                "transitions": len(trajectory["transitions"]),
                "edited": False,
                "planned": False,
                "identified": None,
                "expected_identified": None,
                "alt_identified": None,
                "found_in_search": None,
                "file_identified": None,
                "file_in_search": None,
                "edit_retries": 0,
                "has_diff": False,
                "lint_codes": None,
                "review": False,
                "p_query": 0,
                "p_file": 0,
                "p_code": 0,
                "p_class": 0,
                "p_function": 0,
                "lints": "",
            }

            lint_codes = set()
            search_results_spans = {}
            identified_spans = {}
            planned_spans = {}
            edited_spans = {}

            id_iterations = 0
            search_iterations = 0

            if instance.get("expected_spans"):
                for transition in trajectory["transitions"]:

                    if transition["name"] not in result:
                        result[transition["name"]] = 0
                        result[f"{transition['name']}_cost"] = 0

                    result[transition["name"]] += 1

                    expected_span_str = ""
                    for file_path, span_ids in instance["expected_spans"].items():
                        expected_span_str += f"{file_path}: {span_ids} "

                    transition_result = {
                        "instance_id": instance["instance_id"],
                        "resolved": resolved,
                        "name": transition["name"],
                        "cost": 0,
                        "expected_spans": expected_span_str,
                        "actual_spans": "",
                    }

                    if not transition["actions"]:
                        continue

                    for traj_action in transition["actions"]:
                        result[f"{transition['name']}_cost"] += traj_action.get(
                            "completion_cost", 0
                        )
                        transition_result["cost"] += traj_action.get(
                            "completion_cost", 0
                        )

                    if transition["name"] == "SearchCode":
                        search_iterations += 1

                        action = transition["actions"][-1]

                        if "search_requests" in action["action"]:
                            for search_request in action["action"]["search_requests"]:
                                if search_request.get("query"):
                                    result["p_query"] += 1

                                if search_request.get("file_pattern"):
                                    result["p_file"] += 1

                                if search_request.get("code_snippet"):
                                    result["p_code"] += 1

                                if search_request.get(
                                    "class_name"
                                ) or search_request.get("class_names"):
                                    result["p_class"] += 1

                                if search_request.get(
                                    "function_name"
                                ) or search_request.get("function_names"):
                                    result["p_function"] += 1

                        if "output" in action and action.get("output"):
                            output = action["output"]

                            if "query" in output:
                                result["p_query"] += 1

                            if "file_pattern" in output:
                                result["p_file"] += 1

                            if "code_snippet" in output:
                                result["p_code"] += 1

                            if "class_name" in output or "class_names" in output:
                                result["p_class"] += 1

                            if "function_name" in output or "function_names" in output:
                                result["p_function"] += 1

                            if output.get("ranked_spans"):
                                for ranked_span in output["ranked_spans"]:
                                    if (
                                        ranked_span["file_path"]
                                        not in search_results_spans
                                    ):
                                        search_results_spans[
                                            ranked_span["file_path"]
                                        ] = []
                                    search_results_spans[
                                        ranked_span["file_path"]
                                    ].append(ranked_span["span_id"])

                                if not result["found_in_search"]:
                                    if found_in_expected_spans(
                                        instance, search_results_spans
                                    ) or found_in_alternative_spans(
                                        instance, search_results_spans
                                    ):
                                        result["found_in_search"] = search_iterations

                                if not result["file_in_search"]:
                                    missing_files = get_missing_files(
                                        instance["expected_spans"],
                                        search_results_spans,
                                    )
                                    if not missing_files:
                                        result["file_in_search"] = search_iterations

                    if transition["name"] == "IdentifyCode":
                        id_iterations += 1

                        action = transition["actions"][-1]
                        if action.get("action"):
                            identified_str = ""
                            if action["action"].get("identified_spans"):
                                for span in action["action"]["identified_spans"]:
                                    identified_str += (
                                        f"{span['file_path']}: {span['span_ids']} "
                                    )
                                    if span["file_path"] not in identified_spans:
                                        identified_spans[span["file_path"]] = []

                                    transition_result[
                                        "actual_spans"
                                    ] += f"{span['file_path']}: {','.join(span['span_ids'])} "
                                    for span_id in span["span_ids"]:
                                        identified_spans[span["file_path"]].append(
                                            span_id
                                        )
                            result["identified_spans"] = identified_str

                        if not result["file_identified"]:
                            missing_files = get_missing_files(
                                instance["expected_spans"],
                                identified_spans,
                            )
                            if not missing_files:
                                result["file_identified"] = id_iterations

                        if result[
                            "expected_identified"
                        ] is None and found_in_expected_spans(
                            instance, identified_spans
                        ):
                            result["expected_identified"] = id_iterations

                        if result[
                            "alt_identified"
                        ] is None and found_in_alternative_spans(
                            instance, identified_spans
                        ):
                            result["alt_identified"] = id_iterations

                        if result.get("alt_identified") or result.get(
                            "expected_identified"
                        ):
                            result["identified"] = min(
                                result.get("alt_identified") or 1000,
                                result.get("expected_identified") or 1000,
                            )

                    if transition["name"] == "PlanToCode":
                        action = transition["actions"][-1]["action"]
                        if action.get("action") == "review":
                            result["review"] = True

                        if "file_path" in action:
                            if "span_id" not in action:
                                print(
                                    f"Span id missing in planning action in {instance['instance_id']}"
                                )
                            else:
                                file_path = action["file_path"]
                                if file_path not in planned_spans:
                                    planned_spans[file_path] = []
                                planned_spans[file_path].append(action["span_id"])
                                transition_result["actual_spans"] = (
                                    f"{file_path}: {action['span_id']} "
                                )

                        if not result.get("planned") and (
                            found_in_expected_spans(
                                instance,
                                planned_spans,
                            )
                            or found_in_alternative_spans(instance, planned_spans)
                        ):
                            result["planned"] = True

                    if transition["name"] == "EditCode":
                        result["edit_retries"] = len(transition["actions"]) - 1

                        action = transition["actions"][-1]
                        output = action.get("output", {})

                        if output:
                            edited = output.get("diff")

                            if edited:
                                result["has_diff"] = True

                            for lint in output.get("verification_errors", []):
                                lint_codes.add(lint["code"])

                            if edited and "file_path" in transition["state"]:
                                file_path = transition["state"]["file_path"]
                                if file_path not in edited_spans:
                                    edited_spans[file_path] = []
                                edited_spans[file_path].append(
                                    transition["state"]["span_id"]
                                )
                                transition_result["actual_spans"] = (
                                    f"{file_path}: {transition['state']['span_id']} "
                                )

                            if not result.get("edited") and (
                                found_in_expected_spans(
                                    instance,
                                    edited_spans,
                                )
                                or found_in_alternative_spans(instance, edited_spans)
                            ):
                                result["edited"] = True

                    transitions.append(transition_result)

                if result.get("alt_identified") or result.get("expected_identified"):
                    result["identified"] = min(
                        result.get("alt_identified") or 1000,
                        result.get("expected_identified") or 1000,
                    )

                result["expected_files"] = list(instance["expected_spans"].keys())
                result["edited_files"] = list(edited_spans.keys())
                result["identified_spans"] = sum(
                    [len(v) for v in identified_spans.values()]
                )

            result["lints"] = ",".join(lint_codes)

            if info.get("instance_id", "") in self.report["resolved"]:
                result["status"] = "resolved"
            elif result["edited"]:
                result["status"] = "edited"
            elif result["identified"]:
                result["status"] = "identified"
            elif result["found_in_search"]:
                result["status"] = "found_in_search"
            elif result["file_identified"]:
                result["status"] = "file_identified"
            else:
                result["status"] = ""

            if "error" in info:
                result["error"] = info["error"].split("\n")[0]
            else:
                result["error"] = ""

        except Exception as e:
            raise e

        return result, transitions

    def read_trajectory(self, path) -> Optional[dict]:
        if os.path.exists(path):
            with open(path, "r") as f:
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
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    model_name = model.split("/")[-1]
    return f"{date_str}_{name}_{model_name}"


def generate_md_report(trajectory: dict, instance: dict):
    info = trajectory["info"]
    markdown = f"# {instance['instance_id']}\n"

    markdown += f"\n## Problem statement\n"
    markdown += f"```\n{instance['problem_statement']}\n```\n"

    if "error" in trajectory["info"]:
        markdown += f"\n## Error\n"
        markdown += f"```\n{trajectory['info']['error']}\n```\n"
    else:
        markdown += f"\n## Prediction\n"
        markdown += f"```diff\n{info['submission']}\n```\n"

    markdown += f"\n## Golden patch\n"
    markdown += f"```diff\n{instance['golden_patch']}\n```\n"

    markdown += f"\n## Trajectory\n"

    repo_dir = setup_swebench_repo(instance)
    file_repo = FileRepository(repo_dir)

    for j, step in enumerate(trajectory["transitions"]):

        for i, traj_action in enumerate(step["actions"]):
            markdown += f"### {j+1} {step['name']} ({i+1})\n\n"

            if not traj_action.get("action"):
                continue
            action = traj_action["action"]

            if step["name"] == "PlanToCode":
                if action.get("scratch_pad"):
                    markdown += "*" + action["scratch_pad"] + "*"

                if action.get("instructions"):
                    markdown += f"\n\n * {action['instructions']}"

                if action.get("file_path"):
                    markdown += f"\n * {action['file_path']}"

                if action.get("span_id"):
                    markdown += f"\n * {action['span_id']}"

                if action.get("file_path") and action.get("span_id"):
                    markdown += f"\n\n#### File context \n\n"
                    try:
                        file_context = FileContext(file_repo)
                        file_context.add_span_to_context(
                            action.get("file_path"),
                            action.get("span_id"),
                        )
                        markdown += file_context.create_prompt(
                            show_outcommented_code=True
                        )
                    except Exception as e:
                        print(e)

            if step["name"] == "EditCode":
                markdown += f"#### LLM Response\n\n"
                markdown += f"```\n{action.get('content', '')}\n```\n"

                output = traj_action.get("output")
                if output:
                    if output.get("diff"):
                        markdown += f"#### Diff\n\n"
                        markdown += f"```diff\n{output['diff']}\n```\n"

                    if output.get("errors"):
                        markdown += f"#### Errors\n\n"
                        markdown += f"{output['errors']}\n\n"

                    if output.get("message"):
                        markdown += f"#### Message\n\n"
                        markdown += f"{output['message']}\n\n"

            if step["name"] == "ClarifyCodeChange":
                if action.get("thoughts"):
                    markdown += "*" + action["thoughts"] + "*"

                if action.get("output") and action.get("output").get("start_line"):
                    markdown += f"\n* Start Line: {action['output']['start_line']}\n"
                    markdown += f"\n* End Line: {action['output']['end_line']}\n"

            if step["name"] == "Finished":
                markdown += f"*{action['properties']['message']}*\n"

            if step["name"] == "Rejected":
                markdown += f"*{action['properties']['message']}*\n"

    markdown += f"## Alternative patches\n"
    for alternative in instance["resolved_by"]:
        markdown += f"### {alternative['name']}\n"
        markdown += f"```diff\n{alternative['patch']}\n```\n"

    return markdown
