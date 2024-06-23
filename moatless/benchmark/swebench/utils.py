import logging
import os

from datasets import load_dataset

from moatless.benchmark.utils import (
    get_missing_spans,
    file_spans_to_dict,
    get_missing_files,
)
from moatless.file_context import FileContext
from moatless.repository import FileRepository
from moatless.utils.repo import setup_github_repo
from moatless.workspace import Workspace


def load_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite", split: str = "test"
):
    data = load_dataset(dataset_name, split=split)
    return {d["instance_id"]: d for d in data}


def load_instance(
    instance_id: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
):
    data = load_instances(dataset_name, split=split)
    return data[instance_id]


def sorted_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    sort_by: str = "created_at",
):
    data = load_dataset(dataset_name, split=split)
    instances = list(data)
    instances = sorted(instances, key=lambda x: x[sort_by])
    return instances


def get_repo_dir_name(repo: str):
    return repo.replace("/", "_")


def found_in_expected_spans(instance: dict, spans: dict):
    for file_path, span_ids in instance["expected_spans"].items():
        if not span_ids:
            logging.warning(
                f"{instance['instance_id']} Expected spans for {file_path} is empty"
            )
    missing_spans = get_missing_spans(instance["expected_spans"], spans)
    return not missing_spans


def found_in_alternative_spans(instance: dict, spans: dict):
    if "alternative_spans" not in instance:
        return False
    for alternative_spans in instance["alternative_spans"]:
        for file_path, span_ids in alternative_spans["spans"].items():
            if not span_ids:
                logging.warning(
                    f"{instance['instance_id']} Alternative spans for {file_path} is empty"
                )

        missing_spans = get_missing_spans(alternative_spans["spans"], spans)
        if not missing_spans:
            return True

    return False


def sync_file_context_with_search_trajectory(workspace: Workspace, trajectory: dict):
    for transition in trajectory["transitions"]:
        for action in transition["actions"]:
            if action["action"].get("identified_spans"):
                for span in action["action"]["identified_spans"]:
                    workspace.file_context.add_spans_to_context(
                        span["file_path"], span["span_ids"]
                    )


def verify_search_trajectory(
    trajectory: dict, instance: dict, workspace: Workspace
) -> dict:
    result = {
        "transitions": len(trajectory["transitions"]),
        "identifieed": None,
        "expected_identified": None,
        "alt_identified": None,
        "identified": None,
        "file_identified": None,
        "found_in_search": None,
        "tokens": 0,
        "expanded_imports": False,
        "expanded_related": False,
        "expanded_small_classes": False,
        "expanded_tokens": 0,
    }

    file_context = workspace.create_file_context()
    search_file_context = workspace.create_file_context()

    iterations = 0
    for transition in trajectory["transitions"]:

        if transition["name"] == "SearchCode":
            iterations += 1

        for action in transition["actions"]:
            if (
                "output" in action
                and action.get("output")
                and action["output"].get("ranked_spans")
            ):
                for ranked_span in action["output"]["ranked_spans"]:
                    search_file_context.add_spans_to_context(
                        ranked_span["file_path"], [ranked_span["span_id"]]
                    )

            if action["action"].get("identified_spans"):
                for span in action["action"]["identified_spans"]:
                    file_context.add_spans_to_context(
                        span["file_path"], span["span_ids"]
                    )

            if result["found_in_search"] is None and (
                found_in_expected_spans(
                    instance,
                    file_spans_to_dict(search_file_context.to_files_with_spans()),
                )
                or found_in_alternative_spans(
                    instance, file_spans_to_dict(file_context.to_files_with_spans())
                )
            ):
                result["found_in_search"] = iterations

            if result["file_identified"] is None:
                missing_files = get_missing_files(
                    instance["expected_spans"],
                    file_spans_to_dict(file_context.to_files_with_spans()),
                )
                if not missing_files:
                    result["file_identified"] = iterations

            if result["expected_identified"] is None and found_in_expected_spans(
                instance, file_spans_to_dict(file_context.to_files_with_spans())
            ):
                result["expected_identified"] = iterations

            if result["alt_identified"] is None and found_in_alternative_spans(
                instance, file_spans_to_dict(file_context.to_files_with_spans())
            ):
                result["alt_identified"] = iterations

    if result["expected_identified"] is not None:
        result["identified"] = result["expected_identified"]

    if result["alt_identified"] is not None and (
        result["identified"] is None or result["alt_identified"] < result["identified"]
    ):
        result["identified"] = result["alt_identified"]

    result["tokens"] = file_context.context_size()

    file_context.expand_context_with_init_spans()
    actual_span_dicts = file_spans_to_dict(file_context.to_files_with_spans())

    if found_in_expected_spans(
        instance, actual_span_dicts
    ) or found_in_alternative_spans(instance, actual_span_dicts):
        result["expanded_imports"] = True

    file_context.expand_context_with_related_spans(max_tokens=8000)
    if found_in_expected_spans(
        instance, file_spans_to_dict(file_context.to_files_with_spans())
    ) or found_in_alternative_spans(
        instance, file_spans_to_dict(file_context.to_files_with_spans())
    ):
        result["expanded_related"] = True

    file_context.expand_small_classes(max_tokens=500)
    if found_in_expected_spans(
        instance, file_spans_to_dict(file_context.to_files_with_spans())
    ) or found_in_alternative_spans(
        instance, file_spans_to_dict(file_context.to_files_with_spans())
    ):
        result["expanded_small_classes"] = True

    result["expanded_tokens"] = file_context.context_size()

    result["iterations"] = iterations
    return result


def generate_md_report(trajectory: dict, instance: dict):
    info = trajectory["info"]
    markdown = f"# {info['instance_id']}\n"

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

    for step in trajectory["transitions"]:

        for i, action in enumerate(step["actions"]):
            markdown += f"### {step['name']} ({i})\n\n"

            if step["name"] == "PlanToCode":
                if action.get("action").get("thoughts"):
                    markdown += "*" + action["action"]["thoughts"] + "*"

                if action.get("action", {}).get("action", {}).get("description"):
                    markdown += f"\n\n * {action['action']['action']['description']}"

                if action.get("action", {}).get("action", {}).get("file_path"):
                    markdown += f"\n * {action['action']['action']['file_path']}"

                if action.get("action", {}).get("action", {}).get("span_id"):
                    markdown += f"\n * {action['action']['action']['span_id']}"

                    markdown += f"\n\n#### File context \n\n"

                    file_context = FileContext(file_repo)
                    file_context.add_span_to_context(
                        action["action"]["action"]["file_path"],
                        action["action"]["action"]["span_id"],
                    )

                    markdown += file_context.create_prompt(show_outcommented_code=True)

            if step["name"] == "EditCode":
                markdown += f"#### LLM Response\n\n"
                markdown += f"```\n{action['action']['content']}\n```\n"

                if action.get("output", {}).get("message"):
                    markdown += f"#### Output\n\n"
                    markdown += f"{action['output']['message']}\n\n"

            if step["name"] == "ClarifyCodeChange":
                if action.get("thoughts"):
                    markdown += "*" + action["thoughts"] + "*"

                if action.get("output", {}).get("start_line"):
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


def setup_swebench_repo(instance_data: dict, repo_base_dir: str = "/tmp/repos") -> str:
    repo_dir_name = instance_data["repo"].replace("/", "__")
    github_repo_path = f"swe-bench/{repo_dir_name}"
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data["base_commit"],
        base_dir=repo_base_dir,
    )


def create_workspace(
    instance: dict,
    repo_base_dir: str = "/tmp/repos",
    index_store_dir: str = "/tmp/index_store",
):
    repo_dir = setup_swebench_repo(instance, repo_base_dir=repo_base_dir)
    persist_dir = os.path.join(
        index_store_dir, get_repo_dir_name(instance["instance_id"])
    )
    return Workspace.from_dirs(repo_dir=repo_dir, index_dir=persist_dir)
