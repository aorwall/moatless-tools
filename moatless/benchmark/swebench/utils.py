import os

from datasets import load_dataset

from moatless.benchmark.utils import get_missing_files, get_missing_spans
from moatless.file_context import FileContext
from moatless.repository import FileRepository
from moatless.types import Reject
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


def to_find_result(trajectory: dict, instance: dict, workspace: Workspace) -> dict:
    info = trajectory["info"]

    expected_file = list(instance["expected_spans"].keys())[0]

    result = {
        "instance_id": info["instance_id"],
        "duration": info["duration"],
        "total_cost": info["total_cost"],
        "status": "not_found",
        "expected_file": expected_file,
        "steps": len(trajectory["steps"]),
        "context_window": 0,
        "query": 0,
        "code_snippet": 0,
        "class_name": 0,
        "function_name": 0,
    }

    if "error" in info:
        result["status"] = "error"

    if "resolved_by" in instance:
        result["resolved_by"] = len(instance["resolved_by"])

    for step in trajectory["steps"]:
        for action in step["actions"]:
            for field in action["input"].keys():
                if field in result:
                    result[field] += 1

            if action["name"] == Reject.name():
                result["status"] = "rejected"

    if trajectory.get("output") and trajectory["output"].get("files"):
        file_context = workspace.create_file_context()

        actual_spans = {}
        for span in trajectory["output"]["files"]:
            actual_spans[span["file_path"]] = span["span_ids"]
            file_context.add_spans_to_context(span["file_path"], span["span_ids"])

        result["context_window"] = file_context.context_size()

        missing_files = get_missing_files(instance["expected_spans"], actual_spans)
        if not missing_files:
            result["status"] = "file_found"
        else:
            print(
                f"{instance['instance_id']} failed. Expected {instance['expected_spans'].keys()}, but got {actual_spans.keys()}"
            )

        missing_spans = get_missing_spans(instance["expected_spans"], actual_spans)
        if not missing_spans:
            result["status"] = "gold_patch"
        elif "alternative_spans" in instance:
            for alternative_spans in instance["alternative_spans"]:
                missing_spans = get_missing_spans(
                    alternative_spans["spans"], actual_spans
                )
                if not missing_spans:
                    result["status"] = "alternative_patch"
                    break
    else:
        result["status"] = "rejected"

    return result


def generate_md_report(trajectory: dict, instance: dict):
    info = trajectory["info"]
    markdown = f"# {info['instance_id']}\n"

    markdown += f"## Problem statement\n"
    markdown += f"```\n{instance['problem_statement']}\n```\n"

    if "error" in trajectory["info"]:
        markdown += f"## Error\n"
        markdown += f"```\n{trajectory['info']['error']}\n```\n"
    else:
        markdown += f"## Prediction\n"
        markdown += f"```diff\n{info['submission']}\n```\n"

    markdown += f"## Golden patch\n"
    markdown += f"```diff\n{instance['golden_patch']}\n```\n"

    markdown += f"## Trajectory\n"

    repo_dir = setup_swebench_repo(instance)
    file_repo = FileRepository(repo_dir)

    for step in trajectory["steps"]:
        for action in step["actions"]:
            markdown += f"### {action['name']}\n\n"

            if action["name"] == "code_finder":
                markdown += f"#### Instructions\n"
                markdown += action["input"]["instructions"]

                markdown += f"#### Output\n\n"

                if "message" in action["output"] and action["output"]["message"]:
                    markdown += action["output"]["message"] + "\n\n"

                if "files" in action["output"]:
                    file_context = FileContext(file_repo)

                    for file in action["output"]["files"]:
                        file_context.add_spans_to_context(
                            file["file_path"], file["span_ids"]
                        )

                    markdown += file_context.create_prompt(show_outcommented_code=True)

            if action["name"] == "request_for_change":
                markdown += f"\n* Description: {action['input']['description']}\n"
                markdown += f"* File: {action['input']['file_path']}\n"
                markdown += f"* Span: {action['input']['span_id']}\n"

                if "response" in action["output"]:
                    markdown += f"\n* Response: {action['output']['response']}\n"

            if action["name"] == "specify_lines":
                markdown += f"\n* Start Line: {action['input']['start_line']}\n"
                markdown += f"\n* End Line: {action['input']['end_line']}\n"
                if "response" in action["output"]:
                    markdown += f"\n* Response: {action['output']['response']}\n"

            if action["name"] == "search_replace":
                markdown += f"```python\n{action['input']['replacement_code']}\n```\n"

                if "diff" in action["output"]:
                    markdown += f"#### Diff\n"
                    markdown += f"```diff\n{action['output']['diff']}\n```\n"

            if action["name"] == "finish":
                markdown += action["input"]["reason"] + "\n\n"

            if action["name"] == "reject":
                markdown += action["input"]["reason"] + "\n\n"

            if action["name"] == "coding_task":
                markdown += f"#### Instructions\n"
                markdown += action["input"]["instructions"]

                markdown += f"\n\n * File: {action['input']['file_path']}"
                markdown += f"\n * Start line {action['input']['start_line']}"
                markdown += f"\n * End line {action['input'].get('end_line', '?')}"

                if "relevant_code" in action["input"]:
                    markdown += f"\n\n#### Relevant code\n"
                    file_context = FileContext(file_repo)

                    for file in action["input"]["relevant_code"]:
                        file_context.add_spans_to_context(
                            file["file_path"], file["span_ids"]
                        )

                    markdown += file_context.create_prompt(
                        show_outcommented_code=True, exclude_comments=True
                    )

                markdown += f"\n\n#### Coder writer steps\n\n"

                for i, code_step in enumerate(action["trajectory"]["steps"]):
                    markdown += f"\n\n##### Step {i+1}\n"
                    markdown += f"\n\n###### Search"
                    if "search_code" in code_step["input"]:
                        markdown += (
                            f"\n\n```python\n{code_step['input']['search_code']}\n```\n"
                        )
                    else:
                        markdown += "\n\n```python\nNone\n```\n"

                if "replace" in action["trajectory"]["output"]:
                    markdown += f"\n\n###### Replace"
                    markdown += f"\n\n```python\n{action["trajectory"]["output"]['replace']}\n```\n"

                if "message" in action["output"] and action["output"]["message"]:
                    markdown += f"\n#### Output\n\n"
                    markdown += action["output"]["message"] + "\n\n"

                if "diff" in action["output"]:
                    markdown += f"##### Diff\n"
                    markdown += f"```diff\n{action['output']['diff']}\n```\n"

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
