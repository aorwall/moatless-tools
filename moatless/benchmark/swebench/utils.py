import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from datasets import load_dataset

from moatless.verify.testbed import TestbedVerifier
from testbed.client.client import TestbedClient

from moatless.benchmark.utils import (
    file_spans_to_dict,
    get_missing_files,
    get_missing_spans,
)
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository import FileRepository, GitRepository
from moatless.utils.repo import setup_github_repo
from moatless.workspace import Workspace


logger = logging.getLogger(__name__)




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
                logging.info(
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


def setup_swebench_repo(
    instance_data: Optional[dict] = None,
    instance_id: str = None,
    repo_base_dir: Optional[str] = None,
) -> str:
    assert (
        instance_data or instance_id
    ), "Either instance_data or instance_id must be provided"
    if not instance_data:
        instance_data = load_instance(instance_id)

    if not repo_base_dir:

        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    repo_dir_name = instance_data["repo"].replace("/", "__")
    github_repo_path = f"swe-bench/{repo_dir_name}"
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data["base_commit"],
        base_dir=repo_base_dir,
    )


def create_workspace(
    instance: Optional[dict] = None,
    instance_id: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
    initiate_index: bool = True,
    index_store_dir: Optional[str] = None,
    create_instance_dir: bool = False,
    testbed: Optional[TestbedClient] = None,
    max_file_context_tokens: int = 8000,
    use_perfect_file_context: bool = False,
    use_expected_test_files: bool = False,
):
    """
    Create a workspace for the given SWE-bench instance.
    """
    assert instance or instance_id, "Either instance or instance_id must be provided"
    if not instance:
        instance = load_instance(instance_id)

    if not index_store_dir:
        index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    repo_dir_name = instance["repo"].replace("/", "__")
    repo_url = f"https://github.com/swe-bench/{repo_dir_name}.git"

    if create_instance_dir:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        repo_dir = f"{repo_base_dir}/swe-bench_{instance['instance_id']}_{date_str}"
    else:
        repo_dir = f"{repo_base_dir}/{repo_dir_name}"

    repo = GitRepository.from_repo(
        git_repo_url=repo_url, repo_path=repo_dir, commit=instance["base_commit"]
    )

    if initiate_index:
        code_index = CodeIndex.from_index_name(
            instance["instance_id"], index_store_dir=index_store_dir, file_repo=repo
        )
    else:
        code_index = None

    if testbed:
        verifier = TestbedVerifier(
            testbed=testbed, repository=repo
        )
    else:
        verifier = None

    workspace = Workspace(
        file_repo=repo,
        verifier=verifier,
        code_index=code_index,
        max_file_context_tokens=max_file_context_tokens,
    )

    if use_perfect_file_context and "expected_spans" in instance:
        for file_path, span_ids in instance["expected_spans"].items():
            workspace.file_context.add_spans_to_context(file_path, set(span_ids))

        for resolved_by in instance.get("resolved_by", []):
            if "alternative_spans" in resolved_by:
                for file_path, span_ids in resolved_by["alternative_spans"].items():
                    workspace.file_context.add_spans_to_context(file_path, set(span_ids))

        if "alternative_spans" in instance:
            for alternative_spans in instance["alternative_spans"]:
                for file_path, span_ids in alternative_spans["spans"].items():
                    workspace.file_context.add_spans_to_context(file_path, set(span_ids))

        if use_expected_test_files and "test_file_spans" in instance:
            for file_path, span_ids in instance["test_file_spans"].items():
                workspace.file_context.add_spans_to_context(file_path, set(span_ids))

        workspace.file_context.expand_context_with_related_spans(1000)
        workspace.file_context.expand_classes(500)

    return workspace

def create_index(
    instance: Optional[dict] = None,
    instance_id: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
    index_store_dir: Optional[str] = None,
):
    """
    Create a workspace for the given SWE-bench instance.
    """
    assert instance or instance_id, "Either instance or instance_id must be provided"
    if not instance:
        instance = load_instance(instance_id)

    if not index_store_dir:
        index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    repo_dir_name = instance["repo"].replace("/", "__")

    repo_dir = f"{repo_base_dir}/{repo_dir_name}"
    repo = FileRepository(repo_dir)

    return CodeIndex.from_index_name(
        instance["instance_id"], index_store_dir=index_store_dir, file_repo=repo
    )
