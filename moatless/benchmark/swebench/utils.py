import fcntl
import logging
import os
from typing import Optional

from moatless.benchmark.utils import (
    get_missing_files,
    get_missing_spans,
)
from moatless.index import CodeIndex
from moatless.repository import GitRepository
from moatless.repository.repository import Repository
from moatless.utils.repo import (
    setup_github_repo,
    get_repo_dir_name,
    retry_clone,
)

logger = logging.getLogger(__name__)


def load_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite", split: str = "test"
):
    from datasets import load_dataset

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
    from datasets import load_dataset

    data = load_dataset(dataset_name, split=split)
    instances = list(data)
    instances = sorted(instances, key=lambda x: x[sort_by])
    return instances


def get_repo_dir_name(repo: str):
    return repo.replace("/", "__")


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


def found_in_alternative_files(instance: dict, files: list):
    if "alternative_spans" not in instance:
        return False
    for alternative_spans in instance["alternative_spans"]:
        for file_path, span_ids in alternative_spans["spans"].items():
            if not span_ids:
                logging.info(
                    f"{instance['instance_id']} Alternative spans for {file_path} is empty"
                )

        missing_spans = get_missing_files(alternative_spans["spans"], files)
        if not missing_spans:
            return True

    return False


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


def create_repository(
    instance: Optional[dict] = None,
    instance_id: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
):
    """
    Create a workspace for the given SWE-bench instance.
    """
    assert instance or instance_id, "Either instance or instance_id must be provided"
    if not instance:
        instance = load_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(repo_base_dir), exist_ok=True)

    # Ensure the base directory exists
    os.makedirs(repo_base_dir, exist_ok=True)

    repo_dir_name = get_repo_dir_name(instance["repo"])
    local_repo_path = f"{repo_base_dir}/swe-bench_{repo_dir_name}"
    lock_file_path = f"{local_repo_path}.lock"

    # Ensure the directory for the lock file exists
    os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)

    repo_path = f"{repo_base_dir}/swe-bench_{instance['instance_id']}"
    if os.path.exists(repo_path):
        try:
            logger.info(f"Initializing GitRepository from existing repo {repo_path}")
            return GitRepository(repo_path=repo_path)
        except Exception as e:
            logging.warning(f"Error initializing GitRepository: {e}")

    with open(lock_file_path, "w") as lock_file:
        logging.debug(f"Acquiring lock for {local_repo_path}")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not os.path.exists(local_repo_path):
            # Clone from GitHub if local repo doesn't exist
            github_url = f"https://github.com/swe-bench/{repo_dir_name}.git"
            try:
                retry_clone(github_url, local_repo_path)
                logging.info(f"Cloned {github_url} to {local_repo_path}")
            except Exception as e:
                logger.error(f"Failed to clone after multiple attempts: {e}")
                raise
        logging.debug(f"Releasing lock for {local_repo_path}")
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    repo_url = f"file://{local_repo_path}"

    return GitRepository.from_repo(
        git_repo_url=repo_url, repo_path=repo_path, commit=instance["base_commit"]
    )


def create_index(
    instance: dict,
    repository: Repository | None = None,
    index_store_dir: Optional[str] = None,
):
    """
    Create a workspace for the given SWE-bench instance.
    """
    if not index_store_dir:
        index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")

    if not repository:
        repository = create_repository(instance)

    code_index = CodeIndex.from_index_name(
        instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
    )
    return code_index
