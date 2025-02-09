import fcntl
import logging
import os
import shutil
import subprocess
from typing import Optional
import asyncio

from moatless.benchmark.utils import (
    get_missing_files,
    get_missing_spans,
    get_moatless_instance,
)
from moatless.index import CodeIndex
from moatless.repository import GitRepository
from moatless.repository.repository import Repository
from moatless.utils.repo import (
    setup_github_repo,
    get_repo_dir_name,
    retry_clone,
    async_clone_and_checkout,
    run_git_command,
)

logger = logging.getLogger(__name__)


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
            logging.warning(f"{instance['instance_id']} Expected spans for {file_path} is empty")

    missing_spans = get_missing_spans(instance["expected_spans"], spans)
    return not missing_spans


def found_in_alternative_spans(instance: dict, spans: dict):
    if "alternative_spans" not in instance:
        return False
    for alternative_spans in instance["alternative_spans"]:
        for file_path, span_ids in alternative_spans["spans"].items():
            if not span_ids:
                logging.info(f"{instance['instance_id']} Alternative spans for {file_path} is empty")

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
                logging.info(f"{instance['instance_id']} Alternative spans for {file_path} is empty")

        missing_spans = get_missing_files(alternative_spans["spans"], files)
        if not missing_spans:
            return True

    return False


def setup_swebench_repo(
    instance_data: Optional[dict] = None,
    instance_id: str = None,
    repo_base_dir: Optional[str] = None,
) -> str:
    assert instance_data or instance_id, "Either instance_data or instance_id must be provided"
    if not instance_data:
        instance_data = get_moatless_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    repo_dir_name = instance_data["repo"].replace("/", "__")
    github_repo_path = f"swe-bench/{repo_dir_name}"
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data["base_commit"],
        base_dir=repo_base_dir,
    )


def instance_repo_path(instance_id: str, repo_base_dir: str | None = None) -> str:
    """Get the path to the repository for an instance."""
    if repo_base_dir is None:
        repo_base_dir = os.getenv("MOATLESS_REPO_DIR", "./repos")
    return os.path.join(repo_base_dir, f"swe-bench_{instance_id}")


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
        instance = get_moatless_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    # Convert to absolute path
    repo_base_dir = os.path.abspath(repo_base_dir)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(repo_base_dir), exist_ok=True)

    # Ensure the base directory exists
    os.makedirs(repo_base_dir, exist_ok=True)

    repo_dir_name = get_repo_dir_name(instance["repo"])
    local_repo_path = os.path.join(repo_base_dir, f"swe-bench_{repo_dir_name}")
    lock_file_path = f"{local_repo_path}.lock"

    # Ensure the directory for the lock file exists
    os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)

    repo_path = instance_repo_path(instance["instance_id"], repo_base_dir)
    if os.path.exists(repo_path):
        try:
            # Check if the commit exists in the repo
            import subprocess

            result = subprocess.run(
                ["git", "cat-file", "-e", instance["base_commit"]],
                cwd=repo_path,
                capture_output=True,
                check=True,
            )
            logger.info(f"Found existing repo with commit {instance['base_commit']} at {repo_path}")
            return GitRepository(repo_path=repo_path)
        except subprocess.CalledProcessError:
            logger.warning(f"Existing repo at {repo_path} doesn't have commit {instance['base_commit']}")
            shutil.rmtree(repo_path)
        except Exception as e:
            logging.warning(f"Error checking repository: {e}")
            shutil.rmtree(repo_path)

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

    # Use absolute path for file URL
    repo_url = f"file://{os.path.abspath(local_repo_path)}"

    return GitRepository.from_repo(git_repo_url=repo_url, repo_path=repo_path, commit=instance["base_commit"])


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


async def create_repository_async(
    instance: Optional[dict] = None,
    instance_id: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
):
    """
    Async version of create_repository for the given SWE-bench instance.
    """
    assert instance or instance_id, "Either instance or instance_id must be provided"
    if not instance:
        instance = get_moatless_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    # Convert to absolute and normalized path
    repo_base_dir = os.path.normpath(os.path.abspath(repo_base_dir))
    
    # Ensure directories exist
    os.makedirs(repo_base_dir, exist_ok=True)

    # Check repo directory exists and is writable
    if not os.path.exists(repo_base_dir):
        logger.error(f"Base repo directory {repo_base_dir} does not exist!")
        os.makedirs(repo_base_dir, exist_ok=True)
    
    if not os.access(repo_base_dir, os.W_OK):
        logger.error(f"Base repo directory {repo_base_dir} is not writable!")
        raise PermissionError(f"Cannot write to {repo_base_dir}")

    repo_dir_name = get_repo_dir_name(instance["repo"])
    central_repo_path = os.path.normpath(os.path.join(repo_base_dir, f"swe-bench_{repo_dir_name}"))
    lock_file_path = f"{central_repo_path}.lock"
    instance_repo_path = os.path.normpath(os.path.join(repo_base_dir, f"swe-bench_{instance['instance_id']}"))

    logger.info(f"Central repo path: {central_repo_path}")
    logger.info(f"Checking central repo: exists={os.path.exists(central_repo_path)}, is_dir={os.path.isdir(central_repo_path)}, is_link={os.path.islink(central_repo_path)}")
    if os.path.islink(central_repo_path):
        real_path = os.path.realpath(central_repo_path)
        logger.info(f"Central repo is symlink pointing to: {real_path}")
        central_repo_path = real_path

    # Check if instance repo already exists with correct commit
    if os.path.exists(instance_repo_path):
        try:
            result = await run_git_command(
                ["git", "cat-file", "-e", instance["base_commit"]],
                instance_repo_path
            )
            logger.info(f"Found existing repo with commit {instance['base_commit']} at {instance_repo_path}")
            return GitRepository(repo_path=instance_repo_path)
        except Exception:
            logger.warning(f"Existing repo at {instance_repo_path} invalid, removing")
            shutil.rmtree(instance_repo_path)

    # Lock and setup central repo if needed
    with open(lock_file_path, "w") as lock_file:
        logging.debug(f"Acquiring lock for {central_repo_path}")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not os.path.exists(central_repo_path):
                github_url = f"https://github.com/swe-bench/{repo_dir_name}.git"
                logger.info(f"Creating central repo with full clone from {github_url} to {central_repo_path}")
                await run_git_command(["git", "clone", github_url, central_repo_path])
                # await run_git_command(["git", "fetch", "--unshallow"], central_repo_path)
                logger.info(f"Central repo created at {central_repo_path}")
        finally:
            logging.debug(f"Releasing lock for {central_repo_path}")
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    # Create shallow clone from central repo for this instance
    repo_url = f"file://{os.path.abspath(central_repo_path)}"
    return await GitRepository.from_repo_async(
        git_repo_url=repo_url,
        repo_path=instance_repo_path,
        commit=instance["base_commit"]
    )
