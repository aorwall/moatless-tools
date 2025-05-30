import asyncio
import fcntl
import logging
import os
import shutil
import signal
from typing import Optional

import aiohttp
import requests

from moatless.evaluation.utils import (
    get_missing_files,
    get_missing_spans,
    get_moatless_instance,
)
from moatless.index import CodeIndex
from moatless.repository import GitRepository, FileRepository
from moatless.repository.repository import Repository
from moatless.utils.repo import (
    async_clone_and_checkout,
    get_repo_dir_name,
    retry_clone,
    run_git_command,
    setup_github_repo,
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
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")
    return os.path.join(repo_base_dir, f"swe-bench_{instance_id}")


def github_repo_exists_sync(url: str) -> bool:
    """Synchronous version to check if a GitHub repository exists."""
    try:
        # Strip .git suffix for API check
        api_url = url.replace("https://github.com/", "https://api.github.com/repos/").rstrip(".git")
        response = requests.head(api_url)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Error checking if repo exists: {e}")
        return False


def create_file_repository(
    instance: Optional[dict] = None,
    instance_id: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
):
    repo_path = instance_repo_path(instance["instance_id"], repo_base_dir)
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository {repo_path} does not exist")
    return FileRepository(repo_path=repo_path)


def create_repository(
    instance: Optional[dict] = None,
    instance_id: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
    use_local_repo: bool = False,
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
    github_url = f"https://github.com/swe-bench/{repo_dir_name}.git"

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

    # Try the swe-bench URL format first
    if not github_repo_exists_sync(github_url):
        logger.info(f"Repository {github_url} not found, trying direct GitHub URL")
        # Use the direct GitHub URL without checking if it exists
        github_url = f"https://github.com/{instance['repo']}.git"
        logger.info(f"Using direct GitHub URL: {github_url}")

    local_repo_path = os.path.join(repo_base_dir, f"swe-bench_{repo_dir_name}")
    lock_file_path = f"{local_repo_path}.lock"

    # Ensure the directory for the lock file exists
    os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)

    if use_local_repo:
        with open(lock_file_path, "w") as lock_file:
            logging.debug(f"Acquiring lock for {local_repo_path}")
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not os.path.exists(local_repo_path):
                # Clone from GitHub if local repo doesn't exist
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
    else:
        repo_url = github_url

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


async def create_index_async(
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
        repository = await create_repository_async(instance)

    code_index = await CodeIndex.from_index_name_async(
        instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
    )
    return code_index


def repository_exists(instance: dict, repo_base_dir: str):
    instance_repo_path = os.path.normpath(os.path.join(repo_base_dir, f"swe-bench_{instance['instance_id']}"))
    return os.path.exists(instance_repo_path)


async def github_repo_exists(url: str) -> bool:
    """Check if a GitHub repository exists by making a HEAD request."""
    try:
        # Strip .git suffix for API check
        api_url = url.replace("https://github.com/", "https://api.github.com/repos/").rstrip(".git")
        async with aiohttp.ClientSession() as session:
            async with session.head(api_url) as response:
                return response.status == 200
    except Exception as e:
        logger.warning(f"Error checking if repo exists: {e}")
        return False


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

    logger.info(f"Creating repository for instance {instance['instance_id']}")

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    repo_base_dir = os.path.normpath(os.path.abspath(repo_base_dir))
    os.makedirs(repo_base_dir, exist_ok=True)

    repo_dir_name = get_repo_dir_name(instance["repo"])
    github_url = f"https://github.com/swe-bench/{repo_dir_name}.git"

    # Try the swe-bench URL format first
    if not await github_repo_exists(github_url):
        logger.info(f"Repository {github_url} not found, trying direct GitHub URL")
        # Use the direct GitHub URL without checking if it exists
        github_url = f"https://github.com/{instance['repo']}.git"
        logger.info(f"Using direct GitHub URL: {github_url}")

    central_repo_path = os.path.normpath(os.path.join(repo_base_dir, f"swe-bench_{repo_dir_name}"))
    instance_repo_path = os.path.normpath(os.path.join(repo_base_dir, f"swe-bench_{instance['instance_id']}"))

    # First check if instance repo already exists and is valid
    if os.path.exists(instance_repo_path):
        logger.info(f"Checking if instance repo at {instance_repo_path} is valid")
        try:
            result = await run_git_command(["git", "cat-file", "-e", instance["base_commit"]], instance_repo_path)
            logger.info(f"Found existing valid repo at {instance_repo_path}")
            return GitRepository(repo_path=instance_repo_path)
        except Exception:
            logger.warning(f"Removing invalid repo at {instance_repo_path}")
            shutil.rmtree(instance_repo_path)

    logger.info(f"Setting up central repo at {central_repo_path}")

    # Handle both central repo setup and instance repo creation under the same lock
    # First handle central repo
    if not os.path.exists(central_repo_path):
        logger.info(f"Creating central repo at {central_repo_path}")
        # Use swe-bench organization URL format

        try:
            # Add timeout and process tracking
            clone_task = asyncio.create_task(
                run_git_command(["git", "clone", "--mirror", github_url, central_repo_path])
            )
            try:
                await asyncio.wait_for(clone_task, timeout=300)  # 5 minute timeout
            except asyncio.TimeoutError:
                logger.error("Git clone operation timed out after 5 minutes")
                # Get hanging processes
                ps_result = await run_git_command(["ps", "-ef", "|", "grep", "git-clone"], working_dir="/")
                logger.error(f"Hanging git processes: {ps_result}")
                raise

            fetch_task = asyncio.create_task(run_git_command(["git", "fetch", "--all"], central_repo_path))
            try:
                await asyncio.wait_for(fetch_task, timeout=300)
            except asyncio.TimeoutError:
                logger.error("Git fetch operation timed out after 5 minutes")
                ps_result = await run_git_command(["ps", "-ef", "|", "grep", "git-fetch"], working_dir="/")
                logger.error(f"Hanging git processes: {ps_result}")
                raise

        except Exception as e:
            logger.error(f"Failed to create central repo: {e}")
            if os.path.exists(central_repo_path):
                shutil.rmtree(central_repo_path)
            raise
    else:
        try:
            logger.info("Fetching all from central repo")
            await run_git_command(["git", "fetch", "--all"], central_repo_path)
        except Exception as e:
            logger.error(f"Central repo is invalid: {e}")
            shutil.rmtree(central_repo_path)
            raise

    # Now clone from central repo to instance path (still under the same lock)
    logger.info(f"Cloning from central repo {central_repo_path} to {instance_repo_path}")
    repo = None
    try:
        await async_clone_and_checkout(central_repo_path, instance_repo_path, instance["base_commit"])
        logger.info(f"Cloned from central repo {central_repo_path} to {instance_repo_path}")
        repo = GitRepository(repo_path=instance_repo_path, commit=instance["base_commit"])
    except Exception as e:
        logger.error(f"Failed to create instance repo: {e}")
        if os.path.exists(instance_repo_path):
            shutil.rmtree(instance_repo_path)
        raise

    # Return outside the async with block to ensure lock is released
    return repo


async def cleanup_hanging_git_processes(repo_path: str):
    """Force cleanup any hanging git processes for a specific repo."""
    try:
        # Find all git processes related to this repo
        ps_result = await run_git_command(["ps", "-ef", "|", "grep", repo_path, "|", "grep", "git"], working_dir="/")

        # Kill hanging processes
        for line in ps_result.splitlines():
            if "grep" not in line:  # Skip the grep process itself
                pid = line.split()[1]
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    logger.info(f"Terminated hanging git process {pid}")
                except ProcessLookupError:
                    pass  # Process already terminated

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
