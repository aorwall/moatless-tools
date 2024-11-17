import logging
import os
import random
import subprocess
import time

logger = logging.getLogger(__name__)


def retry_clone(repo_url, repo_dir, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            logger.info(f"Cloning {repo_url} to {repo_dir} (attempt {attempt + 1})")
            result = subprocess.run(
                ["git", "clone", repo_url, repo_dir],
                check=True,
                text=True,
                capture_output=True,
            )
            logger.info(f"Cloned {repo_url} to {repo_dir}. Output: {result.stdout}")
            return
        except subprocess.CalledProcessError as e:
            logger.error(f"Clone attempt {attempt + 1} failed: {e.stderr}")
            if attempt < max_attempts - 1:
                if "Connection reset by peer" in e.stderr or "early EOF" in e.stderr:
                    wait_time = (2**attempt) + (random.randint(0, 1000) / 1000)
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise  # Don't retry for other types of errors
            else:
                raise  # Raise the error on the last attempt


def setup_github_repo(repo: str, base_commit: str, base_dir: str = "/tmp/repos") -> str:
    repo_name = get_repo_dir_name(repo)
    repo_url = f"https://github.com/{repo}.git"
    path = f"{base_dir}/{repo_name}"
    logger.info(
        f"Clone Github repo {repo_url} to {path} and checkout commit {base_commit}"
    )
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory '{path}' was created.")
    maybe_clone(repo_url, path)
    checkout_commit(path, base_commit)
    return path


def get_repo_dir_name(repo: str):
    return repo.replace("/", "_")


def clone_and_checkout(repo_url, repo_dir, commit):
    if os.path.exists(f"{repo_dir}/.git"):
        subprocess.run(
            ["rm", "-rf", repo_dir],
            check=True,
            text=True,
            capture_output=True,
        )

    # Ensure the URL is in the correct format for anonymous access
    if repo_url.startswith("https://github.com/"):
        repo_url = repo_url.replace("https://github.com/", "https://github.com/")

    try:
        logger.info(
            f"Attempting shallow clone of {repo_url} at commit {commit} to {repo_dir}"
        )
        subprocess.run(
            ["git", "clone", "--depth", "1", "--no-single-branch", repo_url, repo_dir],
            check=True,
            text=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", commit],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", commit],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
        logger.info(
            f"Successfully cloned {repo_url} and checked out commit {commit} in {repo_dir}"
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"Shallow clone failed, attempting full clone: {e.stderr}")
        subprocess.run(
            ["git", "clone", repo_url, repo_dir],
            check=True,
            text=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", commit],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
        logger.info(
            f"Successfully cloned {repo_url} and checked out commit {commit} in {repo_dir}"
        )


def maybe_clone(repo_url, repo_dir):
    if not os.path.exists(f"{repo_dir}/.git"):
        logger.info(f"Cloning repo '{repo_url}'")
        try:
            retry_clone(repo_url, repo_dir)
        except Exception as e:
            logger.error(f"Clone failed after multiple attempts: {e}")
            raise ValueError(f"Failed to clone repo '{repo_url}' to '{repo_dir}'")
        logger.info(f"Repo '{repo_url}' was cloned to '{repo_dir}'")


def pull_latest(repo_dir):
    subprocess.run(
        ["git", "pull"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def clean_and_reset_state(repo_dir):
    subprocess.run(
        ["git", "clean", "-fd"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "reset", "--hard"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def create_branch(repo_dir, branch_name):
    try:
        subprocess.run(
            ["git", "branch", branch_name],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def create_and_checkout_branch(repo_dir, branch_name):
    try:
        branches = subprocess.run(
            ["git", "branch"],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.split("\n")
        branches = [branch.strip() for branch in branches]
        if branch_name in branches:
            subprocess.run(
                ["git", "checkout", branch_name],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def commit_changes(repo_dir, commit_message):
    subprocess.run(
        ["git", "commit", "-m", commit_message, "--no-verify"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def checkout_branch(repo_dir, branch_name):
    subprocess.run(
        ["git", "checkout", branch_name],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def push_branch(repo_dir, branch_name):
    subprocess.run(
        ["git", "push", "origin", branch_name, "--no-verify"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def get_diff(repo_dir):
    output = subprocess.run(
        ["git", "diff"], cwd=repo_dir, check=True, text=True, capture_output=True
    )

    return output.stdout


def stage_all_files(repo_dir):
    subprocess.run(
        ["git", "add", "."], cwd=repo_dir, check=True, text=True, capture_output=True
    )


def checkout_commit(repo_dir, commit_hash):
    logger.info(f"Checking out commit {commit_hash} in {repo_dir}")
    try:
        subprocess.run(
            ["git", "reset", "--hard", commit_hash],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def create_and_checkout_new_branch(repo_dir: str, branch_name: str):
    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def setup_repo(repo_url, repo_dir, branch_name="master"):
    maybe_clone(repo_url, repo_dir)
    clean_and_reset_state(repo_dir)
    checkout_branch(repo_dir, branch_name)
    pull_latest(repo_dir)


def clean_and_reset_repo(repo_dir, branch_name="master"):
    clean_and_reset_state(repo_dir)
    checkout_branch(repo_dir, branch_name)
    pull_latest(repo_dir)
