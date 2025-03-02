import asyncio
import hashlib
import logging
import os
import random
import subprocess
import time
from contextlib import contextmanager, asynccontextmanager
from pathlib import Path

import filelock

logger = logging.getLogger(__name__)


def get_repo_lock_path(repo_url: str) -> str:
    """Get the lock file path for a repository URL."""
    url_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    lock_dir = Path("/tmp/repo_locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    return str(lock_dir / f"repo_{url_hash}.lock")


def get_repo_async_lock_path(repo_url: str) -> str:
    """Get the async lock file path for a repository URL."""
    url_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:16]
    lock_dir = Path("/tmp/repo_locks")
    lock_dir.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    return str(lock_dir / f"repo_{url_hash}_async_{pid}.lock")


@contextmanager
def repo_operation_lock(repo_url: str):
    """Context manager to ensure thread-safe git operations on a repository."""
    lock_path = get_repo_lock_path(repo_url)
    lock = filelock.FileLock(lock_path, timeout=60)  # 1 minute timeout for sync operations
    try:
        lock.acquire()
        yield
    finally:
        if lock.is_locked:
            lock.release()


@asynccontextmanager
async def repo_operation_async_lock(repo_url: str):
    """Async context manager for thread-safe git operations."""
    lock_path = get_repo_async_lock_path(repo_url)
    lock = filelock.FileLock(lock_path, timeout=300)  # Increase timeout to 5 minutes

    try:
        # Add exponential backoff retry logic
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                await asyncio.to_thread(lock.acquire)
                break
            except filelock.Timeout:
                if attempt == max_attempts - 1:
                    logger.error(f"Failed to acquire lock for {repo_url} after {max_attempts} attempts")
                    raise
                wait_time = (2**attempt) + (random.random())
                logger.warning(f"Lock acquisition failed, retrying in {wait_time:.1f}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
        yield
    except Exception as e:
        logger.error(f"Error during repo operation: {str(e)}")
        # Clean up stale lock if needed
        if os.path.exists(lock_path):
            try:
                if not lock.is_locked:
                    os.remove(lock_path)
                    logger.info(f"Cleaned up stale lock file: {lock_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up lock file: {cleanup_error}")
        raise
    finally:
        if lock.is_locked:
            try:
                await asyncio.to_thread(lock.release)
            except Exception as e:
                logger.error(f"Error releasing lock: {str(e)}")
                # Force remove lock file in case of release failure
                try:
                    os.remove(lock_path)
                    logger.info(f"Forcibly removed lock file: {lock_path}")
                except Exception:
                    pass


def setup_github_repo(repo: str, base_commit: str, base_dir: str = "/tmp/repos") -> str:
    repo_name = get_repo_dir_name(repo)
    repo_url = f"https://github.com/{repo}.git"
    path = f"{base_dir}/{repo_name}"
    logger.info(f"Clone Github repo {repo_url} to {path} and checkout commit {base_commit}")
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory '{path}' was created.")

    with repo_operation_lock(repo_url):
        maybe_clone(repo_url, path)
        checkout_commit(path, base_commit)
    return path


def get_repo_dir_name(repo: str):
    return repo.replace("/", "_")


def verify_commit_exists(repo_dir: str, commit: str) -> bool:
    """Check if a commit exists in the repository."""
    try:
        subprocess.run(
            ["git", "cat-file", "-e", commit],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, Exception):
        return False


def clone_and_checkout(repo_url, repo_dir, commit):
    with repo_operation_lock(repo_url):
        if os.path.exists(f"{repo_dir}/.git"):
            if verify_commit_exists(repo_dir, commit):
                subprocess.run(
                    ["git", "checkout", commit],
                    cwd=repo_dir,
                    check=True,
                    text=True,
                    capture_output=True,
                )
                logger.info(f"Found existing repo with commit {commit} at {repo_dir}")
                return
            else:
                logger.warning(f"Existing repo at {repo_dir} doesn't have commit {commit}, recloning")
                subprocess.run(
                    ["rm", "-rf", repo_dir],
                    check=True,
                    text=True,
                    capture_output=True,
                )

        try:
            logger.info(f"Attempting shallow clone of {repo_url} at commit {commit} to {repo_dir}")
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
            logger.info(f"Successfully cloned {repo_url} and checked out commit {commit} in {repo_dir}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Shallow clone failed, attempting full clone: {e.stderr}")
            if os.path.exists(f"{repo_dir}/.git") and verify_commit_exists(repo_dir, commit):
                subprocess.run(
                    ["git", "checkout", commit],
                    cwd=repo_dir,
                    check=True,
                    text=True,
                    capture_output=True,
                )
                logger.info(f"Another process created repo with commit {commit}, using that")
                return

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
            logger.info(f"Successfully cloned {repo_url} and checked out commit {commit} in {repo_dir}")


async def async_clone_and_checkout(repo_url, repo_dir, commit):
    """Clone and checkout a specific commit asynchronously."""
    if os.path.exists(f"{repo_dir}/.git"):
        if await asyncio.to_thread(verify_commit_exists, repo_dir, commit):
            logger.info(f"Found existing repo with commit {commit} at {repo_dir}")
            await run_git_command(["git", "checkout", commit], repo_dir)
            logger.info(f"Checked out commit {commit} in {repo_dir}")
            return
        else:
            logger.warning(f"Existing repo at {repo_dir} doesn't have commit {commit}, recloning")
            if os.path.exists(repo_dir):
                import shutil

                shutil.rmtree(repo_dir)

    try:
        if repo_url.startswith("file://"):
            logger.info(f"Starting shallow clone from local repo {repo_url} to {repo_dir}")
            await run_git_command(["git", "clone", "--depth", "1", "--no-single-branch", repo_url, repo_dir])
            await run_git_command(["git", "fetch", "origin", commit], repo_dir)
        else:
            logger.info(f"Starting full clone of {repo_url} to {repo_dir}")
            await async_retry_clone(repo_url, repo_dir)
            try:
                await run_git_command(["git", "fetch", "origin", commit], repo_dir)
            except subprocess.CalledProcessError:
                await run_git_command(["git", "fetch", "--unshallow"], repo_dir)

        await run_git_command(["git", "checkout", commit], repo_dir)

    except Exception as e:
        logger.error(f"Git operation failed: {e}")
        if os.path.exists(repo_dir):
            import shutil

            shutil.rmtree(repo_dir)
        raise


async def async_retry_clone(repo_url, repo_dir, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            logger.info(f"Cloning {repo_url} to {repo_dir} (attempt {attempt + 1})")
            result = await run_git_command(["git", "clone", repo_url, repo_dir])
            logger.info(f"Cloned {repo_url} to {repo_dir}. Output: {result}")
            return
        except subprocess.CalledProcessError as e:
            logger.error(f"Clone attempt {attempt + 1} failed: {e.stderr}")
            if attempt < max_attempts - 1:
                if "Connection reset by peer" in e.stderr or "early EOF" in e.stderr:
                    wait_time = (2**attempt) + (random.randint(0, 1000) / 1000)
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            else:
                raise


async def run_git_command(command, cwd=None):
    """Run a git command asynchronously with timeout."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        # Add timeout to prevent hanging
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Git command timed out after 300s: {' '.join(command)}")

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stdout.decode(), stderr.decode())
        return stdout.decode()
    except Exception as e:
        logger.error(f"Git command failed: {command} - {str(e)}")
        raise


async def maybe_clone_async(repo_url, repo_dir):
    """Clone a repo if it doesn't exist."""
    async with repo_operation_async_lock(repo_url):
        if not os.path.exists(f"{repo_dir}/.git"):
            logger.info(f"Cloning repo '{repo_url}' to '{repo_dir}'")
            try:
                if repo_url.startswith("file://") and not os.path.exists(repo_url[7:]):
                    repo_url = f"https://github.com/{repo_url.split('/')[-1]}.git"
                    logger.info(f"Converting to GitHub URL: {repo_url}")

                await run_git_command(["git", "clone", repo_url, repo_dir])
                logger.info(f"Repo '{repo_url}' was cloned to '{repo_dir}'")
            except Exception as e:
                logger.error(f"Clone failed: {e}")
                if os.path.exists(repo_dir):
                    import shutil

                    shutil.rmtree(repo_dir)
                raise ValueError(f"Failed to clone repo '{repo_url}' to '{repo_dir}'")


def maybe_clone(repo_url, repo_dir):
    with repo_operation_lock(repo_url):
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
    output = subprocess.run(["git", "diff"], cwd=repo_dir, check=True, text=True, capture_output=True)

    return output.stdout


def stage_all_files(repo_dir):
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, text=True, capture_output=True)


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
    with repo_operation_lock(repo_url):
        maybe_clone(repo_url, repo_dir)
        clean_and_reset_state(repo_dir)
        checkout_branch(repo_dir, branch_name)
        pull_latest(repo_dir)


def clean_and_reset_repo(repo_dir, branch_name="master", repo_url=None):
    if repo_url is None:
        # Try to get the repo URL from the git config
        try:
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
            repo_url = result.stdout.strip()
        except subprocess.CalledProcessError:
            logger.warning("Could not get repo URL, operations may not be thread-safe")
            repo_url = f"unknown_{os.path.basename(repo_dir)}"

    with repo_operation_lock(repo_url):
        clean_and_reset_state(repo_dir)
        checkout_branch(repo_dir, branch_name)
        pull_latest(repo_dir)


async def setup_github_repo_async(repo: str, base_commit: str, base_dir: str = "/tmp/repos") -> str:
    repo_name = get_repo_dir_name(repo)
    repo_url = f"https://github.com/{repo}.git"
    path = f"{base_dir}/{repo_name}"
    logger.info(f"Clone Github repo {repo_url} to {path} and checkout commit {base_commit}")
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory '{path}' was created.")

    with repo_operation_lock(repo_url):
        await maybe_clone_async(repo_url, path)
        await run_git_command(["git", "checkout", base_commit], path)
    return path


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
