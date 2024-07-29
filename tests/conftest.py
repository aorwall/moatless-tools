import logging
import os
from dotenv import load_dotenv
import pytest

load_dotenv()
logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--index-store-dir", action="store", default=os.getenv('INDEX_STORE_DIR', '/tmp/index-store'), help="Path for INDEX_STORE_DIR")
    parser.addoption("--repo-dir", action="store", default=os.getenv('REPO_DIR', '/tmp/repo'), help="Path for REPO_DIR")
    parser.addoption("--trajectory-dir", action="store", default=os.getenv('TRAJECTORY_DIR', '/tmp/trajectories'),
                     help="Path for TRAJECTORY_DIR")
    parser.addoption("--prompt-log-dir", action="store", default=os.getenv('PROMPT_LOG_DIR', '/tmp/prompt_logs'),
                        help="Path for PROMPT_LOG_DIR")
    parser.addoption(
        "--run-llm-integration", action="store_true", default=False,
        help="run integration tests that call LLMs"
    )
    parser.addoption(
        "--run-with-index", action="store_true", default=False,
        help="run tests that need vector store index files"
    )


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch, request):
    index_store_dir = request.config.getoption("--index-store-dir")
    repo_dir = request.config.getoption("--repo-dir")
    trajectory_dir = request.config.getoption("--trajectory-dir")
    prompt_log_dir = request.config.getoption("--prompt-log-dir")

    logger.info(f"Setting INDEX_STORE_DIR={index_store_dir}")
    logger.info(f"Setting REPO_DIR={repo_dir}")
    logger.info(f"Setting TRAJECTORY_DIR={trajectory_dir}")
    logger.info(f"Setting PROMPT_LOG_DIR={prompt_log_dir}")

    monkeypatch.setenv('INDEX_STORE_DIR', index_store_dir)
    monkeypatch.setenv('REPO_DIR', repo_dir)
    monkeypatch.setenv('TRAJECTORY_DIR', trajectory_dir)
    monkeypatch.setenv('PROMPT_LOG_DIR', prompt_log_dir)